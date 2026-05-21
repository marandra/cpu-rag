"""Disk cache for per-procedure llama-cpp KV snapshots.

A snapshot is the LlamaState captured immediately after warming the model
with `(system + fulldoc)`. Loading a snapshot at request time avoids the
~80s prompt-eval pass over the fulldoc prefix.

Cache key is a fingerprint of every input that affects the KV state:
model file identity, n_ctx, flash_attn flag, system prompt text, fulldoc
text. Any change → miss → re-warm + re-save.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_key(
    model_path: Path,
    n_ctx: int,
    flash_attn: bool,
    system_prompt: str,
    fulldoc_text: str,
) -> str:
    """Stable SHA256 over inputs that affect the snapshot.

    The model is fingerprinted by (path, size, mtime_ns) — fast and
    sufficient to detect swaps or edits without hashing GBs.
    """
    h = hashlib.sha256()
    stat = model_path.stat()
    h.update(str(model_path.resolve()).encode())
    h.update(f"|size={stat.st_size}|mtime_ns={stat.st_mtime_ns}".encode())
    h.update(f"|n_ctx={n_ctx}|flash_attn={int(flash_attn)}".encode())
    h.update(b"|system=")
    h.update(system_prompt.encode("utf-8"))
    h.update(b"|fulldoc=")
    h.update(fulldoc_text.encode("utf-8"))
    return h.hexdigest()


def cache_path(snapshots_dir: Path, key: str) -> Path:
    return snapshots_dir / f"{key}.pkl"


def load_snapshot(path: Path) -> Any | None:
    """Unpickle a snapshot. Returns None on any failure (caller falls back to live warm)."""
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        logger.exception(f"Snapshot load failed: {path}")
        return None


def meta_path(snapshots_dir: Path, procedure: str) -> Path:
    """Sidecar lives alongside the pkl, keyed by procedure name (not key).

    Procedure-keyed (not key-keyed) so the server can resolve a request's
    `procedure` to its snapshot file by a single sidecar read, without
    enumerating every pkl.
    """
    return snapshots_dir / f"{procedure}.meta.json"


def write_meta(
    snapshots_dir: Path,
    procedure: str,
    snapshot_key: str,
    model_path: Path,
    n_ctx: int,
    fulldoc_path: Path,
    fulldoc_text: str,
) -> bool:
    """Write the sidecar describing one snapshot. Returns True on success."""
    try:
        stat = model_path.stat()
        meta = {
            "procedure": procedure,
            "snapshot_pkl": f"{snapshot_key}.pkl",
            "model_path": str(model_path),
            "model_size": stat.st_size,
            "model_mtime_ns": stat.st_mtime_ns,
            "n_ctx": n_ctx,
            "fulldoc_path": str(fulldoc_path),
            "fulldoc_sha256": hashlib.sha256(fulldoc_text.encode("utf-8")).hexdigest(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        p = meta_path(snapshots_dir, procedure)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tmp.replace(p)
        return True
    except Exception:
        logger.exception(f"Sidecar write failed: procedure={procedure!r}")
        return False


def read_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception(f"Sidecar read failed: {path}")
        return None


def scan_meta(snapshots_dir: Path) -> dict[str, dict]:
    """Return {procedure: meta_dict} for every sidecar whose pkl exists.

    Silently skips orphaned sidecars (pkl deleted). The server uses this at
    startup to learn what it can serve; no LLM load required.
    """
    found: dict[str, dict] = {}
    if not snapshots_dir.is_dir():
        return found
    for meta_file in sorted(snapshots_dir.glob("*.meta.json")):
        meta = read_meta(meta_file)
        if not meta:
            continue
        pkl = snapshots_dir / meta.get("snapshot_pkl", "")
        if not pkl.is_file():
            logger.warning(
                f"Sidecar {meta_file.name} references missing pkl "
                f"{meta.get('snapshot_pkl')!r}; skipping"
            )
            continue
        found[meta["procedure"]] = meta
    return found


def save_snapshot(state: Any, path: Path) -> bool:
    """Pickle a snapshot atomically. Returns False on failure (caller logs and continues)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
        return True
    except Exception:
        logger.exception(f"Snapshot save failed: {path}")
        return False
