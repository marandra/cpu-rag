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
import logging
import pickle
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
