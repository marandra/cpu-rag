"""Shared snapshot build/load logic.

Used by both the server lifespan startup (`app.main`) and the one-shot
generation CLI (`app.generate`). A snapshot is the LlamaState captured
immediately after warming a model with `(system + fulldoc)` for a given
procedure; on this CPU each warm costs ~80s, so the on-disk pickle cache
in `app.snapshot_cache` is what makes container restarts cheap.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_meta(
    settings,
    procedure: str,
    snapshot_key: str,
    fulldoc_path: Path,
    fulldoc_text: str,
) -> None:
    """Write/refresh the sidecar describing this snapshot. Best-effort."""
    from app.snapshot_cache import write_meta

    write_meta(
        snapshots_dir=settings.snapshots_dir,
        procedure=procedure,
        snapshot_key=snapshot_key,
        model_path=settings.model_path,
        n_ctx=settings.n_ctx,
        fulldoc_path=fulldoc_path,
        fulldoc_text=fulldoc_text,
    )


def select_procedures(settings) -> dict[str, Path]:
    """Honor `settings.procedure_filter`; else return all configured procedures."""
    if settings.procedure_filter:
        if settings.procedure_filter not in settings.fulldoc_procedures:
            raise RuntimeError(
                f"procedure_filter={settings.procedure_filter!r} not in "
                f"fulldoc_procedures keys {sorted(settings.fulldoc_procedures)}"
            )
        return {
            settings.procedure_filter: settings.fulldoc_procedures[
                settings.procedure_filter
            ]
        }
    return dict(settings.fulldoc_procedures)


def build_or_load_snapshot(
    llm,
    procedure: str,
    fulldoc_text: str,
    settings,
    fulldoc_path: Path | None = None,
) -> tuple[Any | None, bool]:
    """Return (state, was_cached). state is None only if save_state failed.

    Side effects: mutates the LLM's KV (warm pass or load_state). On a
    cache miss, also writes the pickle to `settings.snapshots_dir`. Always
    refreshes the per-procedure sidecar (idempotent) so the server can
    discover the snapshot without loading it.
    """
    from app.prompt import get_system_prompt
    from app.snapshot_cache import (
        cache_path,
        compute_key,
        load_snapshot,
        save_snapshot,
    )

    system_prompt = get_system_prompt(procedure)
    key = compute_key(
        model_path=settings.model_path,
        n_ctx=settings.n_ctx,
        flash_attn=True,
        system_prompt=system_prompt,
        fulldoc_text=fulldoc_text,
    )
    path = cache_path(settings.snapshots_dir, key)

    if path.exists():
        logger.info(f"Snapshot cache HIT (procedure={procedure!r}): {path.name}")
        state = load_snapshot(path)
        if state is not None:
            try:
                llm.load_state(state)
            except Exception:
                logger.exception(
                    f"load_state failed for cached snapshot {path.name}; "
                    f"falling back to live warm"
                )
            else:
                if fulldoc_path is not None:
                    _ensure_meta(settings, procedure, key, fulldoc_path, fulldoc_text)
                return state, True

    logger.info(
        f"Snapshot cache MISS (procedure={procedure!r}); warming "
        f"(chars={len(fulldoc_text)})..."
    )
    t0 = time.perf_counter()
    # Must match byte-for-byte the prefix used by /query so the cached KV
    # covers everything up to the user question tokens.
    llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: hola"},
        ],
        max_tokens=1,
        temperature=0.1,
    )
    state = llm.save_state()
    warm_s = time.perf_counter() - t0
    logger.info(
        f"Warmed (procedure={procedure!r}) in {warm_s:.1f}s; pickling snapshot..."
    )
    if save_snapshot(state, path):
        logger.info(f"Snapshot saved: {path.name}")
        if fulldoc_path is not None:
            _ensure_meta(settings, procedure, key, fulldoc_path, fulldoc_text)
    return state, False
