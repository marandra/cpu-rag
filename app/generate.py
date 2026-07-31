"""One-shot snapshot generation CLI.

Pre-builds the on-disk KV snapshot cache so a subsequent server start
is HIT-only (no ~80s warmup per procedure). Idempotent: cached
snapshots are skipped via the same fingerprint logic used at startup.
Honors `PROCEDURE_FILTER` to target a single procedure.

Invocation (same image, alternate CMD):

    python -m app.generate
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.snapshot_builder import build_or_load_snapshot, select_procedures

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("app.generate")


def main() -> int:
    from src.llm import load_model

    # A no-op rather than an error, so a bootstrap script can call this
    # unconditionally without knowing the mode. In "memory" the warm happens at
    # server startup; in "off" there is nothing to warm.
    if settings.snapshot_mode != "disk":
        logger.info(
            f"snapshot_mode={settings.snapshot_mode!r} builds no pickles; "
            f"nothing to generate."
        )
        return 0

    procedures = select_procedures(settings)
    logger.info(f"Generating snapshots for: {sorted(procedures)}")

    fulldoc_texts: dict[str, str] = {}
    fulldoc_paths: dict[str, Path] = {}
    for proc, path in procedures.items():
        text = path.read_text(encoding="utf-8")
        fulldoc_texts[proc] = text
        fulldoc_paths[proc] = path
        logger.info(
            f"Fulldoc loaded: procedure={proc!r} chars={len(text)} path={path}"
        )

    logger.info(f"Loading LLM from {settings.model_path}...")
    load_kwargs: dict = {
        "path": str(settings.model_path),
        "n_ctx": settings.n_ctx,
    }
    if settings.n_threads is not None:
        load_kwargs["n_threads"] = settings.n_threads
        logger.info(f"Using n_threads={settings.n_threads} (overridden)")
    llm = load_model(**load_kwargs)
    logger.info(f"LLM ready: {Path(settings.model_path).stem}")

    t0 = time.perf_counter()
    summary: list[tuple[str, bool]] = []
    for proc, text in fulldoc_texts.items():
        state, was_cached = build_or_load_snapshot(
            llm, proc, text, settings, fulldoc_path=fulldoc_paths[proc]
        )
        if state is None:
            logger.error(f"Snapshot generation failed for procedure={proc!r}")
            return 1
        summary.append((proc, was_cached))

    total = time.perf_counter() - t0
    cached = sum(1 for _, c in summary if c)
    built = len(summary) - cached
    logger.info(
        f"Done in {total:.1f}s — {built} built, {cached} cached "
        f"(procedures: {[p for p, _ in summary]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
