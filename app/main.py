"""
FastAPI application with lifespan management.

Models are loaded once at startup and shared across requests.
"""

import asyncio
import logging
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AppState:
    llm: object = None
    model_name: str = ""
    procedures: set = field(default_factory=set)
    fulldoc_texts: dict = field(default_factory=dict)  # procedure -> full markdown text
    snapshots: dict = field(default_factory=dict)  # procedure -> LlamaState
    # Serializes generation across requests: one Llama, one live KV state.
    gen_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app_state = AppState()


def _load_models():
    """Load fulldoc markdowns and the LLM, then build per-procedure KV snapshots."""
    from src.llm import load_model

    if settings.procedure_filter:
        if settings.procedure_filter not in settings.fulldoc_procedures:
            raise RuntimeError(
                f"procedure_filter={settings.procedure_filter!r} not in "
                f"fulldoc_procedures keys {sorted(settings.fulldoc_procedures)}"
            )
        procedures = {
            settings.procedure_filter: settings.fulldoc_procedures[
                settings.procedure_filter
            ]
        }
        logger.info(f"Procedure filter active: only loading {settings.procedure_filter!r}")
    else:
        procedures = settings.fulldoc_procedures

    for proc, path in procedures.items():
        text = path.read_text(encoding="utf-8")
        app_state.fulldoc_texts[proc] = text
        app_state.procedures.add(proc)
        logger.info(
            f"Fulldoc loaded: procedure={proc!r} chars={len(text)} path={path}"
        )

    logger.info(f"Loading LLM from {settings.model_path}...")
    load_kwargs = {"path": str(settings.model_path), "n_ctx": settings.n_ctx}
    if settings.n_threads is not None:
        load_kwargs["n_threads"] = settings.n_threads
        logger.info(f"Using n_threads={settings.n_threads} (overridden)")
    app_state.llm = load_model(**load_kwargs)
    app_state.model_name = Path(settings.model_path).stem
    logger.info(f"LLM ready: {app_state.model_name}")

    _build_snapshots()
    logger.info(f"Snapshots ready for procedures: {sorted(app_state.snapshots)}")


def _build_snapshots() -> None:
    """Build a KV snapshot per procedure.

    For each procedure: check the on-disk cache (Phase 2). Hit → unpickle.
    Miss → live-warm the LLM with the exact `(system + fulldoc)` prefix used
    by `/query`, then `save_state()` and pickle. Snapshots are stored in
    `app_state.snapshots[procedure]` and loaded into the live LLM on each
    request. The LLM's post-loop state is irrelevant — every request begins
    with `load_state`.
    """
    from app.prompt import get_system_prompt
    from app.snapshot_cache import (
        cache_path,
        compute_key,
        load_snapshot,
        save_snapshot,
    )

    snapshots_dir = settings.snapshots_dir

    for proc, text in app_state.fulldoc_texts.items():
        system_prompt = get_system_prompt(proc)
        key = compute_key(
            model_path=settings.model_path,
            n_ctx=settings.n_ctx,
            flash_attn=True,
            system_prompt=system_prompt,
            fulldoc_text=text,
        )
        path = cache_path(snapshots_dir, key)

        if path.exists():
            logger.info(
                f"Snapshot cache HIT (procedure={proc!r}): {path.name}"
            )
            state = load_snapshot(path)
            if state is not None:
                try:
                    app_state.llm.load_state(state)
                except Exception:
                    logger.exception(
                        f"load_state failed for cached snapshot {path.name}; "
                        f"falling back to live warm"
                    )
                else:
                    app_state.snapshots[proc] = state
                    continue

        logger.info(
            f"Snapshot cache MISS (procedure={proc!r}); warming "
            f"(chars={len(text)})..."
        )
        t0 = time.perf_counter()
        # Must match the byte-for-byte prefix used by /query so the cached
        # KV covers everything up to the user's question tokens.
        app_state.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"INFORMACIÓN:\n{text}\n\nPREGUNTA: hola"},
            ],
            max_tokens=1,
            temperature=0.1,
        )
        # Capture immediately, before any other call mutates KV.
        state = app_state.llm.save_state()
        app_state.snapshots[proc] = state
        warm_s = time.perf_counter() - t0
        logger.info(
            f"Warmed (procedure={proc!r}) in {warm_s:.1f}s; pickling snapshot..."
        )
        if save_snapshot(state, path):
            logger.info(f"Snapshot saved: {path.name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    _load_models()
    logger.info("All models loaded, server ready")
    yield
    logger.info("Shutting down...")
    logger.info("Shutdown complete")


app = FastAPI(
    title="CPU-RAG API",
    description="Medical FAQ RAG API with streaming responses (fulldoc mode)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

from app.routes.health import router as health_router
from app.routes.query import router as query_router

app.include_router(health_router)
app.include_router(query_router)
