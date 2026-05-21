"""
FastAPI application with lifespan management.

Models are loaded once at startup and shared across requests.
"""

import asyncio
import logging
import shutil
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
    snapshot_paths: dict = field(default_factory=dict)  # procedure -> Path to pkl
    # Serializes generation across requests: one Llama, one live KV state.
    gen_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


app_state = AppState()


def _load_models():
    """Discover snapshots via sidecar scan, then load the LLM (no warm)."""
    from src.llm import load_model

    from app.snapshot_cache import scan_meta

    sidecars = scan_meta(settings.snapshots_dir)
    if not sidecars:
        raise RuntimeError(
            f"No snapshots found in {settings.snapshots_dir}. Run "
            f"`docker compose --profile generate run --rm rag-generate` first."
        )

    proc_filter = settings.procedure_filter
    if proc_filter:
        logger.info(f"Procedure filter active: only serving {proc_filter!r}")
        if proc_filter not in sidecars:
            raise RuntimeError(
                f"procedure_filter={proc_filter!r} not found in sidecars "
                f"{sorted(sidecars)}"
            )
        sidecars = {proc_filter: sidecars[proc_filter]}

    # Stage snapshots to a local directory if configured. This decouples
    # request-path reads from the (possibly NFS-backed) snapshots_dir.
    stage_dir = Path(settings.snapshot_stage_dir) if settings.snapshot_stage_dir else None
    if stage_dir:
        stage_dir.mkdir(parents=True, exist_ok=True)

    for proc, meta in sidecars.items():
        fulldoc_path = Path(meta["fulldoc_path"])
        text = fulldoc_path.read_text(encoding="utf-8")
        app_state.fulldoc_texts[proc] = text

        src_pkl = settings.snapshots_dir / meta["snapshot_pkl"]
        if stage_dir:
            dst_pkl = stage_dir / meta["snapshot_pkl"]
            src_stat = src_pkl.stat()
            try:
                dst_stat = dst_pkl.stat()
                fresh = (dst_stat.st_size == src_stat.st_size
                         and dst_stat.st_mtime_ns >= src_stat.st_mtime_ns)
            except FileNotFoundError:
                fresh = False
            if not fresh:
                t0 = time.perf_counter()
                shutil.copy2(src_pkl, dst_pkl)
                logger.info(
                    f"Staged snapshot procedure={proc!r} "
                    f"{src_pkl} -> {dst_pkl} "
                    f"({src_stat.st_size/1e6:.0f}MB in "
                    f"{(time.perf_counter()-t0)*1000:.0f}ms)"
                )
            app_state.snapshot_paths[proc] = dst_pkl
        else:
            app_state.snapshot_paths[proc] = src_pkl

        app_state.procedures.add(proc)
        logger.info(
            f"Discovered: procedure={proc!r} chars={len(text)} "
            f"pkl={app_state.snapshot_paths[proc]}"
        )

    logger.info(f"Loading LLM from {settings.model_path}...")
    load_kwargs = {"path": str(settings.model_path), "n_ctx": settings.n_ctx}
    if settings.n_threads is not None:
        load_kwargs["n_threads"] = settings.n_threads
        logger.info(f"Using n_threads={settings.n_threads} (overridden)")
    app_state.llm = load_model(**load_kwargs)
    app_state.model_name = Path(settings.model_path).stem
    logger.info(
        f"LLM ready: {app_state.model_name}; serving procedures: "
        f"{sorted(app_state.procedures)} (snapshots loaded lazily per request)"
    )


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
