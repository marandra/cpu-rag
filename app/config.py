"""
Configuration via pydantic-settings.

Loads from environment variables with .env support.
All settings have sensible defaults except RAG_API_KEY.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required (no default) — env var: RAG_API_KEY
    rag_api_key: str

    # Paths (defaults work in container)
    model_path: Path = Path("./models/Ministral-3-3B-Q4_K_M.gguf")

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Timeouts (seconds)
    generation_timeout: float = 1200.0
    startup_timeout: float = 120.0

    # LLM
    n_ctx: int = 8192
    n_batch: int = 512
    max_tokens: int = 320
    # Optional override for llama-cpp n_threads. None → use src.llm default
    # (min(cpu_count, 9)). Set when pinning to a cpuset smaller than that to
    # avoid thread oversubscription.
    n_threads: int | None = None
    # Optional single-procedure filter. When set, only this procedure is
    # loaded from fulldoc_procedures. Used by multi-instance deployments
    # that pin each procedure to a dedicated cpuset.
    procedure_filter: str | None = None

    # Fulldoc mode: one distilled markdown per procedure, sent in full as
    # context. KV cache prefix-matching makes the (system + fulldoc) prefix
    # cheap to reuse across queries for the same procedure.
    # Diabetes is served by a separate instance and is deliberately absent
    # here, even though corpus/markdown/diabetes.md is still in the repo.
    fulldoc_procedures: dict[str, Path] = {
        "hemorroides": Path("./corpus/markdown/hemorroides.md"),
        "cirugia-abdominal": Path("./corpus/markdown/cirugia-abdominal.md"),
    }

    # Per-procedure KV snapshots (Llama.save_state) are pickled here so we
    # don't re-pay ~80s warmup per procedure on every container start.
    snapshots_dir: Path = Path("./snapshots")

    # At startup, copy pkls from snapshots_dir to this directory and read
    # from there. Decouples request-path I/O from the original (possibly
    # NFS-mounted) snapshots_dir. Empty string disables staging.
    snapshot_stage_dir: str = "/tmp/cpu-rag-snapshots"

    # Identifies this replica in logs and metrics. Set by compose to e.g.
    # "rag-1", "rag-2". Defaults to "rag" for single-instance setups.
    replica_id: str = "rag"

    # Logging
    log_level: str = "INFO"


settings = Settings()
