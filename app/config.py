"""
Configuration via pydantic-settings.

Loads from environment variables with .env support.
All settings have sensible defaults except RAG_API_KEY.
"""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Deployment profiles: each maps the procedures it owns to their distilled
# fulldoc markdown. One image ships every profile; the PROFILE env var picks
# the active one.
#
# The profiles are the two downstream projects sharing this codebase —
# glucowise (diabetes) and aiciblock (the rest). Keeping both here instead of
# forking means one code path to maintain; when the projects split for good,
# forking is just deleting the other entry.
# v2 serving decisions (Fase B, 2026-07-24). Reversion is a one-line change
# here — the originals are kept intact under the same directory, never
# overwritten, so pointing a Path back is the whole rollback.
PROFILES: dict[str, dict[str, Path]] = {
    "glucowise": {
        # v2: v4 distillation (better decision 85.5% vs 83.4%, telegraphy
        # 7% vs 11%; -5% decode at nT=8, not a blocker). Revert -> diabetes.md
        # (the v1 original, kept untouched).
        "diabetes": Path("./corpus/markdown/diabetes.v4.md"),
    },
    "aiciblock": {
        # v2: vA rewrite (form-only, same facts, no invention; fixes the 108
        # anticoagulant break, telegraphy 56%->23%). MUST_REFUSE re-derived
        # unchanged. Revert -> hemorroides.md (the v1 original, kept untouched).
        "hemorroides": Path("./corpus/markdown/hemorroides.vA.md"),
        # v2: served original kept — v4 measured worse (88.7% vs 93.8%,
        # telegraphy x5), does not enter v2.
        "cirugia-abdominal": Path("./corpus/markdown/cirugia-abdominal.md"),
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required (no default) — env var: RAG_API_KEY
    rag_api_key: str

    # Paths (defaults work in container). v2 serves gemma-4-26B (MoE 26B/~4B
    # active); the launcher does not pass MODEL_PATH, so this default is what
    # the pool serves. The 17 GB GGUF is not baked into the image — the `init`
    # stage downloads it (see B2/B3). Override via MODEL_PATH for local dev
    # (e.g. the Ministral GGUF that fits a laptop).
    model_path: Path = Path("./models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")

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
    # Active profile, keyed into PROFILES. Governs both what `app.generate`
    # builds snapshots for and what the server is allowed to serve, so a
    # glucowise instance can never answer an aiciblock procedure even if both
    # sets of snapshots end up in the same directory.
    profile: str = "aiciblock"

    # Optional single-procedure filter, applied on top of the profile. Used
    # by multi-instance deployments that pin each procedure to its own cpuset.
    procedure_filter: str | None = None

    # Which system prompt to serve, keyed into app.prompt.PROMPT_VARIANTS.
    # "v13" is what we deploy; the rest exist so the prompt A/B can put one
    # variant per worker process without editing code. Validated in app.prompt
    # and not here, because config must not import prompt (prompt imports this).
    # A variant changes the snapshot cache key, so each one warms its own
    # pickle and they never collide.
    prompt_variant: str = "v13"

    @field_validator("profile")
    @classmethod
    def _known_profile(cls, v: str) -> str:
        if v not in PROFILES:
            raise ValueError(
                f"Unknown profile {v!r}. Known profiles: {sorted(PROFILES)}"
            )
        return v

    @property
    def fulldoc_procedures(self) -> dict[str, Path]:
        """Procedures owned by the active profile.

        Fulldoc mode: one distilled markdown per procedure, sent in full as
        context. KV cache prefix-matching makes the (system + fulldoc) prefix
        cheap to reuse across queries for the same procedure.
        """
        return dict(PROFILES[self.profile])

    # Per-procedure KV snapshots (Llama.save_state) are pickled under here so
    # we don't re-pay ~80s warmup per procedure on every container start.
    # This is the *root*: the pickles themselves live in a per-profile subdir
    # (see `snapshots_dir` below).
    snapshots_root: Path = Path("./snapshots")

    @property
    def snapshots_dir(self) -> Path:
        """This profile's snapshot directory: `<snapshots_root>/<profile>`.

        The per-profile scoping used to live only in the launchers, which bound
        `./snapshots/$PROFILE` onto the container's `/app/snapshots`. Anything
        importing this module from the *host* therefore resolved to the bare
        root — a directory no deployment serves — and silently built a parallel
        set of pickles there. Deriving the subdir here makes host and container
        agree, and the launchers now bind the root.
        """
        return self.snapshots_root / self.profile

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
