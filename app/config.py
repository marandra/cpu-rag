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
# v2.2 sirve los tres documentos tuteados. La conversión es estrictamente
# gramatical —pronombres y conjugación, ni una palabra de contenido clínico— y se
# verificó por documento que encabezados, viñetas y todas las cifras quedan
# idénticos. Medido en EC2 sobre las 134: 123/134 de decisión, el mejor de todos
# los brazos, con 0 fugas de registro (32 respuestas en tú, ninguna en usted).
# `eval/d1c-tu/`.
#
# El trato lo fija el CORPUS, no el prompt: con estos documentos en usted salían
# 0 de 89 respuestas en tú por mucho que los ejemplos del prompt tutearan. Por eso
# el literal de abstención (`prompt_variant`) tiene que girar con ellos.
#
# Las versiones en usted que sirvió la v2/v2.1 siguen en el disco y archivadas en
# `corpus/archive/*_v2servido_2026-07-27.md`; volver atrás es cambiar estas rutas
# y `prompt_variant` a "d1c".
# Where the per-request KV prefix comes from. See `Settings.snapshot_mode`.
SNAPSHOT_MODES = ("disk", "memory", "off")

PROFILES: dict[str, dict[str, Path]] = {
    "glucowise": {
        # v5 = la v4 destilada, tuteada. (v4 batió a la v1: decisión 85.5% vs
        # 83.4%, telegráficas 7% vs 11%.)
        "diabetes": Path("./corpus/markdown/diabetes.v5-tu.md"),
    },
    "aiciblock": {
        # v2 = la reescritura vA, tuteada. (vA era solo de forma, mismos hechos,
        # sin invención; arregló la frontera 108 y bajó las telegráficas 56%->23%.)
        "hemorroides": Path("./corpus/markdown/hemorroides.v2-tu.md"),
        # v2 = el original tuteado. La destilación v4 se midió peor (88.7% vs
        # 93.8%, telegráficas x5) y no entra.
        "cirugia-abdominal": Path("./corpus/markdown/cirugia-abdominal.v2-tu.md"),
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
    # Validated in app.prompt and not here, because config must not import
    # prompt (prompt imports this). Una variante cambia la clave del snapshot,
    # así que cada una calienta su propio pickle y nunca colisionan.
    #
    # v2.2 sirve "d1c-tu": el literal reescrito, tuteado para acompañar al
    # corpus (ver PROFILES). Medido en EC2 sobre las 134 (2026-07-27):
    # 123/134, y recupera 2 de los 3 "Sí/No" iniciales que el literal costaba
    # en usted. Las 42 abstenciones dejan de ser "No tengo información sobre
    # eso.", que es lo prometido en la respuesta a la auditoría.
    # `eval/d1c-tu/`. La v2.1 servía "d1c", el mismo texto en usted.
    #
    # "v13" sigue aquí y NO se toca: es lo que sirve la v2 entregada, y su
    # snapshot solo se reproduce con el prompt byte a byte.
    prompt_variant: str = "d1c-tu"

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

    @field_validator("snapshot_mode")
    @classmethod
    def _known_snapshot_mode(cls, v: str) -> str:
        if v not in SNAPSHOT_MODES:
            raise ValueError(
                f"Unknown snapshot_mode {v!r}. Known modes: {sorted(SNAPSHOT_MODES)}"
            )
        return v

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

    # Where the per-request KV prefix comes from.
    #
    #   "memory" — warm each procedure once at startup and keep its LlamaState
    #              in RAM. Default, because it removes the generate job, the
    #              ~0.5 GB pickle per procedure and the staging copy, and makes
    #              a corpus edit a restart instead of a rebuild.
    #   "disk"   — pickles in snapshots_dir, unpickled and loaded per request.
    #              What the delivered v1.1/v2.x bundles serve. Still the right
    #              choice for a pool: "memory" makes every replica re-warm on
    #              every start (~80 s per procedure), and N replicas warming at
    #              once on one socket is far worse than N times that.
    #   "off"    — no state at all; each request re-prefills its document
    #              (~70 s here). Diagnostic only on CPU.
    #
    # NOT chosen for request-path speed. Two runs on this laptop, same model and
    # same procedures, disagree on the ordering — memory 180-222 ms vs disk
    # 332-436 ms on 2026-07-30, memory 453-630 ms vs disk 388-422 ms on
    # 2026-07-31 — and they ran under different CPU governors (balanced, then
    # performance), so they are not comparable at all. Neither settles it;
    # a 200 ms difference has to be measured on the target box.
    # What is not noise: "off" costs a full re-prefill (~70 s) whenever the
    # procedure changes, and a pool re-warms once per replica in "memory".
    #
    # The three modes came from the GPU port — see the sibling repo ../gpu-rag,
    # where the same measurement on an A10G kills the machinery outright,
    # because there a re-prefill is 0.2-0.6 s.
    #
    # One behavioural difference worth knowing: a pickle carries a frozen RNG
    # state, so "disk" replays byte-identically across runs. "memory" warms
    # live, so its answers are only deterministic within a process. Nothing in
    # the audited numbers depends on that, but a byte-diff of two runs does.
    snapshot_mode: str = "memory"

    # At startup, copy pkls from snapshots_dir to this directory and read
    # from there. Decouples request-path I/O from the original (possibly
    # NFS-mounted) snapshots_dir. Empty string disables staging.
    # Only meaningful when snapshot_mode == "disk".
    snapshot_stage_dir: str = "/tmp/cpu-rag-snapshots"

    # Identifies this replica in logs and metrics. Set by compose to e.g.
    # "rag-1", "rag-2". Defaults to "rag" for single-instance setups.
    replica_id: str = "rag"

    # Logging
    log_level: str = "INFO"


settings = Settings()
