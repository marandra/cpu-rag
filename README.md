# cpu-rag

CPU-only Q&A service for patient questions about clinical procedures. Each procedure has
a single distilled markdown ("fulldoc") that is sent in full as context to a local
llama.cpp model. No retrieval, no embeddings, no vector DB.

The retrieval-based variant of this project lives in the sibling repo
[`gpu-rag`](../gpu-rag/) (chunking, embeddings, Qdrant, reranker, eval sweeps).

## How it works

```
corpus/markdown/{procedure}/distilled.md ──► load into RAM at startup
                                                │
                                                v
                                  prime KV cache with (SYSTEM + fulldoc)
                                                │
                                                v
            POST /query  ──►  reuse cached prefix + PREGUNTA: <q>  ──►  SSE stream
```

The KV-cache prefix match means the (system + fulldoc) prefix is paid once at warmup and
reused per query. Only the question and the answer cost generation time.

## Endpoints

| Method | Path      | Auth | Description                                                            |
| ------ | --------- | :--: | ---------------------------------------------------------------------- |
| `GET`  | `/health` |  no  | Status, model name, procedures loaded                                  |
| `POST` | `/query`  | yes  | SSE stream (`chunk` / `done` / `error`). Body: `{question, procedure}` |

Auth: `X-API-Key` header (`RAG_API_KEY` in `.env`).

## Configuration

All in `app/config.py` (Settings). Override via environment / `.env`.

| Setting              | Default                               | Notes                                           |
| -------------------- | ------------------------------------- | ----------------------------------------------- |
| `rag_api_key`        | —                                     | Required                                        |
| `model_path`         | `./models/Ministral-3-3B-Q4_K_M.gguf` | GGUF path                                       |
| `n_ctx`              | 32768                                 | Must cover system + fulldoc + question + answer |
| `n_batch`            | 512                                   |                                                 |
| `max_tokens`         | 320                                   | Generation cap                                  |
| `fulldoc_procedures` | `{"diabetes": Path(...)}`             | procedure → markdown path                       |

Generation temperature is fixed at 0.1 in `app/routes/query.py`.

## Procedures

Currently wired:

| Procedure  | Fulldoc                                                       | Tokens |
| ---------- | ------------------------------------------------------------- | -----: |
| `diabetes` | `corpus/markdown/diabetes/GUIA_DIABETES_v3_distilled_3429.md` |  3,429 |

Other procedure source markdowns live under
`corpus/markdown/{hemorroides,fisura-anal,cirugia-abdominal,general}/` and are pending
distillation to fulldoc size (target 2–4K tokens).

## Adding a procedure

1. Distill the source markdown down to ~2–4K tokens (drop bold markup, heading
   numbering; keep semantic content). See `docs/prompt_versions.md` and
   `corpus/markdown/diabetes/` for the reference flow.
2. Drop the file in `corpus/markdown/{procedure}/`.
3. Add an entry to `Settings.fulldoc_procedures` in `app/config.py`.
4. Rebuild and restart the container — warmup will prime the KV cache for the new
   procedure.

## Running

```bash
# Local (uses ./models, ./corpus)
uv sync
uv run uvicorn app.main:app --reload

# Docker
docker compose up -d --build
```

Warmup runs once per procedure at startup and takes ~80s for the 3.4K diabetes fulldoc
on the reference CPU. The service is ready when `/health` returns `status: "healthy"`.

## Repo layout

```
app/
  main.py            FastAPI app + lifespan (model load, KV warmup)
  config.py          Settings
  prompt.py          V13 system prompt template
  auth.py            X-API-Key middleware
  schemas.py         Pydantic models
  routes/
    health.py
    query.py         SSE streaming
src/
  llm.py             llama-cpp wrapper, chat template handling
corpus/markdown/     Source + distilled markdown per procedure
models/              GGUF model files (symlinked into gpu-rag)
tools/
  bench_kv_context.py
  count_tokens.py
  demo_rag.py        Eval-dataset driver against /query
  rag_client.py      HTTP client helpers
  run_eval.py        Baseline eval → reports/
  hpc/               Apptainer pool launcher (HPC)
  sweep/             Performance sweep jobs
eval/
  datasets/          Per-procedure coverage + grayzone eval datasets
reports/             Generated eval + sweep outputs (gitignored)
docs/                Historical notes (HPC setup, prompt history, lessons)
```

## Prompt

V13 (`app/prompt.py`). Procedure-agnostic, single-doc fulldoc framing with `INFORMACIÓN`
/ `PREGUNTA`. Iteration history in `docs/prompt_versions.md`.
