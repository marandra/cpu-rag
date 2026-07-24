# CPU-RAG fulldoc — deliverable v2

Self-contained, CPU-only deployment of the medical FAQ RAG service. Runs on any
modern x86_64 Linux host with Docker (EC2, on-prem, VM).

> **What changed since v1.1**
> - **Model:** gemma-4-26B-A4B (a 26B Mixture-of-Experts, ~4B active) replaces
>   the 3B Ministral. Markedly better answer/refusal decisions and far fewer
>   telegraphic answers, at CPU speeds that stay above the usable floor.
> - **Corpus:** diabetes now serves the richer **v4** document; hemorroides
>   serves the **vA** rewrite (fixes the anticoagulant-instruction defect, much
>   less telegraphic); cirugía-abdominal unchanged.
> - **Packaging:** the 17 GB model is **no longer shipped inside the bundle** —
>   the bootstrap downloads it on first run. The image itself stays ~320 MB.
> - **Profiles:** one image, configured per project (see below).

This is the **portable** image flavor (AVX2 + FMA) — it runs on any x86_64 CPU.
A native flavor (AVX-512 VNNI + AMX-INT8) that gives ~+15–30 % decode on Intel
Sapphire/Emerald Rapids (e.g. EC2 m7i/c7i) is available on request.

## Two projects, one image

The service is configured per **profile**:

| profile     | serves                          | default port |
|-------------|---------------------------------|:------------:|
| `glucowise` | diabetes                        | 8001         |
| `aiciblock` | hemorroides + cirugía-abdominal | 8002         |

In production, run **one profile per server**. For evaluation you can run
**both on one host** — they get separate ports, containers and snapshots and
**share the single downloaded model**.

Each profile is served by **N replicas behind an nginx load balancer**
(`N_REPLICAS`, default **1**). Concurrency == replicas: each replica serializes
one generation at a time, so more replicas = more concurrent users. The topology
is the same at N=1 and N=8 — only the number changes, so the trial and
production run the identical stack. **Each replica loads its own ~16 GB copy of
the model**, so budget ~16–20 GB RAM per replica and keep
`N_THREADS * N_REPLICAS ≤ physical cores`.

## Contents

```
rag-deliverable-v2/
  docker-compose.yml                       # scalable replicas + nginx LB, PROFILE-parametrized
  nginx/nginx.conf                         # load balancer (dynamic upstream)
  profiles/glucowise.env                   # PROFILE=glucowise, RAG_PORT=8001
  profiles/aiciblock.env                   # PROFILE=aiciblock, RAG_PORT=8002
  .env.example                             # copy to .env, set RAG_API_KEY
  fetch_model.sh                           # downloads the model (once, shared)
  load_and_run.sh                          # bootstrap: fetch -> load -> generate -> up --scale
  images/cpu-rag-api-2.0.0-portable.tar    # docker image (~320 MB)
  models/                                   # empty; the model lands here on first run
  corpus/markdown/                          # the 3 served procedure documents
  snapshots/                                # empty; KV snapshots built on first boot
  README.md
```

The model and snapshots are **not** shipped. The model (~16.9 GB) is downloaded
from a public Hugging Face repo on first run (no account/token needed). The
snapshots are KV-cache states tied to this image's exact llama-cpp version, so
they are generated once on the target host.

## Prerequisites

- Docker Engine with the Compose plugin (`docker compose version`)
- **RAM:** ~20 GB per running profile (the model is ~16.9 GB resident). Running
  **both** profiles on one host needs ~40 GB.
- **Disk:** ~25 GB (17 GB model + ~1–2 GB snapshots + image)
- Outbound HTTPS to `huggingface.co` on first run (for the model download)
- More CPU cores = faster generation and higher decode throughput

## Quick start

```bash
# 1. Unpack onto the host
sudo mkdir -p /opt/rag && sudo tar -xzf rag-deliverable-v2.tar.gz -C /opt/rag --strip-components=1
cd /opt/rag

# 2. Configure
cp .env.example .env
# edit .env and set RAG_API_KEY to a long random secret

# 3. Bring up a profile (downloads the model, builds snapshots, starts serving)
./load_and_run.sh glucowise        # diabetes on :8001
# and/or, to also test the other project on the same box:
./load_and_run.sh aiciblock        # hemorroides + cirugía on :8002
```

`load_and_run.sh` is idempotent — re-running skips the model download, skips
already-built snapshots, and restarts the service.

## Manual steps (what the script does)

```bash
P=glucowise                        # or aiciblock
ENVF=profiles/$P.env

# Download the model once (shared by both profiles), verified by sha256.
./fetch_model.sh

# Load the image.
docker load -i images/cpu-rag-api-2.0.0-portable.tar

# The container runs as non-root (uid 1001); make snapshots/ writable by it.
# (best-effort: a previously-built profile's subdir is owned by 1001, not you)
chmod 777 snapshots; chmod -R 777 snapshots 2>/dev/null || true

# Build the KV snapshots once. REQUIRED before `up`.
docker compose --env-file $ENVF -p rag-$P --profile generate run --rm rag-generate

# Start serving: N replicas behind the LB (default 1).
docker compose --env-file $ENVF -p rag-$P up -d --scale rag=${N_REPLICAS:-1}

# Check health (through the load balancer).
curl http://localhost:8001/health     # 8002 for aiciblock
```

## Verify

```bash
curl http://localhost:8001/health
# {"status":"healthy","model":"gemma-4-26B-A4B-it-UD-Q4_K_M",
#  "procedures":["diabetes"]}
```

Query (streaming SSE; requires the `X-API-Key` header):

```bash
curl -N -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAG_API_KEY" \
  -d '{"procedure":"diabetes","question":"¿Qué debo hacer si tengo una hipoglucemia?"}'
```

Response is a Server-Sent Events stream:

- `event: chunk` — `{"text": "..."}` incremental tokens
- `event: done` — `{"request_id":..., "usage": {"completion_tokens":..., "decode_tok_s":..., ...}}`
- `event: error` — `{"code":..., "detail":...}`

Request body: `procedure` (must belong to the running profile) and `question`
(1–1000 chars). Each query is stateless — no conversation memory. Questions
outside the loaded document are refused by design.

## Operations

```bash
P=glucowise; ENVF=profiles/$P.env
docker compose --env-file $ENVF -p rag-$P logs -f rag     # follow logs
docker compose --env-file $ENVF -p rag-$P restart rag     # restart
docker compose --env-file $ENVF -p rag-$P down            # stop
```

## Configuration (.env)

| Variable          | Default                        | Notes                                                            |
|-------------------|--------------------------------|------------------------------------------------------------------|
| `RAG_API_KEY`     | — (required)                   | Secret for the `X-API-Key` header.                               |
| `N_REPLICAS`      | `1`                            | Serving replicas per profile behind the LB. Concurrency == replicas; each holds its own ~16 GB model copy. |
| `N_THREADS`       | all cores (set by the script)  | Serving/decode threads per replica. Keep `N_THREADS * N_REPLICAS ≤` physical cores. |
| `RAG_GEN_THREADS` | `nproc` (set by the script)    | Threads for the one-shot snapshot generation.                    |
| `ALLOWED_ORIGINS` | `["http://localhost:3000"]`    | CORS origins, JSON list.                                          |

Port and profile come from `profiles/<profile>.env`, not `.env`.

## Notes

- **Swapping the served document:** edit the file under `corpus/markdown/`, then
  re-run the generate step — snapshots are content-fingerprinted and rebuild
  automatically.
- **Native (faster) image:** on Sapphire/Emerald Rapids, a VNNI+AMX build gives
  ~+15–30 % decode. Available on request; drop-in replacement for the image tar.
- **Model source:** `unsloth/gemma-4-26B-A4B-it-GGUF` on Hugging Face (public).
  `fetch_model.sh` can point at an internal mirror via `MODEL_URL` (the sha256
  is still enforced).
