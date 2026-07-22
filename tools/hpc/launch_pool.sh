#!/usr/bin/env bash
###############################################################################
# launch_pool.sh — Apptainer pool launcher for the CPU-RAG service on HPC.
#
# Replaces docker-compose on the cluster. Starts N identical RAG replicas plus
# one nginx load balancer, all as unprivileged Apptainer containers on a single
# compute node. Designed to be the body of an sbatch/srun allocation: it starts
# everything, then `wait`s and tears the pool down on exit/signal.
#
# Two facts drive the design vs the compose setup:
#   1. Apptainer shares the host network (no netns) — each replica binds its
#      port directly on the host; no port mapping. nginx talks to 127.0.0.1.
#   2. Unprivileged containers cannot bind ports <1024 — the LB uses a high port.
#
# --pwd /app is load-bearing, do not drop it. Apptainer auto-binds $HOME and
# inherits the host CWD, so from the repo root `uvicorn app.main:app` imports
# the *host's* app/ instead of the image's, and every relative default in
# config.py (./snapshots, ./models) resolves against the host repo — silently
# ignoring the binds below. It fails quietly rather than loudly, because the
# host repo has a ./snapshots/$PROFILE too: you get the host's app/ and the
# host's pkls, and nothing tells you the binds were ignored.
#
# Topology defaults to the validated nT8 N8: 8 replicas x 8 threads, one replica
# pinned per NUMA node (matches 8 NUMA x 8 cores on the Xeon Gold 6430 nodes;
# --membind keeps memory local, avoiding the -17/-20% cross-socket penalty).
#
# Prereqs (run once per shell, also fine in .bashrc):
#   module load singularity/1.4.1                       # apptainer on PATH
#   module load spack/1.1.0 && spack load numactl target=x86_64_v3
#   (numactl is not on the base PATH; without it replicas are NOT NUMA-pinned.
#    The target= disambiguates the v3/v4 builds; a bare `spack load` errors out.)
# Required env:
#   RAG_API_KEY          API key the service requires (no default).
# Common overrides (env vars):
#   PROFILE              Deployment profile to serve          (default aiciblock)
#   SIF_IMAGE            Path to the SPR-native RAG .sif      (default ./rag-spr-native.sif)
#   NGINX_SIF            Path to the nginx .sif               (default ./nginx.sif; built if missing)
#   MODELS_DIR           Host dir with the GGUF model         (default ./models)
#   SNAPSHOTS_DIR        Host *root* of the KV snapshots      (default ./snapshots;
#                        the pkls live in $SNAPSHOTS_DIR/$PROFILE — config.py appends it)
#   N_REPLICAS           Number of RAG replicas               (default 8)
#   N_THREADS            llama threads per replica            (default 8)
#   BASE_PORT            First replica port; replicas use BASE_PORT+i (default 8001)
#   LB_PORT              Port nginx listens on (>=1024)       (default 8080)
#   NUMA_START           First NUMA node; replica i -> NUMA_START+i (default 0)
#   STAGE_ROOT           Local scratch for per-replica snapshot staging (default /tmp/cpu-rag)
#   APP_DIR              Host app/ to mount over the image's       (default: unset)
#
# APP_DIR is the prompt-iteration escape hatch. app/ is baked into the SIF, so
# editing app/prompt.py normally costs a Docker rebuild on a laptop, an scp of
# the tar and an `apptainer build` — half an hour to change one string. Setting
#
#     APP_DIR=./app tools/hpc/launch_pool.sh
#
# mounts the working tree over /app/app instead, so the cycle becomes edit ->
# regenerate snapshot -> relaunch. Snapshots need no manual invalidation: their
# cache key hashes the system prompt (app/snapshot_cache.py), so a changed
# prompt is simply a different key and old snapshots stay valid and reusable.
#
# Deliberately opt-in and never a default: the point of baking app/ is that the
# shipped image is self-contained and reproducible. Leave it unset for anything
# that is not an experiment.
###############################################################################
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# --- config ----------------------------------------------------------------
: "${RAG_API_KEY:?Set RAG_API_KEY (the service requires it)}"
# The profile decides what the replicas serve; it must reach them as an env var
# (the image bakes app/config.py, whose default is aiciblock). It also picks the
# snapshots subdir, but that resolution now happens inside config.py — we bind
# the root and the container appends the profile. Same layout as ./run.sh.
PROFILE="${PROFILE:-aiciblock}"
SIF_IMAGE="${SIF_IMAGE:-./rag-spr-native.sif}"
NGINX_SIF="${NGINX_SIF:-./nginx.sif}"
MODELS_DIR="${MODELS_DIR:-./models}"
SNAPSHOTS_DIR="${SNAPSHOTS_DIR:-./snapshots}"
PROFILE_SNAPSHOTS="${SNAPSHOTS_DIR}/${PROFILE}"
N_REPLICAS="${N_REPLICAS:-8}"
N_THREADS="${N_THREADS:-8}"
BASE_PORT="${BASE_PORT:-8001}"
LB_PORT="${LB_PORT:-8080}"
NUMA_START="${NUMA_START:-0}"
STAGE_ROOT="${STAGE_ROOT:-/tmp/cpu-rag}"
NGINX_RUN="${STAGE_ROOT}/nginx"

# --- preflight --------------------------------------------------------------
command -v apptainer >/dev/null || { echo "ERROR: apptainer not in PATH. Run: module load singularity/1.4.1" >&2; exit 1; }
[[ -f "$SIF_IMAGE" ]] || { echo "ERROR: RAG image not found: $SIF_IMAGE" >&2
  echo "  Build the SPR-native flavor elsewhere and convert, e.g.:" >&2
  echo "    docker build --build-arg CMAKE_FLAGS='<VNNI/AMX flags>' -t cpu-rag-api:1.2.0-native ." >&2
  echo "    docker save cpu-rag-api:1.2.0-native -o rag.tar   # then scp to HPC" >&2
  echo "    apptainer build $SIF_IMAGE docker-archive://rag.tar" >&2
  exit 1; }
[[ -d "$MODELS_DIR" ]]    || { echo "ERROR: models dir not found: $MODELS_DIR" >&2; exit 1; }
[[ -d "$PROFILE_SNAPSHOTS" ]] || { echo "ERROR: snapshots dir not found: $PROFILE_SNAPSHOTS" >&2; exit 1; }
if ! ls "$PROFILE_SNAPSHOTS"/*.pkl >/dev/null 2>&1; then
  echo "ERROR: no *.pkl in $PROFILE_SNAPSHOTS — build snapshots first:" >&2
  echo "    apptainer exec --bind $MODELS_DIR:/app/models,$SNAPSHOTS_DIR:/app/snapshots \\" >&2
  echo "      --env RAG_API_KEY=\$RAG_API_KEY --env PROFILE=$PROFILE \\" >&2
  echo "      $SIF_IMAGE python -m app.generate" >&2
  exit 1
fi

# Optional override of the image's app/ — see APP_DIR in the header.
APP_BIND=()
if [[ -n "${APP_DIR:-}" ]]; then
  [[ -d "$APP_DIR" ]] || { echo "ERROR: APP_DIR is not a directory: $APP_DIR" >&2; exit 1; }
  [[ -f "$APP_DIR/main.py" ]] || { echo "ERROR: $APP_DIR has no main.py — expected the app package, not the repo root." >&2; exit 1; }
  APP_DIR="$(readlink -f "$APP_DIR")"
  APP_BIND=(--bind "${APP_DIR}:/app/app:ro")
  echo "==> APP_DIR override: serving ${APP_DIR} instead of the image's app/"
fi

NUMACTL=()
if command -v numactl >/dev/null; then
  NUMACTL_AVAILABLE=1
else
  echo "WARN: numactl not found — replicas will NOT be NUMA-pinned (expect cross-socket penalty)." >&2
  NUMACTL_AVAILABLE=0
fi

# Absolute paths (apptainer binds need them once we may have cd'd around).
SIF_IMAGE="$(readlink -f "$SIF_IMAGE")"
MODELS_DIR="$(readlink -f "$MODELS_DIR")"
SNAPSHOTS_DIR="$(readlink -f "$SNAPSHOTS_DIR")"

echo "==> Pool: profile ${PROFILE}, ${N_REPLICAS} replicas x ${N_THREADS} threads, LB on :${LB_PORT}"
echo "    snapshots: ${SNAPSHOTS_DIR}/${PROFILE}  (root bound; config.py appends the profile)"
mkdir -p "$STAGE_ROOT" "$NGINX_RUN"/{client_temp,proxy_temp,fastcgi_temp,uwsgi_temp,scgi_temp}

# --- nginx image (build from docker:// if missing — unprivileged build works) -
if [[ ! -f "$NGINX_SIF" ]]; then
  echo "==> nginx image missing; building $NGINX_SIF from docker://nginx:1.27-alpine"
  apptainer build "$NGINX_SIF" docker://nginx:1.27-alpine
fi
NGINX_SIF="$(readlink -f "$NGINX_SIF")"

# --- generate nginx.conf for a host-network pool (127.0.0.1:BASE_PORT+i) -----
NGINX_CONF="${NGINX_RUN}/nginx.conf"
{
  echo "pid ${NGINX_RUN}/nginx.pid;"
  echo "error_log ${NGINX_RUN}/error.log warn;"
  echo "events { worker_connections 256; }"
  echo "http {"
  echo "  client_body_temp_path ${NGINX_RUN}/client_temp;"
  echo "  proxy_temp_path        ${NGINX_RUN}/proxy_temp;"
  echo "  fastcgi_temp_path      ${NGINX_RUN}/fastcgi_temp;"
  echo "  uwsgi_temp_path        ${NGINX_RUN}/uwsgi_temp;"
  echo "  scgi_temp_path         ${NGINX_RUN}/scgi_temp;"
  echo "  access_log ${NGINX_RUN}/access.log;"
  echo "  upstream rag_pool {"
  echo "    least_conn;"
  for ((i=0; i<N_REPLICAS; i++)); do
    echo "    server 127.0.0.1:$((BASE_PORT + i)) max_fails=3 fail_timeout=30s;"
  done
  echo "  }"
  echo "  server {"
  echo "    listen ${LB_PORT};"
  echo "    location = /lb-health { access_log off; return 200 \"ok\\n\"; add_header Content-Type text/plain; }"
  echo "    location / {"
  echo "      proxy_pass http://rag_pool;"
  echo "      proxy_http_version 1.1;"
  echo "      proxy_set_header Host \$host;"
  echo "      proxy_set_header X-Real-IP \$remote_addr;"
  echo "      proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;"
  echo "      proxy_buffering off; proxy_cache off;"      # SSE: stream tokens as produced
  echo "      proxy_set_header Connection \"\";"
  echo "      chunked_transfer_encoding on;"
  echo "      proxy_read_timeout 1200s; proxy_send_timeout 1200s;"
  echo "    }"
  echo "  }"
  echo "}"
} > "$NGINX_CONF"

# --- teardown on exit/signal ------------------------------------------------
PIDS=()
cleanup() {
  echo ""; echo "==> Stopping pool"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- launch replicas --------------------------------------------------------
for ((i=0; i<N_REPLICAS; i++)); do
  port=$((BASE_PORT + i))
  node=$((NUMA_START + i))
  stage="${STAGE_ROOT}/snap-r${i}"
  mkdir -p "$stage"
  pin=()
  (( NUMACTL_AVAILABLE )) && pin=(numactl --cpunodebind="$node" --membind="$node")
  echo "==> replica r${i} -> :${port}  (NUMA ${node}, ${N_THREADS} threads)"
  "${pin[@]}" apptainer exec \
    --pwd /app \
    --bind "${MODELS_DIR}:/app/models:ro" \
    --bind "${SNAPSHOTS_DIR}:/app/snapshots:ro" \
    "${APP_BIND[@]}" \
    --env RAG_API_KEY="$RAG_API_KEY" \
    --env PROFILE="$PROFILE" \
    --env REPLICA_ID="r${i}" \
    --env N_THREADS="$N_THREADS" \
    --env OMP_NUM_THREADS="$N_THREADS" \
    --env GGML_N_THREADS="$N_THREADS" \
    --env OPENBLAS_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env SNAPSHOT_STAGE_DIR="$stage" \
    --env LOG_LEVEL="${LOG_LEVEL:-INFO}" \
    "$SIF_IMAGE" \
    uvicorn app.main:app --host 127.0.0.1 --port "$port" \
    >"${STAGE_ROOT}/r${i}.log" 2>&1 &
  PIDS+=($!)
done

# --- launch nginx LB --------------------------------------------------------
echo "==> nginx LB -> :${LB_PORT}"
apptainer exec --bind "${NGINX_RUN}:${NGINX_RUN}" "$NGINX_SIF" \
  nginx -c "$NGINX_CONF" -g 'daemon off;' \
  >"${STAGE_ROOT}/nginx.log" 2>&1 &
PIDS+=($!)

NODE_HOST="${SLURMD_NODENAME:-$(hostname 2>/dev/null || cat /proc/sys/kernel/hostname)}"
echo ""
echo "==> Pool up. Logs in ${STAGE_ROOT}/. Reach it at:"
echo "      http://${NODE_HOST}:${LB_PORT}/  (set RAG host bind / firewall as needed)"
echo "    Health: curl http://${NODE_HOST}:${LB_PORT}/lb-health"
echo "    Ctrl-C (or scancel) to stop the pool."
wait
