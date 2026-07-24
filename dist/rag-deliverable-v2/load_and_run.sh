#!/usr/bin/env bash
# CPU-RAG fulldoc — deliverable v2 bootstrap.
#
#   ./load_and_run.sh <profile>        glucowise | aiciblock
#
# Per profile, it: downloads the model (once, shared), loads the image, builds
# the KV snapshots, and starts the service. Idempotent — re-running skips the
# model download, skips already-built snapshots, and restarts the service.
#
# Both profiles can run on one host for testing (each gets its own project
# name, port and snapshots subdir):
#   ./load_and_run.sh glucowise        # diabetes            -> :8001
#   ./load_and_run.sh aiciblock        # hemorroides+cirugía -> :8002
set -euo pipefail
cd "$(dirname "$0")"

P="${1:-}"
[[ -n "$P" ]] || { echo "usage: $0 <glucowise|aiciblock>" >&2; exit 1; }
ENVF="profiles/${P}.env"
[[ -f "$ENVF" ]] || { echo "ERROR: unknown profile '$P' (no $ENVF)" >&2; exit 1; }

echo "==> Checking prerequisites"
command -v docker >/dev/null || { echo "ERROR: docker not found" >&2; exit 1; }
docker compose version >/dev/null 2>&1 || { echo "ERROR: 'docker compose' plugin not found" >&2; exit 1; }

if [[ ! -f .env ]]; then
  echo "==> No .env found; creating from .env.example"
  cp .env.example .env
  echo "    EDIT .env and set RAG_API_KEY, then re-run this script." >&2
  exit 1
fi
if grep -q '^RAG_API_KEY=change-me' .env; then
  echo "ERROR: RAG_API_KEY still has its placeholder value. Edit .env first." >&2
  exit 1
fi

PORT="$(grep -E '^RAG_PORT=' "$ENVF" | cut -d= -f2)"; PORT="${PORT:-8000}"
PROJECT="rag-${P}"

# Serving (decode) threads. Honor N_THREADS from .env if the operator pinned it;
# otherwise default to ALL cores. gemma is bandwidth-bound and benefits from
# more threads than the app's built-in cap of 9, so we make the choice explicit
# and visible here rather than letting that cap apply silently.
set -a; # shellcheck disable=SC1091
source .env; set +a
if [[ -z "${N_THREADS:-}" ]]; then
  N_THREADS="$(nproc 2>/dev/null || echo 8)"
  echo "==> N_THREADS not set in .env — serving profile '$P' with $N_THREADS threads (all cores)."
  echo "    Running BOTH profiles on this host? Set N_THREADS to ~half the cores in .env."
else
  echo "==> Serving profile '$P' with N_THREADS=$N_THREADS (from .env)."
fi
export N_THREADS OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"

# Serving replicas behind the load balancer. Concurrency == replicas (each
# serializes one generation at a time). Default 1; raise to serve more
# concurrent users or to benchmark scaling. Each replica holds its own ~16 GB
# copy of the model, so keep N_THREADS * N_REPLICAS <= physical cores and budget
# ~16-20 GB RAM per replica.
N_REPLICAS="${N_REPLICAS:-1}"
echo "==> Serving profile '$P' with N_REPLICAS=$N_REPLICAS (concurrency == replicas)."
if [[ "$N_REPLICAS" -gt 1 ]]; then
  echo "    Each replica loads its own ~16 GB model; budget RAM accordingly."
fi

# 1. Model — downloaded once into ./models, shared by both profiles.
echo "==> Ensuring model is present (downloads ~16.9 GB on first run)"
./fetch_model.sh

# 2. Load the service image.
echo "==> Loading image"
docker load -i images/cpu-rag-api-2.0.0-portable.tar

# 3. The container runs as non-root (uid 1001); make snapshots/ writable by it.
# Best-effort: the top-level dir (which we own) must be writable so the container
# can create its per-profile subdir. Subdirs from a previously-built profile are
# already owned by uid 1001 and don't need re-permissioning — chmod'ing them
# would fail (we're not their owner) and, under `set -e`, abort the run. So we
# tolerate that: this is exactly the "both profiles on one host" path.
echo "==> Ensuring snapshots/ is writable by the container"
mkdir -p snapshots
chmod 777 snapshots
chmod -R 777 snapshots 2>/dev/null || true

# 4. Build the KV snapshots for this profile (one-shot; warms at all cores).
echo "==> Building KV snapshots for profile '$P' (one-shot; minutes on first run)"
RAG_GEN_THREADS="$(nproc 2>/dev/null || echo 8)" \
  docker compose --env-file "$ENVF" -p "$PROJECT" \
  --profile generate run --rm rag-generate

# 5. Start serving: N replicas behind the nginx load balancer.
echo "==> Starting service (profile '$P': $N_REPLICAS replica(s), LB on port $PORT)"
docker compose --env-file "$ENVF" -p "$PROJECT" up -d --scale "rag=$N_REPLICAS"

# 6. Wait for health (through the load balancer).
echo "==> Waiting for health on :$PORT"
for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "OK: profile '$P' healthy on port ${PORT} ($N_REPLICAS replica(s))"
    exit 0
  fi
  sleep 3
done
echo "WARN: health check did not pass in time; inspect with" >&2
echo "      docker compose --env-file $ENVF -p $PROJECT logs -f" >&2
exit 1
