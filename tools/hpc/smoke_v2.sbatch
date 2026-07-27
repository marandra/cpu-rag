#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-v2-smoke
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=cpu-rag-v2-smoke-%j.log
###############################################################################
# smoke_v2.sbatch — one-node end-to-end validation of the v2 (gemma-4) image.
#
# Requests an exclusive cpu node and, on it:
#   1. builds ./rag-v2-spr-native.sif from the scp'd docker-archive tar (once),
#   2. generates the KV snapshots for both profiles at nT=64 (warm at max),
#   3. runs tools/hpc/smoke.sh for each profile (health + one on-topic /query).
#
# The gemma GGUF is already on the cluster (models/); nothing is downloaded.
# Serving config (gemma, diabetes v4, hemorroides vA) is baked into the SIF, so
# only models/ and snapshots/ are bound.
#
#   sbatch tools/hpc/smoke_v2.sbatch
###############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ -f .env ]]; then
  set -a; source ./.env; set +a
fi
: "${RAG_API_KEY:?RAG_API_KEY not set — add it to ./.env}"

module load singularity/1.4.1
module load spack/1.1.0
spack load numactl target=x86_64_v3

SIF=./rag-v2-spr-native.sif
TAR=./rag-v2-spr-native.tar

echo "=== node ${SLURMD_NODENAME} — v2 smoke ==="

# 1. Build the SIF from the docker-archive (idempotent).
if [[ -f "$SIF" ]]; then
  echo "=== $SIF exists — skipping build ==="
else
  [[ -f "$TAR" ]] || { echo "ERROR: $TAR not found (scp it first)"; exit 1; }
  echo "=== apptainer build $SIF from $TAR ==="
  apptainer build "$SIF" "docker-archive://$TAR"
fi

# 2. Generate snapshots for both profiles at nT=64 (warm gemma over the v2
#    corpora). Idempotent: cached snapshots are skipped by fingerprint.
mkdir -p ./snapshots
for P in glucowise aiciblock; do
  echo "=== generate snapshots: profile=$P (nT=64) ==="
  apptainer exec --pwd /app \
    --bind "$(readlink -f ./models):/app/models:ro" \
    --bind "$(readlink -f ./snapshots):/app/snapshots" \
    --env RAG_API_KEY="$RAG_API_KEY" --env PROFILE="$P" \
    --env N_THREADS=64 --env OMP_NUM_THREADS=64 --env GGML_N_THREADS=64 \
    "$SIF" python -m app.generate
done

# 3. End-to-end smoke per profile (single replica, one on-topic /query).
for P in glucowise aiciblock; do
  echo "########## SMOKE profile=$P ##########"
  SIF_IMAGE="$SIF" PROFILE="$P" PORT=8001 bash tools/hpc/smoke.sh \
    || echo "WARN: smoke for $P returned nonzero"
done

echo "=== v2 SMOKE ALL DONE ==="
