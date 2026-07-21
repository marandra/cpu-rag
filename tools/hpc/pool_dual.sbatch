#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-dual
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive                 # whole node: 64 cores / 8 NUMA, split 4+4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=cpu-rag-dual-%j.log
###############################################################################
# pool_dual.sbatch — both deployment profiles on ONE node, side by side.
#
# Use this instead of two pool.sbatch jobs when you want to query glucowise and
# aiciblock in the same session without holding two nodes. Two pool.sbatch jobs
# *cannot* share a node anyway: --exclusive means the second one just queues
# until the first dies.
#
#   sbatch tools/hpc/pool_dual.sbatch
#   squeue -j <jobid> -o "%N"          # the node; both LBs live on it
#
#   curl http://<node>:8080/health     # glucowise (diabetes)
#   curl http://<node>:8081/health     # aiciblock (hemorroides, cirugia)
#
# The split is 4 replicas x 8 threads each. That is the whole node: the rule is
# replicas x N_THREADS <= physical cores (64 here), and breaking it oversubscribes
# and tanks throughput for both pools. Per-query speed is unchanged from the 8x8
# single-profile pool (~11 tok/s) because threads-per-replica is what sets it;
# what halves is how many concurrent queries each profile absorbs.
#
# NUMA: glucowise takes nodes 0-3, aiciblock 4-7, so neither straddles a socket
# (the cross-socket penalty measured -17/-20%).
#
# Each pool needs its own STAGE_ROOT: launch_pool.sh writes its generated
# nginx.conf, pid, logs and per-replica snapshot staging under it, so sharing
# one would have the two pools overwrite each other's config.
###############################################################################
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load settings from .env (export every assignment, like docker-compose env_file).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi
: "${RAG_API_KEY:?RAG_API_KEY not set — add it to ./.env or export it before sbatch}"

# See pool.sbatch for why this is the module and not a bare `spack load`.
module load singularity/1.4.1
module load spack/1.1.0
spack load numactl target=x86_64_v3

REPLICAS_PER_POOL="${REPLICAS_PER_POOL:-4}"
THREADS="${N_THREADS:-8}"

echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME}: 2 profiles x ${REPLICAS_PER_POOL} replicas x ${THREADS} threads"

PIDS=()
cleanup() {
  echo "==> Stopping both pools"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# glucowise: NUMA 0-3, replicas on :8001-8004, LB :8080
PROFILE=glucowise \
  N_REPLICAS="$REPLICAS_PER_POOL" N_THREADS="$THREADS" \
  NUMA_START=0 BASE_PORT=8001 LB_PORT=8080 \
  STAGE_ROOT=/tmp/cpu-rag-glucowise \
  tools/hpc/launch_pool.sh &
PIDS+=($!)

# aiciblock: NUMA 4-7, replicas on :8011-8014, LB :8081
PROFILE=aiciblock \
  N_REPLICAS="$REPLICAS_PER_POOL" N_THREADS="$THREADS" \
  NUMA_START=4 BASE_PORT=8011 LB_PORT=8081 \
  STAGE_ROOT=/tmp/cpu-rag-aiciblock \
  tools/hpc/launch_pool.sh &
PIDS+=($!)

echo ""
echo "==> glucowise -> http://${SLURMD_NODENAME}:8080/  (logs /tmp/cpu-rag-glucowise/)"
echo "==> aiciblock -> http://${SLURMD_NODENAME}:8081/  (logs /tmp/cpu-rag-aiciblock/)"
wait
