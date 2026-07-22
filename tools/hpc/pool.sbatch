#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-pool
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive                 # whole node: 64 cores / 8 NUMA for nT8 N8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00             # raise up to 7-00:00:00 if needed
#SBATCH --output=cpu-rag-pool-%j.log
###############################################################################
# pool.sbatch — enqueue the CPU-RAG Apptainer pool on one cpu-partition node.
#
# RAG_API_KEY (and any other settings) are read from ./.env in the repo root,
# the same file the app uses. Just:
#   sbatch tools/hpc/pool.sbatch
#
# PROFILE picks what the pool serves; pass it per-job rather than putting it in
# .env, so both profiles can be queued at once (give each its own LB_PORT):
#   sbatch --export=ALL,PROFILE=glucowise tools/hpc/pool.sbatch
#   sbatch --export=ALL,PROFILE=aiciblock,LB_PORT=8081 tools/hpc/pool.sbatch
#
# The job holds the node and serves traffic until it hits --time or you scancel.
# launch_pool.sh tears the pool down on the resulting SIGTERM.
#
# Reach the service while the job runs (firewalld is inactive on compute nodes):
#   squeue -j <jobid> -o "%N"      # the node, e.g. eurehpccomputo01
#   curl http://<node>:${LB_PORT:-8080}/lb-health
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

# Activate apptainer + numactl. Prefer the cluster's own module for apptainer:
# it wraps `spack load apptainer target=x86_64_v3` and survives admins changing
# the spec underneath us. Doing that spack load by hand is now a trap — since
# ~2026-07 the tree carries both an x86_64_v3 and an x86_64_v4 build, so a bare
# `spack load apptainer` fails with "matches multiple packages", which under
# `set -e` kills the job before it starts.
module load singularity/1.4.1

# numactl is not in that module and not on the base PATH; without it
# launch_pool falls back to no NUMA pinning (expect the cross-socket penalty).
# Same target= disambiguation applies; v3 runs on every node here.
module load spack/1.1.0
spack load numactl target=x86_64_v3

# Deployment profile. launch_pool.sh binds the snapshots root (./snapshots) and
# passes PROFILE to every replica, which resolves ./snapshots/$PROFILE itself;
# the image's own default is aiciblock.
export PROFILE="${PROFILE:-aiciblock}"

# Pool topology — one replica per NUMA node on the Xeon Gold 6430 node.
export N_REPLICAS="${N_REPLICAS:-8}"
export N_THREADS="${N_THREADS:-8}"
export LB_PORT="${LB_PORT:-8080}"
export NUMA_START="${NUMA_START:-0}"

# $SLURMD_NODENAME, not $(hostname): the module/spack loads above rewrite PATH
# and hostname(1) is no longer on it, so $(hostname) prints an empty node name.
echo "Job $SLURM_JOB_ID on ${SLURMD_NODENAME}: profile ${PROFILE}, ${N_REPLICAS}x${N_THREADS} threads, LB :${LB_PORT}"
exec tools/hpc/launch_pool.sh
