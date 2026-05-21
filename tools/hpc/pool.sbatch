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
# Submit from the repo root:
#   sbatch --export=ALL,RAG_API_KEY=<key> tools/hpc/pool.sbatch
# (or `export RAG_API_KEY=...` first, then `sbatch tools/hpc/pool.sbatch`)
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

: "${RAG_API_KEY:?Pass RAG_API_KEY via --export=ALL,RAG_API_KEY=... or export it before sbatch}"

# Activate apptainer + numactl (both already in the system spack tree). numactl
# is not on the base PATH; without it launch_pool falls back to no NUMA pinning.
module load spack/1.1.0
spack load apptainer
spack load numactl

# Pool topology — one replica per NUMA node on the Xeon Gold 6430 node.
export N_REPLICAS="${N_REPLICAS:-8}"
export N_THREADS="${N_THREADS:-8}"
export LB_PORT="${LB_PORT:-8080}"
export NUMA_START="${NUMA_START:-0}"

echo "Job $SLURM_JOB_ID on $(hostname): ${N_REPLICAS}x${N_THREADS} threads, LB :${LB_PORT}"
exec tools/hpc/launch_pool.sh
