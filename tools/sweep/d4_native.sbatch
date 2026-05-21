#!/usr/bin/env bash
#SBATCH --job-name=d4-native
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=reports/sweep/log/d4_native.%j.out
#SBATCH --error=reports/sweep/log/d4_native.%j.out
#
# Phase D4 (multi-replica throughput) re-run with .venv-native
# (AVX512-VNNI + AMX-INT8) to confirm the (N, nT) decision. The original D4
# (reports/sweep/d4_throughput.csv) used .venv-portable. Same 12 cells:
#   nT4 N16, nT8 N8, nT16 N4, nT32 N2  x  K in {1, N, 2N}.
# Resumable: re-running skips (config_id, question_id, run_idx) already in CSV.
#
# Submit from the project root:
#   cd ~/Projects/cpu-rag && sbatch tools/sweep/d4_native.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/sweep/log

# Pull RAG_API_KEY (and other settings) from .env without printing it.
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "host=${SLURMD_NODENAME:-$(cat /proc/sys/kernel/hostname)} job=$SLURM_JOB_ID cpus=$(nproc) start=$(date -Is)"
echo "lscpu model: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//')"

# Drive with an existing venv's python (the harness launches each replica's own
# venv via --venv). Do NOT use `uv run` -- it syncs the default .venv and
# rebuilds llama-cpp-python with the wrong toolchain. The driver needs stdlib.
.venv-portable/bin/python tools/sweep/run_d4.py \
  --venv .venv-native \
  --out reports/sweep/d4_native.csv \
  --runs 3

echo "done=$(date -Is)"
