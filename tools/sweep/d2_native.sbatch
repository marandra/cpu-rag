#!/usr/bin/env bash
#SBATCH --job-name=d2-native
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --output=reports/sweep/log/d2_native.%j.out
#SBATCH --error=reports/sweep/log/d2_native.%j.out
#
# Focused D2: .venv-avx512 (no VNNI/AMX) vs .venv-native (AVX512-VNNI + AMX-INT8)
# at nT in {8,16,32}. Runs on a compute node (Xeon Gold 6430 SPR) for the
# definitive number. The harness sets taskset/cpuset/threads per config.
#
# Submit from the project root:
#   cd ~/Projects/cpu-rag && sbatch tools/sweep/d2_native.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p reports/sweep/log

# Pull RAG_API_KEY (and any other settings) from .env without printing it.
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "host=${SLURMD_NODENAME:-$(cat /proc/sys/kernel/hostname)} job=$SLURM_JOB_ID cpus=$(nproc) start=$(date -Is)"
echo "lscpu model: $(lscpu | grep -m1 'Model name' | sed 's/.*: *//')"

# Drive with an existing venv's python (the harness launches each config's own
# venv). Do NOT use `uv run` -- it syncs the default .venv and rebuilds
# llama-cpp-python with the wrong toolchain. The driver itself only needs stdlib.
.venv-portable/bin/python tools/sweep/run_sweep.py \
  --phase d2-native \
  --out reports/sweep/d2_native.csv \
  --runs 3

echo "done=$(date -Is)"
