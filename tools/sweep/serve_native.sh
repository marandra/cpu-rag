#!/usr/bin/env bash
# Launch a single RAG replica on an HPC compute node with the .venv-native build
# (AVX512-VNNI + AMX-INT8), pinned to a NUMA-aligned cpuset.
#
# Run from inside a salloc/srun shell ON the compute node, from the project root:
#   srun --pty bash -i        # land on eurehpccomputoNN
#   cd ~/Projects/cpu-rag
#   ./tools/sweep/serve_native.sh
#
# Defaults: nT=8, cores 0-7 (one NUMA node), bind 0.0.0.0:8000 (reachable from the
# login node as http://<computeNN>:8000 -- direct ssh to compute is blocked, but the
# app port is plain TCP over the internal network). Override via env:
#   N_THREADS=8 CPUSET=0-7 HOST=0.0.0.0 PORT=8000 VENV=.venv-native ./tools/sweep/serve_native.sh

set -euo pipefail

N_THREADS="${N_THREADS:-8}"
CPUSET="${CPUSET:-0-7}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
VENV="${VENV:-.venv-native}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: $VENV/bin/python not found. Run from project root; venv must exist." >&2
  exit 1
fi

# Load RAG_API_KEY (and other settings) from .env without printing it.
set -a
# shellcheck disable=SC1091
source .env
set +a

echo "host=$(hostname) venv=$VENV nT=$N_THREADS cpuset=$CPUSET bind=$HOST:$PORT"

# Pin to the cpuset and cap all thread pools to N_THREADS (BLAS pools to 1 so
# they don't oversubscribe). Use the venv python directly -- never `uv run`,
# which would resync .venv and recompile llama-cpp-python with the wrong toolchain.
exec taskset -c "$CPUSET" env \
  N_THREADS="$N_THREADS" \
  OMP_NUM_THREADS="$N_THREADS" \
  GGML_N_THREADS="$N_THREADS" \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "$VENV/bin/python" -m uvicorn app.main:app \
    --host "$HOST" --port "$PORT" --log-level info
