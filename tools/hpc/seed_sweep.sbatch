#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-seed
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=cpu-rag-seed-%j.log
###############################################################################
# seed_sweep.sbatch — measure the real noise floor: vary the RNG seed.
#
# Not a pool: this loads the model in-process on the native venv and generates
# directly, because the seed the service uses is frozen inside the snapshot
# pickle and cannot be overridden through the API. See tools/audit_seed_sweep.py
# for why that is and what it invalidates.
#
#   sbatch tools/hpc/seed_sweep.sbatch
#   SUBSET=control SEEDS=2 TEMPERATURES=0.1 \
#     OUT_DIR=/tmp/seed-smoke sbatch tools/hpc/seed_sweep.sbatch   # ~5 min check
#
# One procedure per NUMA node, 8 threads each — the config the pool serves.
# ---------------------------------------------------------------------------
# The first version ran the three procedures *serially* at N_THREADS=32 with no
# pinning. Thread count changes the reduction order and so the numerics, and 32
# is not what any replica runs, so the sweep measured a configuration we do not
# deploy. Now: `numactl --cpunodebind=N --membind=N` and N_THREADS=8, exactly
# like a pool replica (launch_pool.sh), fanned out over the exclusive node
# instead of run one after another. Cells are dispatched by
# tools/hpc/numa_fanout.sh, shared with the prompt A/B — that one is 5 variants
# x 3 procedures = 15 cells and lands in two waves of 8 with no change here.
#
# What this does NOT buy, measured 2026-07-22 (job 7274): matching the replay
# byte for byte. The frozen-seed column agreed on 4 of 10 questions at nT=8,
# no better than the 29/54 at nT=32, so threads were not the reason. The
# remaining suspect is snapshot provenance — a cell rebuilds the prefix KV in
# this venv, while the pool serves pickles warmed in May by the container.
# That is fine for an A/B, where every arm rebuilds the same way and the
# comparison is internal, but it means the harness is not production: the last
# word on a chosen prompt is still a replay against the real pool.
#
# Node rule: sum(cells x N_THREADS) <= 64, no NUMA node with two tenants. To
# leave the service alive on the same node, hand it the top NUMA nodes and give
# the sweep the rest, e.g. NUMA_COUNT=6 with a pool on NUMA_START=6.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# numactl is not on the base PATH. See launch_pool.sh for why this is the
# module and not a bare `spack load`.
module load spack/1.1.0
spack load numactl target=x86_64_v3

N_THREADS="${N_THREADS:-8}"
SUBSET="${SUBSET:-argued}"
SEEDS="${SEEDS:-8}"                    # the tool prepends the frozen one -> 9
TEMPERATURES="${TEMPERATURES:-0.1,0.8}"
OUT_DIR="${OUT_DIR:-eval/audit_seed_sweep}"
# Never the repo's ./snapshots: the pool re-reads its pkl on every request and
# this tool warms on a MISS. The tool refuses the serving root anyway.
#
# Deliberately NOT called SNAPSHOTS_ROOT: that name is the app.config setting,
# sbatch exports the submitting environment, and setting it would move the
# *configured* root along with the tool's — which is what the tool compares
# against to decide it is being pointed at a live serving directory. Naming
# them the same turns the guard into a false alarm.
SWEEP_SNAPSHOTS_ROOT="${SWEEP_SNAPSHOTS_ROOT:-/tmp/cpu-rag-offline-snapshots}"

export N_THREADS
export OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: subset=${SUBSET} seeds=${SEEDS}" \
     "temps=${TEMPERATURES} threads=${N_THREADS} -> ${OUT_DIR}"

# shellcheck source=tools/hpc/numa_fanout.sh
source tools/hpc/numa_fanout.sh
FANOUT_LOG_DIR="${FANOUT_LOG_DIR:-cpu-rag-seed-${SLURM_JOB_ID}}" fanout_init

# -u or the cell's progress sits in stdout's buffer until the run exits, while
# llama.cpp's own stderr chatter comes through immediately — which makes a
# healthy job look hung.
cell() {  # profile procedure
    fanout_submit "$2" \
        env PROFILE="$1" ./.venv-native/bin/python -u tools/audit_seed_sweep.py \
            --procedure "$2" \
            --subset "$SUBSET" \
            --seeds "$SEEDS" \
            --temperatures "$TEMPERATURES" \
            --snapshots-root "$SWEEP_SNAPSHOTS_ROOT" \
            --out "${OUT_DIR}/$2.json"
}

cell glucowise diabetes
cell aiciblock cirugia-abdominal
cell aiciblock hemorroides

fanout_wait
echo "==> seed sweep done"
