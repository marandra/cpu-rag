#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-ab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=cpu-rag-ab-%j.log
###############################################################################
# prompt_ab.sbatch — the prompt A/B: every variant x every procedure x 9 seeds
# over all 134 audit questions.
#
#   sbatch tools/hpc/prompt_ab.sbatch
#   VARIANTS="v13 v14a" sbatch tools/hpc/prompt_ab.sbatch      # a subset
#
# Then, on the laptop or here:
#   uv run python tools/audit_score.py \
#       --run v13=eval/audit_ab/v13 --run v14a=eval/audit_ab/v14a --baseline v13
#
# Why in-process and not against the pool
# ---------------------------------------
# The seed has to be injectable, and the API has no parameter for it — it comes
# out of the snapshot pickle. `create_chat_completion(seed=...)` is the only way
# to average over seeds, and averaging is the whole method: at t=0.1 the seed
# alone flips the decision on ~30% of the argued questions, so a single draw per
# variant cannot separate a prompt effect from sampling.
#
# Why V13 is an arm and not the versioned replay
# ----------------------------------------------
# A prompt is part of the snapshot cache key, so a variant can never HIT the
# pickles the pool serves — it warms its own. Measured 2026-07-22 (job 7276):
# with the pool's own pickle this harness reproduces the replay 54/54, and with
# a rebuilt one, roughly half. So a V13 arm read from the served replay would
# differ from the variants by provenance as well as by prompt. Every arm
# rebuilds, V13 included. The comparison is internal on purpose; the anchor to
# production is the final replay against the real pool, once a prompt is chosen.
#
# Order matters: cells are dispatched to free NUMA slots in submission order, so
# the long ones (diabetes, 55 questions) go first and the short tail packs into
# the last wave. 18 cells over 8 slots is 3 waves, ~2 h.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# See pool.sbatch for why this is the module and not a bare `spack load`.
module load spack/1.1.0
spack load numactl target=x86_64_v3

VARIANTS="${VARIANTS:-v13 v14a v14b v14c v14d v14e}"
N_THREADS="${N_THREADS:-8}"
SEEDS="${SEEDS:-8}"                    # the tool prepends the frozen one -> 9
SUBSET="${SUBSET:-all}"                # only lower it for a smoke run
TEMPERATURES="${TEMPERATURES:-0.1}"    # what we deploy; the sweep owns 0.8
OUT_ROOT="${OUT_ROOT:-eval/audit_ab}"
# Node-local scratch, one dir per job: each (variant x procedure) warms its own
# ~0.5 GB pickle, so 18 cells is ~9 GB. NOT the repo's ./snapshots — a pool
# re-reads its pkl on every request and the tool refuses that root anyway. See
# the naming note in seed_sweep.sbatch for why it is not called SNAPSHOTS_ROOT.
SWEEP_SNAPSHOTS_ROOT="${SWEEP_SNAPSHOTS_ROOT:-/tmp/cpu-rag-ab-${SLURM_JOB_ID}}"

export N_THREADS
export OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: variants=[${VARIANTS}]" \
     "seeds=${SEEDS} temps=${TEMPERATURES} threads=${N_THREADS} -> ${OUT_ROOT}"

# shellcheck source=tools/hpc/numa_fanout.sh
source tools/hpc/numa_fanout.sh
FANOUT_LOG_DIR="${FANOUT_LOG_DIR:-cpu-rag-ab-${SLURM_JOB_ID}}" fanout_init

cell() {  # variant profile procedure
    mkdir -p "${OUT_ROOT}/$1"
    fanout_submit "$1-$3" \
        env PROFILE="$2" PROMPT_VARIANT="$1" \
            ./.venv-native/bin/python -u tools/audit_seed_sweep.py \
                --procedure "$3" \
                --subset "$SUBSET" \
                --seeds "$SEEDS" \
                --temperatures "$TEMPERATURES" \
                --snapshots-root "$SWEEP_SNAPSHOTS_ROOT" \
                --out "${OUT_ROOT}/$1/$3.json"
}

for v in $VARIANTS; do cell "$v" glucowise diabetes;          done
for v in $VARIANTS; do cell "$v" aiciblock cirugia-abdominal; done
for v in $VARIANTS; do cell "$v" aiciblock hemorroides;       done

fanout_wait
echo "==> prompt A/B done. Score it:"
echo "    uv run python tools/audit_score.py --baseline v13 \\"
for v in $VARIANTS; do echo "        --run ${v}=${OUT_ROOT}/${v} \\"; done
