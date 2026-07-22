#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-model-ab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=05:00:00
#SBATCH --output=cpu-rag-model-ab-%j.log
###############################################################################
# model_ab.sbatch — screen 2 of the model benchmark: does a candidate decide
# better? Same machinery as prompt_ab.sbatch, with the model as the arm instead
# of the prompt: every model x every procedure x 9 seeds over all 134 questions.
#
#   sbatch tools/hpc/model_ab.sbatch
#   MODELS="ministral:Ministral-3-3B-Q4_K_M.gguf" sbatch tools/hpc/model_ab.sbatch
#
# Then:
#   uv run python tools/audit_score.py --baseline ministral \
#       --run ministral=eval/model_ab/ministral --run gemma4=eval/model_ab/gemma4
#
# V13 in every arm, deliberately
# -----------------------------
# V13 is what we serve and what all 134 questions were scored against. A model
# change and a prompt change at once is not attributable, so the prompt is held
# fixed here. V14d — the one prompt change that survived reading the A/B — gets
# retested on top of whichever model wins, not alongside it.
#
# Why the incumbent is re-run and not read off eval/audit_ab/v13
# -------------------------------------------------------------
# That arm was generated on `.venv-native` (llama-cpp 0.3.19) and the candidates
# need `.venv-latest` (0.3.34) — gemma4 does not exist as an architecture before
# it. Comparing across those would confound the model with the build, and this
# project has already paid for that mistake once: the seed sweep's snapshot came
# from a different provenance than the replay and only 29 of 54 answers matched.
# Both arms run here, same venv, same node, same seeds.
#
# Cost: the long cell is diabetes (55 questions x 9 seeds). Ministral does ~5.7
# s/question, and gemma-4 is ~1.9x slower and answers at more length, so budget
# ~50 min and ~2.5 h respectively. Two models x 3 procedures is 6 cells, one
# wave of 8, so the job is as long as its slowest cell.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load spack/1.1.0
spack load numactl target=x86_64_v3

N_THREADS="${N_THREADS:-8}"
VENV="${VENV:-.venv-latest}"
SEEDS="${SEEDS:-8}"                    # the tool prepends the frozen one -> 9
SUBSET="${SUBSET:-all}"
TEMPERATURES="${TEMPERATURES:-0.1}"
OUT_ROOT="${OUT_ROOT:-eval/model_ab}"
MODELS="${MODELS:-
ministral:Ministral-3-3B-Q4_K_M.gguf
gemma4:gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
}"

# Node-local scratch. Each (model x procedure) warms its own pickle — ~0.5 GB
# for Ministral, ~1 GB+ for gemma-4, and more on diabetes because the state
# scales with the prefix. NOT the repo's ./snapshots: a pool re-reads its pkl on
# every request, and the tool refuses that root anyway. Not named
# SNAPSHOTS_ROOT because that is app.config's own setting and sbatch exports the
# submitting environment, which would trip the tool's guard.
SWEEP_SNAPSHOTS_ROOT="${SWEEP_SNAPSHOTS_ROOT:-/tmp/cpu-rag-model-ab-${SLURM_JOB_ID}}"

export N_THREADS
export OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: models=[${MODELS}]" \
     "seeds=${SEEDS} subset=${SUBSET} -> ${OUT_ROOT}"

# shellcheck source=tools/hpc/numa_fanout.sh
source tools/hpc/numa_fanout.sh
FANOUT_LOG_DIR="${FANOUT_LOG_DIR:-cpu-rag-model-ab-${SLURM_JOB_ID}}" fanout_init

cell() {  # label model_file profile procedure
    mkdir -p "${OUT_ROOT}/$1"
    fanout_submit "$1-$4" \
        env PROFILE="$3" MODEL_PATH="models/$2" N_THREADS="$N_THREADS" \
            "./${VENV}/bin/python" -u tools/audit_seed_sweep.py \
                --procedure "$4" \
                --subset "$SUBSET" \
                --seeds "$SEEDS" \
                --temperatures "$TEMPERATURES" \
                --snapshots-root "$SWEEP_SNAPSHOTS_ROOT" \
                --out "${OUT_ROOT}/$1/$4.json"
}

# Longest first, so the short tail packs into whatever slots free up.
for entry in $MODELS; do cell "${entry%%:*}" "${entry#*:}" glucowise diabetes;          done
for entry in $MODELS; do cell "${entry%%:*}" "${entry#*:}" aiciblock cirugia-abdominal; done
for entry in $MODELS; do cell "${entry%%:*}" "${entry#*:}" aiciblock hemorroides;       done

fanout_wait
echo "==> model A/B done. Score it:"
echo "    uv run python tools/audit_score.py --baseline ministral \\"
for entry in $MODELS; do echo "        --run ${entry%%:*}=${OUT_ROOT}/${entry%%:*} \\"; done
