#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-corpus-ab
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=cpu-rag-corpus-ab-%j.log
###############################################################################
# corpus_ab.sbatch — A/B of **corpus editions**, with model and prompt fixed.
#
#   CORPORA="vA:hemorroides.vA.md vB:hemorroides.vB.md" \
#       sbatch tools/hpc/corpus_ab.sbatch
#
# The three existing sbatches each vary one thing and none varies this one:
# prompt_ab varies the prompt (and is pinned to .venv-native, so it cannot load
# gemma4 at all), model_ab varies the model, model_scaling varies threads. This
# varies the fulldoc.
#
# The control arm is NOT re-run. `eval/model_ab/gemma4/<procedure>.json` (job
# 7320) is gemma + V13 + the served corpus, produced by *this same tool* on the
# same venv with the same seeds, and the harness is deterministic between jobs
# (verified 2026-07-22, job 7299: the v13 arm came out byte-identical to job
# 7290's, 1206/1206). Score against that directly.
#
# Why every arm rebuilds its snapshot, and why that is fine: the fulldoc is part
# of the snapshot cache key, so a corpus edition can never HIT a served pickle —
# exactly like a prompt variant. The comparison is internal on purpose.
#
# Cost: hemorroides is the short procedure (31 questions x 9 seeds). Three arms
# fit in one wave of the fan-out, well inside the walltime.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load spack/1.1.0
spack load numactl target=x86_64_v3

N_THREADS="${N_THREADS:-8}"
VENV="${VENV:-.venv-latest}"          # gemma4 needs llama-cpp 0.3.34
MODEL="${MODEL:-gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
PROMPT_VARIANT="${PROMPT_VARIANT:-v13}"
PROCEDURE="${PROCEDURE:-hemorroides}"
PROFILE_FOR="${PROFILE_FOR:-aiciblock}"
SEEDS="${SEEDS:-8}"                    # the tool prepends the frozen one -> 9
SUBSET="${SUBSET:-all}"
TEMPERATURES="${TEMPERATURES:-0.1}"
OUT_ROOT="${OUT_ROOT:-eval/corpus_ab}"

# label:filename, relative to corpus/markdown/. A label becomes a directory
# under OUT_ROOT, so keep it short and stable — it is what audit_score.py shows.
CORPORA="${CORPORA:-
vA:hemorroides.vA.md
vB:hemorroides.vB.md
mio:hemorroides.v4-mio.md
}"

SWEEP_SNAPSHOTS_ROOT="${SWEEP_SNAPSHOTS_ROOT:-/tmp/cpu-rag-corpus-ab-${SLURM_JOB_ID}}"

export N_THREADS
export OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: procedure=${PROCEDURE}"
echo "  model=${MODEL} prompt=${PROMPT_VARIANT} venv=${VENV} -> ${OUT_ROOT}"
echo "  corpora=[${CORPORA}]"

[[ -f "models/${MODEL}" ]] || { echo "ERROR: no models/${MODEL}" >&2; exit 1; }

# Fail before burning a node on a typo: every edition must exist up front.
missing=0
for entry in $CORPORA; do
    doc="corpus/markdown/${entry#*:}"
    [[ -f "$doc" ]] || { echo "ERROR: no such fulldoc: $doc" >&2; missing=1; }
done
(( missing )) && exit 1

# shellcheck source=tools/hpc/numa_fanout.sh
source tools/hpc/numa_fanout.sh
FANOUT_LOG_DIR="${FANOUT_LOG_DIR:-cpu-rag-corpus-ab-${SLURM_JOB_ID}}" fanout_init

cell() {  # label fulldoc_filename
    mkdir -p "${OUT_ROOT}/$1"
    fanout_submit "$1-${PROCEDURE}" \
        env PROFILE="$PROFILE_FOR" MODEL_PATH="models/${MODEL}" \
            PROMPT_VARIANT="$PROMPT_VARIANT" N_THREADS="$N_THREADS" \
            "./${VENV}/bin/python" -u tools/audit_seed_sweep.py \
                --procedure "$PROCEDURE" \
                --fulldoc "corpus/markdown/$2" \
                --subset "$SUBSET" \
                --seeds "$SEEDS" \
                --temperatures "$TEMPERATURES" \
                --snapshots-root "$SWEEP_SNAPSHOTS_ROOT" \
                --out "${OUT_ROOT}/$1/${PROCEDURE}.json"
}

for entry in $CORPORA; do cell "${entry%%:*}" "${entry#*:}"; done

fanout_wait
echo "==> corpus A/B done. Score it against the already-measured control:"
echo "    python3 tools/audit_score.py --baseline control \\"
echo "        --run control=eval/model_ab/gemma4 \\"
for entry in $CORPORA; do echo "        --run ${entry%%:*}=${OUT_ROOT}/${entry%%:*} \\"; done
echo "    # add --exclude-procedure to drop the procedures this job did not run"
