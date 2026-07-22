#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-models
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --output=cpu-rag-models-%j.log
###############################################################################
# model_screen.sbatch — screen 1 of the model benchmark: does a candidate clear
# the 6 tok/s floor at all, and what does it cost per request?
#
#   MODE=solo      sbatch tools/hpc/model_screen.sbatch     # one cell per model
#   MODE=saturated sbatch tools/hpc/model_screen.sbatch     # 8 cells per model
#
# The two modes answer different questions and only one of them is the deliverable
# number:
#
#   solo       one cell per model, each on its own NUMA node. Every candidate
#              loads, its template renders, its answers can be read, and the
#              per-request costs (load_state, pickle) come out. The tok/s is an
#              upper bound: nothing else is competing for memory bandwidth.
#   saturated  8 cells of the *same* model at once, which is the shape we
#              deploy (nT8 N8). Measured 2026-07-22: the incumbent loses almost
#              nothing here, 15.2 solo -> 14.9-15.4 across the 8 cells, so on
#              this node cores and memory bandwidth are NOT what 8-way costs.
#              Read the numbers as a ranking, not as production: the same model
#              serves 11.4 tok/s through the pool, and that gap is the pool's
#              own overhead (HTTP/SSE, the executor, re-reading the pickle per
#              request). The anchor for a candidate is therefore the incumbent's
#              column here, not the 6 tok/s floor directly.
#
# Models are mmap'd, so the 8 cells of one wave share page cache and a 20 GB MoE
# costs ~20 GB on the node, not 160 — which is the whole reason a 35B-A3B is on
# this list at all. Do not interleave models within a wave: that would both
# double the resident set and make the contention un-attributable.
#
# Everything runs on .venv-latest (llama-cpp 0.3.34): gemma4 does not exist as
# an architecture before it, and both Qwen3.5 need it too.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load spack/1.1.0
spack load numactl target=x86_64_v3

MODE="${MODE:-solo}"
N_THREADS="${N_THREADS:-8}"
VENV="${VENV:-.venv-latest}"
OUT_ROOT="${OUT_ROOT:-eval/model_screen/${MODE}}"
REPEAT="${REPEAT:-1}"

# label:model:profile:procedure. One procedure per model is enough for a speed
# screen — cirugia-abdominal, because at 7.6 KB its fulldoc is the middle one,
# so neither the 13 KB diabetes prefill nor the 1.1 KB hemorroides flatters it.
PROCEDURE="${PROCEDURE:-cirugia-abdominal}"
PROFILE_FOR="${PROFILE_FOR:-aiciblock}"
MODELS="${MODELS:-
ministral:Ministral-3-3B-Q4_K_M.gguf
qwen35-4b:Qwen3.5-4B-Q4_K_M.gguf
granite41-3b:granite-4.1-3b-Q4_K_M.gguf
gemma4-26b-a4b:gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
qwen35-35b-a3b:Qwen3.5-35B-A3B-Q4_K_M.gguf
}"

export N_THREADS
export OMP_NUM_THREADS="$N_THREADS" GGML_N_THREADS="$N_THREADS"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: mode=${MODE} threads=${N_THREADS}" \
     "procedure=${PROCEDURE} -> ${OUT_ROOT}"

# shellcheck source=tools/hpc/numa_fanout.sh
source tools/hpc/numa_fanout.sh
LOG_ROOT="${FANOUT_LOG_DIR:-cpu-rag-models-${SLURM_JOB_ID}}"

cell() {  # label model_file replica_index
    mkdir -p "${OUT_ROOT}"
    fanout_submit "$1${3:+-r$3}" \
        env PROFILE="$PROFILE_FOR" MODEL_PATH="models/$2" N_THREADS="$N_THREADS" \
            "./${VENV}/bin/python" -u tools/bench_model.py \
                --procedure "$PROCEDURE" \
                --repeat "$REPEAT" \
                --label "$1" \
                --out "${OUT_ROOT}/$1${3:+-r$3}.json"
}

rc=0
if [[ "$MODE" == "saturated" ]]; then
    # One model per wave, with a barrier between waves. Without the barrier a
    # slot freed early would start the next model while the previous one is
    # still running, and then neither number is "8 replicas of X": the tail of
    # one model would be contending with the head of another, on a metric whose
    # entire point is who you are contending with.
    for entry in $MODELS; do
        label="${entry%%:*}"; file="${entry#*:}"
        FANOUT_LOG_DIR="${LOG_ROOT}/${label}" fanout_init
        for r in $(seq 0 $((${NUMA_COUNT:-8} - 1))); do cell "$label" "$file" "$r"; done
        fanout_wait || rc=1   # a dead candidate must not take the rest with it
    done
else
    FANOUT_LOG_DIR="${LOG_ROOT}" fanout_init
    for entry in $MODELS; do
        cell "${entry%%:*}" "${entry#*:}" ""
    done
    fanout_wait || rc=1
fi

echo "==> model screen done (rc=${rc}): ${OUT_ROOT}"
grep -h "MEDIAN decode\|== \|BELOW FLOOR" "${LOG_ROOT}"/*.log "${LOG_ROOT}"/*/*.log 2>/dev/null || true
exit "$rc"
