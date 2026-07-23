#!/usr/bin/env bash
#SBATCH --job-name=cpu-rag-scaling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=cpu-rag-scaling-%j.log
###############################################################################
# model_scaling.sbatch — the D2/D4 curves for ONE model. Task 2 of the plan.
#
#   MODEL=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf PHASE=d2 sbatch tools/hpc/model_scaling.sbatch
#   MODEL=...                               PHASE=d4 sbatch tools/hpc/model_scaling.sbatch
#
# Why this exists next to model_screen.sbatch: the screen answers "does this
# candidate clear the floor at the shape we deploy (nT8 N8)". This answers
# "how does it scale", which is a different question and a property of the
# *model*, not of the node. The May sweep ([[sweep-done-2026-05-21]]) measured
# Ministral: nT 1..64, knee at 32 with 13.8x, regression at 64. gemma-4 is a
# 26B MoE with ~4B active, so none of that transfers — the screen already saw
# it lose ~15% to 8-way saturation where the dense 2-3 GB models lose nothing.
#
# Two phases, and they answer different things:
#
#   d2  single replica, nT in {1,2,4,8,16,32,64}, in SERIES. Serial on purpose:
#       a second tenant would contend for memory bandwidth and that is the very
#       thing being measured. Gives the speedup curve and the parallel
#       efficiency, which is what decides replicas-vs-threads.
#   d4  saturated, sum(N x nT) = 64, waves of (nT,N) in {(4,16),(8,8),(16,4),
#       (32,2)}. Gives per-user tok/s under load and the aggregate. The May
#       numbers to beat/compare: 28.6 tok/s aggregate at nT16 N4.
#
# Deliberately NOT using numa_fanout.sh: it pins one cell to one NUMA node,
# and here a cell can want up to 64 cores. The pinning rule below is the same
# in spirit (whole NUMA nodes, --cpunodebind paired with --membind) but a cell
# may own several.
#
# Cost note: at nT=1 a 26B MoE warms the (system + fulldoc) prefix at roughly
# one core's prefill rate, so that single cell dominates the phase. Hence
# IDS defaults to two questions — the docstring of bench_model.py is explicit
# that speed barely depends on which questions run, and this phase reads
# timings, not text.
###############################################################################
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module load spack/1.1.0
spack load numactl target=x86_64_v3

PHASE="${PHASE:-d2}"
VENV="${VENV:-.venv-latest}"
MODEL="${MODEL:-gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
LABEL="${LABEL:-$(basename "$MODEL" .gguf)}"
PROCEDURE="${PROCEDURE:-cirugia-abdominal}"
PROFILE_FOR="${PROFILE_FOR:-aiciblock}"
IDS="${IDS:-67,86}"
REPEAT="${REPEAT:-1}"
OUT_ROOT="${OUT_ROOT:-eval/model_scaling/${PHASE}}"
LOG_ROOT="${LOG_ROOT:-cpu-rag-scaling-${SLURM_JOB_ID}}"

D2_THREADS="${D2_THREADS:-1 2 4 8 16 32 64}"
D4_CELLS="${D4_CELLS:-4:16 8:8 16:4 32:2}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT"

echo "Job ${SLURM_JOB_ID} on ${SLURMD_NODENAME}: phase=${PHASE} model=${MODEL}"
echo "  procedure=${PROCEDURE} ids=${IDS} venv=${VENV} -> ${OUT_ROOT}"

[[ -f "models/${MODEL}" ]] || { echo "ERROR: no models/${MODEL}" >&2; exit 1; }

# --- pinning ---------------------------------------------------------------
# 8 cores per NUMA node on the Xeon Gold 6430, 4 nodes per socket. A cell of
# nT threads takes nT/8 whole nodes starting at `first`; anything under 8 gets
# one node. Refuse rather than oversubscribe: a cell that quietly runs nT
# threads on fewer cores does not fail, it just reports a wrong number, which
# is the failure mode this whole file exists to avoid.
nodes_for() {  # first_node n_threads -> "a-b" or "a"
    local first="$1" nt="$2" span
    span=$(( (nt + 7) / 8 ))
    (( span < 1 )) && span=1
    if (( first + span > 8 )); then
        echo "ERROR: cell needs ${span} NUMA nodes from ${first}, node has 8" >&2
        return 1
    fi
    if (( span == 1 )); then echo "$first"; else echo "$first-$(( first + span - 1 ))"; fi
}

run_cell() {  # out_name nodes n_threads
    local name="$1" nodes="$2" nt="$3"
    env PROFILE="$PROFILE_FOR" MODEL_PATH="models/${MODEL}" \
        N_THREADS="$nt" OMP_NUM_THREADS="$nt" GGML_N_THREADS="$nt" \
        OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        numactl --cpunodebind="$nodes" --membind="$nodes" \
        "./${VENV}/bin/python" -u tools/bench_model.py \
            --procedure "$PROCEDURE" \
            --ids "$IDS" \
            --repeat "$REPEAT" \
            --label "${LABEL}_${name}" \
            --out "${OUT_ROOT}/${name}.json" \
        >"${LOG_ROOT}/${name}.log" 2>&1
}

rc=0

if [[ "$PHASE" == "d2" ]]; then
    for nt in $D2_THREADS; do
        nodes="$(nodes_for 0 "$nt")" || { rc=1; continue; }
        name="nT$(printf '%02d' "$nt")"
        echo "==> [$(date +%H:%M:%S)] d2 ${name} on NUMA ${nodes}"
        if run_cell "$name" "$nodes" "$nt"; then
            grep -E "MEDIAN decode|load_state|BELOW FLOOR" "${LOG_ROOT}/${name}.log" || true
        else
            echo "    FAIL ${name} (see ${LOG_ROOT}/${name}.log)"; rc=1
        fi
    done

elif [[ "$PHASE" == "d4" ]]; then
    for cell in $D4_CELLS; do
        nt="${cell%%:*}"; n="${cell#*:}"
        echo "==> [$(date +%H:%M:%S)] d4 wave nT=${nt} N=${n}"
        pids=()
        for ((i = 0; i < n; i++)); do
            # Same formula both ways: under 8 threads several cells share a
            # node (nT=4 -> two per node), at 8 or more each cell takes whole
            # nodes (nT=16 -> nodes 0-1, 2-3, ...).
            first=$(( i * nt / 8 ))
            nodes="$(nodes_for "$first" "$nt")" || { rc=1; continue; }
            name="nT$(printf '%02d' "$nt")_N$(printf '%02d' "$n")_r${i}"
            run_cell "$name" "$nodes" "$nt" &
            pids+=($!)
        done
        # Barrier between waves: a slot freed early must not start the next
        # shape while this one is still running, or neither number is "N
        # replicas of X" — the tail of one wave would contend with the head of
        # the next, on a metric whose entire point is who you contend with.
        for p in "${pids[@]}"; do wait "$p" || rc=1; done
        grep -h "MEDIAN decode" "${LOG_ROOT}/nT$(printf '%02d' "$nt")_N$(printf '%02d' "$n")"_r*.log 2>/dev/null || true
    done
else
    echo "ERROR: PHASE must be d2 or d4, got ${PHASE}" >&2
    exit 2
fi

echo "==> scaling ${PHASE} done (rc=${rc}): ${OUT_ROOT}"
exit "$rc"
