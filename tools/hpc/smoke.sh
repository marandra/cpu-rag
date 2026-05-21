#!/usr/bin/env bash
###############################################################################
# smoke.sh — single-replica end-to-end smoke test of the SPR-native SIF.
# Run on a compute node via srun, e.g.:
#   srun -p cpu -N1 -n1 -c8 -t 00:15:00 tools/hpc/smoke.sh
# Verifies: SIF runs, llama reports VNNI/AMX, snapshot loads, /query streams.
###############################################################################
set -uo pipefail
cd "$(dirname "$0")/../.."

export RAG_API_KEY="${RAG_API_KEY:-smoke-key}"
PORT="${PORT:-8001}"
SIF="${SIF_IMAGE:-./rag-spr-native.sif}"
PROC="${PROC:-hemorroides}"   # shortest prefix → fastest smoke
STAGE="/tmp/cpu-rag/smoke-stage"
mkdir -p "$STAGE"

echo "=== node $(cat /proc/sys/kernel/hostname) ==="
echo "=== llama system info (VNNI/AMX flags) ==="
apptainer exec "$SIF" python - <<'PY'
from llama_cpp import llama_print_system_info
i = llama_print_system_info()
print((i.decode() if hasattr(i, "decode") else i))
PY

echo "=== starting replica on :$PORT (NUMA node 0) ==="
pin=(); command -v numactl >/dev/null && pin=(numactl --cpunodebind=0 --membind=0)
"${pin[@]}" apptainer exec \
  --bind "$(readlink -f ./models):/app/models:ro" \
  --bind "$(readlink -f ./snapshots):/app/snapshots:ro" \
  --env RAG_API_KEY="$RAG_API_KEY" --env REPLICA_ID=smoke \
  --env N_THREADS=8 --env OMP_NUM_THREADS=8 --env GGML_N_THREADS=8 \
  --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env SNAPSHOT_STAGE_DIR="$STAGE" \
  "$SIF" uvicorn app.main:app --host 127.0.0.1 --port "$PORT" \
  >/tmp/cpu-rag/smoke.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait 2>/dev/null' EXIT

echo "=== waiting for /health ==="
ok=0
for i in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 2
done
[[ $ok == 1 ]] || { echo "FAIL: health never came up"; tail -30 /tmp/cpu-rag/smoke.log; exit 1; }
echo "health OK"

echo "=== /query (procedure=$PROC) ==="
t0=$(date +%s.%N)
curl -sS -N "http://127.0.0.1:$PORT/query" \
  -H "X-API-Key: $RAG_API_KEY" -H "Content-Type: application/json" \
  -d "{\"procedure\":\"$PROC\",\"question\":\"¿Puedo ducharme después de la operación?\"}" \
  | tee /tmp/cpu-rag/smoke_answer.txt
t1=$(date +%s.%N)
echo ""
echo "=== elapsed: $(awk "BEGIN{printf \"%.1f\", $t1 - $t0}")s ==="
echo "=== server log tail (tok/s) ==="
grep -iE "tok/s|tokens|generat" /tmp/cpu-rag/smoke.log | tail -5
echo "=== SMOKE DONE ==="
