#!/usr/bin/env bash
###############################################################################
# numa_fanout.sh — run a queue of independent cells, one per NUMA node.
#
# Source this from an sbatch that holds the node with --exclusive. Every cell
# runs under `numactl --cpunodebind=N --membind=N` on a NUMA node nobody else
# is using, so a cell sees the same locality a pool replica sees.
#
#   source tools/hpc/numa_fanout.sh
#   fanout_init                       # NUMA_START / NUMA_COUNT / FANOUT_LOG_DIR
#   fanout_submit diabetes env PROFILE=glucowise ./.venv-native/bin/python ...
#   fanout_wait                       # non-zero if any cell failed
#
# `fanout_submit` blocks while every slot is busy, so a queue longer than
# NUMA_COUNT runs in as many waves as it needs without any bookkeeping by the
# caller. The node rule this enforces one half of: sum(cells x threads) <= 64,
# and no NUMA node with two tenants. The threads half is the caller's — set
# N_THREADS so that NUMA_COUNT * N_THREADS stays within the node.
#
# Why pinning is mandatory here and only a warning in launch_pool.sh: the
# launcher degrades to a slower service, this degrades to a *different
# measurement*. The seed sweep of 2026-07-21 ran unpinned at 32 threads and
# only 29 of 54 answers matched what the pool served; the threads are part of
# the numerics. Export ALLOW_UNPINNED=1 to run anyway (laptop, debugging).
###############################################################################

# Slot i owns NUMA node FANOUT_NUMA_START+i. Parallel arrays, indexed by slot.
FANOUT_PID=()
FANOUT_LABEL=()

fanout_init() {
  FANOUT_NUMA_START="${NUMA_START:-0}"
  FANOUT_NUMA_COUNT="${NUMA_COUNT:-8}"
  # Deliberately a different name than the FANOUT_LOG_DIR the caller sets: a
  # `VAR=x fanout_init` prefix scopes VAR to the call and restores it on
  # return, so writing back into it would leave the rest of the run unset.
  FANOUT_DIR="${FANOUT_LOG_DIR:-fanout-${SLURM_JOB_ID:-$$}}"

  if command -v numactl >/dev/null; then
    FANOUT_PIN=1
  elif [[ -n "${ALLOW_UNPINNED:-}" ]]; then
    FANOUT_PIN=0
    echo "WARN: no numactl and ALLOW_UNPINNED set — cells are NOT pinned, and" >&2
    echo "      their numbers are NOT comparable to what the pool serves." >&2
  else
    echo "ERROR: numactl not in PATH. Run:" >&2
    echo "         module load spack/1.1.0 && spack load numactl target=x86_64_v3" >&2
    echo "       or set ALLOW_UNPINNED=1 if you accept an incomparable run." >&2
    return 1
  fi

  (( FANOUT_PIN )) && { _fanout_preflight || return 1; }

  mkdir -p "$FANOUT_DIR"
  rm -f "$FANOUT_DIR"/.slot*.free "$FANOUT_DIR"/*.rc
  FANOUT_PID=(); FANOUT_LABEL=()
  local s
  for ((s = 0; s < FANOUT_NUMA_COUNT; s++)); do FANOUT_PID[s]=""; done
  echo "==> fanout: ${FANOUT_NUMA_COUNT} slots on NUMA" \
       "${FANOUT_NUMA_START}-$((FANOUT_NUMA_START + FANOUT_NUMA_COUNT - 1))," \
       "logs in ${FANOUT_DIR}/"
}

_fanout_expand() {  # "0-3,8" -> "0 1 2 3 8"
  local part a b i out=()
  local -a parts
  IFS=',' read -ra parts <<<"$1"
  for part in "${parts[@]}"; do
    if [[ "$part" == *-* ]]; then
      a="${part%-*}"; b="${part#*-}"
      for ((i = a; i <= b; i++)); do out+=("$i"); done
    else
      out+=("$part")
    fi
  done
  echo "${out[*]}"
}

# Check the job really owns every NUMA node we are about to pin to, with enough
# cores on each. Slurm hands out fragmented cores unless the job is --exclusive,
# and the two failure modes are not equally visible: pinning to a node we own
# *no* cores of dies loudly, but pinning to one we own *some* of just runs
# N_THREADS threads on fewer cores — oversubscribed, silent, and it would look
# like a slow prompt rather than a broken measurement. So refuse up front.
_fanout_preflight() {
  local allowed cpu node n slot bad=0
  declare -A own=()
  allowed="$(grep -m1 Cpus_allowed_list /proc/self/status | awk '{print $2}')"
  for cpu in $(_fanout_expand "$allowed"); do own[$cpu]=1; done

  for ((slot = 0; slot < FANOUT_NUMA_COUNT; slot++)); do
    node=$((FANOUT_NUMA_START + slot))
    if [[ ! -r "/sys/devices/system/node/node${node}/cpulist" ]]; then
      echo "ERROR: NUMA node ${node} does not exist on $(uname -n)." >&2
      bad=1; continue
    fi
    n=0
    for cpu in $(_fanout_expand "$(<"/sys/devices/system/node/node${node}/cpulist")"); do
      [[ -n "${own[$cpu]:-}" ]] && ((++n))
    done
    if ((n < ${N_THREADS:-1})); then
      echo "ERROR: NUMA node ${node}: this job owns ${n} of its cores, need" \
           "${N_THREADS:-1}. Cpus_allowed_list=${allowed}" >&2
      bad=1
    fi
  done

  if ((bad)); then
    echo "       Fan-out needs whole NUMA nodes: submit with --exclusive, or" >&2
    echo "       narrow the range with NUMA_START / NUMA_COUNT." >&2
    return 1
  fi
}

# A slot is free if it never ran, or if its cell dropped the done marker. Going
# by the marker and not by `kill -0` on purpose: an exited child stays a zombie
# until it is waited for, so liveness alone would never report it free.
_fanout_free_slot() {
  local s
  for ((s = 0; s < FANOUT_NUMA_COUNT; s++)); do
    if [[ -z "${FANOUT_PID[s]}" || -f "${FANOUT_DIR}/.slot${s}.free" ]]; then
      echo "$s"; return 0
    fi
  done
  return 1
}

fanout_submit() {  # label cmd...
  local label="$1"; shift
  local slot
  until slot="$(_fanout_free_slot)"; do sleep 5; done

  local node=$((FANOUT_NUMA_START + slot))
  local log="${FANOUT_DIR}/${label}.log"
  local pin=()
  (( FANOUT_PIN )) && pin=(numactl --cpunodebind="$node" --membind="$node")

  rm -f "${FANOUT_DIR}/.slot${slot}.free"
  echo "==> [$(date +%H:%M:%S)] ${label} -> slot ${slot} (NUMA ${node})  ${log}"
  (
    set +e
    "${pin[@]}" "$@" >"$log" 2>&1
    echo "$?" >"${FANOUT_DIR}/${label}.rc"
    touch "${FANOUT_DIR}/.slot${slot}.free"
  ) &
  FANOUT_PID[slot]=$!
  FANOUT_LABEL[slot]="$label"
}

fanout_wait() {
  wait
  local failed=() rc f label
  for f in "${FANOUT_DIR}"/*.rc; do
    [[ -e "$f" ]] || continue
    label="$(basename "$f" .rc)"
    rc="$(cat "$f")"
    if [[ "$rc" == "0" ]]; then
      echo "    ok    ${label}"
    else
      echo "    FAIL  ${label} (rc=${rc}, see ${FANOUT_DIR}/${label}.log)"
      failed+=("$label")
    fi
  done
  if ((${#failed[@]})); then
    echo "==> ${#failed[@]} cell(s) failed: ${failed[*]}" >&2
    return 1
  fi
  echo "==> all cells ok"
}
