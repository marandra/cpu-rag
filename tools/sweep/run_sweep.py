"""
Sweep harness for Phase D.

Launches the RAG server with a sequence of (venv, n_threads, cpuset) configs,
sends a small fixed query set, appends one CSV row per measured request with
fsync. Resumable: skips (config_id, procedure, run_idx) combos already in the
CSV. Designed to run on an HPC compute node with bare Python venvs (no docker).

Usage:
  python tools/sweep/run_sweep.py --phase d2 --out reports/sweep/d2_threads.csv

Phases:
  d2  thread sweep, single replica
  d3  numa alignment (best N_T from d2)
  d4  multi-replica throughput

Phase D1 (snapshot read cost) is measured implicitly: the load_state_ms column
captures it on every row.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


# ---- Calibrated queries (output 18-24 tokens, deterministic at T=0.1) -------

QUERIES = [
    ("diabetes", "q_metformina",
     "¿para qué sirve la metformina?"),
    ("cirugia-abdominal", "q_calmantes",
     "¿cada cuánto se administran los calmantes después de la operación?"),
    ("hemorroides", "q_beneficios",
     "¿qué beneficios tiene operarse las hemorroides?"),
]


# ---- Config matrix ---------------------------------------------------------

@dataclass
class Config:
    venv: str            # path to venv root (e.g. .venv-portable)
    n_threads: int
    cpuset: str          # taskset -c argument, e.g. "0-7" or "0-3,16-19"
    config_id: str       # stable id used for CSV resume
    note: str = ""

    @property
    def env(self) -> dict[str, str]:
        return {
            "N_THREADS": str(self.n_threads),
            "OMP_NUM_THREADS": str(self.n_threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "GGML_N_THREADS": str(self.n_threads),
        }


def build_d2_configs() -> list[Config]:
    """Single-replica thread sweep, three builds, NUMA-aligned cpusets."""
    configs = []
    thread_counts = [1, 2, 4, 8, 16, 32, 64]
    for venv in (".venv-portable", ".venv-avx512", ".venv-native"):
        for nT in thread_counts:
            cpuset = _aligned_cpuset(nT)
            cid = f"{Path(venv).name}_nT{nT:02d}"
            configs.append(Config(venv=venv, n_threads=nT,
                                  cpuset=cpuset, config_id=cid))
    return configs


def build_d2_native_focused_configs() -> list[Config]:
    """Focused .venv-native vs .venv-avx512 comparison around the D2 knee.
    Only the thread counts where the previous sweep mattered (8/16/32) so
    we can confirm or reject a VNNI/AMX gain quickly before committing to a
    full re-sweep."""
    configs = []
    for venv in (".venv-avx512", ".venv-native"):
        for nT in (8, 16, 32):
            cpuset = _aligned_cpuset(nT)
            cid = f"{Path(venv).name}_nT{nT:02d}"
            configs.append(Config(venv=venv, n_threads=nT,
                                  cpuset=cpuset, config_id=cid,
                                  note="d2-native-focused"))
    return configs


def build_d3_configs(venv: str) -> list[Config]:
    """NUMA penalty across thread counts on Xeon Gold 6430 (8 cpus/node, 32/socket, 64/2 sockets)."""
    cfgs = [
        # N_T=8: fits in 1 NUMA node — clearest "aligned vs broken"
        Config(venv, 8, "0-7",          "d3_n08_aligned",     "1 NUMA node"),
        Config(venv, 8, "0-3,8-11",     "d3_n08_cross_numa",  "2 NUMA, same socket"),
        Config(venv, 8, "0-3,32-35",    "d3_n08_cross_sock",  "2 sockets"),
        # N_T=16: needs 2 NUMA nodes; cross-socket should be ~25% slower
        Config(venv, 16, "0-15",        "d3_n16_aligned",     "2 NUMA, same socket"),
        Config(venv, 16, "0-7,32-39",   "d3_n16_cross_sock",  "2 NUMA, cross socket"),
        # N_T=32: full socket vs cross-socket
        Config(venv, 32, "0-31",        "d3_n32_aligned",     "single socket"),
        Config(venv, 32, "0-15,32-47",  "d3_n32_cross_sock",  "split across sockets"),
    ]
    return cfgs


def _aligned_cpuset(n_threads: int) -> str:
    """NUMA-aligned cpuset for D2. 8 cores per NUMA node on Xeon Gold 6430."""
    # 0..7 = numa0, 8..15 = numa1, ..., 56..63 = numa7
    return f"0-{n_threads - 1}"


# ---- Server lifecycle ------------------------------------------------------

def find_free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def launch_server(cfg: Config, port: int, replica_id: str = "rag",
                  log_path: Path | None = None) -> subprocess.Popen:
    """Spawn `taskset -c <cpuset> .venv/bin/python -m uvicorn ...`."""
    env = os.environ.copy()
    env.update(cfg.env)
    env["REPLICA_ID"] = replica_id
    # Stage to a config-specific tmp dir so concurrent configs don't collide.
    env["SNAPSHOT_STAGE_DIR"] = f"/tmp/cpu-rag-stage-{port}"

    py = Path(cfg.venv) / "bin" / "python"
    cmd = [
        "taskset", "-c", cfg.cpuset,
        str(py), "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", str(port),
        "--log-level", "info", "--no-access-log",
    ]
    log_fh = open(log_path, "wb") if log_path else subprocess.DEVNULL
    return subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)


def wait_health(port: int, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


# ---- Client (SSE) ----------------------------------------------------------

def send_query(port: int, procedure: str, question: str,
               request_id: str, api_key: str, timeout: float = 180.0) -> dict:
    """One SSE request. Returns the 'done' event's usage dict + a wall_ms field."""
    body = json.dumps({"procedure": procedure, "question": question}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/query", data=body,
        headers={
            "Content-Type": "application/json",
            "X-API-Key": api_key,
            "X-Request-ID": request_id,
            "Accept": "text/event-stream",
        },
    )
    t0 = time.perf_counter()
    ttft_ms = 0.0
    out_tokens = 0
    done_payload = None
    cur_event = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").rstrip()
            if line.startswith("event:"):
                cur_event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                payload = line[5:].lstrip()
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue
                if cur_event == "chunk":
                    if out_tokens == 0:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    out_tokens += 1
                elif cur_event == "done":
                    done_payload = ev
                elif cur_event == "error":
                    raise RuntimeError(f"server error: {ev}")
    wall_ms = (time.perf_counter() - t0) * 1000
    if done_payload is None:
        raise RuntimeError("stream ended without done event")
    out = dict(done_payload.get("usage", {}))
    out["ttft_ms_client"] = round(ttft_ms)
    out["wall_ms"] = round(wall_ms)
    out["request_id"] = done_payload.get("request_id", request_id)
    out["replica_id"] = done_payload.get("replica_id", "unknown")
    return out


# ---- CSV writer (append + fsync per row) -----------------------------------

CSV_FIELDS = [
    "timestamp", "phase", "config_id", "venv", "n_threads", "cpuset",
    "n_replicas", "procedure", "question_id", "run_idx",
    "ttft_ms_client", "load_state_ms", "prefill_ms", "decode_ms",
    "output_tokens", "decode_tok_s", "wall_ms",
    "replica_id", "request_id", "note",
]


def open_csv(path: Path) -> tuple[csv.DictWriter, set[tuple[str, str, int]]]:
    """Open CSV for append; return (writer, set of existing keys)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    new_file = not path.exists() or path.stat().st_size == 0
    if not new_file:
        with path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add((row["config_id"], row["question_id"],
                              int(row["run_idx"])))
    fh = path.open("a", buffering=1)  # line-buffered
    writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
    if new_file:
        writer.writeheader()
        fh.flush()
        os.fsync(fh.fileno())
    return writer, existing, fh


def append_row(writer: csv.DictWriter, fh, row: dict) -> None:
    writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})
    fh.flush()
    os.fsync(fh.fileno())


# ---- Orchestrator ----------------------------------------------------------

def run_config(cfg: Config, phase: str, runs: int, writer, fh,
               existing: set, api_key: str, log_dir: Path) -> None:
    port = find_free_port()
    log_path = log_dir / f"{cfg.config_id}.server.log"
    proc = launch_server(cfg, port, replica_id="rag-1", log_path=log_path)
    try:
        if not wait_health(port, timeout=90):
            print(f"  [SKIP] {cfg.config_id} server never became healthy")
            return
        # Per-procedure: 1 warmup (discarded) + `runs` measured.
        for proc_name, qid, question in QUERIES:
            for run_idx in range(runs + 1):  # 0 = warmup
                key = (cfg.config_id, qid, run_idx)
                if key in existing:
                    continue
                t_total = time.perf_counter()
                try:
                    res = send_query(port, proc_name, question,
                                     request_id=f"{cfg.config_id}-{qid}-r{run_idx}",
                                     api_key=api_key)
                except Exception as e:
                    print(f"  [ERR] {cfg.config_id} {qid} r{run_idx}: {e}")
                    continue
                if run_idx == 0:
                    print(f"  [warm] {cfg.config_id:24s} {proc_name:18s} "
                          f"out={res['completion_tokens']:2d} tok "
                          f"decode={res['decode_ms']:5d}ms ({res['decode_tok_s']:.2f} tok/s)")
                    continue
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "phase": phase,
                    "config_id": cfg.config_id,
                    "venv": Path(cfg.venv).name,
                    "n_threads": cfg.n_threads,
                    "cpuset": cfg.cpuset,
                    "n_replicas": 1,
                    "procedure": proc_name,
                    "question_id": qid,
                    "run_idx": run_idx,
                    "ttft_ms_client": res["ttft_ms_client"],
                    "load_state_ms": res["load_state_ms"],
                    "prefill_ms": res["prefill_ms"],
                    "decode_ms": res["decode_ms"],
                    "output_tokens": res["completion_tokens"],
                    "decode_tok_s": res["decode_tok_s"],
                    "wall_ms": res["wall_ms"],
                    "replica_id": res["replica_id"],
                    "request_id": res["request_id"],
                    "note": cfg.note,
                }
                append_row(writer, fh, row)
                print(f"  [meas] {cfg.config_id:24s} {proc_name:18s} r{run_idx} "
                      f"out={row['output_tokens']:2d} "
                      f"load={row['load_state_ms']:4d}ms "
                      f"prefill={row['prefill_ms']:5d}ms "
                      f"decode={row['decode_ms']:5d}ms "
                      f"({row['decode_tok_s']:.2f} tok/s)")
    finally:
        stop_server(proc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["d2", "d3", "d2-native"])
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--runs", type=int, default=3,
                    help="Measured runs per (config, procedure)")
    ap.add_argument("--api-key", default=os.environ.get("RAG_API_KEY", ""))
    ap.add_argument("--d3-venv", default=".venv-portable")
    ap.add_argument("--filter", help="Substring match on config_id")
    args = ap.parse_args()

    if not args.api_key:
        print("RAG_API_KEY missing (env or --api-key)", file=sys.stderr)
        return 2

    out = Path(args.out)
    log_dir = out.parent / "log" / args.phase
    log_dir.mkdir(parents=True, exist_ok=True)

    writer, existing, fh = open_csv(out)

    if args.phase == "d2":
        configs = build_d2_configs()
    elif args.phase == "d2-native":
        configs = build_d2_native_focused_configs()
    else:
        configs = build_d3_configs(args.d3_venv)

    if args.filter:
        configs = [c for c in configs if args.filter in c.config_id]
        print(f"Filter: {args.filter} -> {len(configs)} configs")

    print(f"Phase {args.phase}: {len(configs)} configs, "
          f"runs={args.runs}, existing rows={len(existing)}")
    print(f"CSV: {out}")
    print(f"Logs: {log_dir}")
    print()

    t_start = time.time()
    try:
        for i, cfg in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] {cfg.config_id}  "
                  f"venv={cfg.venv} nT={cfg.n_threads} cpuset={cfg.cpuset}")
            run_config(cfg, args.phase, args.runs, writer, fh,
                       existing, args.api_key, log_dir)
    finally:
        fh.close()
        elapsed = time.time() - t_start
        print(f"\nDone in {elapsed/60:.1f} min. CSV: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
