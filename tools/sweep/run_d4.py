"""
Phase D4 — multi-replica throughput sweep.

Launches N uvicorn replicas pinned to disjoint NUMA-aligned cpusets, then
saturates them with K concurrent SSE clients. One CSV row per request,
fsynced. Reanudable.

Each (N, n_threads, K) cell = K concurrent in-flight requests, distributed
round-robin across the N replicas. We send 3 batches of K requests each
(total = 3K requests) so each replica handles roughly 3K/N requests.

Configs are derived from the D2 knee:
  (n_threads=4,  N=16, K∈{1,4,8,16,32})
  (n_threads=8,  N=8,  K∈{1,4,8,16})
  (n_threads=16, N=4,  K∈{1,4,8})
  (n_threads=32, N=2,  K∈{1,2,4})

NUMA-aligned cpusets: 8 cores per NUMA node on Xeon Gold 6430. Replicas
that fit in a single NUMA node get one each; replicas larger than 8 cores
span multiple nodes within a socket.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Reuse client / launcher helpers from run_sweep.py
sys.path.insert(0, str(Path(__file__).parent))
from run_sweep import (
    QUERIES, find_free_port, launch_server, wait_health, stop_server,
    send_query, Config, CSV_FIELDS, open_csv, append_row,
)


# ---- NUMA-aligned cpuset list for N replicas ------------------------------

NUMA_NODES = [
    (0, 7), (8, 15), (16, 23), (24, 31),       # socket 0
    (32, 39), (40, 47), (48, 55), (56, 63),    # socket 1
]


def cpuset_for_replica(rep_idx: int, n_threads: int, n_replicas: int) -> str:
    """Pin replica `rep_idx` (0..N-1) to a NUMA-aligned slice of `n_threads` cores."""
    cores_per_replica = 64 // n_replicas
    if cores_per_replica < n_threads:
        raise ValueError(f"Cannot fit N={n_replicas} replicas of {n_threads} threads in 64 cores")
    start = rep_idx * cores_per_replica
    # Use the first n_threads cores of that slice (more deterministic than scattering).
    return f"{start}-{start + n_threads - 1}"


# ---- D4 cell definition ---------------------------------------------------

@dataclass
class Cell:
    n_threads: int
    n_replicas: int
    concurrency: int      # K concurrent in-flight at a time
    config_id: str
    venv: str = ".venv-portable"

    def __post_init__(self):
        if self.concurrency > self.n_replicas * 4:
            # Cap K to a reasonable multiple of N (queue depth 4)
            pass


def build_d4_cells(venv: str) -> list[Cell]:
    cells = []
    for nT, N in [(4, 16), (8, 8), (16, 4), (32, 2)]:
        # K = 1, N (perfect parallel), and 2N (queue depth 2) when feasible
        ks = sorted(set([1, N, min(2 * N, 32)]))
        for K in ks:
            cells.append(Cell(
                n_threads=nT, n_replicas=N, concurrency=K, venv=venv,
                config_id=f"d4_nT{nT:02d}_N{N:02d}_K{K:02d}",
            ))
    return cells


# ---- Multi-replica orchestration ------------------------------------------

def launch_pool(cell: Cell, log_dir: Path) -> list[tuple[subprocess.Popen, int, str]]:
    """Return list of (proc, port, replica_id)."""
    procs = []
    for r in range(cell.n_replicas):
        port = find_free_port()
        cpuset = cpuset_for_replica(r, cell.n_threads, cell.n_replicas)
        cfg = Config(venv=cell.venv, n_threads=cell.n_threads,
                     cpuset=cpuset, config_id=cell.config_id)
        replica_id = f"r{r:02d}"
        log_path = log_dir / f"{cell.config_id}_{replica_id}.log"
        proc = launch_server(cfg, port, replica_id=replica_id, log_path=log_path)
        procs.append((proc, port, replica_id))
    # Wait for all healthy
    for proc, port, rid in procs:
        if not wait_health(port, timeout=120):
            for p, _, _ in procs:
                stop_server(p)
            raise RuntimeError(f"replica {rid} on port {port} never healthy")
    return procs


def warm_pool(procs: list, api_key: str) -> None:
    """Send 1 query per (replica, procedure) in parallel so snapshots are loaded."""
    jobs = [
        (port, proc_name, q, f"warm-{rid}-{qid}")
        for _, port, rid in procs
        for proc_name, qid, q in QUERIES
    ]
    print(f"  warming {len(jobs)} (replica × procedure) slots in parallel...")
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=min(len(procs), 16)) as ex:
        futs = [ex.submit(send_query, port, p, q, rid, api_key, 120)
                for port, p, q, rid in jobs]
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as e:
                print(f"    warmup error: {e}")
    print(f"  warmed in {time.perf_counter()-t0:.1f}s")


def run_cell(cell: Cell, runs_per_query: int, writer, fh, existing: set,
             api_key: str, log_dir: Path) -> None:
    procs = launch_pool(cell, log_dir)
    try:
        warm_pool(procs, api_key)
        ports = [p for _, p, _ in procs]

        # Build the request list: each (procedure, query) repeated runs_per_query times,
        # cycled across replicas round-robin (by request index).
        tasks = []
        for run_idx in range(1, runs_per_query + 1):
            for q_idx, (proc_name, qid, question) in enumerate(QUERIES):
                tasks.append((proc_name, qid, question, run_idx, q_idx))

        # Filter tasks already done
        todo = [
            t for t in tasks
            if (cell.config_id, t[1], t[3]) not in existing
        ]
        if not todo:
            print(f"  all {len(tasks)} tasks already done; skipping")
            return

        K = cell.concurrency
        executor = cf.ThreadPoolExecutor(max_workers=K)
        in_flight = {}
        results = []
        wall_start = time.perf_counter()

        def submit(task, port):
            proc_name, qid, question, run_idx, q_idx = task
            req_id = f"{cell.config_id}-{qid}-r{run_idx}"
            t0 = time.perf_counter()
            try:
                res = send_query(port, proc_name, question, req_id, api_key, timeout=300)
                res["_task"] = task
                res["_port"] = port
                res["_client_wall_ms"] = round((time.perf_counter() - t0) * 1000)
                return res
            except Exception as e:
                return {"_task": task, "_port": port, "_error": str(e)}

        # Round-robin assignment of tasks to replicas, K in flight
        next_port_idx = 0
        task_iter = iter(todo)
        # Prime
        try:
            for _ in range(min(K, len(todo))):
                task = next(task_iter)
                port = ports[next_port_idx % len(ports)]
                next_port_idx += 1
                fut = executor.submit(submit, task, port)
                in_flight[fut] = task

            while in_flight:
                done_fut = next(cf.as_completed(in_flight))
                task = in_flight.pop(done_fut)
                res = done_fut.result()
                results.append(res)
                # Print progress + write CSV
                if "_error" in res:
                    print(f"  [ERR] {task} -> {res['_error']}")
                    continue
                proc_name, qid, _, run_idx, _ = res["_task"]
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "phase": "d4",
                    "config_id": cell.config_id,
                    "venv": cell.venv.lstrip("./"),
                    "n_threads": cell.n_threads,
                    "cpuset": f"N={cell.n_replicas} pool",
                    "n_replicas": cell.n_replicas,
                    "procedure": proc_name,
                    "question_id": qid,
                    "run_idx": run_idx,
                    "ttft_ms_client": res.get("ttft_ms_client", 0),
                    "load_state_ms": res.get("load_state_ms", 0),
                    "prefill_ms": res.get("prefill_ms", 0),
                    "decode_ms": res.get("decode_ms", 0),
                    "output_tokens": res.get("completion_tokens", 0),
                    "decode_tok_s": res.get("decode_tok_s", 0.0),
                    "wall_ms": res.get("_client_wall_ms", 0),
                    "replica_id": res.get("replica_id", "?"),
                    "request_id": res.get("request_id", ""),
                    "note": f"K={cell.concurrency}",
                }
                append_row(writer, fh, row)
                print(f"  [K{cell.concurrency:02d}] {cell.config_id} {proc_name:18s} "
                      f"r{run_idx} via {row['replica_id']} "
                      f"decode={row['decode_ms']:5d}ms ({row['decode_tok_s']:.2f} tok/s)")
                # Submit next
                try:
                    nxt = next(task_iter)
                    port = ports[next_port_idx % len(ports)]
                    next_port_idx += 1
                    fut = executor.submit(submit, nxt, port)
                    in_flight[fut] = nxt
                except StopIteration:
                    pass
        finally:
            executor.shutdown(wait=True)

        wall_end = time.perf_counter() - wall_start
        n_ok = sum(1 for r in results if "_error" not in r)
        print(f"  Cell wall: {wall_end:.1f}s, {n_ok} reqs OK, "
              f"aggregate {n_ok / wall_end:.2f} req/s")
    finally:
        for p, _, _ in procs:
            stop_server(p)
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", type=int, default=3,
                    help="Measured runs per (procedure)")
    ap.add_argument("--venv", default=".venv-portable")
    ap.add_argument("--api-key", default=os.environ.get("RAG_API_KEY", ""))
    ap.add_argument("--filter")
    args = ap.parse_args()

    if not args.api_key:
        print("RAG_API_KEY missing", file=sys.stderr)
        return 2

    out = Path(args.out)
    log_dir = out.parent / "log" / "d4"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer, existing, fh = open_csv(out)
    cells = build_d4_cells(args.venv)
    if args.filter:
        cells = [c for c in cells if args.filter in c.config_id]

    print(f"Phase d4: {len(cells)} cells, runs={args.runs}, "
          f"existing rows={len(existing)}")
    print(f"CSV: {out}\n")

    t0 = time.time()
    try:
        for i, cell in enumerate(cells, 1):
            print(f"\n[{i}/{len(cells)}] {cell.config_id}  "
                  f"nT={cell.n_threads} N={cell.n_replicas} K={cell.concurrency}")
            run_cell(cell, args.runs, writer, fh, existing,
                     args.api_key, log_dir)
    finally:
        fh.close()
        print(f"\nDone in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
