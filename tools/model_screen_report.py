"""Turn a directory of bench_model.py outputs into the screen-1 table.

    uv run python tools/model_screen_report.py eval/model_screen/solo
    uv run python tools/model_screen_report.py eval/model_screen/saturated

Both modes land in the same shape: one JSON per cell, and in `saturated` there
are N cells of the same model (label `<model>-r<i>`), which is the number that
decides. The floor is per user under saturation, so the saturated column
aggregates across replicas as **median of the per-cell medians**, and prints the
spread — a wide spread across identical cells means the node was not evenly
loaded and the number is not yet trustworthy.

Columns beyond tok/s are here because they are what a model swap actually costs
us and no leaderboard reports them: `load_state` is paid on *every* request
(205 ms today), the pickle is what caps variants-per-node, and prefix tokens
show a tokenizer charging more for the same corpus.
"""

from __future__ import annotations

import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

FLOOR = 6.0        # tok/s per user, worst case
GOOD = 11.0        # what the incumbent gets at 8x8


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    cells = sorted(root.glob("*.json"))
    if not cells:
        print(f"no cells in {root}", file=sys.stderr)
        return 1

    by_model: dict[str, list[dict]] = defaultdict(list)
    for f in cells:
        d = json.loads(f.read_text(encoding="utf-8"))
        # `<label>-r<i>` in saturated mode, plain `<label>` in solo.
        by_model[re.sub(r"-r\d+$", "", d["label"])].append(d)

    print(f"# model screen — {root}   ({len(cells)} cells, "
          f"{len(by_model)} models)\n")
    hdr = (f"{'model':16s} {'arch':10s} {'n':>2s} {'tok/s':>7s} {'spread':>13s} "
           f"{'ttft':>8s} {'load_state':>10s} {'pickle':>9s} {'prefix':>7s} "
           f"{'prefill':>8s} {'warm':>6s} {'think':>5s}  verdict")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for label, ds in sorted(by_model.items(),
                            key=lambda kv: -statistics.median(
                                d["median_decode_tok_s"] for d in kv[1])):
        toks = [d["median_decode_tok_s"] for d in ds]
        med = statistics.median(toks)
        d0 = ds[0]
        warm = statistics.median(d["warm_s"] for d in ds)
        ttft = statistics.median(
            statistics.median(r["ttft_ms"] for r in d["runs"]) for d in ds)
        # Computed here and not only in the bench so cells recorded before the
        # check existed are still judged by it — the per-run ttft was always in
        # the payload. A question is ~30 tokens on a prefix of thousands: with
        # the prefix reused ttft is a fraction of the warm, and at warm-scale
        # the snapshot restore bought nothing.
        reuse = ttft < 0.25 * warm * 1000
        verdict = ("PASS" if med >= GOOD else
                   "thin" if med >= FLOOR else "BELOW FLOOR")
        if not reuse:
            verdict = "NO PREFIX REUSE"
        leaks = sum(d["think_leaks"] for d in ds)
        if leaks:
            verdict += f"  !! {leaks} think leaks"
        print(f"{label:16s} {d0['arch']:10s} {len(ds):2d} {med:7.2f} "
              f"{min(toks):6.2f}-{max(toks):<6.2f} "
              f"{ttft:7.0f}ms "
              f"{statistics.median(d['median_load_state_ms'] for d in ds):8.0f}ms "
              f"{statistics.median(d['pickle_bytes'] for d in ds)/2**20:7.0f}MiB "
              f"{d0['prefix_tokens']:7d} "
              f"{statistics.median(d['prefill_tok_s'] for d in ds):8.1f} "
              f"{warm:5.0f}s "
              f"{leaks:5d}  {verdict}")
        rows.append((label, ds))

    print("\n## answers — the half a timing cannot screen")
    for label, ds in rows:
        print(f"\n### {label}")
        for r in ds[0]["runs"]:
            ans = " ".join(r["answer"].split())
            print(f"  q{r['id']:<4} {r['completion_tokens']:4d} tok  {ans[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
