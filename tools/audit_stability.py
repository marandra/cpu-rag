"""Measure how much the answer-vs-refuse decision moves between identical runs.

Runs on the cluster under `./.venv-native/bin/python`, like audit_replay.py.

    ./.venv-native/bin/python tools/audit_stability.py \
        --procedure diabetes --api-url http://<node>:8080 --runs 10

Why this exists, and why it comes before any prompt or model work:

The triage found three question pairs that ask the same thing in different words
and get opposite treatment (112/128, 91/103, 30/6), plus two more where one
answer keeps a rule's boundary and its twin breaks it (67/86, 87/88). That is
not a missing instruction — it is variance. And while the variance is unmeasured,
no A/B is interpretable: "V14 fixed five questions" and "V14 changed nothing and
we sampled a noisy process twice" look identical.

So this asks every question N times against one unchanged pool — same prompt,
same snapshot, same model — and reports two things:

  1. The noise floor: how much the headline refusal count moves run to run. Any
     future comparison has to beat this to mean anything.
  2. The partition that decides where to spend effort. Cross-referencing the
     triage verdicts splits the over-refusals into

       stably refused  the system reads the fulldoc the same way every time and
                       still refuses -> deterministic, and the prompt can move it
       oscillating     the same question lands on both sides across runs -> no
                       rule can fix this; it is sampling or model capability

Questions come from the versioned replays, not from the auditors' spreadsheet,
so this runs from a fresh clone. Output goes to eval/audit_stability/ for the
same reason the replays live in eval/: temperature is 0.1 with no fixed seed, so
a run can be replaced but never reproduced.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audit_replay import ask
from audit_triage import REPLAY_DIR, TRIAGE, refused
from rag_client import get_health, load_dotenv, resolve_api_url

STABILITY_DIR = "eval/audit_stability"

# The pairs that motivated the measurement: same material, opposite outcome.
# Ordered (handled, defective) and checked against TRIAGE below, because the
# whole point is reading which member moves — getting the polarity backwards
# would invert the conclusion.
#
#   sobre-rechazo  the defective member refuses what the fulldoc covers (SR)
#   frontera       both answer; the defective member breaks a rule's boundary
CONTROL_PAIRS = {
    "sobre-rechazo": [(112, 128), (91, 103), (30, 6)],
    "frontera": [(86, 67), (88, 87)],
}


def _check_pair_polarity() -> None:
    """Fail loudly if a pair is entered backwards or a verdict moves under it."""
    for handled, defective in CONTROL_PAIRS["sobre-rechazo"]:
        if TRIAGE[defective][0] != "SR":
            raise SystemExit(
                f"pair ({handled}, {defective}): expected {defective} to be the "
                f"over-refusal, but its verdict is {TRIAGE[defective][0]}"
            )
        if TRIAGE[handled][0] in ("SR", "FN"):
            raise SystemExit(
                f"pair ({handled}, {defective}): {handled} is a refusal "
                f"({TRIAGE[handled][0]}), so it cannot be the handled member"
            )
    for handled, defective in CONTROL_PAIRS["frontera"]:
        for qid in (handled, defective):
            if TRIAGE[qid][0] in ("SR", "FN"):
                raise SystemExit(
                    f"pair ({handled}, {defective}): {qid} is a refusal "
                    f"({TRIAGE[qid][0]}); boundary pairs must both answer"
                )
        if TRIAGE[defective][0] != "DEF":
            raise SystemExit(
                f"pair ({handled}, {defective}): expected {defective} to be the "
                f"defect, but its verdict is {TRIAGE[defective][0]}"
            )


_check_pair_polarity()


def load_questions(replays: Path, procedure: str) -> list[dict]:
    """Question records for one procedure, taken from its versioned replay."""
    payload = json.loads((replays / f"{procedure}.json").read_text(encoding="utf-8"))
    return [
        {"id": r["id"], "question": r["question"], "procedure": r["procedure"]}
        for r in payload["rows"]
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--procedure", required=True)
    p.add_argument("--api-url", default=None)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent requests; match the pool's replica count.")
    p.add_argument("--replays", default=REPLAY_DIR)
    p.add_argument("--out", default=None,
                   help=f"Default {STABILITY_DIR}/<procedure>.json")
    args = p.parse_args()

    load_dotenv()
    api_url = resolve_api_url(args.api_url)

    recs = load_questions(Path(args.replays), args.procedure)
    if not recs:
        print(f"No questions for procedure {args.procedure!r}", file=sys.stderr)
        return 1

    health = get_health(api_url)
    print(f"API {api_url}  profile={health.get('profile','?')}  "
          f"model={health.get('model','?')}")
    if args.procedure not in (health.get("procedures") or []):
        print(f"ERROR: the service does not serve {args.procedure!r} — "
              f"wrong profile or wrong pool.", file=sys.stderr)
        return 1

    # runs[i][id] -> answer text for pass i
    runs: list[dict[int, str]] = []
    failures = 0
    t0 = time.perf_counter()
    for i in range(args.runs):
        rt = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda r: ask(r, api_url), recs))
        bad = [r["id"] for r in rows if r["our_error"]]
        failures += len(bad)
        runs.append({r["id"]: r["our_answer"] for r in rows if not r["our_error"]})
        n_ref = sum(1 for r in rows if not r["our_error"] and refused(r["our_answer"]))
        print(f"  run {i+1}/{args.runs}: {n_ref}/{len(rows)} refusals "
              f"({time.perf_counter()-rt:.0f}s)"
              + (f"  FAILED {bad}" if bad else ""))
    wall = time.perf_counter() - t0

    # --- per question: how many of N runs refused ----------------------------
    per_q: dict[int, dict] = {}
    for rec in recs:
        qid = rec["id"]
        answers = [r[qid] for r in runs if qid in r]
        refusals = [refused(a) for a in answers]
        verdict = TRIAGE.get(qid, ("?", "", ""))[0]
        per_q[qid] = {
            "id": qid,
            "question": rec["question"],
            "verdict": verdict,
            "n": len(answers),
            "refused": sum(refusals),
            "answers": answers,
        }

    counted = [q for q in per_q.values() if q["n"] > 0]
    oscillating = [q for q in counted if 0 < q["refused"] < q["n"]]
    stable_ref = [q for q in counted if q["refused"] == q["n"]]
    stable_ans = [q for q in counted if q["refused"] == 0]

    payload = {
        "procedure": args.procedure,
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "api_url": api_url,
        "profile": health.get("profile"),
        "model": health.get("model"),
        "runs": args.runs,
        "failures": failures,
        "questions": list(per_q.values()),
    }
    dest = Path(args.out or f"{STABILITY_DIR}/{args.procedure}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    # --- report --------------------------------------------------------------
    per_run_refusals = [
        sum(1 for a in r.values() if refused(a)) for r in runs
    ]
    print(f"\nWrote {dest}  ({args.runs} runs x {len(recs)} Q, {wall/60:.1f} min)")
    if failures:
        print(f"WARNING: {failures} request failures across all runs")

    print(f"\nNoise floor — refusals per run: {per_run_refusals}")
    if len(per_run_refusals) > 1:
        sd = statistics.stdev(per_run_refusals)
        print(f"  mean {statistics.mean(per_run_refusals):.1f}  "
              f"sd {sd:.1f}  range {min(per_run_refusals)}-{max(per_run_refusals)}")
        print(f"  => a prompt change must move the count by more than ~{2*sd:.0f} "
              "to be distinguishable from noise.")

    print(f"\nDecision stability over {len(counted)} questions:")
    print(f"  stably answered : {len(stable_ans)}")
    print(f"  stably refused  : {len(stable_ref)}")
    print(f"  OSCILLATING     : {len(oscillating)}")

    # The diagnostic: which over-refusals are deterministic and which are noise.
    sr = [q for q in counted if q["verdict"] == "SR"]
    if sr:
        sr_osc = [q for q in sr if 0 < q["refused"] < q["n"]]
        print(f"\nOver-refusals (SR) in this procedure: {len(sr)}")
        print(f"  stably refused -> prompt can address : {len(sr) - len(sr_osc)}")
        print(f"  oscillating    -> not a prompt problem: {len(sr_osc)}"
              f"  {sorted(q['id'] for q in sr_osc)}")

    print("\nControl pairs (expect the defective member to oscillate):")
    for kind, pairs in CONTROL_PAIRS.items():
        for good, bad in pairs:
            cells = []
            for qid in (good, bad):
                q = per_q.get(qid)
                cells.append(f"{qid}={q['refused']}/{q['n']}" if q else f"{qid}=n/a")
            print(f"  {kind:14s} {cells[0]:>12s}  vs  {cells[1]:>12s}")

    if oscillating:
        print("\nOscillating questions (refused/runs):")
        for q in sorted(oscillating, key=lambda q: -q["refused"]):
            print(f"  {q['id']:>4}  {q['refused']}/{q['n']}  [{q['verdict']}]  "
                  f"{q['question'][:64]}")

    by_verdict = Counter(q["verdict"] for q in oscillating)
    if by_verdict:
        print(f"\nOscillating by triage verdict: {dict(by_verdict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
