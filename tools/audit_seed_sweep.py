"""Find the real noise floor: vary the RNG seed and the temperature.

Runs on the cluster, on an idle node, against the model directly — no pool, no
API, no change to `app/`:

    PROFILE=glucowise ./.venv-native/bin/python tools/audit_seed_sweep.py \
        --procedure diabetes --seeds 8

Why this exists
---------------

`tools/audit_stability.py` measured zero variance: 10 passes x 134 questions,
every generation byte-identical. That is not the model being confident. It is a
**frozen seed**, introduced by accident by the snapshot mechanism:

  1. `LlamaState` pickles a `seed` field, and `Llama.load_state()` restores it
     (`llama_cpp/llama.py`: `self._seed = state.seed`).
  2. `app/routes/query.py` calls `load_state()` before *every* generation, to
     put the KV back to the (system + fulldoc) prefix.
  3. `Llama._create_completion()` with no explicit seed does
     `set_seed(random.Random(self._seed).randint(0, 2**32))`.

So step 1 resets `_seed` to the constant stored in the pickle before every
request, and step 3 derives the same number from it every time. Every answer
this service has ever produced for a given procedure used one single seed.

That matters for more than trivia. Editing the system prompt changes the
snapshot key (`app/snapshot_cache.py`), so a prompt variant is a *different
pickle with a different stored seed*. Comparing V13 against V14 would compare
prompt and seed at once, and the noise floor of 0 would not apply to it. This
script measures the thing that comparison actually needs: how much the answer
moves when only the seed moves, at the temperature we deploy.

The temperature axis is the control. If answers barely move across seeds at
0.1 but move a lot at 0.8, the frozen seed was masking little; if they move at
0.1 too, every single-seed number we have — theirs and ours — is one draw.

Questions come from the versioned replays, like audit_stability.py. The default
subset is the questions the triage actually argues about (the control pairs and
the over-refusals), because this generates in one stream and paying for all 134
buys little.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_stability import CONTROL_PAIRS
from audit_triage import REPLAY_DIR, TRIAGE, refused

SWEEP_DIR = "eval/audit_seed_sweep"

# Never the repo's ./snapshots: a pool re-reads its pkl on every request, and
# this tool warms (and writes) on a cache MISS.
SCRATCH_SNAPSHOTS = "/tmp/cpu-rag-offline-snapshots"

# Fixed so a re-run is comparable; arbitrary otherwise.
SEEDS = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def subset_ids(procedure_ids: set[int], subset: str) -> list[int]:
    """Which questions to sweep. Ordered, so the report reads consistently."""
    control = {q for pairs in CONTROL_PAIRS.values() for pair in pairs for q in pair}
    if subset == "all":
        want = procedure_ids
    elif subset == "control":
        want = control
    else:  # "argued": the control pairs plus every over-refusal and defect
        want = control | {i for i, v in TRIAGE.items() if v[0] in ("SR", "DEF")}
    return sorted(procedure_ids & want)


def load_questions(replays: Path, procedure: str) -> dict[int, str]:
    payload = json.loads((replays / f"{procedure}.json").read_text(encoding="utf-8"))
    return {r["id"]: r["question"] for r in payload["rows"]}


def load_replay_answers(replays: Path, procedure: str) -> dict[int, str]:
    payload = json.loads((replays / f"{procedure}.json").read_text(encoding="utf-8"))
    return {r["id"]: r.get("our_answer") or "" for r in payload["rows"]}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--procedure", required=True)
    p.add_argument("--subset", default="argued",
                   choices=("argued", "control", "all"))
    p.add_argument("--seeds", type=int, default=8,
                   help=f"How many of {SEEDS} to use.")
    p.add_argument("--temperatures", default="0.1",
                   help="Comma-separated. 0.1 is what we deploy.")
    p.add_argument("--fulldoc", default=None,
                   help="Override the procedure's distilled markdown. For the "
                        "corpus A/B: the fulldoc is part of the snapshot cache "
                        "key, so a variant warms its own pickle and cannot "
                        "collide with the base one.")
    p.add_argument("--replays", default=REPLAY_DIR)
    p.add_argument("--out", default=None)
    p.add_argument("--snapshots-root", default=SCRATCH_SNAPSHOTS,
                   help="Scratch root for snapshots this run may build. Must "
                        "NOT be the repo's ./snapshots — a pool serves from "
                        "there and re-reads the pkl on every request.")
    args = p.parse_args()

    temps = [float(t) for t in args.temperatures.split(",")]
    seeds = SEEDS[: args.seeds]

    from app.config import settings
    from app.prompt import get_system_prompt
    from app.snapshot_builder import build_or_load_snapshot
    from src.llm import load_model

    # `settings.snapshots_dir` resolves to <root>/<profile>, and the default
    # root is the one the pools serve. Redirect to scratch before anything can
    # build: this tool warms on a MISS and would overwrite a live pkl.
    root = Path(args.snapshots_root).resolve()
    if root == Path(settings.snapshots_root).resolve():
        print(f"ERROR: --snapshots-root is the serving root ({root}). Point it "
              f"at scratch, e.g. --snapshots-root {SCRATCH_SNAPSHOTS}",
              file=sys.stderr)
        return 1
    settings.snapshots_root = root

    proc = args.procedure
    if proc not in settings.fulldoc_procedures:
        print(f"ERROR: PROFILE={settings.profile!r} does not own {proc!r}. "
              f"It owns {sorted(settings.fulldoc_procedures)}.", file=sys.stderr)
        return 1

    questions = load_questions(Path(args.replays), proc)
    ids = subset_ids(set(questions), args.subset)
    print(f"profile={settings.profile} procedure={proc} "
          f"questions={len(ids)} seeds={seeds} temps={temps}")

    fulldoc_path = Path(args.fulldoc) if args.fulldoc else settings.fulldoc_procedures[proc]
    if not fulldoc_path.is_file():
        print(f"ERROR: no such fulldoc: {fulldoc_path}", file=sys.stderr)
        return 1
    fulldoc_text = fulldoc_path.read_text(encoding="utf-8")
    if args.fulldoc:
        print(f"fulldoc override: {fulldoc_path} ({len(fulldoc_text)} car)")
    system_prompt = get_system_prompt(proc)

    load_kwargs: dict = {"path": str(settings.model_path), "n_ctx": settings.n_ctx}
    if settings.n_threads is not None:
        load_kwargs["n_threads"] = settings.n_threads
    print(f"Loading {Path(settings.model_path).stem}...")
    llm = load_model(**load_kwargs)

    # Same snapshot the service uses; a MISS here would warm it, which is fine
    # but slow. Reusing it keeps this comparable to the replays.
    state, cached = build_or_load_snapshot(
        llm, proc, fulldoc_text, settings, fulldoc_path=fulldoc_path
    )
    if state is None:
        print("ERROR: no snapshot", file=sys.stderr)
        return 1
    # The seed the service actually uses: load_state restores state.seed, then
    # _create_completion derives this from it. Sweeping it as an extra condition
    # gives a free check that this venv reproduces what the container served —
    # if the "frozen" column matches the replay, the numerics agree.
    frozen = random.Random(state.seed).randint(0, 2**32)
    seeds = [frozen] + [s for s in seeds if s != frozen]
    print(f"Snapshot {'HIT' if cached else 'BUILT'}; stored seed={state.seed}, "
          f"frozen generation seed={frozen}")

    # results[(temp, seed)][qid] -> answer
    results: dict[tuple[float, int], dict[int, str]] = {}
    t0 = time.perf_counter()
    for temp in temps:
        for seed in seeds:
            cell: dict[int, str] = {}
            ct = time.perf_counter()
            for qid in ids:
                # Exactly what app/routes/query.py does per request: restore the
                # prefix KV first, or the previous answer stays in context.
                llm.load_state(state)
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content":
                         f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: {questions[qid]}"},
                    ],
                    max_tokens=settings.max_tokens,
                    temperature=temp,
                    seed=seed,
                )
                cell[qid] = out["choices"][0]["message"]["content"] or ""
            results[(temp, seed)] = cell
            n_ref = sum(1 for a in cell.values() if refused(a))
            print(f"  t={temp} seed={seed:>3}: {n_ref}/{len(ids)} refusals "
                  f"({time.perf_counter()-ct:.0f}s)")
    wall = time.perf_counter() - t0

    payload = {
        "procedure": proc,
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "profile": settings.profile,
        "model": Path(settings.model_path).stem,
        "snapshot_seed": state.seed,
        "subset": args.subset,
        "seeds": seeds,
        "temperatures": temps,
        "questions": [
            {
                "id": qid,
                "question": questions[qid],
                "verdict": TRIAGE.get(qid, ("?",))[0],
                "answers": {f"{t}|{s}": results[(t, s)][qid]
                            for t in temps for s in seeds},
            }
            for qid in ids
        ],
    }
    dest = Path(args.out or f"{SWEEP_DIR}/{proc}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\nWrote {dest}  ({wall/60:.1f} min)")

    # Does this venv reproduce what the container served? The frozen column is
    # the same seed the pool used, so any mismatch is build numerics, not
    # sampling — llama-cpp 0.3.19 native here vs 0.3.23 in the delivered image.
    if 0.1 in temps:
        replay = load_replay_answers(Path(args.replays), proc)
        cell = results[(0.1, frozen)]
        agree = sum(1 for q in ids if cell[q] == replay.get(q))
        print(f"\nFrozen seed vs the served replay: {agree}/{len(ids)} identical")

    # --- what the seed actually moves -----------------------------------------
    for temp in temps:
        cells = [results[(temp, s)] for s in seeds]
        per_seed_ref = [sum(1 for a in c.values() if refused(a)) for c in cells]
        flipped, reworded = [], []
        for qid in ids:
            answers = [c[qid] for c in cells]
            if len({refused(a) for a in answers}) > 1:
                flipped.append(qid)
            if len(set(answers)) > 1:
                reworded.append(qid)
        print(f"\n=== temperature {temp} over {len(seeds)} seeds, {len(ids)} questions")
        print(f"  refusals per seed : {per_seed_ref}")
        print(f"  answer text moved : {len(reworded)}/{len(ids)} "
              f"({len(reworded)/len(ids):.0%})  {reworded}")
        print(f"  DECISION flipped  : {len(flipped)}/{len(ids)} "
              f"({len(flipped)/len(ids):.0%})  {flipped}")
        if flipped:
            print("  -> these are seed noise, not a property of the prompt.")
        else:
            print("  -> the decision is seed-independent at this temperature.")

        print("  control pairs (refusals across seeds):")
        for kind, pairs in CONTROL_PAIRS.items():
            for good, bad in pairs:
                if good not in results[(temp, seeds[0])]:
                    continue
                cnt = {q: sum(1 for c in cells if refused(c[q])) for q in (good, bad)}
                print(f"    {kind:14s} {good}={cnt[good]}/{len(seeds)}  "
                      f"vs  {bad}={cnt[bad]}/{len(seeds)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
