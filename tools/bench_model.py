"""Screen a candidate model on the shape the service actually runs.

    MODEL_PATH=models/Qwen3.5-4B-Q4_K_M.gguf PROFILE=glucowise N_THREADS=8 \
        ./.venv-latest/bin/python tools/bench_model.py \
            --procedure diabetes --out bench/qwen35-4b.diabetes.json

Screen 1 of [[plan-b-model-benchmark]]: kill anything under the 6 tok/s floor
before spending quality compute on it. It measures the four costs a model swap
moves, and three of them no leaderboard reports:

  * **decode tok/s** — the floor is >6 worst case, ~11 is good, and what counts
    is per-user under saturation, not a single idle process. Run this 8x in
    parallel, one per NUMA node, to get that number (see model_screen.sbatch).
  * **`load_state` ms** — `app/routes/query.py` restores the prefix KV before
    *every* generation. 205 ms per request for Ministral. More layers / more KV
    heads pay this on every single query, not once at warm.
  * **pickle MiB** — ~460-710 MB per (prompt x procedure) today, and that, not
    cores, is what caps how many variants and procedures fit on a node.
  * **prefix tokens** — a different tokenizer means a different prefill and a
    different KV for the same corpus.

Deliberately does not write snapshots. It warms in-process and keeps the state
in RAM, so it can never touch the pickles a pool is serving (the layout fix of
2026-07-22 made the default root resolve to the live one).

Speed is only half a screen: it also dumps every answer, because a model that
is fast and does not follow the REGLA is not a candidate. `<think>` leakage and
language drift show up there, and both are silent in the timings.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

REPLAY_DIR = "eval/audit_replay"

# Questions to generate. Not arbitrary: these are the boundary defects the
# audit work isolated (67 de-scope, 87/108 subject, 29/26/84 fusion, 105
# disjunction) plus a control pair per procedure — 67/86 and 87/88 for the
# boundary, 30/6 and 91/103 and 112/128 for over-refusal. Speed barely depends
# on which questions run, but the dumped text does, and these are the ones we
# can read against a known failure. Ids are global across the 134 and the
# procedures own disjoint ranges: diabetes 1-55, cirugia 56-103, hemorroides
# 104-134. Falls back to the first N ids of the replay.
PROBE_IDS = {
    "diabetes": [29, 26, 4, 30, 6, 52],
    "cirugia-abdominal": [67, 86, 87, 84, 91, 103],
    "hemorroides": [108, 105, 112, 128, 111, 117],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--procedure", required=True)
    p.add_argument("--ids", default=None,
                   help="Comma-separated question ids. Default: the boundary "
                        "probes and control pairs for this procedure.")
    p.add_argument("--replays", default=REPLAY_DIR)
    p.add_argument("--repeat", type=int, default=1,
                   help="Passes over the id list. >1 tightens the median.")
    p.add_argument("--seed", type=int, default=1894574933,
                   help="The frozen generation seed diabetes has always used, "
                        "so a re-run is comparable. Screening is not an A/B: "
                        "one seed is fine because we are measuring speed, and "
                        "the text is read, not scored.")
    p.add_argument("--fulldoc", default=None,
                   help="Override the profile's configured fulldoc, e.g. "
                        "corpus/markdown/diabetes.v4.md, to measure warm and "
                        "tok/s for a corpus variant. The prefix is (system + "
                        "fulldoc), so a bigger doc is a longer prefill and a "
                        "slower decode — exactly what this flag exists to size.")
    p.add_argument("--label", default=None, help="Tag for the output row.")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import llama_cpp

    from app.config import settings
    from app.prompt import get_system_prompt
    from src.llm import load_model

    proc = args.procedure
    if proc not in settings.fulldoc_procedures:
        print(f"ERROR: PROFILE={settings.profile!r} does not own {proc!r}. "
              f"It owns {sorted(settings.fulldoc_procedures)}.", file=sys.stderr)
        return 1

    model_path = Path(settings.model_path)
    if not model_path.is_file():
        print(f"ERROR: no such model: {model_path}", file=sys.stderr)
        return 1
    label = args.label or model_path.stem

    fulldoc_path = Path(args.fulldoc) if args.fulldoc \
        else settings.fulldoc_procedures[proc]
    if not fulldoc_path.is_file():
        print(f"ERROR: no such fulldoc: {fulldoc_path}", file=sys.stderr)
        return 1
    fulldoc_text = fulldoc_path.read_text(encoding="utf-8")
    system_prompt = get_system_prompt(proc)

    payload = json.loads(
        (Path(args.replays) / f"{proc}.json").read_text(encoding="utf-8"))
    questions = {r["id"]: r["question"] for r in payload["rows"]}
    if args.ids:
        ids = [int(i) for i in args.ids.split(",")]
    else:
        ids = [i for i in PROBE_IDS.get(proc, []) if i in questions]
        ids = ids or sorted(questions)[:6]

    n_threads = settings.n_threads or None
    print(f"== {label}  procedure={proc}  threads={n_threads or 'default'}  "
          f"n_ctx={settings.n_ctx}  max_tokens={settings.max_tokens}")
    print(f"   fulldoc {fulldoc_path} ({len(fulldoc_text)} car), "
          f"system prompt {settings.prompt_variant}, {len(ids)} questions "
          f"x {args.repeat}")

    load_kwargs: dict = {"path": str(model_path), "n_ctx": settings.n_ctx}
    if settings.n_threads is not None:
        load_kwargs["n_threads"] = settings.n_threads
    t0 = time.perf_counter()
    llm = load_model(**load_kwargs)
    load_s = time.perf_counter() - t0

    md = llm.metadata
    arch = md.get("general.architecture", "?")
    meta = {k: v for k, v in md.items()
            if k.startswith(f"{arch}.") and ("block_count" in k or "head_count" in k
                                             or "expert" in k or "ssm." in k)}
    # The decisive property for *this* service, and no leaderboard reports it.
    # llama.cpp cannot partially evict a recurrent/hybrid memory, so
    # `Llama.generate`'s prefix-match hit fails ("partial kv removal not
    # supported, re-evaluating full prompt") and every request re-prefills the
    # whole (system + fulldoc) prefix instead of restoring it. Our whole design
    # is that restore. Measured below as ttft vs warm, this flag says why.
    recurrent = bool(llama_cpp.llama_model_is_recurrent(llm._model.model))
    hybrid = bool(llama_cpp.llama_model_is_hybrid(llm._model.model))
    print(f"   load {load_s:6.1f} s | arch={arch} "
          f"size={md.get('general.size_label', '?')} "
          f"chat_format={llm.chat_format!r} "
          f"recurrent={recurrent} hybrid={hybrid}")
    print(f"   {meta}")

    # Warm the same prefix snapshot_builder does, byte for byte, so the KV
    # holds everything up to the user's question. Kept in RAM: writing it out
    # is what the snapshots-root guard exists to prevent.
    t0 = time.perf_counter()
    llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: hola"},
        ],
        max_tokens=1, temperature=0.1, seed=args.seed,
    )
    warm_s = time.perf_counter() - t0
    n_prefix = llm.n_tokens
    print(f"   warm {warm_s:6.1f} s | prefix {n_prefix} tok "
          f"({n_prefix / warm_s:6.1f} tok/s prefill)")

    t0 = time.perf_counter()
    state = llm.save_state()
    save_s = time.perf_counter() - t0
    pickle_bytes = len(pickle.dumps(state))
    print(f"   save_state {save_s:5.1f} s | pickle {pickle_bytes / 2**20:8.1f} MiB")

    rows = []
    for _ in range(args.repeat):
        for qid in ids:
            t0 = time.perf_counter()
            llm.load_state(state)
            restore_ms = (time.perf_counter() - t0) * 1000

            # Streamed for the same reason query.py streams: it separates the
            # question's prefill from decode. Lumping them understates tok/s
            # and hides a model whose prefill is the real cost.
            gen_start = time.perf_counter()
            first_t = None
            chunks = []
            for chunk in llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content":
                     f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: {questions[qid]}"},
                ],
                max_tokens=settings.max_tokens, temperature=0.1,
                seed=args.seed, stream=True,
            ):
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    if first_t is None:
                        first_t = time.perf_counter()
                    chunks.append(text)
            end = time.perf_counter()
            if first_t is None:  # produced nothing at all
                first_t = end
            answer = "".join(chunks)
            ntok = len(chunks)
            decode_s = end - first_t
            tok_s = (ntok - 1) / decode_s if ntok > 1 and decode_s > 0 else 0.0
            rows.append({
                "id": qid, "question": questions[qid], "answer": answer,
                "load_state_ms": restore_ms,
                "ttft_ms": (first_t - gen_start) * 1000,
                "decode_s": decode_s, "completion_tokens": ntok,
                "decode_tok_s": tok_s,
                "has_think": "<think>" in answer or "<|channel>" in answer,
            })
            print(f"   q{qid:<4} load_state {restore_ms:6.0f} ms | ttft "
                  f"{(first_t-gen_start)*1000:7.0f} ms | {ntok:4d} tok "
                  f"@ {tok_s:6.2f} tok/s"
                  + ("  !! THINK LEAK" if rows[-1]["has_think"] else ""))

    med = statistics.median(r["decode_tok_s"] for r in rows)
    med_ls = statistics.median(r["load_state_ms"] for r in rows)
    med_ttft = statistics.median(r["ttft_ms"] for r in rows)
    # A question adds ~30 tokens to a prefix of thousands, so with the prefix
    # reused, ttft is a fraction of the warm. At warm-scale, the restore bought
    # nothing and the model re-prefilled the whole document.
    reuse = med_ttft < 0.25 * warm_s * 1000
    print(f"   MEDIAN decode {med:7.2f} tok/s | load_state {med_ls:6.0f} ms "
          f"| ttft {med_ttft:7.0f} ms vs warm {warm_s*1000:7.0f} ms "
          f"| verdict {'PASS' if med >= 6 else 'BELOW FLOOR'}")
    if not reuse:
        print("   !! NO PREFIX REUSE: ttft is warm-scale, so load_state is not "
              "buying a cached prefix. Expected on recurrent/hybrid memory — "
              "llama.cpp cannot partially evict it, so every request re-prefills "
              "the whole fulldoc. Disqualifying for this service.")

    out = {
        "label": label, "model": model_path.name, "procedure": proc,
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "host": os.uname().nodename,
        "arch": arch, "size_label": md.get("general.size_label"),
        "arch_meta": meta, "chat_format": llm.chat_format,
        "is_recurrent": recurrent, "is_hybrid": hybrid,
        "prefix_reuse": reuse, "median_ttft_ms": med_ttft,
        "n_threads": n_threads, "n_ctx": settings.n_ctx,
        "max_tokens": settings.max_tokens, "prompt_variant": settings.prompt_variant,
        "model_load_s": load_s, "warm_s": warm_s, "prefix_tokens": n_prefix,
        "prefill_tok_s": n_prefix / warm_s,
        "save_state_s": save_s, "pickle_bytes": pickle_bytes,
        "median_decode_tok_s": med, "median_load_state_ms": med_ls,
        "think_leaks": sum(1 for r in rows if r["has_think"]),
        "runs": rows,
    }
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"   wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
