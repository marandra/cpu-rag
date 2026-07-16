"""Run baseline eval over coverage + grayzone datasets for a procedure.

Hits the dockerized /query (SSE), reassembles each answer, computes
deterministic metrics (keyword coverage, refusal correctness) and writes
a Markdown report per dataset under ./reports/.

Usage:
    uv run python tools/run_eval.py --procedure diabetes
    uv run python tools/run_eval.py --procedure all
    uv run python tools/run_eval.py --dataset eval/datasets/eval_dataset_diabetes_coverage.json \\
        --procedure-slug diabetes --label diabetes_coverage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

# Allow `tools/rag_client.py` to be importable when run from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_client import get_health, load_dotenv, resolve_api_url, stream_query

PROCEDURES = ["diabetes", "hemorroides", "cirugia-abdominal"]
SETS = ["coverage", "grayzone"]
DATASETS_DIR = Path("./eval/datasets")
REFUSAL = "no tengo información sobre eso"
REPORTS_DIR = Path("./reports")


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def keyword_hits(answer: str, keywords: list[str]) -> list[bool]:
    a = _normalize(answer)
    return [_normalize(k) in a for k in keywords]


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def run_one_query(item: dict, api_url: str, procedure_slug: str) -> dict:
    query = item["query"]
    t_start = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    usage: dict = {}
    error: dict | None = None

    try:
        for ev in stream_query(query, procedure_slug, api_url=api_url):
            if ev.event == "chunk":
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(ev.data.get("text", ""))
            elif ev.event == "done":
                usage = ev.data.get("usage", {})
            elif ev.event == "error":
                error = ev.data
                break
    except Exception as e:
        return {"error": "client_exception", "detail": str(e),
                "response": "".join(parts)}

    t_end = time.perf_counter()
    response = "".join(parts)
    if error:
        return {"error": error.get("code", "error"),
                "detail": error.get("detail", ""),
                "response": response}

    ttft = (t_first or t_end) - t_start
    total = t_end - t_start
    gen_s = usage.get("generation_ms", 0) / 1000.0
    n_tok = usage.get("completion_tokens", 0)
    decode_s = (t_end - t_first) if t_first else 0.0
    tps = (n_tok - 1) / decode_s if decode_s > 0 and n_tok > 1 else 0.0

    return {
        "response": response, "ttft": ttft, "total": total,
        "generation_s": gen_s, "n_tokens": n_tok, "tps": tps,
    }


def score(item: dict, res: dict) -> dict:
    """Per-item deterministic metrics."""
    answer = res.get("response", "")
    answer_norm = _normalize(answer)
    refused = _normalize(REFUSAL) in answer_norm

    if item["answerable"]:
        hits = keyword_hits(answer, item.get("expected_keywords") or [])
        cov = sum(hits) / len(hits) if hits else 1.0
        return {
            "answerable": True,
            "refused": refused,                  # false-positive refusal if True
            "keyword_hits": hits,
            "keyword_coverage": cov,
            "correct_refusal": None,
        }
    else:
        return {
            "answerable": False,
            "refused": refused,
            "keyword_hits": None,
            "keyword_coverage": None,
            "correct_refusal": refused,
        }


def run_dataset(dataset_path: Path, procedure_slug: str,
                api_url: str, label: str, health: dict) -> Path:
    items = load_dataset(dataset_path)
    print(f"\n=== {label}: {len(items)} queries -> procedure={procedure_slug} ===\n")
    results: list[tuple[dict, dict, dict]] = []

    for i, item in enumerate(items, 1):
        print(f"[{i}/{len(items)}] {item['query'][:70]}")
        res = run_one_query(item, api_url, procedure_slug)
        sc = score(item, res)
        if res.get("error"):
            print(f"   ERROR {res['error']}: {res.get('detail','')}")
        else:
            if sc["answerable"]:
                print(f"   cov={sc['keyword_coverage']:.2f} "
                      f"refused={sc['refused']} "
                      f"tok={res['n_tokens']} tps={res['tps']:.1f} "
                      f"total={res['total']:.1f}s")
            else:
                print(f"   refusal={'OK' if sc['correct_refusal'] else 'FAIL'} "
                      f"tok={res['n_tokens']} tps={res['tps']:.1f} "
                      f"total={res['total']:.1f}s")
        results.append((item, res, sc))

    return write_report(label, results, health, dataset_path, procedure_slug, api_url)


def write_report(label: str, results, health: dict,
                 dataset_path: Path, procedure_slug: str, api_url: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"eval_{label}.md"
    L: list[str] = []
    L.append(f"# eval baseline — {label} — "
             f"{datetime.now().isoformat(timespec='seconds')}")
    L.append("")
    L.append("## Setup")
    L.append(f"- API: `{api_url}`")
    L.append(f"- Procedure: `{procedure_slug}`")
    L.append(f"- Dataset: `{dataset_path}`")
    L.append(f"- Model: `{health.get('model', '?')}`")
    L.append(f"- Queries run: {len(results)}")
    L.append("")

    n = len(results)
    n_err = sum(1 for _, r, _ in results if r.get("error"))
    n_ok = n - n_err
    answerable = [(i, r, s) for i, r, s in results if s["answerable"]]
    not_answerable = [(i, r, s) for i, r, s in results if not s["answerable"]]

    if answerable:
        covs = [s["keyword_coverage"] for _, r, s in answerable if not r.get("error")]
        avg_cov = sum(covs) / len(covs) if covs else 0.0
        full_cov = sum(1 for c in covs if c == 1.0)
        false_refuse = sum(1 for _, r, s in answerable
                           if not r.get("error") and s["refused"])
    else:
        avg_cov = 0.0; full_cov = 0; false_refuse = 0

    if not_answerable:
        correct_ref = sum(1 for _, r, s in not_answerable
                          if not r.get("error") and s["correct_refusal"])
    else:
        correct_ref = 0

    toks = [r["n_tokens"] for _, r, _ in results if not r.get("error")]
    tps_all = [r["tps"] for _, r, _ in results
               if not r.get("error") and r.get("tps", 0) > 0]
    totals = [r["total"] for _, r, _ in results if not r.get("error")]
    avg_tps = sum(tps_all) / len(tps_all) if tps_all else 0.0
    avg_tok = sum(toks) / len(toks) if toks else 0.0
    max_tok = max(toks) if toks else 0
    avg_total = sum(totals) / len(totals) if totals else 0.0

    L.append("## Summary")
    L.append(f"- Successful: {n_ok} / {n}  (errors: {n_err})")
    if answerable:
        L.append(f"- Answerable: {len(answerable)} — "
                 f"avg keyword coverage: {avg_cov:.2f}, "
                 f"full-coverage (=1.0): {full_cov}/{len(answerable)}, "
                 f"false refusals: {false_refuse}")
    if not_answerable:
        L.append(f"- Non-answerable: {len(not_answerable)} — "
                 f"correct refusals: {correct_ref}/{len(not_answerable)}")
    L.append(f"- Tokens: avg={avg_tok:.0f}, max={max_tok}")
    L.append(f"- Speed: avg gen={avg_tps:.2f} tok/s, avg total={avg_total:.1f}s")
    L.append("")

    L.append("## Queries")
    for i, (item, res, sc) in enumerate(results, 1):
        scope = "IN-SCOPE" if item["answerable"] else "OOS"
        L.append("")
        L.append(f"### {i}. [{scope}] [{item['category']}] {item['query']}")
        L.append(f"- Intent: {item['intent']}")
        if item.get("test_focus"):
            L.append(f"- Test focus: {item['test_focus']}")
        if item.get("expected_keywords"):
            L.append(f"- Expected keywords: {item['expected_keywords']}")
        if sc["answerable"] and sc["keyword_hits"] is not None:
            hit_str = ", ".join(f"{k}={'Y' if h else 'N'}"
                                for k, h in zip(item["expected_keywords"],
                                                sc["keyword_hits"]))
            L.append(f"- Coverage: {sc['keyword_coverage']:.2f} ({hit_str})")
        else:
            L.append(f"- Refusal: {'OK' if sc.get('correct_refusal') else 'FAIL'}")
        L.append("")
        if res.get("error"):
            L.append(f"**ERROR** `{res['error']}`: {res.get('detail','')}")
            if res.get("response"):
                L.append("")
                L.append("```")
                L.append(res["response"])
                L.append("```")
            continue
        L.append("**Response:**")
        L.append("")
        L.append("```")
        L.append((res.get("response") or "").rstrip() or "(empty)")
        L.append("```")
        L.append(
            f"- Timing: total={res.get('total',0):.2f}s "
            f"TTFT={res.get('ttft',0):.2f}s "
            f"gen={res.get('generation_s',0):.2f}s "
            f"tokens={res.get('n_tokens',0)} ({res.get('tps',0):.1f} tok/s)"
        )

    out.write_text("\n".join(L) + "\n")
    print(f"\nReport: {out}")
    return out


def main():
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", default=None,
                   help="API base URL. Defaults to $RAG_API_URL (incl. .env) "
                        "or http://localhost:8000.")
    p.add_argument("--procedure", default="all",
                   help="diabetes | hemorroides | cirugia-abdominal | all")
    p.add_argument("--set", default="all", choices=["coverage", "grayzone", "all"])
    p.add_argument("--dataset", help="Override: explicit dataset path")
    p.add_argument("--procedure-slug",
                   help="When --dataset is used, the API procedure slug")
    p.add_argument("--label", help="When --dataset is used, label for report file")
    args = p.parse_args()
    args.api_url = resolve_api_url(args.api_url)

    try:
        health = get_health(args.api_url)
    except Exception as e:
        print(f"Cannot reach API at {args.api_url}: {e}")
        return 1
    print(f"API: {args.api_url}  model={health.get('model')}  "
          f"procedures={health.get('procedures')}")

    if args.dataset:
        run_dataset(Path(args.dataset),
                    args.procedure_slug or args.procedure,
                    args.api_url,
                    args.label or Path(args.dataset).stem,
                    health)
        return 0

    procs = PROCEDURES if args.procedure == "all" else [args.procedure]
    sets = SETS if args.set == "all" else [args.set]
    for proc in procs:
        for s in sets:
            path = DATASETS_DIR / f"eval_dataset_{proc}_{s}.json"
            if not path.exists():
                print(f"  skip (missing): {path}")
                continue
            run_dataset(path, proc, args.api_url, f"{proc}_{s}", health)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
