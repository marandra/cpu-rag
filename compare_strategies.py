"""
Compare retrieval strategies side-by-side on the eval dataset.

Usage:
    python compare_strategies.py                          # all strategies
    python compare_strategies.py vector hybrid+rerank     # specific ones
    python compare_strategies.py --llm-judge              # + LLM scoring
    python compare_strategies.py --top-k 10               # change top_k
    python compare_strategies.py --tokenizer spacy        # BM25 tokenizer
    python compare_strategies.py --tokenizer whitespace   # baseline tokenizer
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from src.embeddings import _get_client
from src.evaluation import (
    RetrievalResult,
    evaluate_retrieval,
    llm_judge,
    load_eval_dataset,
)
from src.retrieval import STRATEGIES, build_retriever

# --- Configuration ---
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"
TOP_K = 5
EVAL_DATASET = "eval_dataset.json"  # override with --dataset <path>
LLM_JUDGE_MODEL = "gpt-4o-mini"
OUTPUT_JSON = "comparison_results.json"

# --- Parse CLI args ---
use_llm_judge = "--llm-judge" in sys.argv
argv_clean = [a for a in sys.argv[1:] if not a.startswith("--")]

top_k = TOP_K
eval_dataset = EVAL_DATASET
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--top-k" and i < len(sys.argv) - 1:
        top_k = int(sys.argv[i + 1])
    if arg == "--dataset" and i < len(sys.argv) - 1:
        eval_dataset = sys.argv[i + 1]

tokenizer_name = "spacy"  # default
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--tokenizer" and i < len(sys.argv) - 1:
        tokenizer_name = sys.argv[i + 1]

# Which strategies to compare
argv_clean = [a for a in argv_clean if not a.endswith(".json")]
selected = [s for s in argv_clean if s in STRATEGIES]
if not selected:
    selected = list(STRATEGIES)


def avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def run_strategy(
    strategy: str, dataset, client, top_k: int, use_judge: bool
) -> dict:
    """Run one strategy on the full dataset, return summary dict."""
    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy}")
    print(f"{'=' * 60}")

    retriever = build_retriever(client, COLLECTION_NAME, strategy=strategy, tokenizer=tokenizer_name)

    results: list[RetrievalResult] = []
    judge_scores = []
    latencies = []

    for i, eq in enumerate(dataset):
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(eq.query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)

        r = evaluate_retrieval(eq, retrieved)
        results.append(r)

        line = f"  [{i+1:2d}/{len(dataset)}] P@3={r.precision_at_3:.2f} R@5={r.recall_at_5:.2f} MRR={r.mrr:.2f} KW={r.keyword_coverage:.2f} ({elapsed_ms:.0f}ms)"

        if use_judge:
            chunks = [doc.page_content for doc, _ in retrieved]
            j = llm_judge(eq.query, chunks, model=LLM_JUDGE_MODEL)
            judge_scores.append(j)
            if j.error is None:
                line += f" F={j.faithfulness:.2f} R={j.relevance:.2f}"

        print(line)

    summary = {
        "strategy": strategy,
        "num_queries": len(results),
        "top_k": top_k,
        "P@3": avg([r.precision_at_3 for r in results]),
        "P@5": avg([r.precision_at_5 for r in results]),
        "R@5": avg([r.recall_at_5 for r in results]),
        "MRR": avg([r.mrr for r in results]),
        "keyword_cov": avg([r.keyword_coverage for r in results]),
        "score_top1": avg([r.top_score for r in results]),
        "score_avg": avg([r.avg_score for r in results]),
        "latency_ms_avg": avg(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
    }

    if judge_scores:
        valid = [j for j in judge_scores if j.error is None]
        if valid:
            summary["faithfulness"] = avg([j.faithfulness for j in valid])
            summary["relevance"] = avg([j.relevance for j in valid])

    # Per-query detail for JSON output
    summary["per_query"] = [asdict(r) for r in results]

    return summary


# --- Main ---
dataset = load_eval_dataset(eval_dataset)
print(f"Dataset: {len(dataset)} queries from {eval_dataset}")
print(f"Strategies: {', '.join(selected)}")
print(f"Top-K: {top_k}")
print(f"BM25 tokenizer: {tokenizer_name}")

client = _get_client(QDRANT_PATH)
t_global = time.perf_counter()

all_summaries = []
for strategy in selected:
    try:
        summary = run_strategy(strategy, dataset, client, top_k, use_llm_judge)
        all_summaries.append(summary)
    except Exception as e:
        print(f"\n  SKIPPED {strategy}: {e}\n")

client.close()
total_time = time.perf_counter() - t_global

# --- Comparison table ---
metrics = ["P@3", "P@5", "R@5", "MRR", "keyword_cov", "score_top1", "latency_ms_avg"]
if use_llm_judge:
    metrics += ["faithfulness", "relevance"]

col_w = 14
header = f"{'Metric':<18}" + "".join(f"{s['strategy']:>{col_w}}" for s in all_summaries)
sep = "-" * len(header)

print(f"\n\n{'=' * len(header)}")
print("  COMPARISON MATRIX")
print(f"{'=' * len(header)}")
print(header)
print(sep)

for m in metrics:
    vals = [s.get(m, 0.0) for s in all_summaries]
    best = max(vals) if m != "latency_ms_avg" else min(vals)
    row = f"{m:<18}"
    for v in vals:
        fmt = f"{v:.0f}" if "latency" in m else f"{v:.3f}"
        marker = " *" if v == best and len(all_summaries) > 1 else "  "
        row += f"{fmt + marker:>{col_w}}"
    print(row)

print(sep)
print(f"Total time: {total_time:.1f}s\n")

# --- Per-category comparison ---
categories = sorted(set(r.category for r in dataset if r.category))
if categories and len(all_summaries) > 1:
    print(f"\n{'=' * 60}")
    print("  PER-CATEGORY BREAKDOWN (P@3)")
    print(f"{'=' * 60}")

    cat_header = f"{'Category':<18}" + "".join(f"{s['strategy']:>{col_w}}" for s in all_summaries)
    print(cat_header)
    print("-" * len(cat_header))

    for cat in categories:
        row = f"{cat:<18}"
        for s in all_summaries:
            cat_queries = [q for q in s["per_query"] if q["category"] == cat]
            val = avg([q["precision_at_3"] for q in cat_queries]) if cat_queries else 0.0
            row += f"{val:.3f}".rjust(col_w)
        print(row)
    print()

# --- Save JSON ---
output = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "config": {"top_k": top_k, "dataset": eval_dataset, "num_queries": len(dataset)},
    "strategies": [{k: v for k, v in s.items() if k != "per_query"} for s in all_summaries],
    "per_query": {s["strategy"]: s["per_query"] for s in all_summaries},
}
Path(OUTPUT_JSON).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Full results saved to {OUTPUT_JSON}")
