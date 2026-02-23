"""
Evaluation pipeline: run eval dataset through retrieval and score quality.

Usage:
    python eval_retrieval.py                              # vector (default)
    python eval_retrieval.py --strategy hybrid             # hybrid retrieval
    python eval_retrieval.py --strategy hybrid+rerank      # with reranker
    python eval_retrieval.py --llm-judge                   # + LLM-as-judge
"""

import sys
import time

from src.embeddings import _get_client
from src.evaluation import (
    evaluate_retrieval,
    generate_report,
    llm_judge,
    load_eval_dataset,
)
from src.retrieval import STRATEGIES, build_retriever

# --- Configuration ---
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 384
TOP_K = 5
EVAL_DATASET = "eval_dataset.json"
REPORT_OUTPUT = "eval_report.md"
LLM_JUDGE_MODEL = "gpt-4o-mini"

# --- Parse CLI args ---
use_llm_judge = "--llm-judge" in sys.argv

strategy = "vector"
for arg in sys.argv[1:]:
    if arg.startswith("--strategy"):
        if "=" in arg:
            strategy = arg.split("=", 1)[1]
        else:
            idx = sys.argv.index(arg)
            if idx + 1 < len(sys.argv):
                strategy = sys.argv[idx + 1]

if strategy not in STRATEGIES:
    print(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")
    sys.exit(1)

# --- Load dataset ---
dataset = load_eval_dataset(EVAL_DATASET)
print(f"Dataset: {len(dataset)} consultas de {EVAL_DATASET}")

client = _get_client(QDRANT_PATH)

print(f"Building retriever (strategy={strategy})...")
retriever = build_retriever(client, COLLECTION_NAME, strategy=strategy)

retrieval_results = []
judge_results = [] if use_llm_judge else None

t_total = time.perf_counter()

for i, eq in enumerate(dataset):
    t0 = time.perf_counter()
    results = retriever.retrieve(eq.query, top_k=TOP_K)
    elapsed = (time.perf_counter() - t0) * 1000

    # Retrieval metrics
    r = evaluate_retrieval(eq, results)
    retrieval_results.append(r)

    status = f"  [{i+1}/{len(dataset)}] P@3={r.precision_at_3:.2f} R@5={r.recall_at_5:.2f} MRR={r.mrr:.2f} KW={r.keyword_coverage:.2f} ({elapsed:.0f}ms)"

    # LLM judge
    if use_llm_judge:
        chunks = [doc.page_content for doc, _ in results]
        j = llm_judge(eq.query, chunks, model=LLM_JUDGE_MODEL)
        judge_results.append(j)
        if j.error is None:
            status += f" F={j.faithfulness:.2f} R={j.relevance:.2f}"
        else:
            status += f" LLM_ERR"

    print(status)

total_elapsed = time.perf_counter() - t_total

# Generate report
report = generate_report(retrieval_results, judge_results, REPORT_OUTPUT)
print(f"\nInforme guardado en {REPORT_OUTPUT} ({total_elapsed:.1f}s total)")

# Print summary to console
avg = lambda vals: sum(vals) / len(vals) if vals else 0.0
print(f"\n{'=' * 50}")
print(f"  Strategy: {strategy}")
print(f"  P@3: {avg([r.precision_at_3 for r in retrieval_results]):.2f}")
print(f"  P@5: {avg([r.precision_at_5 for r in retrieval_results]):.2f}")
print(f"  R@5: {avg([r.recall_at_5 for r in retrieval_results]):.2f}")
print(f"  MRR: {avg([r.mrr for r in retrieval_results]):.2f}")
print(f"  Keywords: {avg([r.keyword_coverage for r in retrieval_results]):.2f}")
if judge_results:
    valid = [j for j in judge_results if j.error is None]
    if valid:
        print(f"  Faithfulness: {avg([j.faithfulness for j in valid]):.2f}")
        print(f"  Relevance: {avg([j.relevance for j in valid]):.2f}")

client.close()
