"""
Retrieval test tool: run queries and inspect results.

Usage:
    python test_retrieval.py                                    # vector, queries from queries.txt
    python test_retrieval.py --strategy hybrid                  # hybrid retrieval
    python test_retrieval.py --strategy hybrid+rerank           # with reranker
    python test_retrieval.py "¿pregunta concreta?"              # single ad-hoc query
    python test_retrieval.py --strategy bm25 "¿pregunta?"      # combined
"""

import sys
import time
from pathlib import Path

from src.embeddings import _get_client
from src.retrieval import STRATEGIES, build_retriever

# --- Configuration ---
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"
VECTOR_SIZE = 384
TOP_K = 5
QUERIES_FILE = "queries.txt"


def load_queries(path: str) -> list[str]:
    """Load queries from file (one per line, # comments)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def print_results(query: str, results, elapsed_ms: float):
    """Print results for a single query."""
    print(f"\n{'─' * 70}")
    print(f"Q: {query}")
    print(f"  {len(results)} resultados en {elapsed_ms:.0f} ms")

    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        source = Path(meta.get("source", "?")).name
        section = meta.get("section", "")
        tokens = meta.get("num_tokens", "?")
        text = doc.page_content[:150].replace("\n", " ")

        print(f"\n  [{i}] score={score:.4f}  {source}  §{section}  ({tokens} tok)")
        print(f"      {text}...")


# --- Parse CLI args ---
strategy = "vector"
positional_args = []

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg.startswith("--strategy"):
        if "=" in arg:
            strategy = arg.split("=", 1)[1]
        elif i + 1 < len(sys.argv):
            i += 1
            strategy = sys.argv[i]
    else:
        positional_args.append(arg)
    i += 1

if strategy not in STRATEGIES:
    print(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")
    sys.exit(1)

# --- Main ---
client = _get_client(QDRANT_PATH)

print(f"Building retriever (strategy={strategy})...")
retriever = build_retriever(client, COLLECTION_NAME, strategy=strategy)

if positional_args:
    queries = [" ".join(positional_args)]
else:
    queries = load_queries(QUERIES_FILE)
    print(f"Cargadas {len(queries)} consultas de {QUERIES_FILE}")

all_scores = []
total_time = 0.0

for query in queries:
    t0 = time.perf_counter()
    results = retriever.retrieve(query, top_k=TOP_K)
    elapsed = (time.perf_counter() - t0) * 1000
    total_time += elapsed

    print_results(query, results, elapsed)
    all_scores.extend(s for _, s in results)

# Summary
print(f"\n{'=' * 70}")
print(f"Resumen (strategy={strategy})")
print(f"  Consultas: {len(queries)}")
print(f"  Tiempo total: {total_time:.0f} ms ({total_time/len(queries):.0f} ms/consulta)")
if all_scores:
    print(f"  Scores: min={min(all_scores):.4f}  avg={sum(all_scores)/len(all_scores):.4f}  max={max(all_scores):.4f}")

client.close()
