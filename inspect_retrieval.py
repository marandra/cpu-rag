"""
Inspect retrieved chunks for a single query.

Usage:
    python inspect_retrieval.py "¿Qué es la cirugía de hemorroides?"
    python inspect_retrieval.py "¿Qué riesgos tiene?" --strategy hybrid --top-k 10
"""

import argparse

from src.embeddings import _get_client
from src.retrieval import STRATEGIES, build_retriever

QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"


def main():
    parser = argparse.ArgumentParser(description="Inspect retrieval results for a query")
    parser.add_argument("query", help="Query string")
    parser.add_argument("--strategy", default="hybrid", choices=STRATEGIES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tokenizer", default="spacy")
    args = parser.parse_args()

    client = _get_client(QDRANT_PATH)
    retriever = build_retriever(
        client, COLLECTION_NAME, strategy=args.strategy, tokenizer=args.tokenizer
    )
    results = retriever.retrieve(args.query, top_k=args.top_k)

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy} | top_k: {args.top_k}")
    print(f"Results: {len(results)}")

    for i, (doc, score) in enumerate(results, 1):
        src = doc.metadata.get("source", "?")
        print(f"\n{'=' * 60}")
        print(f"[{i}] score={score:.4f}  src={src}")
        print(f"{'=' * 60}")
        print(doc.page_content)

    client.close()


if __name__ == "__main__":
    main()
