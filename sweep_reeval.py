"""
Re-evaluate sweep configs with updated metrics.

Processes all retrieval variants for a single chunk config in one job,
avoiding Qdrant file lock conflicts when running in parallel.

Usage:
    python sweep_reeval.py --chunk-index 0    # single chunk config
    python sweep_reeval.py --all               # all chunk configs sequentially
    python sweep_reeval.py --aggregate         # aggregate results
"""

import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from sweep import FIXED, SWEEP_PARAMS, SweepConfig, generate_configs

# Group configs by chunk_id
def configs_by_chunk() -> dict[str, list[tuple[int, SweepConfig]]]:
    """Group all configs by chunk_id. Returns {chunk_id: [(global_index, config), ...]}."""
    groups: dict[str, list[tuple[int, SweepConfig]]] = {}
    for i, cfg in enumerate(generate_configs()):
        groups.setdefault(cfg.chunk_id, []).append((i, cfg))
    return groups


def run_chunk_group(chunk_index: int):
    """Evaluate all retrieval variants for a single chunk config."""
    from src.embeddings import _get_client, collection_count
    from src.evaluation import evaluate_retrieval, load_eval_dataset
    from src.retrieval import build_retriever

    groups = configs_by_chunk()
    chunk_ids = sorted(groups.keys())

    if chunk_index >= len(chunk_ids):
        print(f"Chunk index {chunk_index} out of range (max {len(chunk_ids) - 1})")
        return

    chunk_id = chunk_ids[chunk_index]
    configs = groups[chunk_id]
    results_dir = Path(FIXED["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chunk config: {chunk_id} ({len(configs)} retrieval variants)")

    # Open Qdrant once for this chunk config
    qdrant_path = Path(FIXED["qdrant_base"]) / chunk_id
    client = _get_client(str(qdrant_path))
    n_points = collection_count(client, FIXED["collection_name"])
    print(f"  DB: {n_points} points")

    # Load eval dataset once
    dataset = load_eval_dataset(FIXED["eval_dataset"])

    for global_index, cfg in configs:
        t0 = time.perf_counter()

        # Check if already done
        out_path = results_dir / f"{cfg.config_id}.json"
        if out_path.exists():
            print(f"  SKIP {cfg.config_id} (exists)")
            continue

        # Build retriever + evaluate
        retriever = build_retriever(
            client,
            FIXED["collection_name"],
            strategy=cfg.strategy,
            tokenizer=cfg.tokenizer,
        )

        avg = lambda vals: sum(vals) / len(vals) if vals else 0.0
        results = []
        for eq in dataset:
            retrieved = retriever.retrieve(eq.query, top_k=cfg.top_k)
            r = evaluate_retrieval(eq, retrieved)
            results.append(r)

        elapsed = time.perf_counter() - t0

        summary = {
            "config": asdict(cfg),
            "config_id": cfg.config_id,
            "config_index": global_index,
            "num_chunks": n_points,
            "num_queries": len(results),
            "metrics": {
                "P@3": avg([r.precision_at_3 for r in results]),
                "P@5": avg([r.precision_at_5 for r in results]),
                "R@5": avg([r.recall_at_5 for r in results]),
                "MRR": avg([r.mrr for r in results]),
                "keyword_cov": avg([r.keyword_coverage for r in results]),
            },
            "elapsed_s": round(elapsed, 1),
        }

        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  {cfg.config_id}: P@3={summary['metrics']['P@3']:.3f} "
              f"MRR={summary['metrics']['MRR']:.3f} ({elapsed:.1f}s)")

    client.close()
    print(f"Done: {chunk_id}")


def aggregate():
    """Re-use sweep.py aggregate logic."""
    from sweep import aggregate_results
    aggregate_results()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-index", type=int, help="Index of chunk config (0-63)")
    parser.add_argument("--all", action="store_true", help="Run all chunk configs")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
    elif args.all:
        groups = configs_by_chunk()
        for i in range(len(groups)):
            run_chunk_group(i)
    elif args.chunk_index is not None:
        run_chunk_group(args.chunk_index)
    else:
        parser.print_help()
