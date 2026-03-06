"""
Parameter sweep: systematic evaluation of all pipeline configurations.

Rebuilds chunks + embeddings + evaluates for each configuration.
Designed for parallel execution via Slurm (one config per job).

Usage:
    # Run a single configuration (by index or explicit params)
    python sweep.py --config-index 0
    python sweep.py --max-tokens 512 --overlap 0.25 --strategy hybrid --tokenizer spacy

    # List all configurations
    python sweep.py --list

    # Run all configurations sequentially (for testing, not for HPC)
    python sweep.py --all

    # Aggregate results from completed runs
    python sweep.py --aggregate
"""

import itertools
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Sweep parameter space ---

SWEEP_PARAMS = {
    "max_tokens": [256, 384, 512, 768],
    "overlap_ratio": [0.0, 0.10, 0.20, 0.30],
    "contextual_prefix": [True, False],
    "llm_enrichment": [True, False],
    "strategy": ["vector", "bm25", "hybrid"],
    "tokenizer": ["whitespace", "spacy"],
    "top_k": [3, 5, 10],
}

# Fixed parameters (not swept)
FIXED = {
    "eval_dataset": "eval_dataset_realistic.json",
    "markdown_dir": "markdown",
    "qdrant_base": "./sweep_qdrant",  # each config gets its own subdirectory
    "chunks_base": "./sweep_chunks",
    "results_dir": "./sweep_results",
    "collection_name": "medical_docs",
    "vector_size": 384,
}


@dataclass
class SweepConfig:
    max_tokens: int
    overlap_ratio: float
    contextual_prefix: bool
    llm_enrichment: bool
    strategy: str
    tokenizer: str
    top_k: int

    @property
    def config_id(self) -> str:
        """Short identifier for this configuration."""
        cp = "cp" if self.contextual_prefix else "nocp"
        llm = "llm" if self.llm_enrichment else "nollm"
        return f"mt{self.max_tokens}_ov{self.overlap_ratio:.2f}_{cp}_{llm}_{self.strategy}_{self.tokenizer}_k{self.top_k}"

    @property
    def chunk_id(self) -> str:
        """Identifier for the chunking config (shared across retrieval variants)."""
        cp = "cp" if self.contextual_prefix else "nocp"
        llm = "llm" if self.llm_enrichment else "nollm"
        return f"mt{self.max_tokens}_ov{self.overlap_ratio:.2f}_{cp}_{llm}"


def generate_configs() -> list[SweepConfig]:
    """Generate all parameter combinations."""
    keys = list(SWEEP_PARAMS.keys())
    configs = []
    for values in itertools.product(*SWEEP_PARAMS.values()):
        params = dict(zip(keys, values))
        # Skip: tokenizer only matters for bm25/hybrid
        if params["strategy"] == "vector" and params["tokenizer"] != "spacy":
            continue
        configs.append(SweepConfig(**params))
    return configs


def run_config(cfg: SweepConfig, config_index: int | None = None) -> dict:
    """Run a single configuration: chunk -> embed -> evaluate."""
    from glob import glob

    from src.chunks import ChunkConfig, process_markdown
    from src.embeddings import (
        VectorStoreConfig,
        _get_client,
        collection_count,
        embed_documents,
        ensure_collection,
        ingest,
        load_chunks,
    )
    from src.evaluation import evaluate_retrieval, load_eval_dataset
    from src.retrieval import build_retriever

    t_start = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"  Config: {cfg.config_id}")
    if config_index is not None:
        print(f"  Index:  {config_index}")
    print(f"{'='*60}")

    # --- Paths ---
    chunks_dir = Path(FIXED["chunks_base"]) / cfg.chunk_id
    qdrant_path = Path(FIXED["qdrant_base"]) / cfg.chunk_id
    results_dir = Path(FIXED["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Chunk (reuse if same chunk config already built) ---
    if not list(chunks_dir.glob("*_chunks.json")):
        print(f"  Chunking (max_tokens={cfg.max_tokens}, overlap={cfg.overlap_ratio})...")
        chunks_dir.mkdir(parents=True, exist_ok=True)

        md_files = sorted(glob(f"{FIXED['markdown_dir']}/*.md"))
        chunk_config = ChunkConfig(
            max_tokens=cfg.max_tokens,
            overlap_ratio=cfg.overlap_ratio,
            output_dir=str(chunks_dir),
            contextual_prefix=cfg.contextual_prefix,
            llm_enrichment=cfg.llm_enrichment,
        )
        total_chunks = 0
        for f in md_files:
            docs = process_markdown(f, chunk_config)
            total_chunks += len(docs)
        print(f"  -> {total_chunks} chunks")
    else:
        print(f"  Chunks exist at {chunks_dir}, reusing.")

    # --- Step 2: Embed (reuse if same chunk config already embedded) ---
    client = _get_client(str(qdrant_path))
    existing = [c.name for c in client.get_collections().collections]

    if FIXED["collection_name"] not in existing:
        print(f"  Embedding...")
        documents = load_chunks(str(chunks_dir))
        texts = [doc.page_content for doc in documents]
        vectors = embed_documents(texts, dimensionality=FIXED["vector_size"])

        vs_config = VectorStoreConfig(
            path=str(qdrant_path),
            collection_name=FIXED["collection_name"],
            vector_size=FIXED["vector_size"],
        )
        ensure_collection(client, vs_config)
        ingest(client, FIXED["collection_name"], documents, vectors)
        n_points = collection_count(client, FIXED["collection_name"])
        print(f"  -> {n_points} points indexed")
    else:
        n_points = collection_count(client, FIXED["collection_name"])
        print(f"  DB exists ({n_points} points), reusing.")

    # --- Step 3: Evaluate ---
    print(f"  Evaluating (strategy={cfg.strategy}, tokenizer={cfg.tokenizer}, top_k={cfg.top_k})...")
    dataset = load_eval_dataset(FIXED["eval_dataset"])
    retriever = build_retriever(
        client,
        FIXED["collection_name"],
        strategy=cfg.strategy,
        tokenizer=cfg.tokenizer,
    )

    results = []
    for eq in dataset:
        retrieved = retriever.retrieve(eq.query, top_k=cfg.top_k)
        r = evaluate_retrieval(eq, retrieved)
        results.append(r)

    client.close()

    # --- Aggregate ---
    avg = lambda vals: sum(vals) / len(vals) if vals else 0.0
    elapsed = time.perf_counter() - t_start

    summary = {
        "config": asdict(cfg),
        "config_id": cfg.config_id,
        "config_index": config_index,
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

    # Save individual result
    out_path = results_dir / f"{cfg.config_id}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"  P@3={summary['metrics']['P@3']:.3f}  R@5={summary['metrics']['R@5']:.3f}  "
          f"MRR={summary['metrics']['MRR']:.3f}  KW={summary['metrics']['keyword_cov']:.3f}  "
          f"({elapsed:.1f}s)")
    return summary


def aggregate_results():
    """Aggregate all individual result files into a summary table."""
    results_dir = Path(FIXED["results_dir"])
    result_files = sorted(results_dir.glob("*.json"))

    if not result_files:
        print("No results found.")
        return

    # Load all results
    all_results = []
    for f in result_files:
        if f.name == "sweep_summary.json":
            continue
        data = json.loads(f.read_text(encoding="utf-8"))
        all_results.append(data)

    if not all_results:
        print("No results found.")
        return

    # Sort by P@3 descending
    all_results.sort(key=lambda x: x["metrics"]["P@3"], reverse=True)

    # Print table
    metrics = ["P@3", "P@5", "R@5", "MRR", "keyword_cov"]
    header = f"{'#':>3} {'chunks':>6} {'max_tok':>7} {'overlap':>7} {'prefix':>6} {'llm':>5} {'strategy':>10} {'tokenizer':>10} {'top_k':>5}"
    header += "".join(f"  {m:>10}" for m in metrics)
    print(header)
    print("-" * len(header))

    for i, r in enumerate(all_results):
        c = r["config"]
        row = (
            f"{i+1:>3} "
            f"{r['num_chunks']:>6} "
            f"{c['max_tokens']:>7} "
            f"{c['overlap_ratio']:>7.2f} "
            f"{'Y' if c['contextual_prefix'] else 'N':>6} "
            f"{'Y' if c['llm_enrichment'] else 'N':>5} "
            f"{c['strategy']:>10} "
            f"{c['tokenizer']:>10} "
            f"{c['top_k']:>5}"
        )
        for m in metrics:
            row += f"  {r['metrics'][m]:>10.3f}"
        print(row)

    # Save aggregate
    agg_path = results_dir / "sweep_summary.json"
    agg_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(all_results)} results aggregated -> {agg_path}")


# --- CLI ---

if __name__ == "__main__":
    configs = generate_configs()

    if "--list" in sys.argv:
        print(f"Total configurations: {len(configs)}")
        # Count unique chunk configs
        chunk_ids = set(c.chunk_id for c in configs)
        print(f"Unique chunk configs (require re-chunking + embedding): {len(chunk_ids)}")
        llm_configs = sum(1 for c in chunk_ids if "llm" in c and "nollm" not in c)
        print(f"  of which need LLM enrichment (OpenAI API calls): {llm_configs}")
        print(f"\nFirst 10:")
        for i, cfg in enumerate(configs[:10]):
            print(f"  [{i:3d}] {cfg.config_id}")
        print(f"  ...")
        print(f"  [{len(configs)-1:3d}] {configs[-1].config_id}")
        sys.exit(0)

    if "--aggregate" in sys.argv:
        aggregate_results()
        sys.exit(0)

    if "--all" in sys.argv:
        print(f"Running all {len(configs)} configurations sequentially...")
        for i, cfg in enumerate(configs):
            run_config(cfg, config_index=i)
        aggregate_results()
        sys.exit(0)

    # Single config by index
    config_index = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--config-index" and i < len(sys.argv) - 1:
            config_index = int(sys.argv[i + 1])

    if config_index is not None:
        if config_index >= len(configs):
            print(f"Config index {config_index} out of range (0-{len(configs)-1})")
            sys.exit(1)
        run_config(configs[config_index], config_index=config_index)
        sys.exit(0)

    # Single config by explicit params
    params = {}
    param_map = {
        "--max-tokens": ("max_tokens", int),
        "--overlap": ("overlap_ratio", float),
        "--prefix": ("contextual_prefix", lambda x: x.lower() == "true"),
        "--llm-enrichment": ("llm_enrichment", lambda x: x.lower() == "true"),
        "--strategy": ("strategy", str),
        "--tokenizer": ("tokenizer", str),
        "--top-k": ("top_k", int),
    }
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg in param_map and i < len(sys.argv) - 1:
            key, cast = param_map[arg]
            params[key] = cast(sys.argv[i + 1])

    if params:
        # Fill defaults
        defaults = {
            "max_tokens": 512, "overlap_ratio": 0.25, "contextual_prefix": True,
            "llm_enrichment": True, "strategy": "hybrid", "tokenizer": "spacy", "top_k": 5,
        }
        defaults.update(params)
        cfg = SweepConfig(**defaults)
        run_config(cfg)
    else:
        print("Usage:")
        print("  python sweep.py --list                    # list all configs")
        print("  python sweep.py --config-index 0          # run config by index")
        print("  python sweep.py --all                     # run all sequentially")
        print("  python sweep.py --aggregate               # aggregate results")
        print("  python sweep.py --max-tokens 512 --strategy hybrid  # explicit params")
