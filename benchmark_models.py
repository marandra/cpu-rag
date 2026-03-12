"""
Model comparison benchmark: all available GGUF models with best prompt config.

Tests speed (tok/s, TTFT, total time) and quality (span coverage) across
the same set of queries with identical retrieval results.

Usage:
    uv run python benchmark_models.py
    uv run python benchmark_models.py --n-threads 9
"""

import argparse
import json
import os
import sys
import time

from src.embeddings import _get_client
from src.llm import load_model
from src.retrieval import build_retriever

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# --- Config ---
QDRANT_PATH = "./sweep_qdrant/mt512_ov0.20_cp_llm"
COLLECTION = "medical_docs"
STRATEGY = "hybrid"
TOKENIZER = "spacy"
TOP_K = 5
N_CTX = 2048
MAX_TOKENS = 256

# Best prompt config (no system prompt + instructive template)
SYSTEM_PROMPT = None
PROMPT_TEMPLATE = (
    "A continuación tienes fragmentos de documentos médicos. "
    "Úsalos para responder la pregunta del paciente con todos los detalles relevantes.\n\n"
    "FRAGMENTOS:\n{context}\n\n"
    "PREGUNTA DEL PACIENTE: {query}\n\n"
    "RESPUESTA DETALLADA:"
)

# Test queries (same as prompt tuning benchmark)
TEST_QUERY_IDS = [0, 5, 9, 13, 19, 26, 30, 40]

# All models to test
MODELS = [
    ("Qwen3-0.6B",    "./models/Qwen3-0.6B-Q4_K_M.gguf"),
    ("Qwen3.5-0.8B",  "./models/Qwen3.5-0.8B-Q4_K_M.gguf"),       # needs llama.cpp 'qwen35' support
    ("qwen2.5-0.5b",  "./models/qwen2.5-0.5b-instruct-Q4_K_M.gguf"),
    ("qwen2.5-1.5b",  "./models/qwen2.5-1.5b-instruct-Q4_K_M.gguf"),
    ("gemma-3-1b",    "./models/gemma-3-1b-it-Q4_K_M.gguf"),
    ("SmolLM2-1.7B",  "./models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"),
    ("Qwen3.5-2B",    "./models/Qwen3.5-2B-Q4_K_M.gguf"),         # needs llama.cpp 'qwen35' support
    ("llama-3.2-1b",  "./models/llama-3.2-1b-instruct-Q4_K_M.gguf"),
    ("llama-3.2-3b",  "./models/llama-3.2-3b-instruct-Q4_K_M.gguf"),
    ("gemma-3n-E2B",  "./models/google_gemma-3n-E2B-it-Q4_K_M.gguf"),
    # New models (2026-03-12)
    ("Pleias-RAG-1B",   "./models/Pleias-RAG-1B.gguf"),
    ("granite-4.0-1b",  "./models/granite-4.0-1b-Q4_K_M.gguf"),
    ("SmolLM3-3B",      "./models/SmolLM3-3B-Q4_K_M.gguf"),
    ("Ministral-3B",    "./models/Ministral-3-3B-Q4_K_M.gguf"),   # needs llama.cpp 'mistral3' support
]

# Models that need /no_think to disable thinking tokens
NEEDS_NO_THINK = {"SmolLM3-3B", "Qwen3-0.6B"}


def _load_with_chatml_fallback(model_path, n_ctx, n_threads):
    """Load a model whose embedded chat template has unsupported Jinja tags."""
    import re
    from llama_cpp import Llama
    from llama_cpp import llama_chat_format

    # Monkey-patch to strip {% generation %}/{% endgeneration %} before compile
    _orig_init = llama_chat_format.Jinja2ChatFormatter.__init__

    def _patched_init(self, *args, **kwargs):
        if "template" in kwargs and kwargs["template"]:
            kwargs["template"] = re.sub(
                r"\{%[-\s]*(?:end)?generation\s*[-]?%\}", "", kwargs["template"]
            )
        elif args:
            args = list(args)
            if isinstance(args[0], str):
                args[0] = re.sub(
                    r"\{%[-\s]*(?:end)?generation\s*[-]?%\}", "", args[0]
                )
        _orig_init(self, *args, **kwargs)

    llama_chat_format.Jinja2ChatFormatter.__init__ = _patched_init
    try:
        model = Llama(
            model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
            n_threads_batch=os.cpu_count() or n_threads,
            n_batch=512, flash_attn=True, verbose=False,
        )
    finally:
        llama_chat_format.Jinja2ChatFormatter.__init__ = _orig_init
    return model


def load_test_queries(ids):
    with open("eval_dataset_realistic.json") as f:
        data = json.load(f)
    return [data[i] for i in ids]


def build_prompt(query, chunks):
    context = "\n---\n".join(
        doc.metadata.get("page_content_compact") or doc.page_content
        for doc, _ in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def check_span_coverage(answer, spans):
    answer_lower = answer.lower()
    hits = 0
    for span in spans:
        words = [w for w in span.lower().split() if len(w) > 3]
        if not words:
            hits += 1
            continue
        matched = sum(1 for w in words if w in answer_lower)
        if matched / len(words) >= 0.4:
            hits += 1
    return hits / len(spans) if spans else 0


def run_model(model_name, model_path, queries, query_chunks, n_threads):
    """Benchmark a single model. Returns results dict."""
    print(f"\n{'=' * 70}")
    print(f"MODEL: {model_name}")
    print(f"  path: {model_path}")
    print(f"  size: {os.path.getsize(model_path) / 1e9:.2f} GB")
    print(f"{'=' * 70}")

    # Load model
    t0 = time.perf_counter()
    try:
        model = load_model(model_path, n_ctx=N_CTX, n_threads=n_threads)
    except Exception as e:
        # Some models have Jinja templates with unsupported tags (e.g. {% generation %}).
        # Retry with a patched metadata to strip the bad template.
        if "unknown tag" in str(e) or "generation" in str(e):
            print(f"  Template error, retrying with chatml workaround...")
            try:
                model = _load_with_chatml_fallback(model_path, N_CTX, n_threads)
            except Exception as e2:
                print(f"  FAILED to load (retry): {e2}")
                return None
        else:
            print(f"  FAILED to load: {e}")
            return None
    load_time = time.perf_counter() - t0
    print(f"  Load time: {load_time:.1f}s")

    # Warmup
    try:
        messages = [{"role": "user", "content": "Hola, responde breve."}]
        model.create_chat_completion(messages=messages, max_tokens=10, temperature=0)
    except Exception as e:
        print(f"  FAILED warmup: {e}")
        del model
        return None

    # Run queries
    total_coverage = 0
    total_comp_tokens = 0
    total_time = 0
    total_ttft = 0
    query_results = []

    for q in queries:
        chunks = query_chunks[q["query"]]
        prompt = build_prompt(q["query"], chunks)

        messages = []
        if model_name in NEEDS_NO_THINK:
            messages.append({"role": "system", "content": "/no_think"})
        elif SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": prompt})

        # Stream to get TTFT
        t_start = time.perf_counter()
        ttft = None
        full_text = ""
        n_tokens = 0

        try:
            for chunk in model.create_chat_completion(
                messages=messages, max_tokens=MAX_TOKENS, temperature=0, stream=True,
            ):
                text = chunk["choices"][0]["delta"].get("content", "")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t_start
                    full_text += text
                    n_tokens += 1
        except Exception as e:
            print(f"  FAILED on query: {e}")
            query_results.append({"error": str(e)})
            continue

        gen_time = time.perf_counter() - t_start
        speed = n_tokens / gen_time if gen_time > 0 else 0
        ttft = ttft or gen_time

        spans = q.get("relevant_spans", [])
        coverage = check_span_coverage(full_text, spans)

        total_coverage += coverage
        total_comp_tokens += n_tokens
        total_time += gen_time
        total_ttft += ttft

        query_results.append({
            "query": q["query"][:60],
            "coverage": round(coverage, 3),
            "tokens": n_tokens,
            "time": round(gen_time, 1),
            "ttft": round(ttft, 2),
            "speed": round(speed, 1),
        })

        cov_pct = coverage * 100
        print(f"  Q: {q['query'][:50]:50s} cov={cov_pct:5.1f}% tok={n_tokens:3d} "
              f"{gen_time:5.1f}s ttft={ttft:4.1f}s {speed:5.1f}t/s")

    n = len(queries)
    avg_cov = total_coverage / n * 100 if n else 0
    avg_tok = total_comp_tokens / n if n else 0
    avg_time = total_time / n if n else 0
    avg_ttft = total_ttft / n if n else 0
    avg_speed = total_comp_tokens / total_time if total_time > 0 else 0

    print(f"\n  >>> AVG: cov={avg_cov:.1f}%  tok={avg_tok:.0f}  "
          f"time={avg_time:.1f}s  ttft={avg_ttft:.1f}s  {avg_speed:.1f}t/s")

    del model

    return {
        "model": model_name,
        "size_gb": round(os.path.getsize(model_path) / 1e9, 2),
        "load_time": round(load_time, 1),
        "avg_coverage": round(avg_cov, 1),
        "avg_tokens": round(avg_tok, 1),
        "avg_time": round(avg_time, 1),
        "avg_ttft": round(avg_ttft, 2),
        "avg_speed": round(avg_speed, 2),
        "total_time": round(total_time, 1),
        "queries": query_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Model comparison benchmark")
    parser.add_argument("--n-threads", type=int, default=9)
    parser.add_argument("--output", default="benchmark_models_results.json")
    parser.add_argument(
        "--models", nargs="+", metavar="NAME",
        help="Run only these model names (e.g. --models Pleias-RAG-1B granite-4.0-1b)",
    )
    args = parser.parse_args()

    models_to_run = MODELS
    if args.models:
        models_to_run = [(n, p) for n, p in MODELS if n in args.models]
        unknown = set(args.models) - {n for n, _ in MODELS}
        if unknown:
            print(f"WARNING: unknown model names: {unknown}")

    print(f"n_threads: {args.n_threads}")
    print(f"Models to test: {len(models_to_run)}")

    # Load queries and retrieve chunks (shared across all models)
    queries = load_test_queries(TEST_QUERY_IDS)
    print(f"\nTest queries: {len(queries)}")
    for i, q in enumerate(queries):
        print(f"  {i}. {q['query'][:70]}")

    print("\nLoading retriever and pre-fetching chunks...")
    client = _get_client(QDRANT_PATH)
    retriever = build_retriever(client, COLLECTION, strategy=STRATEGY, tokenizer=TOKENIZER)

    query_chunks = {}
    for q in queries:
        query_chunks[q["query"]] = retriever.retrieve(q["query"], top_k=TOP_K)
    client.close()
    print("  Done.")

    # Run each model
    all_results = []
    for model_name, model_path in models_to_run:
        if not os.path.exists(model_path):
            print(f"\nSKIPPING {model_name}: {model_path} not found")
            continue
        result = run_model(model_name, model_path, queries, query_chunks, args.n_threads)
        if result:
            all_results.append(result)
            # Save incrementally so a crash doesn't lose completed results
            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"  [saved {len(all_results)} results to {args.output}]")

    # Summary table
    print(f"\n{'=' * 90}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Model':20s} {'Size':>6s} {'Cov%':>6s} {'tok/s':>7s} {'TTFT':>6s} "
          f"{'Time/q':>7s} {'Tok/q':>6s} {'Load':>5s}")
    print(f"{'─' * 90}")

    for r in sorted(all_results, key=lambda x: x["avg_speed"], reverse=True):
        print(f"{r['model']:20s} {r['size_gb']:5.1f}G {r['avg_coverage']:5.1f}% "
              f"{r['avg_speed']:6.1f}  {r['avg_ttft']:5.1f}s "
              f"{r['avg_time']:6.1f}s {r['avg_tokens']:5.0f}  {r['load_time']:4.1f}s")

    # Quality-speed tradeoff
    print(f"\n{'─' * 90}")
    print("Sorted by coverage (quality):")
    print(f"{'─' * 90}")
    for r in sorted(all_results, key=lambda x: x["avg_coverage"], reverse=True):
        marker = " <<<" if r["avg_coverage"] >= 75 and r["avg_speed"] >= 5 else ""
        print(f"{r['model']:20s} cov={r['avg_coverage']:5.1f}%  "
              f"speed={r['avg_speed']:5.1f} tok/s  "
              f"time={r['avg_time']:5.1f}s{marker}")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
