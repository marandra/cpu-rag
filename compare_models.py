"""
Cross-model comparison: same retrieval + prompt → all models.

For each query, retrieves chunks once, builds one prompt, then runs every
available model on that same prompt.  Outputs a Markdown report with full
answers and timing for manual quality analysis.

Usage:
    uv run python compare_models.py
    uv run python compare_models.py --procedure hemorroides
    uv run python compare_models.py --models granite-1b gemma-3n
    uv run python compare_models.py --query-ids 4 0 1 19 41 44
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

from src.embeddings import _get_client
from src.llm import load_model
from src.retrieval import build_retriever

from inspect_rag import (
    MODEL_ALIASES, PROCEDURES,
    QDRANT_PATH, COLLECTION_NAME,
    RERANK_MODEL, RERANK_CANDIDATES,
    SYSTEM_PROMPT_TEMPLATE, PROMPT_TEMPLATE,
)

sys.stdout.reconfigure(line_buffering=True)

# --- Config ---
STRATEGY = "hybrid+rerank"
TOKENIZER = "spacy"
TOP_K = 5
N_CTX = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.3

# Models that need /no_think prefix to suppress thinking tokens
NEEDS_NO_THINK = {"qwen3-0.6b"}

# Models that fail to load with current llama-cpp-python (skip them)
SKIP_MODELS = {"qwen3.5-2b", "qwen3.5-0.8b", "smollm3", "ministral"}

# Representative queries: minimal, anxious, low-literacy, GPC-pain, off-topic, medical-OOS
DEFAULT_QUERY_IDS = [4, 0, 1, 19, 41, 44]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text_for_prompt(doc) -> str:
    """Pick the best text variant for the LLM prompt (same as inspect_rag)."""
    return (
        doc.metadata.get("page_content_compact")
        or doc.metadata.get("page_content_original")
        or doc.page_content
    )


def build_prompt(query: str, chunks) -> str:
    context = "\n---\n".join(_chunk_text_for_prompt(doc) for doc, _ in chunks)
    return PROMPT_TEMPLATE.format(context=context, query=query)


def _load_model_quiet(path, n_ctx, n_threads):
    """Load model with stderr suppression."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        model = load_model(path, n_ctx=n_ctx, n_threads=n_threads)
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)
    return model


def generate_timed(model, messages, max_tokens, temperature):
    """Stream generation for timing measurement. Returns result dict."""
    t_start = time.perf_counter()
    ttft = None
    full_text = ""
    n_tokens = 0

    for chunk in model.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True,
    ):
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            full_text += text
            n_tokens += 1

    total_time = time.perf_counter() - t_start
    ttft = ttft or total_time
    decode_time = total_time - ttft
    speed = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0

    return {
        "answer": full_text.strip(),
        "tokens": n_tokens,
        "ttft": round(ttft, 2),
        "decode_time": round(decode_time, 2),
        "total_time": round(total_time, 2),
        "speed": round(speed, 1),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(results, queries, query_prompts, query_chunks,
                 system_prompt, procedure, output_path):
    """Write Markdown comparison report."""
    lines = []
    w = lines.append

    w("# Model Comparison Report\n")
    w(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Procedure**: {procedure or '(none)'}  ")
    w(f"**Strategy**: {STRATEGY} | top_k={TOP_K} | rerank_candidates={RERANK_CANDIDATES}  ")
    w(f"**Temperature**: {TEMPERATURE} | **Max tokens**: {MAX_TOKENS}  ")
    model_names = [n for n in results]
    w(f"**Models**: {len(model_names)} | **Queries**: {len(queries)}\n")
    w("---\n")

    # ---- Per-query sections ----
    for qi, q in enumerate(queries):
        w(f"## Q{qi + 1}: \"{q['query']}\"\n")
        answerable = "sí" if q.get("answerable", True) else "**no**"
        w(f"**Intent**: {q.get('intent', '?')}  ")
        w(f"**Category**: {q.get('category', '?')} | "
          f"**Answerable**: {answerable} | "
          f"**Profile**: {q.get('profile', '?')}")

        kw = q.get("expected_keywords", [])
        if kw:
            w(f"**Expected keywords**: {', '.join(kw)}")

        spans = q.get("relevant_spans", [])
        if spans:
            w(f"\n**Expected spans** ({len(spans)}):")
            for s in spans:
                w(f"- _{s}_")

        if not q.get("answerable", True):
            w("\n> **Expected behavior**: model should refuse — "
              "\"No tengo información sobre eso\" or similar.")

        # Retrieved chunks
        chunks = query_chunks[q["query"]]
        w(f"\n<details><summary>Retrieved chunks ({len(chunks)})</summary>\n")
        for ci, (doc, score) in enumerate(chunks, 1):
            src = doc.metadata.get("source", "?")
            text = _chunk_text_for_prompt(doc)
            if len(text) > 200:
                text = text[:200] + "…"
            w(f"{ci}. **{src}** (score={score:.3f}) — {text}\n")
        w("</details>\n")

        # Prompt
        w("<details><summary>Full prompt sent to models</summary>\n")
        w("```")
        w(f"[SYSTEM]\n{system_prompt}\n")
        w(f"[USER]\n{query_prompts[q['query']]}")
        w("```\n")
        w("</details>\n")

        # Timing table
        w("### Timing\n")
        w("| Model | Total | TTFT | Decode | tok/s | Tokens |")
        w("|-------|------:|-----:|-------:|------:|-------:|")
        for name in model_names:
            r = results[name].get(q["query"])
            if r and "error" not in r:
                w(f"| {name} | {r['total_time']}s | {r['ttft']}s | "
                  f"{r['decode_time']}s | {r['speed']} | {r['tokens']} |")
            elif r:
                w(f"| {name} | ERROR | — | — | — | — |")

        # Answers
        w("\n### Answers\n")
        for name in model_names:
            r = results[name].get(q["query"])
            if r and "error" not in r:
                answer = r["answer"].replace("\n", "\n> ")
                w(f"**{name}**:")
                w(f"> {answer}\n")
            elif r:
                w(f"**{name}**: ⚠ ERROR — {r.get('error', '?')}\n")

        w("---\n")

    # ---- Summary table ----
    w("## Summary\n")
    w("| Model | Size | Load | Warmup | Avg Total | Avg tok/s | Avg Tokens |")
    w("|-------|-----:|-----:|-------:|----------:|----------:|-----------:|")

    for name in model_names:
        meta = results[name].get("_meta", {})
        qr = [v for k, v in results[name].items()
              if k != "_meta" and isinstance(v, dict) and "error" not in v]
        if not qr:
            continue
        avg_total = sum(r["total_time"] for r in qr) / len(qr)
        avg_speed = sum(r["speed"] for r in qr) / len(qr)
        avg_tokens = sum(r["tokens"] for r in qr) / len(qr)
        w(f"| {name} | {meta.get('size_gb', '?')}G | "
          f"{meta.get('load_time', '?')}s | {meta.get('warmup_time', '?')}s | "
          f"{avg_total:.1f}s | {avg_speed:.1f} | {avg_tokens:.0f} |")

    w("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model comparison: same retrieval + prompt → all models",
    )
    procedures_list = ", ".join(f"{k} ({v})" for k, v in PROCEDURES.items())
    parser.add_argument(
        "--procedure", default="hemorroides",
        choices=list(PROCEDURES),
        help=f"Procedure context key (default: hemorroides). Available: {procedures_list}",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="ALIAS",
        help="Run only these model aliases (default: all)",
    )
    parser.add_argument(
        "--query-ids", nargs="+", type=int, default=DEFAULT_QUERY_IDS,
        help=f"Query indices from eval_dataset_realistic.json (default: {DEFAULT_QUERY_IDS})",
    )
    parser.add_argument("--n-threads", type=int, default=9)
    parser.add_argument("--output", default="comparison_report.md")
    args = parser.parse_args()

    # Resolve models (skip known-broken ones)
    models_to_run = [(a, p) for a, p in MODEL_ALIASES.items() if a not in SKIP_MODELS]
    if args.models:
        models_to_run = [(a, p) for a, p in models_to_run if a in args.models]
        unknown = set(args.models) - set(MODEL_ALIASES.keys())
        if unknown:
            print(f"WARNING: unknown model aliases: {unknown}")

    # Resolve procedure
    procedure = PROCEDURES[args.procedure]
    clause = f" que responde preguntas de pacientes sobre {procedure}"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(procedure_clause=clause)

    # Load queries
    with open("eval_dataset_realistic.json") as f:
        all_queries = json.load(f)
    queries = [all_queries[i] for i in args.query_ids]

    print(f"Models: {len(models_to_run)}")
    print(f"Queries: {len(queries)}")
    for i, q in enumerate(queries):
        tag = "✓" if q.get("answerable", True) else "✗"
        print(f"  {i + 1}. [{tag}] {q['query'][:70]}")
    print(f"Procedure: {args.procedure} ({procedure})")
    print(f"Output: {args.output}")
    print()

    # ── Phase 1: Setup retrieval ──
    print("Loading retrieval pipeline...", end="", flush=True)
    t0 = time.perf_counter()
    client = _get_client(QDRANT_PATH)
    retriever = build_retriever(
        client, COLLECTION_NAME, strategy=STRATEGY, tokenizer=TOKENIZER,
        rerank_model=RERANK_MODEL, rerank_candidates=RERANK_CANDIDATES,
    )
    if hasattr(retriever, "preload"):
        retriever.preload()
    print(f" {time.perf_counter() - t0:.1f}s")

    # ── Phase 2: Retrieve + build prompts (once per query) ──
    print("Retrieving chunks...", end="", flush=True)
    t0 = time.perf_counter()
    query_chunks = {}
    query_prompts = {}
    for q in queries:
        rq = f"{procedure}: {q['query']}"
        chunks = retriever.retrieve(rq, top_k=TOP_K)
        query_chunks[q["query"]] = chunks
        query_prompts[q["query"]] = build_prompt(q["query"], chunks)
    client.close()
    print(f" {time.perf_counter() - t0:.1f}s")

    # ── Phase 3: Generate across models ──
    results = {}  # {alias: {"_meta": {...}, query_text: result_dict}}

    for alias, path in models_to_run:
        if not os.path.exists(path):
            print(f"\nSKIPPING {alias}: {path} not found")
            continue

        size_gb = round(os.path.getsize(path) / 1e9, 2)
        print(f"\n{'─' * 60}")
        print(f"  {alias} ({size_gb}G)")
        print(f"{'─' * 60}")

        # Load
        t0 = time.perf_counter()
        try:
            model = _load_model_quiet(path, N_CTX, args.n_threads)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue
        load_time = round(time.perf_counter() - t0, 1)
        print(f"  Loaded in {load_time}s")

        # System prompt (prepend /no_think for models that need it)
        effective_sp = system_prompt
        if alias in NEEDS_NO_THINK:
            effective_sp = "/no_think\n\n" + system_prompt

        # Warmup
        t0 = time.perf_counter()
        try:
            model.create_chat_completion(
                messages=[
                    {"role": "system", "content": effective_sp},
                    {"role": "user", "content": "FRAGMENTOS:\nTexto de ejemplo.\n\nPREGUNTA: hola"},
                ],
                max_tokens=1, temperature=0,
            )
        except Exception as e:
            print(f"  FAILED warmup: {e}")
            del model
            continue
        warmup_time = round(time.perf_counter() - t0, 1)
        print(f"  Warmup: {warmup_time}s")

        model_results = {
            "_meta": {"size_gb": size_gb, "load_time": load_time, "warmup_time": warmup_time},
        }

        for q in queries:
            messages = [
                {"role": "system", "content": effective_sp},
                {"role": "user", "content": query_prompts[q["query"]]},
            ]
            try:
                result = generate_timed(model, messages, MAX_TOKENS, TEMPERATURE)
                model_results[q["query"]] = result
                tag = "✓" if q.get("answerable", True) else "✗"
                print(f"  [{tag}] {q['query'][:40]:40s} "
                      f"{result['total_time']:5.1f}s {result['speed']:5.1f}t/s "
                      f"{result['tokens']:3d}tok")
            except Exception as e:
                model_results[q["query"]] = {"error": str(e)}
                print(f"  [!] {q['query'][:40]:40s} ERROR: {e}")

        results[alias] = model_results
        del model

    # ── Phase 4: Write report ──
    print(f"\nWriting report to {args.output}...")
    write_report(results, queries, query_prompts, query_chunks,
                 system_prompt, procedure, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
