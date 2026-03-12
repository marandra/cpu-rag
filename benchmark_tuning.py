"""
Benchmark: gemma-3n-E2B prompt tuning + CPU thread scaling.

Phase 1: Prompt variants — test different system prompts and prompt templates
Phase 2: CPU scaling — n_threads 1-12 with best prompt

Usage:
    uv run python benchmark_tuning.py --phase prompts
    uv run python benchmark_tuning.py --phase threads
    uv run python benchmark_tuning.py --phase all
"""

import argparse
import json
import os
import time

from src.embeddings import _get_client
from src.llm import load_model
from src.retrieval import build_retriever

# --- Config ---
MODEL_PATH = "./models/granite-4.0-1b-Q4_K_M.gguf"
QDRANT_PATH = "./sweep_qdrant/mt512_ov0.20_cp_llm"
COLLECTION = "medical_docs"
STRATEGY = "hybrid"
TOKENIZER = "spacy"
TOP_K = 5
N_CTX = 2048

# Diverse test queries (indices into eval dataset): surgery details, recovery,
# pain management, perioperative care, specific conditions
TEST_QUERY_IDS = [0, 5, 9, 13, 19, 26, 30, 40]

# --- Prompt Variants ---

SYSTEM_PROMPTS = {
    "baseline": (
        "Responde SOLO con la información de los fragmentos. "
        "Si no aparece, di 'No disponible'. Español. 2-3 frases."
    ),
    "exhaustive": (
        "Eres un asistente médico. Responde SOLO con información de los fragmentos proporcionados. "
        "Incluye TODOS los datos relevantes: tiempos, cantidades, pasos, recomendaciones. "
        "Si la información no aparece, di 'No disponible'. Responde en español."
    ),
    "structured": (
        "Eres un asistente médico. Responde SOLO con información de los fragmentos. "
        "Organiza la respuesta en puntos claros. Incluye todos los detalles relevantes "
        "(tiempos, dosis, procedimientos, recomendaciones). Español."
    ),
    "patient_friendly": (
        "Eres un asistente que ayuda a pacientes. Responde SOLO con información de los fragmentos. "
        "Usa lenguaje sencillo y claro. Incluye toda la información relevante sin omitir detalles "
        "importantes como tiempos, cantidades o pasos a seguir. Español."
    ),
    "no_system": None,  # No system prompt at all
}

PROMPT_TEMPLATES = {
    "compact": (
        "FRAGMENTOS:\n{context}\n\nPREGUNTA: {query}\nRESPUESTA:"
    ),
    "instructive": (
        "A continuación tienes fragmentos de documentos médicos. "
        "Úsalos para responder la pregunta del paciente con todos los detalles relevantes.\n\n"
        "FRAGMENTOS:\n{context}\n\n"
        "PREGUNTA DEL PACIENTE: {query}\n\n"
        "RESPUESTA DETALLADA:"
    ),
    "numbered": (
        "Fragmentos de referencia:\n{context_numbered}\n\n"
        "Pregunta: {query}\n\n"
        "Responde usando la información de los fragmentos anteriores. "
        "Incluye todos los detalles relevantes."
    ),
}

# Prompt template × system prompt combinations to test
PROMPT_CONFIGS = [
    # (name, system_prompt_key, template_key, max_tokens, text_source)
    ("baseline", "baseline", "compact", 80, "compact"),
    ("exhaustive_compact", "exhaustive", "compact", 256, "compact"),
    ("exhaustive_instruct", "exhaustive", "instructive", 256, "compact"),
    ("structured_compact", "structured", "compact", 256, "compact"),
    ("structured_instruct", "structured", "instructive", 256, "compact"),
    ("patient_compact", "patient_friendly", "compact", 256, "compact"),
    ("patient_instruct", "patient_friendly", "instructive", 256, "compact"),
    ("exhaustive_numbered", "exhaustive", "numbered", 256, "compact"),
    ("no_sys_instruct", "no_system", "instructive", 256, "compact"),
    # Test full text (not compact) as context
    ("exhaustive_full", "exhaustive", "compact", 256, "full"),
    ("structured_full", "structured", "instructive", 256, "full"),
]


def load_test_queries(ids: list[int]) -> list[dict]:
    with open("eval_dataset_realistic.json") as f:
        data = json.load(f)
    return [data[i] for i in ids]


def chunk_text(doc, source: str = "compact") -> str:
    if source == "compact":
        return (
            doc.metadata.get("page_content_compact")
            or doc.metadata.get("page_content_original")
            or doc.page_content
        )
    elif source == "full":
        return doc.page_content
    else:  # original
        return doc.metadata.get("page_content_original") or doc.page_content


def build_prompt(query, chunks, template_key, text_source):
    template = PROMPT_TEMPLATES[template_key]
    if "{context_numbered}" in template:
        context = "\n".join(
            f"[{i}] {chunk_text(doc, text_source)}"
            for i, (doc, _) in enumerate(chunks, 1)
        )
        return template.format(context_numbered=context, query=query)
    else:
        context = "\n---\n".join(
            chunk_text(doc, text_source) for doc, _ in chunks
        )
        return template.format(context=context, query=query)


def run_generation(model, prompt, system_prompt, max_tokens):
    """Generate and return (text, usage, time_s)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    t0 = time.perf_counter()
    resp = model.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.0,
    )
    elapsed = time.perf_counter() - t0

    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    return text, usage, elapsed


def check_span_coverage(answer: str, spans: list[str]) -> dict:
    """Check how many relevant spans are mentioned in the answer."""
    answer_lower = answer.lower()
    hits = 0
    details = []
    for span in spans:
        # Fuzzy: check if key words from span appear in answer
        span_words = [w for w in span.lower().split() if len(w) > 3]
        if not span_words:
            hits += 1
            details.append((span[:60], True))
            continue
        matched = sum(1 for w in span_words if w in answer_lower)
        ratio = matched / len(span_words)
        hit = ratio >= 0.4
        if hit:
            hits += 1
        details.append((span[:60], hit))
    return {
        "hits": hits,
        "total": len(spans),
        "coverage": hits / len(spans) if spans else 0,
        "details": details,
    }


def phase_prompts(model, retriever, queries):
    """Test all prompt configurations on test queries."""
    print("=" * 70)
    print("PHASE 1: PROMPT TUNING")
    print("=" * 70)

    # Pre-retrieve chunks for all queries (same for all prompt configs)
    print("\nRetrieving chunks for test queries...")
    query_chunks = {}
    for q in queries:
        results = retriever.retrieve(q["query"], top_k=TOP_K)
        query_chunks[q["query"]] = results

    # Warmup model
    print("Warming up model...")
    run_generation(model, "Hola", "Responde breve.", 10)

    results_all = []

    for cfg_name, sys_key, tmpl_key, max_tok, txt_src in PROMPT_CONFIGS:
        sys_prompt = SYSTEM_PROMPTS[sys_key]
        print(f"\n{'─' * 70}")
        print(f"Config: {cfg_name}")
        print(f"  system={sys_key}, template={tmpl_key}, max_tokens={max_tok}, text={txt_src}")
        print(f"{'─' * 70}")

        cfg_results = {
            "config": cfg_name,
            "system_key": sys_key,
            "template_key": tmpl_key,
            "max_tokens": max_tok,
            "text_source": txt_src,
            "queries": [],
        }
        total_coverage = 0
        total_tokens = 0
        total_time = 0

        for q in queries:
            chunks = query_chunks[q["query"]]
            prompt = build_prompt(q["query"], chunks, tmpl_key, txt_src)
            text, usage, elapsed = run_generation(model, prompt, sys_prompt, max_tok)

            spans = q.get("relevant_spans", [])
            coverage = check_span_coverage(text, spans)

            comp_tokens = usage.get("completion_tokens", 0)
            speed = comp_tokens / elapsed if elapsed > 0 else 0

            total_coverage += coverage["coverage"]
            total_tokens += comp_tokens
            total_time += elapsed

            cfg_results["queries"].append({
                "query": q["query"][:60],
                "answer": text,
                "coverage": coverage["coverage"],
                "spans_hit": f"{coverage['hits']}/{coverage['total']}",
                "comp_tokens": comp_tokens,
                "time_s": round(elapsed, 2),
                "speed": round(speed, 1),
            })

            cov_pct = coverage["coverage"] * 100
            print(f"  Q: {q['query'][:55]:55s} cov={cov_pct:5.1f}%  "
                  f"tok={comp_tokens:3d}  {elapsed:.1f}s  {speed:.1f}t/s")

        n = len(queries)
        avg_cov = total_coverage / n * 100
        avg_tok = total_tokens / n
        avg_time = total_time / n
        cfg_results["avg_coverage"] = round(avg_cov, 1)
        cfg_results["avg_tokens"] = round(avg_tok, 1)
        cfg_results["avg_time"] = round(avg_time, 2)
        cfg_results["total_time"] = round(total_time, 1)

        print(f"\n  >>> AVG: coverage={avg_cov:.1f}%  tokens={avg_tok:.0f}  time={avg_time:.1f}s")
        results_all.append(cfg_results)

    # Summary table
    print(f"\n{'=' * 70}")
    print("PROMPT TUNING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Config':30s} {'Cov%':>6s} {'Tok':>5s} {'Time':>6s}")
    print(f"{'─' * 50}")
    for r in sorted(results_all, key=lambda x: x["avg_coverage"], reverse=True):
        print(f"{r['config']:30s} {r['avg_coverage']:5.1f}% {r['avg_tokens']:5.0f} {r['avg_time']:5.1f}s")

    return results_all


def phase_threads(queries, best_prompt_config=None):
    """CPU thread scaling: n_threads 1-12."""
    print(f"\n{'=' * 70}")
    print("PHASE 2: CPU THREAD SCALING (n_threads 1-12)")
    print(f"{'=' * 70}")

    # Use best prompt config or default to exhaustive_instruct
    if best_prompt_config:
        cfg_name, sys_key, tmpl_key, max_tok, txt_src = best_prompt_config
    else:
        cfg_name = "exhaustive_instruct"
        sys_key, tmpl_key, max_tok, txt_src = "exhaustive", "instructive", 256, "compact"

    sys_prompt = SYSTEM_PROMPTS[sys_key]
    print(f"Using prompt config: {cfg_name}")

    # Pick 3 queries for speed (threads sweep is slow due to model reloads)
    test_qs = queries[:3]

    # Pre-retrieve
    client = _get_client(QDRANT_PATH)
    retriever = build_retriever(client, COLLECTION, strategy=STRATEGY, tokenizer=TOKENIZER)
    query_chunks = {}
    for q in test_qs:
        query_chunks[q["query"]] = retriever.retrieve(q["query"], top_k=TOP_K)
    client.close()

    thread_counts = list(range(1, 13))
    results_all = []

    for n_threads in thread_counts:
        print(f"\n--- n_threads={n_threads} ---")
        model = load_model(MODEL_PATH, n_ctx=N_CTX, n_threads=n_threads)

        # Warmup
        run_generation(model, "Hola", "Breve.", 10)

        total_tokens = 0
        total_time = 0
        total_prompt_tokens = 0

        for q in test_qs:
            chunks = query_chunks[q["query"]]
            prompt = build_prompt(q["query"], chunks, tmpl_key, txt_src)
            text, usage, elapsed = run_generation(model, prompt, sys_prompt, max_tok)
            comp_tokens = usage.get("completion_tokens", 0)
            prompt_tokens = usage.get("prompt_tokens", 0)
            total_tokens += comp_tokens
            total_time += elapsed
            total_prompt_tokens += prompt_tokens

        avg_speed = total_tokens / total_time if total_time > 0 else 0
        avg_time = total_time / len(test_qs)
        avg_tok = total_tokens / len(test_qs)
        avg_prompt = total_prompt_tokens / len(test_qs)

        result = {
            "n_threads": n_threads,
            "avg_speed": round(avg_speed, 2),
            "avg_time": round(avg_time, 2),
            "avg_comp_tokens": round(avg_tok, 1),
            "avg_prompt_tokens": round(avg_prompt, 1),
            "total_time": round(total_time, 1),
        }
        results_all.append(result)
        print(f"  avg: {avg_speed:.2f} tok/s, {avg_time:.1f}s/query, {avg_tok:.0f} tok/query")

        del model

    # Summary
    print(f"\n{'=' * 70}")
    print("THREAD SCALING SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Threads':>8s} {'tok/s':>8s} {'time/q':>8s} {'tokens':>8s}")
    print(f"{'─' * 35}")
    best = max(results_all, key=lambda x: x["avg_speed"])
    for r in results_all:
        marker = " <<<" if r == best else ""
        print(f"{r['n_threads']:>8d} {r['avg_speed']:>7.2f} {r['avg_time']:>7.1f}s {r['avg_comp_tokens']:>7.0f}{marker}")

    return results_all


def main():
    parser = argparse.ArgumentParser(description="Prompt tuning + CPU scaling benchmark")
    parser.add_argument("--phase", choices=["prompts", "threads", "all"], default="all")
    parser.add_argument("--output", default="benchmark_tuning_results.json")
    args = parser.parse_args()

    queries = load_test_queries(TEST_QUERY_IDS)
    print(f"Test queries: {len(queries)}")
    for i, q in enumerate(queries):
        print(f"  {i}. {q['query'][:70]}")

    all_results = {}

    if args.phase in ("prompts", "all"):
        client = _get_client(QDRANT_PATH)
        retriever = build_retriever(client, COLLECTION, strategy=STRATEGY, tokenizer=TOKENIZER)
        model = load_model(MODEL_PATH, n_ctx=N_CTX)
        prompt_results = phase_prompts(model, retriever, queries)
        all_results["prompts"] = prompt_results
        del model
        client.close()

        # Determine best config for thread scaling
        best = max(prompt_results, key=lambda x: x["avg_coverage"])
        best_cfg = None
        for cfg_name, sys_key, tmpl_key, max_tok, txt_src in PROMPT_CONFIGS:
            if cfg_name == best["config"]:
                best_cfg = (cfg_name, sys_key, tmpl_key, max_tok, txt_src)
                break
    else:
        best_cfg = None

    if args.phase in ("threads", "all"):
        thread_results = phase_threads(queries, best_prompt_config=best_cfg)
        all_results["threads"] = thread_results

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
