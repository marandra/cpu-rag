"""
Compare 3 candidate models × 2 prompt versions (V1 and V4).

Runs the full eval dataset (54 queries) for each of the 6 combos,
shares retriever setup, reuses loaded models across prompt versions,
and produces a consolidated comparison report.

Models: granite-1b, gemma-3n, ministral
Prompts: V1 (simple), V4 (strict with examples)

Usage:
    uv run python eval_candidates.py
    uv run python eval_candidates.py --min-score -3.0
    uv run python eval_candidates.py --output eval_candidates_report.md
"""

import argparse
import json
import os
import re
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
    PROMPT_TEMPLATE,
)

sys.stdout.reconfigure(line_buffering=True)

STRATEGY = "hybrid+rerank"
TOKENIZER = "spacy"
TOP_K = 5
N_CTX = 2048
MAX_TOKENS = 256
TEMPERATURE = 0.3

# ---------------------------------------------------------------------------
# Model + prompt combos to evaluate
# ---------------------------------------------------------------------------

PROMPT_V1 = (
    "Eres un asistente médico{procedure_clause}.\n\n"
    "Reglas:\n"
    "1. Usa SOLO la información de los fragmentos proporcionados. No inventes ni añadas datos externos.\n"
    "2. Si los fragmentos no contienen la respuesta, responde: \"No tengo información sobre eso.\"\n"
    "3. Responde en un párrafo breve y directo.\n"
    "4. Habla directamente al paciente.\n"
    "5. No menciones los fragmentos ni tu razonamiento. Ve directo a la respuesta."
)

PROMPT_V4 = (
    "Eres un asistente médico{procedure_clause}.\n\n"
    "REGLAS ESTRICTAS:\n"
    "1. Si los fragmentos contienen información relacionada con la pregunta, "
    "DEBES responder usando SOLO esa información.\n"
    "2. NUNCA uses tu conocimiento general. PROHIBIDO inventar datos, medicamentos, "
    "precios o plazos que NO estén en los fragmentos.\n"
    "3. Si los fragmentos NO contienen información relevante para la pregunta, "
    "responde: \"No tengo información sobre eso.\"\n"
    "4. Si la pregunta no tiene relación con cirugía y cuidados perioperatorios, "
    "responde: \"No tengo información sobre eso.\"\n"
    "5. Responde en un párrafo breve y directo al paciente.\n"
    "6. No menciones los fragmentos ni tu razonamiento.\n\n"
    "Ejemplos de preguntas que DEBES rechazar con "
    "\"No tengo información sobre eso.\":\n"
    "- Preguntas sobre temas no médicos (geografía, tecnología, cocina...)\n"
    "- Preguntas médicas cuya respuesta NO aparece en los fragmentos\n"
    "- Preguntas sobre costes, seguros, bajas laborales, trámites o segunda opinión\n"
    "- Preguntas sobre quién eres o qué haces"
)

COMBOS = [
    ("granite-1b", "v1", PROMPT_V1),
    ("granite-1b", "v4", PROMPT_V4),
    ("gemma-3n",   "v1", PROMPT_V1),
    ("gemma-3n",   "v4", PROMPT_V4),
    ("ministral",  "v1", PROMPT_V1),
    ("ministral",  "v4", PROMPT_V4),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text_for_prompt(doc) -> str:
    return (
        doc.metadata.get("page_content_compact")
        or doc.metadata.get("page_content_original")
        or doc.page_content
    )


def build_prompt(query, chunks):
    context = "\n---\n".join(_chunk_text_for_prompt(doc) for doc, _ in chunks)
    return PROMPT_TEMPLATE.format(context=context, query=query)


def generate(model, messages, max_tokens, temperature):
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
    total = time.perf_counter() - t_start
    ttft = ttft or total
    decode = total - ttft
    tok_s = n_tokens / decode if decode > 0 else 0
    clean_text = re.sub(r"<think>.*?</think>\s*", "", full_text, flags=re.DOTALL).strip()
    return {
        "answer": clean_text,
        "tokens": n_tokens,
        "total_time": round(total, 2),
        "ttft": round(ttft, 2),
        "tok_s": round(tok_s, 1),
    }


def is_refusal(answer):
    lower = answer.lower().strip()
    refusal_phrases = [
        "no tengo información",
        "no dispongo de información",
        "no cuento con información",
        "no puedo responder",
        "fuera de mi ámbito",
        "no está dentro de",
    ]
    return any(p in lower for p in refusal_phrases)


def keyword_score(answer, expected_keywords):
    if not expected_keywords:
        return None
    lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in lower)
    return hits / len(expected_keywords)


def classify(answerable, refused, kw_score):
    if answerable:
        if refused:
            return "FALSE_REFUSAL"
        elif kw_score is not None and kw_score >= 0.25:
            return "GOOD"
        elif kw_score is not None and kw_score > 0:
            return "PARTIAL"
        else:
            return "MISS"
    else:
        return "CORRECT_REFUSAL" if refused else "OOS_LEAK"


def summarize(results):
    in_scope = [r for r in results if r["answerable"]]
    oos = [r for r in results if not r["answerable"]]
    return {
        "n_in": len(in_scope),
        "n_oos": len(oos),
        "good": sum(1 for r in in_scope if r["verdict"] == "GOOD"),
        "partial": sum(1 for r in in_scope if r["verdict"] == "PARTIAL"),
        "miss": sum(1 for r in in_scope if r["verdict"] == "MISS"),
        "false_refusal": sum(1 for r in in_scope if r["verdict"] == "FALSE_REFUSAL"),
        "false_refusal_auto": sum(1 for r in in_scope if r["verdict"] == "FALSE_REFUSAL" and r["auto_refused"]),
        "oos_correct": sum(1 for r in oos if r["verdict"] == "CORRECT_REFUSAL"),
        "oos_auto": sum(1 for r in oos if r["auto_refused"]),
        "oos_leak": sum(1 for r in oos if r["verdict"] == "OOS_LEAK"),
        "avg_time": sum(r["total_time"] for r in results) / len(results),
        "avg_tok_s": sum(r["tok_s"] for r in results if r["tok_s"] > 0) / max(1, sum(1 for r in results if r["tok_s"] > 0)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(all_results, eval_data, procedure, min_score, output_path):
    lines = []
    w = lines.append

    w("# Candidate Model Comparison\n")
    w(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    w(f"**Procedure**: {procedure}  ")
    w(f"**Strategy**: {STRATEGY} | top_k={TOP_K} | rerank_candidates={RERANK_CANDIDATES}  ")
    w(f"**Min score**: {min_score}  ")
    w(f"**Queries**: {len(eval_data)} ({sum(1 for q in eval_data if q['answerable'])} in-scope, "
      f"{sum(1 for q in eval_data if not q['answerable'])} OOS)\n")

    combos = list(all_results.keys())

    # --- Summary table ---
    w("## Summary\n")
    w("| Metric | " + " | ".join(combos) + " |")
    w("|--------|" + "|".join(["-------:"] * len(combos)) + "|")

    summaries = {c: summarize(all_results[c]) for c in combos}
    metrics = [
        ("In-scope: GOOD", "good"),
        ("In-scope: PARTIAL", "partial"),
        ("In-scope: MISS", "miss"),
        ("In-scope: FALSE REFUSAL", "false_refusal"),
        ("  (auto-refused)", "false_refusal_auto"),
        ("OOS: CORRECT REFUSAL", "oos_correct"),
        ("  (auto-refused)", "oos_auto"),
        ("OOS: LEAK", "oos_leak"),
        ("Avg time (s)", "avg_time"),
        ("Avg tok/s", "avg_tok_s"),
    ]
    for label, key in metrics:
        vals = []
        for c in combos:
            v = summaries[c][key]
            if key in ("avg_time", "avg_tok_s"):
                vals.append(f"{v:.1f}")
            elif key in ("good", "oos_correct"):
                total = summaries[c]["n_in"] if "In-scope" in label else summaries[c]["n_oos"]
                vals.append(f"**{v}**/{total}")
            else:
                total = summaries[c]["n_in"] if "In-scope" in label or "auto-refused" in label else summaries[c]["n_oos"]
                vals.append(f"{v}/{total}")
        w(f"| {label} | " + " | ".join(vals) + " |")

    # --- Per-query comparison table ---
    w("\n## Per-query results\n")
    header = "| # | Scope | Category | Query | " + " | ".join(combos) + " |"
    sep = "|---|-------|----------|-------|" + "|".join(["-----"] * len(combos)) + "|"
    w(header)
    w(sep)

    markers = {
        "GOOD": "GOOD", "PARTIAL": "PARTIAL", "MISS": "MISS",
        "FALSE_REFUSAL": "FALSE_REF", "CORRECT_REFUSAL": "OK_REF", "OOS_LEAK": "LEAK",
    }

    for i, q in enumerate(eval_data):
        scope = "IN" if q["answerable"] else "OOS"
        query_short = q["query"][:45].replace("|", "\\|")
        cat = q.get("category", "?")
        cells = []
        for c in combos:
            r = all_results[c][i]
            m = markers[r["verdict"]]
            kw = f" {r['kw_score']:.0%}" if r["kw_score"] is not None else ""
            ar = " (AR)" if r["auto_refused"] else ""
            cells.append(f"{m}{kw}{ar}")
        w(f"| {i+1} | {scope} | {cat} | {query_short} | " + " | ".join(cells) + " |")

    # --- Failure details ---
    w("\n## Failure details\n")

    for combo in combos:
        failures = [r for r in all_results[combo]
                    if r["verdict"] in ("FALSE_REFUSAL", "OOS_LEAK", "MISS")]
        if not failures:
            w(f"### {combo}: no failures\n")
            continue
        w(f"### {combo}\n")
        for r in failures:
            tag = r["verdict"]
            ar = " [AUTO]" if r["auto_refused"] else ""
            w(f"- **{tag}{ar}** — Q{r['idx']+1} `{r['query'][:60]}`")
            if r["verdict"] == "OOS_LEAK":
                w(f"  > {r['answer'][:150]}")
        w("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare candidate models with best prompts")
    parser.add_argument("--min-score", type=float, default=-3.0,
                        help="Reranker score threshold (default: -3.0)")
    parser.add_argument("--procedure", default="hemorroides", choices=list(PROCEDURES))
    parser.add_argument("--n-threads", type=int, default=9)
    parser.add_argument("--output", default="eval_candidates_report.md")
    args = parser.parse_args()

    procedure = PROCEDURES[args.procedure]
    clause = f" que responde preguntas de pacientes sobre {procedure}"
    min_score = args.min_score

    # Load eval dataset
    with open("eval_dataset_realistic.json") as f:
        eval_data = json.load(f)

    print(f"Eval: {len(eval_data)} queries, min_score={min_score}")
    print(f"Combos: {', '.join(f'{m}+{p}' for m, p, _ in COMBOS)}\n")

    # Setup retriever (shared across all models)
    print("Loading retriever...", end="", flush=True)
    t0 = time.perf_counter()
    client = _get_client(QDRANT_PATH)
    retriever = build_retriever(
        client, COLLECTION_NAME, strategy=STRATEGY, tokenizer=TOKENIZER,
        rerank_model=RERANK_MODEL, rerank_candidates=RERANK_CANDIDATES,
    )
    if hasattr(retriever, "preload"):
        retriever.preload()
    print(f" {time.perf_counter() - t0:.1f}s")

    # Pre-compute retrieval for all queries (same for all models)
    print("Retrieving chunks for all queries...", end="", flush=True)
    t0 = time.perf_counter()
    query_chunks = {}
    query_prompts = {}
    for q in eval_data:
        rq = f"{procedure}: {q['query']}"
        chunks = retriever.retrieve(rq, top_k=TOP_K, min_score=min_score)
        query_chunks[q["query"]] = chunks
        if chunks:
            query_prompts[q["query"]] = build_prompt(q["query"], chunks)
    print(f" {time.perf_counter() - t0:.1f}s\n")

    # Run each model+prompt combo
    all_results = {}
    loaded_model = None
    loaded_alias = None

    for model_alias, prompt_ver, prompt_template in COMBOS:
        combo_label = f"{model_alias}+{prompt_ver}"
        system_prompt = prompt_template.format(procedure_clause=clause)
        model_path = MODEL_ALIASES[model_alias]

        print(f"{'=' * 60}")
        print(f"  {combo_label}")
        print(f"{'=' * 60}")

        # Load model (reuse if same model as previous combo)
        if loaded_alias != model_alias:
            if loaded_model is not None:
                del loaded_model
            print(f"  Loading {model_alias}...", end="", flush=True)
            t0 = time.perf_counter()
            devnull = os.open(os.devnull, os.O_WRONLY)
            old_stderr = os.dup(2)
            os.dup2(devnull, 2)
            try:
                loaded_model = load_model(model_path, n_ctx=N_CTX, n_threads=args.n_threads)
            finally:
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stderr)
            loaded_alias = model_alias
            print(f" {time.perf_counter() - t0:.1f}s")
        else:
            print(f"  Reusing {model_alias} (already loaded)")
        model = loaded_model

        # Warmup
        print("  Warmup...", end="", flush=True)
        t0 = time.perf_counter()
        model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "FRAGMENTOS:\nTexto.\n\nPREGUNTA: hola"},
            ],
            max_tokens=1, temperature=0,
        )
        print(f" {time.perf_counter() - t0:.1f}s\n")

        # Evaluate
        results = []
        for i, q in enumerate(eval_data):
            query = q["query"]
            answerable = q["answerable"]
            category = q.get("category", "?")
            chunks = query_chunks[query]
            n_chunks = len(chunks)

            if not chunks:
                answer = "No tengo información sobre eso."
                auto_refused = True
                gen = {"total_time": 0, "tokens": 0, "ttft": 0, "tok_s": 0}
            else:
                auto_refused = False
                prompt = query_prompts[query]
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                gen = generate(model, messages, MAX_TOKENS, TEMPERATURE)
                answer = gen["answer"]

            refused = is_refusal(answer) or auto_refused
            kw = keyword_score(answer, q.get("expected_keywords", []))
            verdict = classify(answerable, refused, kw)

            marker = {"GOOD": "✓", "PARTIAL": "~", "MISS": "✗",
                      "FALSE_REFUSAL": "✗R", "CORRECT_REFUSAL": "✓R", "OOS_LEAK": "✗L"}[verdict]
            status = "AR" if auto_refused else ("R" if refused else "A")
            kw_str = f"kw={kw:.0%}" if kw is not None else ""
            print(f"  [{i+1:2d}] {marker:3s} {status:2s} chunks={n_chunks} {kw_str:6s} "
                  f"{gen['total_time']:5.1f}s  {category:20s}  {query[:50]}")

            results.append({
                "idx": i, "query": query, "answerable": answerable,
                "category": category, "n_chunks": n_chunks,
                "auto_refused": auto_refused, "refused": refused,
                "verdict": verdict, "kw_score": kw,
                "answer": answer, "total_time": gen["total_time"],
                "tok_s": gen.get("tok_s", 0),
            })

        s = summarize(results)
        print(f"\n  IN-SCOPE: {s['good']} good, {s['partial']} partial, "
              f"{s['miss']} miss, {s['false_refusal']} false_ref")
        print(f"  OOS: {s['oos_correct']}/{s['n_oos']} correct, {s['oos_leak']} leaks")
        print(f"  Avg time: {s['avg_time']:.1f}s | Avg tok/s: {s['avg_tok_s']:.1f}\n")

        all_results[combo_label] = results

    client.close()

    # Write consolidated report
    print(f"Writing report to {args.output}...")
    write_report(all_results, eval_data, procedure, min_score, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
