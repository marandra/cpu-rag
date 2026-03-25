"""
Interactive RAG: load once, ask many questions with streaming output.

Shows per-stage timing (retrieval, TTFT, generation) and a final summary.

Usage:
    python inspect_rag.py
    python inspect_rag.py --model gemma-3n
    python inspect_rag.py --strategy hybrid --top-k 5
    python inspect_rag.py --retrieval-only
"""

import argparse
import os
import time

from src.embeddings import _get_client
from src.llm import load_model
from src.retrieval import STRATEGIES, build_retriever

# --- Defaults ---
QDRANT_PATH = "./qdrant_manual"
COLLECTION_NAME = "medical_docs"
DEFAULT_MODEL = "granite-1b"
DEFAULT_STRATEGY = "hybrid"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 256

# --- Model aliases (--model accepts short name or full path) ---
MODEL_ALIASES = {
    "granite-1b":   "./models/granite-4.0-1b-Q4_K_M.gguf",
    "gemma-3n":     "./models/google_gemma-3n-E2B-it-Q4_K_M.gguf",
    "gemma-1b":     "./models/gemma-3-1b-it-Q4_K_M.gguf",
    "qwen3.5-2b":   "./models/Qwen3.5-2B-Q4_K_M.gguf",
    "qwen3.5-0.8b": "./models/Qwen3.5-0.8B-Q4_K_M.gguf",
    "qwen3-0.6b":   "./models/Qwen3-0.6B-Q4_K_M.gguf",
    "qwen2.5-1.5b": "./models/qwen2.5-1.5b-instruct-Q4_K_M.gguf",
    "qwen2.5-0.5b": "./models/qwen2.5-0.5b-instruct-Q4_K_M.gguf",
    "llama-3b":     "./models/llama-3.2-3b-instruct-Q4_K_M.gguf",
    "llama-1b":     "./models/llama-3.2-1b-instruct-Q4_K_M.gguf",
    "smollm2":      "./models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    "smollm3":      "./models/SmolLM3-3B-Q4_K_M.gguf",
    "ministral":    "./models/Ministral-3-3B-Q4_K_M.gguf",
    "pleias":       "./models/Pleias-RAG-1B.gguf",
}

# --- Reranker options (uncomment one) ---
# RERANK_MODEL = "BAAI/bge-reranker-v2-m3"                    # 568M, multilingual, best quality
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 66M, multilingual (mMARCO, includes Spanish), fast
# RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L6-H384-v1"   # 22M, multilingual, fastest
RERANK_CANDIDATES = 8  # chunks scored by reranker; best top_k are kept

SYSTEM_PROMPT_TEMPLATE = (
    "Eres un asistente médico{procedure_clause}.\n\n"
    "Reglas:\n"
    "1. Usa SOLO la información de los fragmentos proporcionados. "
    "No inventes ni añadas datos externos.\n"
    "2. Si los fragmentos no contienen la respuesta, responde: "
    '"No tengo información sobre eso."\n'
    "3. Responde en un párrafo breve y directo.\n"
    "4. Habla directamente al paciente.\n"
    "5. No menciones los fragmentos ni tu razonamiento. "
    "Ve directo a la respuesta."
)

PROMPT_TEMPLATE = "FRAGMENTOS:\n{context}\n\nPREGUNTA: {query}"


def _chunk_text_for_prompt(doc) -> str:
    """Pick the best text variant for the LLM prompt."""
    return (
        doc.metadata.get("page_content_compact")
        or doc.metadata.get("page_content_original")
        or doc.page_content
    )


def build_prompt(query: str, chunks: list[tuple]) -> str:
    context = "\n---\n".join(
        _chunk_text_for_prompt(doc) for doc, _score in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def print_chunk(i: int, doc, score: float):
    """Display a retrieved chunk with color-coded sections.

    [i] score + source    — bold white
    [full] original text  — yellow (the actual document text)
    [compact] summary     — cyan
    questions + concerns  — cyan
    """
    src = doc.metadata.get("source", "?")
    original = doc.metadata.get("page_content_original", "")
    compact = doc.metadata.get("page_content_compact", "")
    questions = doc.metadata.get("doc2query_questions", [])
    concerns = doc.metadata.get("doc2query_concerns", [])

    # Header
    print(f"\033[1m  [{i}] score={score:.4f}  src={src}\033[0m")
    print()

    # Original text (yellow)
    print(f"\033[33m  [full] {original or doc.page_content}\033[0m")
    print()

    # Compact summary (cyan)
    if compact:
        print(f"\033[36m  [compact] {compact}\033[0m")
        print()

    # Questions (dim)
    if questions:
        print(f"\033[2m  Preguntas frecuentes:\033[0m")
        for q in questions:
            print(f"\033[2m  {q}\033[0m")
        print()

    # Concerns (dim)
    if concerns:
        print(f"\033[2m  El paciente dice:\033[0m")
        for c in concerns:
            print(f"\033[2m  {c}\033[0m")

    print()


def stream_and_print(model, messages: list[dict], max_tokens: int, t_question: float):
    """Stream generation, printing tokens as they arrive.

    Returns (ttft, decode_time, n_tokens, text).
    """
    t_first_token = None
    full_text = ""
    n_tokens = 0

    for chunk in model.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.3, stream=True,
    ):
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            print(text, end="", flush=True)
            full_text += text
            n_tokens += 1

    t_last_token = time.perf_counter()
    print()  # newline after streamed output

    ttft = (t_first_token or t_last_token) - t_question
    decode_time = t_last_token - (t_first_token or t_last_token)
    return ttft, decode_time, n_tokens, full_text


def run_interactive(retriever, model, top_k: int, max_tokens: int, system_prompt: str,
                     *, retrieval_only: bool = False, procedure: str | None = None):
    """Interactive loop: one query at a time (no conversation history)."""
    timing_history = []
    print("\nType your question (or 'exit' / Ctrl+D to quit).\n")

    while True:
        try:
            query = input("\033[1m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query or query.lower() in ("exit", "quit", "q"):
            break

        t_question = time.perf_counter()

        # --- Retrieval (augment query with procedure context) ---
        retrieval_query = f"{procedure}: {query}" if procedure else query
        t0 = time.perf_counter()
        results = retriever.retrieve(retrieval_query, top_k=top_k)
        t_retrieval = time.perf_counter() - t0

        # Show chunks
        print(f"\n\033[2m── retrieved {len(results)} chunks in {t_retrieval:.2f}s ──\033[0m\n")
        for i, (doc, score) in enumerate(results, 1):
            print_chunk(i, doc, score)

        if retrieval_only:
            timing_history.append({
                "query": query,
                "tag": f"q{len(timing_history) + 1}",
                "retrieval": t_retrieval,
            })
            continue

        # --- Build prompt and generate ---
        prompt = build_prompt(query, results)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_tokens = len(model.tokenize(prompt.encode("utf-8"), add_bos=False))

        # Show system prompt + user prompt together
        print(f"\033[2m── prompt ({prompt_tokens} tokens) ──\033[0m")
        print(f"\033[35m{system_prompt}\033[0m")
        print()
        print(f"\033[36m{prompt}\033[0m")

        # --- Generate (streaming) ---
        tag = f"q{len(timing_history) + 1}"
        print(f"\n\033[2m── generating ({tag}) ──\033[0m\n")
        ttft, decode_time, n_tokens, text = stream_and_print(model, messages, max_tokens, t_question)

        # Timing
        speed = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0
        total = ttft + decode_time
        print(
            f"\n\033[32m  TTFT={ttft:.2f}s  decode={decode_time:.2f}s  "
            f"total={total:.2f}s  tokens={n_tokens}  {speed:.1f} tok/s\033[0m\n"
        )

        timing_history.append({
            "query": query,
            "tag": tag,
            "retrieval": t_retrieval,
            "ttft": ttft,
            "decode_time": decode_time,
            "total": total,
            "n_tokens": n_tokens,
            "prompt_tokens": prompt_tokens,
            "speed": speed,
        })

    return timing_history


def print_summary(history: list[dict], setup_times: dict):
    """Print final timing summary."""
    print("=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    # Setup
    print(f"\n  Setup")
    print(f"  {'─' * 40}")
    for label, t in setup_times.items():
        print(f"  {label:25s} {t:>8.2f}s")

    if not history:
        return

    # Per-query table
    retrieval_only = "ttft" not in history[0]
    print(f"\n  Queries ({len(history)} total)")
    if retrieval_only:
        print(f"  {'─' * 25}")
        print(f"  {'#':5s} {'Retrieval':>10s}")
        print(f"  {'─' * 25}")
        for h in history:
            print(f"  {h['tag']:5s} {h['retrieval']:>9.2f}s")
        if len(history) > 1:
            avg_ret = sum(h["retrieval"] for h in history) / len(history)
            print(f"  {'─' * 25}")
            print(f"  {'avg':5s} {avg_ret:>9.2f}s")
        print()
        return

    print(f"  {'─' * 65}")
    print(f"  {'#':5s} {'Retrieval':>10s} {'TTFT':>8s} {'Decode':>8s} {'Total':>8s} {'Tokens':>7s} {'tok/s':>7s}")
    print(f"  {'─' * 65}")
    for h in history:
        print(
            f"  {h['tag']:5s} {h['retrieval']:>9.2f}s {h['ttft']:>7.2f}s "
            f"{h['decode_time']:>7.2f}s {h['total']:>7.2f}s "
            f"{h['n_tokens']:>7d} {h['speed']:>6.1f}"
        )

    # Averages
    if history:
        h_list = history
        avg_ret = sum(h["retrieval"] for h in h_list) / len(h_list)
        avg_ttft = sum(h["ttft"] for h in h_list) / len(h_list)
        avg_decode = sum(h["decode_time"] for h in h_list) / len(h_list)
        avg_total = sum(h["total"] for h in h_list) / len(h_list)
        avg_tok = sum(h["n_tokens"] for h in h_list) / len(h_list)
        avg_speed = sum(h["speed"] for h in h_list) / len(h_list)
        print(f"  {'─' * 65}")
        print(
            f"  {'avg':5s} {avg_ret:>9.2f}s {avg_ttft:>7.2f}s "
            f"{avg_decode:>7.2f}s {avg_total:>7.2f}s "
            f"{avg_tok:>7.0f} {avg_speed:>6.1f}"
        )
    print()


def resolve_model(name: str) -> str:
    """Resolve a model alias to its path, or return the name as-is if it's a path."""
    return MODEL_ALIASES.get(name, name)


def main():
    aliases_list = ", ".join(MODEL_ALIASES)
    parser = argparse.ArgumentParser(
        description="Interactive RAG chat with timing",
        epilog=f"Model aliases: {aliases_list}",
    )
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=STRATEGIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tokenizer", default="spacy")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="Model alias (e.g. gemma-3n) or path to .gguf file",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument(
        "--procedure",
        default=None,
        help="Procedure context, e.g. 'cirugía de hemorroides'. "
        "Helps the model resolve ambiguous questions.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip model loading and generation; only show retrieved chunks.",
    )
    args = parser.parse_args()

    model_path = resolve_model(args.model)

    # --- Build system prompt ---
    if args.procedure:
        procedure_clause = f" que responde preguntas de pacientes sobre {args.procedure}"
    else:
        procedure_clause = " que responde preguntas de pacientes"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(procedure_clause=procedure_clause)

    setup_times = {}

    print(f"\033[2mLoading qdrant...\033[0m", end="", flush=True)
    t0 = time.perf_counter()
    client = _get_client(QDRANT_PATH)
    setup_times["Qdrant client"] = time.perf_counter() - t0
    print(f"\033[2m {setup_times['Qdrant client']:.2f}s\033[0m")

    print(f"\033[2mLoading retriever ({args.strategy}, {args.tokenizer})...\033[0m", end="", flush=True)
    t0 = time.perf_counter()
    retriever = build_retriever(
        client, COLLECTION_NAME, strategy=args.strategy, tokenizer=args.tokenizer,
        rerank_model=RERANK_MODEL, rerank_candidates=RERANK_CANDIDATES,
    )
    if hasattr(retriever, "preload"):
        retriever.preload()
    setup_times["Retriever"] = time.perf_counter() - t0
    print(f"\033[2m {setup_times['Retriever']:.2f}s\033[0m")

    model = None
    if not args.retrieval_only:
        print(f"\033[2mLoading model ({args.model} → {model_path})...\033[0m", end="", flush=True)
        t0 = time.perf_counter()
        # Suppress llama.cpp C++ stderr warnings (e.g. n_ctx < n_ctx_train)
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            model = load_model(model_path, n_ctx=args.n_ctx)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        setup_times["LLM"] = time.perf_counter() - t0
        print(f"\033[2m {setup_times['LLM']:.2f}s\033[0m")

        # Warmup: exercise prefill with a prompt similar in length to a real
        # RAG query so mmap pages are loaded and caches are hot.
        print(f"\033[2mWarming up model...\033[0m", end="", flush=True)
        t0 = time.perf_counter()
        warmup_prompt = "FRAGMENTOS:\n" + "Texto médico de ejemplo. " * 60 + "\n\nPREGUNTA: pregunta"
        model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": warmup_prompt},
            ],
            max_tokens=1, temperature=0,
        )
        setup_times["Warmup"] = time.perf_counter() - t0
        print(f"\033[2m {setup_times['Warmup']:.2f}s\033[0m")

    print(f"\033[2mtop_k={args.top_k}  max_tokens={args.max_tokens}  n_ctx={args.n_ctx}\033[0m")

    # --- Interactive loop ---
    try:
        history = run_interactive(
            retriever, model, args.top_k, args.max_tokens, system_prompt,
            retrieval_only=args.retrieval_only, procedure=args.procedure,
        )
    except Exception:
        import traceback
        traceback.print_exc()
        history = []

    # --- Summary ---
    print_summary(history, setup_times)

    client.close()


if __name__ == "__main__":
    main()
