"""
Interactive RAG chat: load once, ask many questions with streaming output.

Shows per-stage timing (retrieval, TTFT, generation) and a final summary.

Usage:
    python inspect_rag.py
    python inspect_rag.py --strategy hybrid --top-k 5
    python inspect_rag.py --model models/google_gemma-3n-E2B-it-Q4_K_M.gguf
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
DEFAULT_MODEL = "./models/granite-4.0-1b-Q4_K_M.gguf"
DEFAULT_STRATEGY = "hybrid"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 256

# --- Reranker options (uncomment one) ---
# RERANK_MODEL = "BAAI/bge-reranker-v2-m3"                    # 568M, multilingual, best quality
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # 66M, multilingual (mMARCO, includes Spanish), fast
# RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L6-H384-v1"   # 22M, multilingual, fastest
RERANK_CANDIDATES = 8  # chunks scored by reranker; best top_k are kept
MAX_HISTORY_TURNS = 3

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


def build_messages(
    system_prompt: str, chat_history: list[tuple[str, str]], current_prompt: str
) -> list[dict]:
    """Build messages list with system prompt, conversation history, and current prompt.

    Previous turns include only the bare question (no chunks) so the model has
    conversational context without wasting tokens re-reading old fragments.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for q, a in chat_history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": current_prompt})
    return messages


def stream_and_print(model, messages: list[dict], max_tokens: int, t_question: float):
    """Stream generation, printing tokens as they arrive.

    t_question is the perf_counter timestamp when the user submitted the question.
    Returns (ttft, decode_time, n_tokens, text) where:
      - ttft: time from question submission to first visible token
      - decode_time: time from first token to last token (pure decode speed)
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
    """Interactive chat loop with conversation history. Returns timing dicts."""
    timing_history = []
    chat_history: list[tuple[str, str]] = []  # (bare_query, response) pairs
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
            src = doc.metadata.get("source", "?")
            compact = doc.metadata.get("page_content_compact", "")
            print(f"\033[1;2m  [{i}] score={score:.4f}  src={src}\033[0m")
            if compact and compact != doc.page_content:
                print(f"\033[36m  [compact] {compact}\033[0m")
            print(f"\033[36m  [full] {doc.page_content}\033[0m")
            print()

        if retrieval_only:
            timing_history.append({
                "query": query,
                "tag": f"q{len(timing_history) + 1}",
                "retrieval": t_retrieval,
            })
            continue

        # --- Build prompt and messages ---
        prompt = build_prompt(query, results)
        messages = build_messages(system_prompt, chat_history, prompt)
        prompt_tokens = len(model.tokenize(prompt.encode("utf-8"), add_bos=False))

        # Show prompt
        hist_note = f", history={len(chat_history)} turns" if chat_history else ""
        print(f"\033[2m── prompt ({prompt_tokens} tokens{hist_note}) ──\033[0m")
        print(f"\033[33m{prompt}\033[0m")

        # --- Generate (streaming) ---
        tag = f"q{len(timing_history) + 1}"
        print(f"\n\033[2m── generating ({tag}) ──\033[0m\n")
        ttft, decode_time, n_tokens, text = stream_and_print(model, messages, max_tokens, t_question)

        # Save conversation for context in follow-up questions
        chat_history.append((query, text))

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


def main():
    parser = argparse.ArgumentParser(description="Interactive RAG chat with timing")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY, choices=STRATEGIES)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--tokenizer", default="spacy")
    parser.add_argument("--model", default=DEFAULT_MODEL)
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
        print(f"\033[2mLoading model ({args.model})...\033[0m", end="", flush=True)
        t0 = time.perf_counter()
        # Suppress llama.cpp C++ stderr warnings (e.g. n_ctx < n_ctx_train)
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        try:
            model = load_model(args.model, n_ctx=args.n_ctx)
        finally:
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stderr)
        setup_times["LLM"] = time.perf_counter() - t0
        print(f"\033[2m {setup_times['LLM']:.2f}s\033[0m")

    print(f"\033[2mtop_k={args.top_k}  max_tokens={args.max_tokens}  n_ctx={args.n_ctx}\033[0m")
    if not args.retrieval_only:
        print(f"\033[2m── system prompt ──\033[0m")
        print(f"\033[35m{system_prompt}\033[0m\n")

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
