"""
Combined benchmark: RAG-realistic vs full stuffed context.

For each model, measures TTFT and total generation time in both scenarios.
Model loading excluded from timing.
"""

import time
from pathlib import Path

from src.llm import load_model

# --- RAG scenario: ~350 tokens input ---
RAG_SYSTEM = (
    "Eres un asistente médico especializado. Responde la pregunta del usuario "
    "de forma concisa (2-3 frases) basándote únicamente en los fragmentos "
    "proporcionados. Si no encuentras la respuesta, dilo."
)

RAG_CHUNKS = """Fragmento 1: Las hemorroides son dilataciones de los paquetes vasculares del canal anal. Se clasifican en internas (por encima de la línea dentada) y externas. Los síntomas principales son sangrado, prolapso, prurito y dolor.

Fragmento 2: El tratamiento conservador incluye dieta rica en fibra, hidratación y pomadas tópicas. Para hemorroides grado II-III se recomiendan técnicas ambulatorias como ligadura con bandas elásticas.

Fragmento 3: La cirugía (hemorroidectomía) se reserva para hemorroides grado III-IV que no responden a tratamiento conservador. La técnica de Longo (hemorroidopexia) es una alternativa menos dolorosa con recuperación más rápida."""

# --- Stuffed scenario: all 4 docs ---
MARKDOWN_DIR = Path("markdown")

STUFFED_SYSTEM_PREFIX = (
    "Eres un asistente médico. Responde la pregunta del usuario de forma concisa "
    "(2-3 frases) basándote únicamente en la documentación proporcionada.\n\n"
    "DOCUMENTACIÓN:\n"
)

QUERY = "¿Cuándo se recomienda cirugía para las hemorroides?"

MODELS = [
    "models/Qwen3-0.6B-Q4_K_M.gguf",
    "models/qwen2.5-0.5b-instruct-Q4_K_M.gguf",
    "models/gemma-3-1b-it-Q4_K_M.gguf",
    "models/llama-3.2-1b-instruct-Q4_K_M.gguf",
    "models/qwen2.5-1.5b-instruct-Q4_K_M.gguf",
    "models/Qwen3.5-2B-Q4_K_M.gguf",
    "models/llama-3.2-3b-instruct-Q4_K_M.gguf",
]

MAX_TOKENS = 80
RUNS = 2  # averaged (after 1 warmup)


def load_stuffed_docs():
    text = ""
    for f in sorted(MARKDOWN_DIR.glob("*.md")):
        text += f"\n\n--- {f.stem} ---\n\n" + f.read_text()
    return text


def run_streaming(llm, messages, max_tokens):
    """Run streaming generation, return (ttft, total_time, n_tokens, text)."""
    t_start = time.perf_counter()
    ttft = None
    full_text = ""
    n_tokens = 0

    for chunk in llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.3, stream=True,
    ):
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            if ttft is None:
                ttft = time.perf_counter() - t_start
            full_text += text
            n_tokens += 1

    total = time.perf_counter() - t_start
    return ttft or 0, total, n_tokens, full_text


def benchmark_scenario(llm, messages, label):
    """Warmup + N runs, print results. Returns avg dict."""
    results = []
    last_text = ""
    for i in range(RUNS + 1):
        ttft, total, n_tok, text = run_streaming(llm, messages, MAX_TOKENS)
        speed = n_tok / total if total > 0 else 0
        tag = "warmup" if i == 0 else f"run {i}"
        print(f"    [{tag}] TTFT={ttft:.2f}s  total={total:.2f}s  "
              f"tok={n_tok}  {speed:.1f} tok/s")
        if i > 0:
            results.append({"ttft": ttft, "total": total, "tokens": n_tok})
            last_text = text

    avg_ttft = sum(r["ttft"] for r in results) / len(results)
    avg_total = sum(r["total"] for r in results) / len(results)
    avg_tok = sum(r["tokens"] for r in results) / len(results)
    avg_speed = avg_tok / avg_total if avg_total > 0 else 0

    print(f"    AVG: TTFT={avg_ttft:.2f}s  total={avg_total:.2f}s  "
          f"tok={avg_tok:.0f}  {avg_speed:.1f} tok/s")
    print(f"    Response: {last_text[:250]}")
    return {"ttft": avg_ttft, "total": avg_total, "tokens": avg_tok, "speed": avg_speed}


def main():
    docs_text = load_stuffed_docs()
    stuffed_system = STUFFED_SYSTEM_PREFIX + docs_text

    print(f"RAG context: ~{len(RAG_SYSTEM) + len(RAG_CHUNKS)} chars")
    print(f"Stuffed context: ~{len(stuffed_system)} chars")
    print(f"Max tokens: {MAX_TOKENS}  |  Runs: {RUNS} + warmup")
    print("=" * 85)

    summary = []

    for model_path in MODELS:
        name = Path(model_path).stem
        if not Path(model_path).exists():
            print(f"\nSKIP {name} (not found)")
            continue

        print(f"\n{'─' * 85}")
        print(f"Model: {name}")
        print(f"{'─' * 85}")

        # Pick n_ctx: 8192 for stuffed (needs ~7k), 2048 for RAG
        # Load once at 8192 to handle both scenarios
        llm = load_model(model_path, n_ctx=8192)

        # Count tokens for both scenarios
        rag_input = RAG_SYSTEM + "\n\n" + RAG_CHUNKS + "\n\n" + QUERY
        stuffed_input = stuffed_system + "\n\n" + QUERY
        rag_tok = len(llm.tokenize(rag_input.encode("utf-8"), add_bos=False))
        stuffed_tok = len(llm.tokenize(stuffed_input.encode("utf-8"), add_bos=False))
        print(f"  Input tokens — RAG: {rag_tok}  |  Stuffed: {stuffed_tok}")

        # RAG scenario
        rag_msgs = [
            {"role": "system", "content": RAG_SYSTEM + "\n\n" + RAG_CHUNKS},
            {"role": "user", "content": QUERY},
        ]
        print(f"\n  [RAG scenario]")
        rag_res = benchmark_scenario(llm, rag_msgs, "RAG")

        # Stuffed scenario
        if stuffed_tok + MAX_TOKENS > 8192:
            print(f"\n  [Stuffed scenario] SKIP — {stuffed_tok} + {MAX_TOKENS} > 8192")
            stuffed_res = None
        else:
            stuffed_msgs = [
                {"role": "system", "content": stuffed_system},
                {"role": "user", "content": QUERY},
            ]
            print(f"\n  [Stuffed scenario]")
            stuffed_res = benchmark_scenario(llm, stuffed_msgs, "Stuffed")

        summary.append((name, rag_tok, stuffed_tok, rag_res, stuffed_res))
        del llm

    # Final comparison table
    print("\n" + "=" * 85)
    print("SUMMARY")
    print("=" * 85)
    print(f"{'Model':<35s} {'RAG':>12s} {'Stuffed':>12s} {'Slowdown':>10s}")
    print(f"{'':35s} {'(total s)':>12s} {'(total s)':>12s} {'factor':>10s}")
    print("─" * 85)
    for name, rag_tok, stuffed_tok, rag_res, stuffed_res in summary:
        rag_t = f"{rag_res['total']:.1f}s"
        if stuffed_res:
            st_t = f"{stuffed_res['total']:.1f}s"
            factor = f"{stuffed_res['total'] / rag_res['total']:.1f}x"
        else:
            st_t = "SKIP"
            factor = "—"
        print(f"{name:<35s} {rag_t:>12s} {st_t:>12s} {factor:>10s}")

    print("\nDone.")


if __name__ == "__main__":
    main()
