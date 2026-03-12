"""
Benchmark: RAG-realistic scenario on CPU.

Simulates: ~200 tok system prompt + 3 distilled chunks (~50 tok each) + short generation.
Measures TTFT (time-to-first-token) and total generation time.
Model loading is excluded from timing.
"""

import time
from pathlib import Path

from src.llm import load_model

# --- Simulated RAG context ---
# ~200 tokens system prompt + ~150 tokens of retrieved chunks = ~350 tokens input
SYSTEM_PROMPT = (
    "Eres un asistente médico especializado. Responde la pregunta del usuario "
    "de forma concisa (2-3 frases) basándote únicamente en los fragmentos "
    "proporcionados. Si no encuentras la respuesta, dilo."
)

CHUNKS = """Fragmento 1: Las hemorroides son dilataciones de los paquetes vasculares del canal anal. Se clasifican en internas (por encima de la línea dentada) y externas. Los síntomas principales son sangrado, prolapso, prurito y dolor.

Fragmento 2: El tratamiento conservador incluye dieta rica en fibra, hidratación y pomadas tópicas. Para hemorroides grado II-III se recomiendan técnicas ambulatorias como ligadura con bandas elásticas.

Fragmento 3: La cirugía (hemorroidectomía) se reserva para hemorroides grado III-IV que no responden a tratamiento conservador. La técnica de Longo (hemorroidopexia) es una alternativa menos dolorosa con recuperación más rápida."""

QUERY = "¿Cuándo se recomienda cirugía para las hemorroides?"

# --- Models to benchmark ---
MODELS = [
    "models/gemma-3-1b-it-Q4_K_M.gguf",
    "models/llama-3.2-1b-instruct-Q4_K_M.gguf",
    "models/qwen2.5-0.5b-instruct-Q4_K_M.gguf",
    "models/qwen2.5-1.5b-instruct-Q4_K_M.gguf",
    "models/llama-3.2-3b-instruct-Q4_K_M.gguf",
    "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
]

MAX_TOKENS = 80  # 2-3 sentences
N_CTX = 2048     # small context, plenty for RAG
RUNS = 2         # average over N runs (after warmup)


def benchmark_model(model_path: str):
    name = Path(model_path).stem
    if not Path(model_path).exists():
        print(f"  SKIP {name} (not found)")
        return None

    # Load model (excluded from timing)
    llm = load_model(model_path, n_ctx=N_CTX)

    # Count input tokens
    full_prompt = SYSTEM_PROMPT + "\n\n" + CHUNKS + "\n\n" + QUERY
    input_tokens = len(llm.tokenize(full_prompt.encode("utf-8"), add_bos=False))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + CHUNKS},
        {"role": "user", "content": QUERY},
    ]

    results = []
    for run_i in range(RUNS + 1):  # +1 for warmup
        t_start = time.perf_counter()
        ttft = None
        full_text = ""
        completion_tokens = 0

        for chunk in llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            stream=True,
        ):
            text = chunk["choices"][0]["delta"].get("content", "")
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t_start
                full_text += text
                completion_tokens += 1  # approximate (one chunk ≈ one token)

        t_total = time.perf_counter() - t_start

        if run_i == 0:
            label = "warmup"
        else:
            label = f"run {run_i}"
            results.append({
                "ttft": ttft or 0,
                "total": t_total,
                "tokens": completion_tokens,
            })

        gen_speed = completion_tokens / t_total if t_total > 0 else 0
        print(f"  [{label}] TTFT={ttft:.2f}s  total={t_total:.2f}s  "
              f"tokens={completion_tokens}  speed={gen_speed:.1f} tok/s")

    # Average over runs
    if results:
        avg_ttft = sum(r["ttft"] for r in results) / len(results)
        avg_total = sum(r["total"] for r in results) / len(results)
        avg_tokens = sum(r["tokens"] for r in results) / len(results)
        avg_speed = avg_tokens / avg_total if avg_total > 0 else 0

        print(f"  --- AVG: TTFT={avg_ttft:.2f}s  total={avg_total:.2f}s  "
              f"tokens={avg_tokens:.0f}  speed={avg_speed:.1f} tok/s")
        print(f"  Response: {full_text[:300]}")

    del llm
    return results


def main():
    print(f"System prompt: ~{len(SYSTEM_PROMPT)} chars")
    print(f"Chunks: ~{len(CHUNKS)} chars")
    print(f"Query: ~{len(QUERY)} chars")
    print(f"Max generation tokens: {MAX_TOKENS}")
    print(f"n_ctx: {N_CTX}")
    print(f"Runs per model: {RUNS} (+ 1 warmup)")
    print("=" * 80)

    for model_path in MODELS:
        name = Path(model_path).stem
        print(f"\n{'─' * 80}")
        print(f"Model: {name}")
        print(f"{'─' * 80}")
        benchmark_model(model_path)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
