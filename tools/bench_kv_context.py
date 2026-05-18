"""
Bench: measure prefill cost and warm gen speed as a function of context size.

For each target token count N, slice the first N tokens of the diabetes doc,
use it as the "FRAGMENTOS:" payload, then:
  1. Warmup call (max_tokens=1) — pays cold prefill of N + ~system tokens
  2. Real call with a different question and max_tokens=80 — KV cache prefix
     matches everything up to "PREGUNTA: " (only the new question tokens are
     re-evaluated). TTFT measured from streaming; gen speed from inter-token
     time across the rest of the stream.

Single Llama instance shared across sizes (n_ctx=32K). Between sizes, we just
change the FRAGMENTOS payload — KV is invalidated only where it diverges.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prompt import get_system_prompt
from src.llm import load_model

MODEL_PATH = "./models/Ministral-3-3B-Q4_K_M.gguf"
DOC_PATH = Path("./corpus/markdown/diabetes/GUIA_DIABETES_v2.md")
SIZES = [1500, 3000, 6000]
WARMUP_QUESTION = "hola"
TEST_QUESTION = "¿Qué es la diabetes tipo 2?"
GEN_MAX_TOKENS = 80


def build_user(frag_text: str, question: str) -> str:
    return f"INFORMACIÓN:\n{frag_text}\n\nPREGUNTA: {question}"


def run_call(llm, system: str, user: str, max_tokens: int):
    """Stream a chat completion, return (ttft_s, total_s, n_tokens, text)."""
    t0 = time.perf_counter()
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.1,
        stream=True,
    )
    t_first = None
    n = 0
    parts = []
    for chunk in stream:
        text = chunk["choices"][0]["delta"].get("content", "")
        if not text:
            continue
        if t_first is None:
            t_first = time.perf_counter()
        n += 1
        parts.append(text)
    t_end = time.perf_counter()
    ttft = (t_first - t0) if t_first is not None else None
    total = t_end - t0
    return ttft, total, n, "".join(parts)


def main():
    print(f"Loading model {MODEL_PATH} (n_ctx=32768)...", flush=True)
    t0 = time.perf_counter()
    llm = load_model(path=MODEL_PATH, n_ctx=32768)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    system_prompt = get_system_prompt("diabetes")
    doc_text = DOC_PATH.read_text(encoding="utf-8")

    # Tokenize doc with the model's own tokenizer for exact slicing
    full_tokens = llm.tokenize(doc_text.encode("utf-8"), add_bos=False, special=False)
    print(f"Full doc tokens: {len(full_tokens)}", flush=True)

    rows = []
    for size in SIZES:
        if size > len(full_tokens):
            print(f"Skip size={size} > full doc {len(full_tokens)}")
            continue
        slice_toks = full_tokens[:size]
        slice_text = llm.detokenize(slice_toks).decode("utf-8", errors="replace")
        print(f"\n=== size={size} tokens (slice chars={len(slice_text)}) ===",
              flush=True)

        # Cold prefill: warmup with question="hola", max_tokens=1
        u_warm = build_user(slice_text, WARMUP_QUESTION)
        t_w0 = time.perf_counter()
        _ttft_w, total_w, n_w, _ = run_call(llm, system_prompt, u_warm, max_tokens=1)
        cold_prefill = time.perf_counter() - t_w0
        print(f"  cold prefill (max_tokens=1): {cold_prefill:.1f}s", flush=True)

        # Warm call: different question, same FRAGMENTOS. KV prefix matches
        # everything up to "PREGUNTA: ", only new question tokens re-evaluated.
        u_test = build_user(slice_text, TEST_QUESTION)
        ttft, total, n, text = run_call(llm, system_prompt, u_test,
                                        max_tokens=GEN_MAX_TOKENS)
        # Gen speed across all tokens except the first (which includes the
        # tail-prefill of the new question tokens).
        if n >= 2 and ttft is not None:
            gen_speed = (n - 1) / (total - ttft) if (total - ttft) > 0 else float("nan")
        else:
            gen_speed = float("nan")
        print(f"  warm: ttft={ttft:.2f}s total={total:.1f}s n_tok={n} "
              f"gen_speed={gen_speed:.2f} tok/s", flush=True)
        print(f"  sample: {text[:120]!r}", flush=True)

        rows.append({
            "size_tok": size,
            "cold_prefill_s": round(cold_prefill, 2),
            "ttft_s": round(ttft, 2) if ttft else None,
            "warm_total_s": round(total, 2),
            "n_tokens": n,
            "gen_speed_tok_s": round(gen_speed, 2),
        })

    print("\n=== SUMMARY ===")
    print(f"{'size':>6} {'cold_pref':>10} {'ttft':>7} {'total':>7} "
          f"{'tokens':>7} {'tok/s':>7}")
    for r in rows:
        print(f"{r['size_tok']:>6} {r['cold_prefill_s']:>10.1f} "
              f"{(r['ttft_s'] or 0):>7.2f} {r['warm_total_s']:>7.1f} "
              f"{r['n_tokens']:>7} {r['gen_speed_tok_s']:>7.2f}")


if __name__ == "__main__":
    main()
