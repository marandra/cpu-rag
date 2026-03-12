"""Benchmark CPU generation speed with stuffed context at different n_ctx."""

import time
from pathlib import Path

from src.llm import load_model, generate

MARKDOWN_DIR = Path("markdown")
MODELS = [
    "models/llama-3.2-1b-instruct-Q4_K_M.gguf",
    "models/llama-3.2-3b-instruct-Q4_K_M.gguf",
    "models/qwen2.5-0.5b-instruct-Q4_K_M.gguf",
    "models/qwen2.5-1.5b-instruct-Q4_K_M.gguf",
    "models/gemma-3-1b-it-Q4_K_M.gguf",
    "models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
]
CTX_SIZES = [4096, 8192]
MAX_TOKENS = 256
QUERY = "¿Cuáles son los tratamientos disponibles para las hemorroides y cuándo se recomienda cirugía?"

# Load all docs once
docs_text = ""
for f in sorted(MARKDOWN_DIR.glob("*.md")):
    docs_text += f"\n\n--- {f.stem} ---\n\n" + f.read_text()

system_prompt = (
    "Eres un asistente médico. Responde la pregunta del usuario basándote "
    "únicamente en la documentación proporcionada.\n\n"
    "DOCUMENTACIÓN:\n" + docs_text
)

print(f"System prompt length: {len(system_prompt)} chars")
print(f"Query length: {len(QUERY)} chars")
print(f"Max generation tokens: {MAX_TOKENS}")
print("=" * 90)

for model_path in MODELS:
    model_name = Path(model_path).stem
    if not Path(model_path).exists():
        print(f"\n SKIP {model_name} (file not found)")
        continue

    for n_ctx in CTX_SIZES:
        print(f"\n{'─' * 90}")
        print(f"Model: {model_name}  |  n_ctx: {n_ctx}")
        print(f"{'─' * 90}")

        # Load model
        t0 = time.perf_counter()
        try:
            llm = load_model(model_path, n_ctx=n_ctx)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue
        load_time = time.perf_counter() - t0

        # Count input tokens with model's own tokenizer
        full_input = system_prompt + "\n" + QUERY
        input_tokens = len(llm.tokenize(full_input.encode("utf-8"), add_bos=False))
        print(f"  Load time:    {load_time:.1f}s")
        print(f"  Input tokens: {input_tokens} (n_ctx={n_ctx})")

        if input_tokens >= n_ctx - MAX_TOKENS:
            print(f"  SKIP: input ({input_tokens}) + max_tokens ({MAX_TOKENS}) > n_ctx ({n_ctx})")
            del llm
            continue

        # Generate
        t0 = time.perf_counter()
        try:
            text, usage = generate(
                llm, QUERY, system_prompt=system_prompt,
                max_tokens=MAX_TOKENS, temperature=0.3,
            )
        except Exception as e:
            print(f"  GENERATE ERROR: {e}")
            del llm
            continue
        gen_time = time.perf_counter() - t0

        prompt_tokens = usage.get("prompt_tokens", "?")
        completion_tokens = usage.get("completion_tokens", 0)
        tok_per_sec = completion_tokens / gen_time if gen_time > 0 else 0

        print(f"  Prompt tokens:     {prompt_tokens}")
        print(f"  Completion tokens: {completion_tokens}")
        print(f"  Generation time:   {gen_time:.1f}s")
        print(f"  Speed:             {tok_per_sec:.1f} tok/s")
        print(f"  Response preview:  {text[:200]}...")

        del llm

print("\n" + "=" * 90)
print("Done.")
