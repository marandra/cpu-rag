"""
LLM wrapper for local inference via llama-cpp-python (CPU-only).
"""

from __future__ import annotations

import os

from llama_cpp import Llama

_DEFAULT_THREADS = min(os.cpu_count() or 4, 12)


def load_model(path: str, n_ctx: int = 2048, n_threads: int = _DEFAULT_THREADS) -> Llama:
    """Load a GGUF model from disk with CPU speed optimizations."""
    return Llama(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=os.cpu_count() or n_threads,
        n_batch=512,
        flash_attn=True,
        verbose=False,
    )


def generate_stream(
    model: Llama,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
):
    """Stream a response token by token. Yields text chunks."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    for chunk in model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ):
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            yield text


def generate(
    model: Llama,
    prompt: str,
    system_prompt: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> tuple[str, dict]:
    """Generate a response. Returns (text, stats) where stats has timing info."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    return text, usage
