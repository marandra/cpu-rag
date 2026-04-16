"""
LLM wrapper for local inference via llama-cpp-python (CPU-only).
"""

from __future__ import annotations

import os
import re

from llama_cpp import Llama
import llama_cpp.llama_chat_format as _lcf
from jinja2 import TemplateSyntaxError

_DEFAULT_THREADS = min(os.cpu_count() or 4, 9)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n"
    "{% endfor %}<|im_start|>assistant\n"
)


def load_model(path: str, n_ctx: int = 2048, n_threads: int = _DEFAULT_THREADS) -> Llama:
    """Load a GGUF model from disk with CPU speed optimizations."""
    kwargs = dict(
        model_path=path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_threads_batch=os.cpu_count() or n_threads,
        n_batch=512,
        flash_attn=True,
        verbose=False,
    )
    try:
        return Llama(**kwargs)
    except TemplateSyntaxError:
        # Models with unsupported Jinja tags (e.g. SmolLM3 {% generation %})
        # Monkey-patch to fall back to chatml for unparseable templates.
        original_init = _lcf.Jinja2ChatFormatter.__init__

        def _safe_init(self, *args, **kw):
            try:
                original_init(self, *args, **kw)
            except TemplateSyntaxError:
                kw["template"] = _CHATML_TEMPLATE
                original_init(self, *args, **kw)

        _lcf.Jinja2ChatFormatter.__init__ = _safe_init
        try:
            return Llama(**kwargs)
        finally:
            _lcf.Jinja2ChatFormatter.__init__ = original_init


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
    text = _THINK_RE.sub("", text).strip()
    usage = response.get("usage", {})
    return text, usage
