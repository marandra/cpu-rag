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


def _force_no_thinking() -> None:
    """Render every chat template with `enable_thinking=False`.

    Hybrid-thinking families gate reasoning on a template variable, and
    `Llama.create_chat_completion` (0.3.34) has no `**kwargs`, so the value
    cannot be passed per call — it has to be bound to the formatter.

    Leaving it undefined is not "the safe default", it is *whatever that
    template happens to do*, and the candidates disagree with each other.
    Measured 2026-07-22 by rendering our own message shape:

        Qwen3.5-4B        `enable_thinking is defined and is true`  -> off
        Qwen3.5-35B-A3B   `enable_thinking is defined and is false` -> ON
        gemma-4-26B       `enable_thinking | default(false)`        -> off
        granite-4.1, Ministral   no mention                         -> n/a

    Two models of the same family, opposite defaults. An unforced 35B would
    open `<think>` and spend the whole max_tokens budget reasoning, which
    reads as a broken model rather than a mis-called one.

    Templates that ignore the variable render byte-identically, so this is a
    no-op for what we serve today — the snapshot cache keys on the prompt, and
    a byte of drift would orphan every pickle the pools serve.
    """
    if getattr(_lcf.Jinja2ChatFormatter, "_cpu_rag_no_thinking", False):
        return
    original_call = _lcf.Jinja2ChatFormatter.__call__

    def _no_thinking_call(self, **kw):
        kw.setdefault("enable_thinking", False)
        return original_call(self, **kw)

    _lcf.Jinja2ChatFormatter.__call__ = _no_thinking_call
    _lcf.Jinja2ChatFormatter._cpu_rag_no_thinking = True


def load_model(
    path: str,
    n_ctx: int = 2048,
    n_threads: int = _DEFAULT_THREADS,
    thinking: bool = False,
) -> Llama:
    """Load a GGUF model from disk with CPU speed optimizations."""
    if not thinking:
        _force_no_thinking()
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
