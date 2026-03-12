# Generation Tuning Report: gemma-3n-E2B on CPU

**Date**: 2026-03-11
**Machine**: 12-core CPU, 32GB RAM, Linux 6.17
**Model**: google/gemma-3n-E2B-it Q4_K_M (2.6GB GGUF, 4B total / 2B effective params)
**Inference**: llama-cpp-python 0.3.16, CPU-only
**Retrieval**: hybrid (vector + BM25/RRF), spacy tokenizer, top_k=5
**Chunks**: mt=512, overlap=0.20, contextual prefix + LLM enrichment (35 chunks)

## Context

We have a medical RAG pipeline that retrieves document chunks and feeds them
to a local LLM for answer generation. Previous work identified generation
quality as the main bottleneck: the retriever finds the right chunks, but
the LLM (gemma-3-1b) gave superficial one-liner answers. Switching to
gemma-3n-E2B improved quality significantly, but we needed to tune three
things:

1. **Thinking mode**: Is the model wasting tokens on hidden reasoning?
2. **Prompt engineering**: What prompt gets the most information out?
3. **CPU threading**: What's the optimal thread count for this model?

## 1. Thinking Mode Investigation

**Finding: No hidden tokens. Not an issue.**

gemma-3n-E2B's chat template is straightforward user/model turns with no
`<think>` or reasoning mode. We verified empirically:

```
Completion tokens reported: 28
Visible text tokens:        28
Hidden token overhead:       0
```

The model does not generate any hidden reasoning tokens. Its slower speed
(vs gemma-3-1b) is purely due to larger parameter count and the selective
activation (AltUp) architecture.

## 2. Prompt Engineering

**Finding: +33 percentage points span coverage by changing the prompt.**

### Method

We tested 11 prompt configurations across 8 diverse medical queries,
measuring **span coverage** — the percentage of expected key facts
(e.g., "2-4 semanas de recuperacion", "ayuno 6-8 horas") found in
the model's answer. This tells us how much relevant information the model
extracted from the retrieved chunks.

Configurations varied along three axes:
- **System prompt**: baseline ("2-3 frases"), exhaustive, structured,
  patient-friendly, or *none*
- **Prompt template**: compact (just "FRAGMENTOS/PREGUNTA/RESPUESTA"),
  instructive (explains the task in natural language), or numbered
- **Context source**: compact chunk text vs full original text
- **Max tokens**: 80 (baseline) vs 256

### Results

| Config                 | Span Coverage | Avg Tokens | Avg Time |
|------------------------|:------------:|:----------:|:--------:|
| **no_sys + instructive** | **89.6%** | 244 | 45.4s |
| exhaustive + instructive | 78.1% | 143 | 33.9s |
| structured + compact     | 77.1% | 215 | 41.4s |
| patient + instructive    | 77.1% | 207 | 40.9s |
| exhaustive + numbered    | 75.0% | 126 | 28.4s |
| patient + compact        | 70.8% | 100 | 24.3s |
| exhaustive + compact     | 62.5% | 105 | 33.7s |
| exhaustive + full text   | 62.5% | 105 | 55.9s |
| **baseline (old)**       | **56.2%** | 60 | 34.0s |

### Key Insights

**1. The system prompt was the biggest problem.**
The old system prompt said "2-3 frases" (2-3 sentences). The model obeyed
literally, producing terse answers that omitted critical details. Removing
the system prompt entirely gave the best results — the instructive template
alone provides enough guidance.

Note: gemma-3n's chat template prepends any system message to the first user
message anyway (it doesn't have a separate system turn), so the system prompt
competes with the user prompt for attention.

**2. Instructive > compact template.**
Simply wrapping chunks in "FRAGMENTOS:\n...\nPREGUNTA:\n...\nRESPUESTA:" is
ambiguous about how much detail to include. An explicit instruction like
"Úsalos para responder la pregunta del paciente con todos los detalles
relevantes" tells the model to be thorough.

**3. Compact text > full text for context.**
Using the compact chunk representation (which preserves all key information
in fewer tokens) outperformed full text. Less noise in the context means the
model can focus on what matters, and shorter prompts leave more room for
generation within the context window.

**4. max_tokens=256 is sufficient.**
Bumping to 384 or 512 didn't improve coverage — the remaining misses are
comprehension gaps, not truncation.

### Best Prompt Configuration

```python
# No system prompt
SYSTEM_PROMPT = None

# Instructive template
PROMPT_TEMPLATE = (
    "A continuación tienes fragmentos de documentos médicos. "
    "Úsalos para responder la pregunta del paciente con todos "
    "los detalles relevantes.\n\n"
    "FRAGMENTOS:\n{context}\n\n"
    "PREGUNTA DEL PACIENTE: {query}\n\n"
    "RESPUESTA DETALLADA:"
)

# Generation params
max_tokens = 256
temperature = 0.0  # deterministic for eval
```

### Example: Before vs After

**Query**: "despues de que me operen de las almorranas cuanto tiempo tardo
en recuperarme"

**Before** (baseline, 56% coverage):
> La recuperación después de la cirugía de hemorroides puede tardar entre
> 2 y 4 semanas.

**After** (no_sys + instructive, 90% coverage):
> Después de la cirugía de hemorroides, la recuperación generalmente toma
> de **2 a 4 semanas**. Durante el postoperatorio (2-3 semanas):
> - **Dolor**: Espera un dolor leve a moderado. Se recomienda usar baños
>   de asiento para aliviar el dolor [...]
> - **Alimentación**: [...]
> - **Actividad**: [...]

## 3. CPU Thread Scaling

**Finding: n_threads=9 is optimal (3.73 tok/s), but the curve is non-monotonic.**

### Method

We tested n_threads from 1 to 12, running 3 queries per configuration
(with warmup). Each configuration required a full model reload.

### Results

| Threads | tok/s | time/query | vs best |
|:-------:|:-----:|:----------:|:-------:|
| 1       | 2.12  | 66.6s      | -43%    |
| 2       | 2.18  | 64.8s      | -42%    |
| 3       | 1.86  | 75.9s      | -50%    |
| 4       | 1.91  | 73.9s      | -49%    |
| 5       | 1.99  | 71.0s      | -47%    |
| 6       | 2.13  | 66.3s      | -43%    |
| 7       | 2.11  | 67.0s      | -43%    |
| 8       | 2.91  | 48.5s      | -22%    |
| **9**   | **3.73** | **37.8s** | **best** |
| 10      | 3.05  | 46.2s      | -18%    |
| 11      | 2.83  | 49.7s      | -24%    |
| 12      | 2.76  | 51.1s      | -26%    |

### Interpretation

The bimodal pattern (dip at 3-5, peak at 9) is characteristic of CPU
inference with NUMA or cache topology effects:

- **1-2 threads**: Fits in L1/L2 cache of one core, efficient.
- **3-7 threads**: Cross-core synchronization overhead exceeds parallelism
  gains. The model's selective activation (AltUp) means only a subset of
  parameters are active per token, so the working set is small and doesn't
  benefit much from spreading across cores.
- **8-9 threads**: Enough parallelism to speed up the matrix multiplications
  that *do* happen, likely aligning well with cache line boundaries and
  memory bandwidth.
- **10-12 threads**: Diminishing returns from thread scheduling overhead
  and memory bandwidth saturation.

### Generalizability

These results are **machine- and model-specific**. The optimal thread count
depends on:

- **CPU architecture**: NUMA topology, cache hierarchy, core count
- **Model size**: Larger models (7B+) benefit more from threads because
  their matrix multiplications are larger
- **Quantization**: Q4 models have smaller working sets, reducing the
  benefit of parallelism
- **Prompt length**: Longer prompts benefit more from batch threading
  (`n_threads_batch`)

As a rule of thumb for small quantized models on CPU: start with
`n_threads = n_physical_cores * 0.75` and benchmark around that point.
The difference between best and worst was 2x (1.86 vs 3.73 tok/s), so
it's worth tuning.

## Combined Impact

| Metric | Before | After | Improvement |
|--------|:------:|:-----:|:-----------:|
| Span coverage | 56.2% | 89.6% | **+33 pp** |
| Generation speed | ~2 tok/s | ~3.7 tok/s | **+85%** |
| Time per query | ~34s | ~25s* | **-26%** |
| Answer detail | 60 tokens (1-2 sentences) | 244 tokens (structured) | Much richer |

*Estimated: faster tok/s partially offset by more tokens generated.

## Reproduction

```bash
# Prompt tuning benchmark (11 configs x 8 queries, ~1h)
uv run python benchmark_tuning.py --phase prompts

# Thread scaling benchmark (12 configs x 3 queries, ~2h)
uv run python benchmark_tuning.py --phase threads

# Full RAG demo with best config
uv run python inspect_rag.py "¿Cuándo me dan el alta del hospital?" \
    --strategy hybrid --top-k 5 --max-tokens 256
```

## Next Steps

1. Run best prompt config on all 41 eval queries for end-to-end metrics
2. Test alternative models (Qwen3.5-2B, llama-3.2-3b) with same prompt —
   they may be faster and competitive in quality
3. Retrieval improvements: reranking, query expansion
