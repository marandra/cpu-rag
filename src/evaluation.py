"""
Evaluation module for RAG retrieval quality.

Two evaluation modes:
1. Retrieval metrics (no LLM): precision, recall, MRR, keyword coverage
2. LLM-as-judge (OpenAI API): faithfulness and relevance scoring
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document


# --- Eval dataset ---


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""

    query: str
    expected_keywords: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
    category: str = ""


def load_eval_dataset(path: str | Path) -> list[EvalQuery]:
    """Load evaluation dataset from JSON file.

    Extra fields beyond EvalQuery's known fields are silently ignored,
    so the same dataset can carry metadata (profile, difficulty, etc.).
    """
    known = {f.name for f in EvalQuery.__dataclass_fields__.values()}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [EvalQuery(**{k: v for k, v in item.items() if k in known}) for item in data]


# --- Retrieval metrics ---


def precision_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """Fraction of top-K results that are relevant."""
    top_k = retrieved_sources[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for s in top_k if any(rel in s for rel in relevant_sources))
    return hits / len(top_k)


def recall_at_k(retrieved_sources: list[str], relevant_sources: list[str], k: int) -> float:
    """Fraction of relevant documents found in top-K."""
    if not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    found = sum(1 for rel in relevant_sources if any(rel in s for s in top_k))
    return found / len(relevant_sources)


def mrr(retrieved_sources: list[str], relevant_sources: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result."""
    for i, source in enumerate(retrieved_sources):
        if any(rel in source for rel in relevant_sources):
            return 1.0 / (i + 1)
    return 0.0


def keyword_coverage(texts: list[str], expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found in retrieved texts."""
    if not expected_keywords:
        return 0.0
    combined = " ".join(texts).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


@dataclass
class RetrievalResult:
    """Metrics for a single query retrieval."""

    query: str
    category: str
    num_results: int
    top_score: float
    avg_score: float
    precision_at_3: float
    precision_at_5: float
    recall_at_5: float
    mrr: float
    keyword_coverage: float


def evaluate_retrieval(
    query: EvalQuery,
    results: list[tuple[Document, float]],
) -> RetrievalResult:
    """Compute retrieval metrics for a single query."""
    sources = [r[0].metadata.get("source", "") for r in results]
    scores = [r[1] for r in results]
    texts = [r[0].page_content for r in results]

    return RetrievalResult(
        query=query.query,
        category=query.category,
        num_results=len(results),
        top_score=scores[0] if scores else 0.0,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        precision_at_3=precision_at_k(sources, query.relevant_sources, 3),
        precision_at_5=precision_at_k(sources, query.relevant_sources, 5),
        recall_at_5=recall_at_k(sources, query.relevant_sources, 5),
        mrr=mrr(sources, query.relevant_sources),
        keyword_coverage=keyword_coverage(texts, query.expected_keywords),
    )


# --- LLM-as-judge (OpenAI) ---

FAITHFULNESS_PROMPT = """Eres un evaluador de sistemas de búsqueda médica.

Dada una consulta del paciente y los fragmentos recuperados, evalúa si los fragmentos
contienen información relevante y correcta para responder la consulta.

Consulta: {query}

Fragmentos recuperados:
{chunks}

Evalúa con una puntuación de 0.0 a 1.0:
- 1.0: Los fragmentos contienen toda la información necesaria para responder correctamente
- 0.7: Los fragmentos contienen información parcial pero útil
- 0.4: Los fragmentos tienen algo de información relevante pero insuficiente
- 0.0: Los fragmentos no contienen información relevante

Responde SOLO con un JSON válido (sin markdown):
{{"score": <float>, "reasoning": "<explicación breve en español>"}}"""

RELEVANCE_PROMPT = """Eres un evaluador de sistemas de búsqueda médica.

Dada una consulta del paciente y los fragmentos recuperados, evalúa si los fragmentos
abordan directamente la pregunta del paciente.

Consulta: {query}

Fragmentos recuperados:
{chunks}

Evalúa con una puntuación de 0.0 a 1.0:
- 1.0: Los fragmentos responden directamente a la pregunta
- 0.7: Los fragmentos son relevantes pero no responden completamente
- 0.4: Los fragmentos tienen relación tangencial con la pregunta
- 0.0: Los fragmentos no tienen relación con la pregunta

Responde SOLO con un JSON válido (sin markdown):
{{"score": <float>, "reasoning": "<explicación breve en español>"}}"""


@dataclass
class JudgeResult:
    """LLM judge score for a single query."""

    faithfulness: float = 0.0
    faithfulness_reasoning: str = ""
    relevance: float = 0.0
    relevance_reasoning: str = ""
    error: str | None = None


def llm_judge(
    query: str,
    chunks: list[str],
    model: str = "gpt-4o-mini",
) -> JudgeResult:
    """
    Use OpenAI to judge retrieval quality.
    Requires OPENAI_API_KEY env var.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return JudgeResult(error="openai package not installed")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return JudgeResult(error="OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    chunks_text = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))

    result = JudgeResult()

    for prompt_template, score_field, reasoning_field in [
        (FAITHFULNESS_PROMPT, "faithfulness", "faithfulness_reasoning"),
        (RELEVANCE_PROMPT, "relevance", "relevance_reasoning"),
    ]:
        try:
            prompt = prompt_template.format(query=query, chunks=chunks_text)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=256,
            )
            text = response.choices[0].message.content.strip()
            parsed = json.loads(text)
            setattr(result, score_field, float(parsed["score"]))
            setattr(result, reasoning_field, parsed.get("reasoning", ""))
        except Exception as e:
            result.error = f"{score_field}: {e}"

    return result


# --- Report generation ---


def generate_report(
    results: list[RetrievalResult],
    judge_results: list[JudgeResult] | None = None,
    output_path: str | Path = "eval_report.md",
) -> str:
    """Generate a markdown evaluation report."""
    lines = ["# Informe de evaluación de retrieval\n"]

    # Aggregate metrics
    n = len(results)
    if n == 0:
        lines.append("Sin resultados.\n")
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        return "\n".join(lines)

    avg = lambda vals: sum(vals) / len(vals) if vals else 0.0

    lines.append("## Métricas agregadas\n")
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Consultas evaluadas | {n} |")
    lines.append(f"| Precision@3 (media) | {avg([r.precision_at_3 for r in results]):.2f} |")
    lines.append(f"| Precision@5 (media) | {avg([r.precision_at_5 for r in results]):.2f} |")
    lines.append(f"| Recall@5 (media) | {avg([r.recall_at_5 for r in results]):.2f} |")
    lines.append(f"| MRR (media) | {avg([r.mrr for r in results]):.2f} |")
    lines.append(f"| Keyword coverage (media) | {avg([r.keyword_coverage for r in results]):.2f} |")
    lines.append(f"| Score top-1 (media) | {avg([r.top_score for r in results]):.3f} |")
    lines.append(f"| Score medio | {avg([r.avg_score for r in results]):.3f} |")

    # LLM judge aggregates
    if judge_results and any(j.error is None for j in judge_results):
        valid = [j for j in judge_results if j.error is None]
        lines.append(f"| Faithfulness LLM (media) | {avg([j.faithfulness for j in valid]):.2f} |")
        lines.append(f"| Relevance LLM (media) | {avg([j.relevance for j in valid]):.2f} |")

    # Per-category breakdown
    categories = sorted(set(r.category for r in results if r.category))
    if categories:
        lines.append("\n## Por categoría\n")
        lines.append("| Categoría | N | P@3 | P@5 | R@5 | MRR | Keywords |")
        lines.append("|-----------|---|-----|-----|-----|-----|----------|")
        for cat in categories:
            cat_results = [r for r in results if r.category == cat]
            cn = len(cat_results)
            lines.append(
                f"| {cat} | {cn} "
                f"| {avg([r.precision_at_3 for r in cat_results]):.2f} "
                f"| {avg([r.precision_at_5 for r in cat_results]):.2f} "
                f"| {avg([r.recall_at_5 for r in cat_results]):.2f} "
                f"| {avg([r.mrr for r in cat_results]):.2f} "
                f"| {avg([r.keyword_coverage for r in cat_results]):.2f} |"
            )

    # Per-query detail
    lines.append("\n## Detalle por consulta\n")
    for i, r in enumerate(results):
        lines.append(f"### {i+1}. {r.query}\n")
        lines.append(f"- **Categoría:** {r.category}")
        lines.append(f"- **Resultados:** {r.num_results} | Score top: {r.top_score:.3f} | Avg: {r.avg_score:.3f}")
        lines.append(f"- **P@3:** {r.precision_at_3:.2f} | **P@5:** {r.precision_at_5:.2f} | **R@5:** {r.recall_at_5:.2f} | **MRR:** {r.mrr:.2f}")
        lines.append(f"- **Keyword coverage:** {r.keyword_coverage:.2f}")

        if judge_results and i < len(judge_results):
            j = judge_results[i]
            if j.error is None:
                lines.append(f"- **Faithfulness:** {j.faithfulness:.2f} — {j.faithfulness_reasoning}")
                lines.append(f"- **Relevance:** {j.relevance:.2f} — {j.relevance_reasoning}")
            else:
                lines.append(f"- **LLM judge error:** {j.error}")

        lines.append("")

    report = "\n".join(lines)
    Path(output_path).write_text(report, encoding="utf-8")
    return report
