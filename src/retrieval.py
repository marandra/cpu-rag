"""
Modular retrieval strategies for RAG pipeline.

Strategy pattern: all retrievers share a common interface.
Compose them freely: vector, BM25, hybrid (RRF), reranked.

Usage:
    retriever = build_retriever(client, collection, strategy="hybrid+rerank")
    results = retriever.retrieve("¿Qué riesgos tiene la cirugía?", top_k=3)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from langchain_core.documents import Document

from src.embeddings import embed_query, search


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Retriever(ABC):
    """Abstract base for all retrieval strategies."""

    @abstractmethod
    def _retrieve(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        """Return top_k (Document, score) pairs for a query."""

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float | None = None
    ) -> list[tuple[Document, float]]:
        """Retrieve documents, optionally dropping results below *min_score*."""
        results = self._retrieve(query, top_k)
        if min_score is not None:
            results = [(doc, s) for doc, s in results if s >= min_score]
        return results


# ---------------------------------------------------------------------------
# Vector retriever (wraps existing Qdrant search)
# ---------------------------------------------------------------------------


class VectorRetriever(Retriever):
    """Dense vector search via Qdrant."""

    def __init__(
        self,
        client,
        collection_name: str,
        embed_fn: Callable[[str], list[float]] = embed_query,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embed_fn = embed_fn

    def _retrieve(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        vector = self.embed_fn(query)
        return search(self.client, self.collection_name, vector, top_k=top_k)


# ---------------------------------------------------------------------------
# BM25 retriever (keyword search)
# ---------------------------------------------------------------------------


class BM25Retriever(Retriever):
    """Sparse keyword search using BM25Okapi."""

    def __init__(
        self,
        corpus: list[Document],
        tokenizer: Callable[[str], list[str]] | None = None,
    ):
        from rank_bm25 import BM25Okapi

        self.corpus = corpus
        self.tokenizer = tokenizer or self._default_tokenizer
        tokenized = [self.tokenizer(doc.page_content) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _default_tokenizer(text: str) -> list[str]:
        return text.lower().split()

    @classmethod
    def from_qdrant(
        cls,
        client,
        collection_name: str,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> BM25Retriever:
        """Build corpus by scrolling all points from a Qdrant collection."""
        documents: list[Document] = []
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for pt in points:
                documents.append(
                    Document(
                        page_content=pt.payload["page_content"],
                        metadata={
                            **pt.payload.get("metadata", {}),
                            "page_content_compact": pt.payload.get(
                                "page_content_compact", ""
                            ),
                        },
                    )
                )
            if offset is None:
                break
        return cls(documents, tokenizer=tokenizer)

    def _retrieve(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        # Indices sorted by descending score
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :top_k
        ]
        return [(self.corpus[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# Hybrid retriever (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------


class HybridRetriever(Retriever):
    """Combine two retrievers using Reciprocal Rank Fusion (RRF)."""

    def __init__(
        self,
        vector: Retriever,
        keyword: Retriever,
        rrf_k: int = 60,
        candidate_factor: int = 3,
    ):
        self.vector = vector
        self.keyword = keyword
        self.rrf_k = rrf_k
        self.candidate_factor = candidate_factor

    def _rrf_merge(
        self,
        results_a: list[tuple[Document, float]],
        results_b: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        """Merge two ranked lists via RRF. Uses page_content as identity key."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, _score) in enumerate(results_a):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, (doc, _score) in enumerate(results_b):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            doc_map[key] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_map[key], score) for key, score in ranked]

    def _retrieve(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        n_candidates = top_k * self.candidate_factor
        vec_results = self.vector.retrieve(query, top_k=n_candidates)
        kw_results = self.keyword.retrieve(query, top_k=n_candidates)
        return self._rrf_merge(vec_results, kw_results, top_k)


# ---------------------------------------------------------------------------
# Reranked retriever (cross-encoder)
# ---------------------------------------------------------------------------


class RerankedRetriever(Retriever):
    """Wraps any retriever and reranks candidates with a cross-encoder."""

    _model_cache: dict[str, object] = {}

    def __init__(
        self,
        base: Retriever,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        candidates: int = 10,
    ):
        self.base = base
        self.model_name = model_name
        self.candidates = candidates

    def _get_model(self):
        """Lazy-load cross-encoder (singleton per model_name)."""
        if self.model_name not in self._model_cache:
            from sentence_transformers import CrossEncoder

            self._model_cache[self.model_name] = CrossEncoder(self.model_name)
        return self._model_cache[self.model_name]

    def _retrieve(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        candidates = self.base.retrieve(query, top_k=self.candidates)
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, doc.page_content) for doc, _score in candidates]
        rerank_scores = model.predict(pairs)

        scored = list(zip(candidates, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (doc, float(rerank_score))
            for (doc, _orig_score), rerank_score in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

STRATEGIES = ("vector", "bm25", "hybrid", "hybrid+rerank")


def build_retriever(
    client,
    collection_name: str,
    strategy: str = "vector",
    embed_fn: Callable[[str], list[float]] = embed_query,
    tokenizer: Callable[[str], list[str]] | str | None = None,
    rerank_model: str = "BAAI/bge-reranker-v2-m3",
    rerank_candidates: int = 10,
) -> Retriever:
    """
    Build a composed retriever from a strategy string.

    Strategies:
        "vector"         — dense vector search only
        "bm25"           — BM25 keyword search only
        "hybrid"         — vector + BM25 with RRF
        "hybrid+rerank"  — hybrid + cross-encoder reranking

    tokenizer can be a callable or a string name from src.tokenizers
    ("whitespace", "whitespace+accent", "spacy"). Default: "spacy".
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")

    vector = VectorRetriever(client, collection_name, embed_fn=embed_fn)

    if strategy == "vector":
        return vector

    # Resolve tokenizer: string name → callable
    if tokenizer is None:
        tokenizer = "spacy"
    if isinstance(tokenizer, str):
        from src.tokenizers import get_tokenizer
        tokenizer = get_tokenizer(tokenizer)

    bm25 = BM25Retriever.from_qdrant(client, collection_name, tokenizer=tokenizer)

    if strategy == "bm25":
        return bm25

    hybrid = HybridRetriever(vector=vector, keyword=bm25)

    if strategy == "hybrid":
        return hybrid

    # hybrid+rerank
    return RerankedRetriever(
        base=hybrid,
        model_name=rerank_model,
        candidates=rerank_candidates,
    )
