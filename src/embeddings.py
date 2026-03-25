"""
Embedding and vector store for RAG pipeline.

Embedding: Nomic-Embed-v1.5 (local, CPU)
Vector DB: Qdrant (local storage)

Both are isolated behind simple functions to ease future swaps.
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document


# --- Stderr suppression (CUDA warnings on CPU) ---


@contextmanager
def _suppress_stderr():
    """Suppress C-level stderr (hides CUDA library warnings on CPU)."""
    stderr_fd = sys.stderr.fileno()
    saved = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved, stderr_fd)
        os.close(saved)


with _suppress_stderr():
    from langchain_nomic import NomicEmbeddings


# --- Embedding model ---

_model: NomicEmbeddings | None = None


def _get_model(dimensionality: int = 768) -> NomicEmbeddings:
    """Get or initialize the embedding model (singleton)."""
    global _model
    if _model is None:
        with _suppress_stderr():
            _model = NomicEmbeddings(
                model="nomic-embed-text-v1.5",
                dimensionality=dimensionality,
                inference_mode="local",
                device="cpu",
            )
    return _model


def embed_documents(texts: list[str], dimensionality: int = 768) -> list[list[float]]:
    """Embed a list of texts. Returns list of vectors."""
    if not texts:
        return []
    model = _get_model(dimensionality)
    with _suppress_stderr():
        return model.embed_documents(texts)


def embed_query(text: str, dimensionality: int = 768) -> list[float]:
    """Embed a single query text. Returns one vector."""
    model = _get_model(dimensionality)
    with _suppress_stderr():
        return model.embed_query(text)


# --- Vector store (Qdrant) ---


@dataclass
class VectorStoreConfig:
    """Qdrant configuration."""

    path: str = "./qdrant_data"
    collection_name: str = "medical_docs"
    vector_size: int = 768
    distance: str = "Cosine"


def _get_client(path: str):
    """Create a Qdrant client for local storage."""
    from qdrant_client import QdrantClient

    Path(path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=path)


def ensure_collection(client, config: VectorStoreConfig) -> None:
    """Create collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams

    collections = [c.name for c in client.get_collections().collections]
    if config.collection_name in collections:
        return

    distance_map = {"Cosine": Distance.COSINE, "Euclidean": Distance.EUCLID, "Dot": Distance.DOT}
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(
            size=config.vector_size,
            distance=distance_map[config.distance],
        ),
    )


def ingest(
    client,
    collection_name: str,
    documents: list[Document],
    vectors: list[list[float]],
    batch_size: int = 100,
) -> int:
    """Upsert documents + vectors into Qdrant. Returns count ingested."""
    from qdrant_client.models import PointStruct

    if len(documents) != len(vectors):
        raise ValueError(f"documents ({len(documents)}) and vectors ({len(vectors)}) length mismatch")

    ingested = 0
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i : i + batch_size]
        batch_vecs = vectors[i : i + batch_size]

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "page_content": doc.page_content,
                    "page_content_compact": doc.metadata.get("page_content_compact", doc.page_content),
                    "metadata": doc.metadata,
                },
            )
            for doc, vec in zip(batch_docs, batch_vecs, strict=True)
            if vec
        ]

        if points:
            client.upsert(collection_name=collection_name, points=points)
            ingested += len(points)

    return ingested


def search(
    client,
    collection_name: str,
    query_vector: list[float],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Search Qdrant. Returns list of (Document, score)."""
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        (
            Document(
                page_content=hit.payload["page_content"],
                metadata={
                    **hit.payload["metadata"],
                    "page_content_compact": hit.payload.get("page_content_compact", ""),
                },
            ),
            hit.score,
        )
        for hit in response.points
    ]


def collection_count(client, collection_name: str) -> int:
    """Return number of points in a collection."""
    info = client.get_collection(collection_name)
    return info.points_count


# --- Load chunks ---


def load_chunks(chunks_dir: str | Path) -> list[Document]:
    """Load all *_chunks.json files from a directory."""
    path = Path(chunks_dir)
    documents = []
    for f in sorted(path.glob("*_chunks.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        docs = [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        documents.extend(docs)
        print(f"  {f.name}: {len(docs)} chunks")
    return documents
