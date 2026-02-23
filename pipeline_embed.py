"""
Embedding pipeline: chunk JSONs -> Qdrant vector store.

Usage:
    python pipeline_embed.py
"""

import time

from src.embeddings import (
    VectorStoreConfig,
    _get_client,
    collection_count,
    embed_documents,
    ensure_collection,
    ingest,
    load_chunks,
)

# --- Configuration ---
CHUNKS_DIR = "./chunks"
VECTOR_SIZE = 384
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"

# --- Load chunks ---
# compact & doc2query are already in metadata (from pipeline_chunks.py LLM enrichment)
print("Loading chunks...")
documents = load_chunks(CHUNKS_DIR)
if not documents:
    print("No chunks found.")
    raise SystemExit(1)

texts = [doc.page_content for doc in documents]

# --- Embed ---
print(f"\nEmbedding {len(texts)} chunks (dim={VECTOR_SIZE})...")
t0 = time.perf_counter()
vectors = embed_documents(texts, dimensionality=VECTOR_SIZE)
elapsed = time.perf_counter() - t0

docs_per_sec = len(vectors) / elapsed if elapsed > 0 else 0
ms_per_doc = (elapsed / len(vectors)) * 1000 if vectors else 0
print(f"  {len(vectors)} embeddings in {elapsed:.1f}s ({docs_per_sec:.1f} docs/s, {ms_per_doc:.0f} ms/doc)")

# --- Store in Qdrant ---
print(f"\nStoring in Qdrant ({QDRANT_PATH})...")
config = VectorStoreConfig(
    path=QDRANT_PATH,
    collection_name=COLLECTION_NAME,
    vector_size=VECTOR_SIZE,
)
client = _get_client(config.path)
# Drop collection to ensure fresh re-ingest (payload schema may have changed)
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    client.delete_collection(COLLECTION_NAME)
    print(f"  Dropped existing collection '{COLLECTION_NAME}'")
ensure_collection(client, config)

t0 = time.perf_counter()
count = ingest(client, COLLECTION_NAME, documents, vectors)
store_elapsed = time.perf_counter() - t0

total = collection_count(client, COLLECTION_NAME)
print(f"  {count} ingested in {store_elapsed:.2f}s (collection total: {total})")
