"""
RAG pipeline: query → retrieve → generate answer.

Usage:
    python main_rag.py "¿Cuáles son los riesgos de la colecistectomía?"
    python main_rag.py          # interactive mode
"""

import sys
import time

from src.embeddings import _get_client
from src.llm import generate_stream, load_model
from src.retrieval import build_retriever

# --- Configuration ---
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "medical_docs"
MODEL_PATH = "./models/llama-3.2-1b-instruct-Q4_K_M.gguf"
TOP_K = 3
RETRIEVAL_STRATEGY = "vector"  # vector | bm25 | hybrid | hybrid+rerank

SYSTEM_PROMPT = "Responde SOLO con la información de los fragmentos. Si no aparece, di 'No disponible'. Español."

PROMPT_TEMPLATE = """FRAGMENTOS:
{context}

PREGUNTA: {query}
RESPUESTA:"""


def _chunk_text_for_prompt(doc) -> str:
    """Pick the best text variant for the LLM prompt (no synthetic questions)."""
    return (
        doc.metadata.get("page_content_compact")
        or doc.metadata.get("page_content_original")
        or doc.page_content
    )


def build_prompt(query: str, chunks: list[tuple]) -> str:
    """Format the prompt with retrieved chunks (compact version when available)."""
    context = "\n---\n".join(
        _chunk_text_for_prompt(doc) for doc, _score in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def ask(query: str, retriever, model) -> str:
    """Full RAG pass: retrieve → build prompt → stream answer."""
    t0 = time.perf_counter()
    results = retriever.retrieve(query, top_k=TOP_K)
    prompt = build_prompt(query, results)
    t1 = time.perf_counter()

    chunks = []
    print()
    for chunk in generate_stream(model, prompt, system_prompt=SYSTEM_PROMPT):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print()

    t2 = time.perf_counter()
    gen_time = t2 - t1
    completion_tokens = len(chunks)  # 1 chunk ≈ 1 token in llama-cpp streaming
    tok_per_sec = completion_tokens / gen_time if gen_time > 0 else 0
    print(
        f"[gen={completion_tokens} tok  {tok_per_sec:.1f} tok/s  {t2 - t0:.1f}s total]"
    )
    return "".join(chunks)


def main():
    print("Connecting to Qdrant...")
    client = _get_client(QDRANT_PATH)

    print(f"Building retriever (strategy={RETRIEVAL_STRATEGY})...")
    retriever = build_retriever(client, COLLECTION_NAME, strategy=RETRIEVAL_STRATEGY)

    print("Loading LLM...")
    model = load_model(MODEL_PATH)

    if len(sys.argv) > 1:
        query = sys.argv[1]
        print(f"\nQuery: {query}")
        ask(query, retriever, model)
    else:
        print("\nInteractive mode (Ctrl+C to exit)\n")
        while True:
            try:
                query = input(">>> ").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                break
            if not query:
                continue
            ask(query, retriever, model)
            print()


if __name__ == "__main__":
    main()
