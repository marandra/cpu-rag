"""
Demo RAG: run eval dataset queries interactively for demonstrations.

Shows retrieved chunks, prompt, and streaming response for each query.
Can filter by category or run all queries sequentially.

Usage:
    python demo_rag.py                          # interactive menu
    python demo_rag.py --category procedimiento # only procedimiento queries
    python demo_rag.py --category OOS           # only out-of-scope queries
    python demo_rag.py --list-categories        # show available categories
    python demo_rag.py --auto                   # run all queries automatically
"""

import argparse
import json
import os
import time

from src.embeddings import _get_client
from src.llm import load_model
from src.retrieval import build_retriever

# --- Config (same as inspect_rag.py) ---
QDRANT_PATH = "./qdrant_manual"
COLLECTION_NAME = "medical_docs"
DEFAULT_MODEL = "ministral"
DEFAULT_STRATEGY = "hybrid+rerank"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 256
DEFAULT_MIN_SCORE = None  # score filtering disabled by default
EVAL_DATASET = "./eval_dataset_realistic.json"

MODEL_ALIASES = {
    "granite-1b":   "./models/granite-4.0-1b-Q4_K_M.gguf",
    "gemma-3n":     "./models/google_gemma-3n-E2B-it-Q4_K_M.gguf",
    "ministral":    "./models/Ministral-3-3B-Q4_K_M.gguf",
}

RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
RERANK_CANDIDATES = 8

PROCEDURES = {
    "hemorroides": "cirugía de hemorroides",
    "fisura":      "cirugía de fisura anal",
}

SYSTEM_PROMPT_TEMPLATE = (
    "Eres un asistente médico{procedure_clause}.\n\n"
    "REGLAS ESTRICTAS:\n"
    "1. Si los fragmentos contienen información relacionada con la pregunta, "
    "DEBES responder usando SOLO esa información.\n"
    "2. NUNCA uses tu conocimiento general. PROHIBIDO inventar datos, medicamentos, "
    "precios o plazos que NO estén en los fragmentos.\n"
    "3. Si los fragmentos NO contienen información relevante para la pregunta, "
    'responde: "No tengo información sobre eso."\n'
    "4. Si la pregunta no tiene relación con cirugía y cuidados perioperatorios, "
    'responde: "No tengo información sobre eso."\n'
    "5. Responde en un párrafo breve y directo al paciente.\n"
    "6. No menciones los fragmentos ni tu razonamiento.\n\n"
    "Ejemplos de preguntas que DEBES rechazar con "
    '"No tengo información sobre eso.":\n'
    "- Preguntas sobre temas no médicos (geografía, tecnología, cocina...)\n"
    "- Preguntas médicas cuya respuesta NO aparece en los fragmentos\n"
    "- Preguntas sobre costes, seguros, bajas laborales, trámites o segunda opinión\n"
    "- Preguntas sobre quién eres o qué haces"
)

PROMPT_TEMPLATE = "FRAGMENTOS:\n{context}\n\nPREGUNTA: {query}"


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def filter_by_procedure(dataset: list[dict], procedure: str) -> list[dict]:
    """Filter dataset to only include questions applicable to this procedure.

    Questions with no 'procedure' field are generic (apply to all).
    Questions with a 'procedure' field only apply to that specific procedure.
    """
    return [
        item for item in dataset
        if item.get("procedure") is None or item.get("procedure") == procedure
    ]


def get_categories(dataset: list[dict]) -> dict[str, list[dict]]:
    """Group queries by category."""
    categories = {}
    for item in dataset:
        cat = item["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
    return categories


def _chunk_text_for_prompt(doc) -> str:
    return (
        doc.metadata.get("page_content_compact")
        or doc.metadata.get("page_content_original")
        or doc.page_content
    )


def build_prompt(query: str, chunks: list[tuple]) -> str:
    context = "\n---\n".join(
        _chunk_text_for_prompt(doc) for doc, _score in chunks
    )
    return PROMPT_TEMPLATE.format(context=context, query=query)


def print_query_header(item: dict, index: int, total: int):
    """Print query metadata before running."""
    scope = "IN-SCOPE" if item["answerable"] else "OOS"
    scope_color = "\033[32m" if item["answerable"] else "\033[31m"

    print()
    print("=" * 70)
    print(f"\033[1mQuery {index}/{total}\033[0m  [{scope_color}{scope}\033[0m]  "
          f"category=\033[33m{item['category']}\033[0m  "
          f"profile=\033[35m{item['profile']}\033[0m")
    print("=" * 70)
    print()
    print(f"\033[1m> {item['query']}\033[0m")
    print(f"\033[2m  Intent: {item['intent']}\033[0m")
    print()


def run_query(item: dict, retriever, model, system_prompt: str,
              procedure: str | None, min_score: float, max_tokens: int) -> dict:
    """Run a single query and return results."""
    query = item["query"]
    t_start = time.perf_counter()

    # Retrieval
    retrieval_query = f"{procedure}: {query}" if procedure else query
    t0 = time.perf_counter()
    results = retriever.retrieve(retrieval_query, top_k=DEFAULT_TOP_K, min_score=min_score)
    t_retrieval = time.perf_counter() - t0

    # Handle zero chunks
    if not results:
        t_total = time.perf_counter() - t_start
        print(f"\033[33m[AUTO-REFUSE: no chunks above min_score={min_score}]\033[0m\n")
        print("No tengo información sobre eso.")
        print(f"\n\033[32m  retrieval={t_retrieval:.2f}s  total={t_total:.2f}s  (auto-refused)\033[0m\n")
        return {"response": "No tengo información sobre eso.", "auto_refused": True}

    # Build prompt
    prompt = build_prompt(query, results)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Generate (streaming)
    t0 = time.perf_counter()
    t_first_token = None
    full_text = ""
    n_tokens = 0

    for chunk in model.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=0.3, stream=True,
    ):
        text = chunk["choices"][0]["delta"].get("content", "")
        if text:
            if t_first_token is None:
                t_first_token = time.perf_counter()
            print(text, end="", flush=True)
            full_text += text
            n_tokens += 1

    t_end = time.perf_counter()
    print()

    # Timing stats
    ttft = (t_first_token or t_end) - t0
    decode_time = t_end - (t_first_token or t_end)
    t_total = t_end - t_start
    speed = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0

    print(f"\n\033[32m  retrieval={t_retrieval:.2f}s  ttft={ttft:.2f}s  "
          f"decode={decode_time:.2f}s  total={t_total:.2f}s  "
          f"tokens={n_tokens}  {speed:.1f} tok/s\033[0m\n")

    return {"response": full_text, "auto_refused": False}


def interactive_menu(categories: dict[str, list[dict]]) -> list[dict] | None:
    """Show category menu and return selected queries."""
    print("\n\033[1mCategorías disponibles:\033[0m\n")

    # Separate in-scope and OOS
    in_scope = {k: v for k, v in categories.items() if v[0]["answerable"]}
    oos = {k: v for k, v in categories.items() if not v[0]["answerable"]}

    # Build numbered menu
    menu = []  # list of (number, name, queries)
    n = 1

    print("  \033[32mIn-scope:\033[0m")
    for cat, items in sorted(in_scope.items()):
        print(f"    \033[1m{n:2d}\033[0m. {cat:20s} ({len(items)} queries)")
        menu.append((str(n), cat, items))
        n += 1

    print("\n  \033[31mOut-of-scope:\033[0m")
    for cat, items in sorted(oos.items()):
        print(f"    \033[1m{n:2d}\033[0m. {cat:20s} ({len(items)} queries)")
        menu.append((str(n), cat, items))
        n += 1

    # Special options
    all_queries = [item for items in categories.values() for item in items]
    in_scope_queries = [item for items in in_scope.values() for item in items]
    oos_queries = [item for items in oos.values() for item in items]

    print("\n  \033[33mEspecial:\033[0m")
    print(f"    \033[1m{n:2d}\033[0m. {'ALL':20s} (todas: {len(all_queries)} queries)")
    menu.append((str(n), "all", all_queries))
    n += 1
    print(f"    \033[1m{n:2d}\033[0m. {'IN-SCOPE':20s} (todas in-scope: {len(in_scope_queries)} queries)")
    menu.append((str(n), "in-scope", in_scope_queries))
    n += 1
    print(f"    \033[1m{n:2d}\033[0m. {'OOS':20s} (todas OOS: {len(oos_queries)} queries)")
    menu.append((str(n), "oos", oos_queries))

    print()
    choice = input("\033[1mSelecciona número o nombre (q=salir): \033[0m").strip().lower()

    if choice in ('q', 'quit', 'exit', ''):
        return None

    # Check by number
    for num, name, queries in menu:
        if choice == num or choice == name:
            return queries

    print(f"\033[31mOpción no válida: {choice}\033[0m")
    return interactive_menu(categories)


def main():
    parser = argparse.ArgumentParser(description="Demo RAG with eval dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--category", help="Filter by category (or ALL, IN-SCOPE, OOS)")
    parser.add_argument("--list-categories", action="store_true", help="List categories and exit")
    parser.add_argument("--auto", action="store_true", help="Run all queries without pausing")
    parser.add_argument("--procedure", default="hemorroides", choices=list(PROCEDURES))
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)
    args = parser.parse_args()

    # Load dataset and filter by procedure
    full_dataset = load_dataset(EVAL_DATASET)
    dataset = filter_by_procedure(full_dataset, args.procedure)
    categories = get_categories(dataset)

    excluded = len(full_dataset) - len(dataset)
    if excluded > 0:
        print(f"\033[2m(Filtered {excluded} queries not applicable to {args.procedure})\033[0m")

    if args.list_categories:
        print(f"\nCategories for procedure '{args.procedure}':\n")
        for cat, items in sorted(categories.items()):
            scope = "in-scope" if items[0]["answerable"] else "OOS"
            print(f"  {cat:20s} {len(items):3d} queries  ({scope})")
        return

    # Select queries
    if args.category:
        cat = args.category.lower()
        if cat == 'all':
            queries = dataset
        elif cat == 'in-scope':
            queries = [item for item in dataset if item["answerable"]]
        elif cat == 'oos':
            queries = [item for item in dataset if not item["answerable"]]
        elif cat in categories:
            queries = categories[cat]
        else:
            print(f"Unknown category: {args.category}")
            return
    else:
        queries = interactive_menu(categories)
        if queries is None:
            return

    print(f"\n\033[2mSelected {len(queries)} queries\033[0m\n")

    # Setup
    procedure = PROCEDURES[args.procedure]
    procedure_clause = f" que responde preguntas de pacientes sobre {procedure}"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(procedure_clause=procedure_clause)

    model_path = MODEL_ALIASES.get(args.model, args.model)

    print(f"\033[2mLoading qdrant...\033[0m", end="", flush=True)
    client = _get_client(QDRANT_PATH)
    print(f"\033[2m done\033[0m")

    print(f"\033[2mLoading retriever...\033[0m", end="", flush=True)
    retriever = build_retriever(
        client, COLLECTION_NAME, strategy=DEFAULT_STRATEGY,
        rerank_model=RERANK_MODEL, rerank_candidates=RERANK_CANDIDATES,
    )
    if hasattr(retriever, "preload"):
        retriever.preload()
    print(f"\033[2m done\033[0m")

    print(f"\033[2mLoading model ({args.model})...\033[0m", end="", flush=True)
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        model = load_model(model_path, n_ctx=2048)
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)
    print(f"\033[2m done\033[0m")

    # Run queries
    for i, item in enumerate(queries, 1):
        print_query_header(item, i, len(queries))

        try:
            run_query(item, retriever, model, system_prompt,
                     procedure, args.min_score, args.max_tokens)
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")

        # Pause between queries unless --auto
        if not args.auto and i < len(queries):
            try:
                input("\n\033[2mPress Enter for next query (or Ctrl+C to stop)...\033[0m")
            except KeyboardInterrupt:
                print("\n\nStopped.")
                break

    print("\n\033[1mDemo complete.\033[0m\n")
    client.close()


if __name__ == "__main__":
    main()
