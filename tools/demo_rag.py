"""
Demo RAG client: run eval dataset queries against the dockerized API.

Streams responses via SSE and shows per-query timing.

Usage:
    python demo_rag.py                          # interactive menu
    python demo_rag.py --category procedimiento # only procedimiento queries
    python demo_rag.py --category OOS           # only out-of-scope queries
    python demo_rag.py --list-categories        # show available categories
    python demo_rag.py --auto                   # run all queries automatically
    python demo_rag.py --procedure hemorroides
    python demo_rag.py --dataset ./eval/datasets/eval_dataset_hemorroides_coverage.json
    python demo_rag.py --auto --output run.md      # also save a report
"""

import argparse
import json
import os
import sys
import time

# Make tools/rag_client importable regardless of the cwd it's launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag_client import (
    get_health,
    load_dotenv,
    resolve_api_url,
    stream_query,
)

PROCEDURES = ["hemorroides", "cirugia-abdominal"]
DATASETS_DIR = "./eval/datasets"


def default_dataset(procedure: str) -> str:
    """Coverage dataset path for a procedure slug (matches run_eval naming)."""
    return f"{DATASETS_DIR}/eval_dataset_{procedure}_coverage.json"


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def get_categories(dataset: list[dict]) -> dict[str, list[dict]]:
    categories: dict[str, list[dict]] = {}
    for item in dataset:
        categories.setdefault(item["category"], []).append(item)
    return categories


def print_query_header(item: dict, index: int, total: int):
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


def run_query(item: dict, api_url: str, procedure_slug: str) -> dict:  # noqa: C901
    """Run a single query, stream tokens, return result summary."""
    query = item["query"]
    t_start = time.perf_counter()
    t_first_token: float | None = None
    text_parts: list[str] = []
    usage: dict = {}
    error: dict | None = None

    try:
        for ev in stream_query(query, procedure_slug, api_url=api_url):
            if ev.event == "chunk":
                if t_first_token is None:
                    t_first_token = time.perf_counter()
                token = ev.data.get("text", "")
                print(token, end="", flush=True)
                text_parts.append(token)
            elif ev.event == "done":
                usage = ev.data.get("usage", {})
            elif ev.event == "error":
                error = ev.data
                break
    except KeyboardInterrupt:
        print("\n\033[33m  [interrupted]\033[0m")
        return {"interrupted": True}
    except Exception as e:
        print(f"\n\033[31mClient error: {e}\033[0m\n")
        return {"error": str(e)}

    t_end = time.perf_counter()

    if error:
        code = error.get("code", "error")
        detail = error.get("detail", "")
        print(f"\n\033[31m  [{code}] {detail}\033[0m\n")
        return {"error": code, "detail": detail, "response": "".join(text_parts)}

    if text_parts:
        print()

    ttft = (t_first_token or t_end) - t_start
    total = t_end - t_start
    n_tokens = usage.get("completion_tokens", len(text_parts))
    decode_s = (t_end - t_first_token) if t_first_token else 0.0
    tps = (n_tokens - 1) / decode_s if decode_s > 0 and n_tokens > 1 else 0.0

    print(
        f"\033[32m  total={total:.2f}s  TTFT={ttft:.2f}s  "
        f"tokens={n_tokens} ({tps:.1f} tok/s)\033[0m\n"
    )
    return {
        "response": "".join(text_parts),
        "ttft": ttft,
        "total": total,
        "n_tokens": n_tokens,
        "tps": tps,
    }


def interactive_menu(categories: dict[str, list[dict]]) -> list[dict] | None:
    print("\n\033[1mCategorías disponibles:\033[0m\n")

    in_scope = {k: v for k, v in categories.items() if v[0]["answerable"]}
    oos = {k: v for k, v in categories.items() if not v[0]["answerable"]}

    menu: list[tuple[str, str, list[dict]]] = []
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

    all_q = [it for items in categories.values() for it in items]
    in_q = [it for items in in_scope.values() for it in items]
    oos_q = [it for items in oos.values() for it in items]

    print("\n  \033[33mEspecial:\033[0m")
    print(f"    \033[1m{n:2d}\033[0m. {'ALL':20s} (todas: {len(all_q)} queries)")
    menu.append((str(n), "all", all_q)); n += 1
    print(f"    \033[1m{n:2d}\033[0m. {'IN-SCOPE':20s} (todas in-scope: {len(in_q)} queries)")
    menu.append((str(n), "in-scope", in_q)); n += 1
    print(f"    \033[1m{n:2d}\033[0m. {'OOS':20s} (todas OOS: {len(oos_q)} queries)")
    menu.append((str(n), "oos", oos_q))

    print()
    choice = input("\033[1mSelecciona número o nombre (q=salir): \033[0m").strip().lower()
    if choice in ("q", "quit", "exit", ""):
        return None
    for num, name, queries in menu:
        if choice == num or choice == name:
            return queries
    print(f"\033[31mOpción no válida: {choice}\033[0m")
    return interactive_menu(categories)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Demo RAG client (Docker API)")
    parser.add_argument("--api-url", default=None,
                        help="API base URL. Defaults to $RAG_API_URL (incl. .env) "
                             "or http://localhost:8000.")
    parser.add_argument("--procedure", default="hemorroides", choices=PROCEDURES)
    parser.add_argument("--dataset",
                        help="Path to a custom eval dataset JSON. Overrides the "
                             "default dataset selected by --procedure.")
    parser.add_argument("--category", help="Filter by category (or ALL, IN-SCOPE, OOS)")
    parser.add_argument("--list-categories", action="store_true")
    parser.add_argument("--auto", action="store_true",
                        help="Run all queries without pausing")
    parser.add_argument("--output",
                        help="Write a Markdown report of the run to this path "
                             "(in addition to the console output).")
    args = parser.parse_args()
    args.api_url = resolve_api_url(args.api_url)

    dataset_path = args.dataset or default_dataset(args.procedure)
    dataset = load_dataset(dataset_path)
    categories = get_categories(dataset)
    print(f"\033[2mLoaded {len(dataset)} queries from {dataset_path}\033[0m")

    if args.list_categories:
        print(f"\nCategories for procedure '{args.procedure}':\n")
        for cat, items in sorted(categories.items()):
            scope = "in-scope" if items[0]["answerable"] else "OOS"
            print(f"  {cat:20s} {len(items):3d} queries  ({scope})")
        return 0

    if args.category:
        cat = args.category.lower()
        if cat == "all":
            queries = dataset
        elif cat == "in-scope":
            queries = [it for it in dataset if it["answerable"]]
        elif cat == "oos":
            queries = [it for it in dataset if not it["answerable"]]
        elif (match := next((k for k in categories if k.lower() == cat), None)):
            queries = categories[match]
        else:
            print(f"Unknown category: {args.category}")
            return 1
    else:
        queries = interactive_menu(categories)
        if queries is None:
            return 0

    print(f"\n\033[2mSelected {len(queries)} queries\033[0m\n")

    procedure_slug = args.procedure

    try:
        health = get_health(args.api_url)
    except Exception as e:
        print(f"\033[31mCannot reach API at {args.api_url}: {e}\033[0m")
        return 1

    print(f"\033[2mAPI: {args.api_url}  model={health.get('model')}\033[0m")
    available = health.get("procedures") or []
    if available and procedure_slug not in available:
        print(f"\033[31m  warning: procedure {procedure_slug!r} not loaded on server "
              f"(available: {available})\033[0m")

    results: list[tuple[dict, dict]] = []
    for i, item in enumerate(queries, 1):
        print_query_header(item, i, len(queries))
        try:
            result = run_query(item, args.api_url, procedure_slug)
        except Exception as e:
            print(f"\033[31mError: {e}\033[0m")
            result = {"error": "client_exception", "detail": str(e)}
        results.append((item, result))

        if not args.auto and i < len(queries):
            try:
                input("\n\033[2mPress Enter for next query (or Ctrl+C to stop)...\033[0m")
            except KeyboardInterrupt:
                print("\n\nStopped.")
                break

    if args.output:
        try:
            write_report(args.output, results, args, health, dataset_path)
            print(f"\033[2mReport written to {args.output}\033[0m")
        except Exception as e:
            print(f"\033[31mFailed to write report: {e}\033[0m")

    print("\n\033[1mDemo complete.\033[0m\n")
    return 0


def write_report(path: str, results: list[tuple[dict, dict]],
                 args, health: dict, dataset_path: str) -> None:
    """Write a Markdown report of the run."""
    from datetime import datetime

    lines: list[str] = []
    lines.append(f"# demo_rag run — {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- API: `{args.api_url}`")
    lines.append(f"- Procedure: `{args.procedure}`")
    lines.append(f"- Dataset: `{dataset_path}`")
    lines.append(f"- Model: `{health.get('model', '?')}`")
    lines.append(f"- Queries run: {len(results)}")
    lines.append("")

    refusal = "no tengo información sobre eso"
    n_ok = n_err = n_refusals = 0
    for _item, res in results:
        if res.get("error"):
            n_err += 1
            continue
        n_ok += 1
        if refusal in (res.get("response") or "").lower():
            n_refusals += 1
    lines.append("## Summary")
    lines.append(f"- Successful: {n_ok}")
    lines.append(f"- Errors: {n_err}")
    lines.append(f"- Refusals (\"No tengo información sobre eso\"): {n_refusals}")
    lines.append("")

    lines.append("## Queries")
    for i, (item, res) in enumerate(results, 1):
        scope = "IN-SCOPE" if item["answerable"] else "OOS"
        lines.append("")
        lines.append(f"### {i}. [{scope}] [{item['category']}] {item['query']}")
        lines.append(f"- Intent: {item['intent']}")
        if item.get("test_focus"):
            lines.append(f"- Test focus: {item['test_focus']}")
        if item.get("expected_keywords"):
            lines.append(f"- Expected keywords: {item['expected_keywords']}")
        lines.append("")
        if res.get("error"):
            lines.append(f"**ERROR** `{res['error']}`: {res.get('detail', '')}")
            if res.get("response"):
                lines.append("")
                lines.append("Partial response:")
                lines.append("")
                lines.append("```")
                lines.append(res["response"])
                lines.append("```")
            continue
        lines.append("**Response:**")
        lines.append("")
        lines.append("```")
        lines.append(res.get("response", "").rstrip() or "(empty)")
        lines.append("```")
        lines.append(
            f"- Timing: total={res.get('total', 0):.2f}s "
            f"TTFT={res.get('ttft', 0):.2f}s "
            f"tokens={res.get('n_tokens', 0)} ({res.get('tps', 0):.1f} tok/s)"
        )

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
