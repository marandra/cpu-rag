"""Dump a dataset's question/answer pairs to plain Markdown for human review.

Not an evaluation: no scoring, no keyword coverage — that is run_eval.py's job.
This just asks every question and writes what came back, in order.

Queries run concurrently because the HPC pool is N independent replicas behind
least_conn; one in-flight request per replica is exactly what it is sized for.
Order is restored on write, so the file always follows the dataset.

    ./.venv-native/bin/python dump_qa.py \
        --dataset eval/datasets/eval_dataset_diabetes_coverage.json \
        --api-url http://eurehpccomputo01:8080 \
        --out reports/qa_diabetes_coverage.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rag_client import get_health, load_dotenv, resolve_api_url, stream_query


def ask(idx: int, item: dict, api_url: str, procedure: str) -> dict:
    """Ask one question. Never raises — a failed row is reported, not fatal."""
    question = item["query"]
    chunks: list[str] = []
    usage: dict = {}
    error: str | None = None
    t0 = time.perf_counter()
    try:
        for ev in stream_query(question, procedure, api_url=api_url, timeout=300.0):
            if ev.event == "chunk":
                chunks.append(ev.data.get("text", ""))
            elif ev.event == "done":
                usage = ev.data.get("usage", {})
            elif ev.event == "error":
                error = str(ev.data)
    except Exception as e:  # network, timeout, malformed stream
        error = f"{type(e).__name__}: {e}"
    return {
        "idx": idx,
        "question": question,
        "answer": "".join(chunks).strip(),
        "usage": usage,
        "error": error,
        "elapsed": time.perf_counter() - t0,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--api-url", default=None)
    p.add_argument("--procedure", default=None,
                   help="Defaults to the dataset items' own `procedure` field.")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8,
                   help="Concurrent requests; match the pool's replica count.")
    args = p.parse_args()

    load_dotenv()
    api_url = resolve_api_url(args.api_url)

    items = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    procedure = args.procedure or items[0].get("procedure")
    if not procedure:
        print("Cannot infer procedure; pass --procedure", file=sys.stderr)
        return 1

    health = get_health(api_url)
    print(f"API {api_url}  profile={health.get('profile','?')}  "
          f"procedures={health.get('procedures')}")
    if procedure not in (health.get("procedures") or []):
        print(f"ERROR: the service does not serve {procedure!r} — "
              f"wrong profile or wrong pool.", file=sys.stderr)
        return 1

    print(f"Asking {len(items)} questions ({procedure}) with {args.workers} workers...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(
            lambda t: ask(t[0], t[1], api_url, procedure),
            enumerate(items, start=1),
        ))
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["idx"])   # concurrency scrambles completion order

    failed = [r for r in rows if r["error"]]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")

    out = [
        f"# {procedure} — preguntas y respuestas",
        "",
        f"Dataset: `{Path(args.dataset).name}` ({len(rows)} preguntas)  ",
        f"Generado: {stamp}  ",
        f"Servicio: `{api_url}` — perfil `{health.get('profile','?')}`, "
        f"modelo `{health.get('model','?')}`",
        "",
        "Respuestas tal cual las devolvió el servicio, sin editar.",
        "",
    ]
    if failed:
        out += [f"> **{len(failed)} de {len(rows)} preguntas fallaron** "
                f"(marcadas abajo).", ""]

    for r in rows:
        out += ["---", "", f"## {r['idx']}. {r['question']}", ""]
        if r["error"]:
            out += [f"**ERROR:** {r['error']}", ""]
        else:
            out += [r["answer"] or "_(respuesta vacía)_", ""]

    out.append("---")
    out.append("")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(out), encoding="utf-8")

    toks = sum(r["usage"].get("completion_tokens", 0) for r in rows)
    print(f"Wrote {dest}  ({len(rows)} Q, {toks} tokens, {wall:.1f}s wall)")
    if failed:
        print(f"WARNING: {len(failed)} failed: {[r['idx'] for r in failed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
