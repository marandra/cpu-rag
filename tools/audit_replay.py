"""Re-ask the third-party audit's questions against our pools.

Runs on the cluster (the LB is not reachable from the laptop) under
`./.venv-native/bin/python`. Input is audit_extract.py's JSON; output is the
same records with our fresh answer merged in as `our_answer`.

    ./.venv-native/bin/python tools/audit_replay.py \
        --questions reports/audit_questions.json \
        --api-url http://eurehpccomputo01:8080 \
        --procedure diabetes \
        --out reports/audit_replay_diabetes.json

One pool serves one profile, so this runs once per procedure and each run picks
out only the rows for that procedure. Generation is temperature 0.1 with no
fixed seed: answers are equivalent in behaviour, never identical byte for byte.
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


def ask(rec: dict, api_url: str) -> dict:
    """Ask one question. Never raises — a failed row is reported, not fatal."""
    chunks: list[str] = []
    usage: dict = {}
    error: str | None = None
    t0 = time.perf_counter()
    try:
        for ev in stream_query(rec["question"], rec["procedure"],
                               api_url=api_url, timeout=300.0):
            if ev.event == "chunk":
                chunks.append(ev.data.get("text", ""))
            elif ev.event == "done":
                usage = ev.data.get("usage", {})
            elif ev.event == "error":
                error = str(ev.data)
    except Exception as e:  # network, timeout, malformed stream
        error = f"{type(e).__name__}: {e}"
    out = dict(rec)
    out["our_answer"] = "".join(chunks).strip()
    out["our_usage"] = usage
    out["our_error"] = error
    out["our_elapsed"] = round(time.perf_counter() - t0, 2)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="reports/audit_questions.json")
    p.add_argument("--api-url", default=None)
    p.add_argument("--procedure", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent requests; match the pool's replica count.")
    args = p.parse_args()

    load_dotenv()
    api_url = resolve_api_url(args.api_url)

    all_recs = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    recs = [r for r in all_recs if r["procedure"] == args.procedure]
    if not recs:
        print(f"No questions for procedure {args.procedure!r}", file=sys.stderr)
        return 1

    health = get_health(api_url)
    print(f"API {api_url}  profile={health.get('profile','?')}  "
          f"procedures={health.get('procedures')}")
    if args.procedure not in (health.get("procedures") or []):
        print(f"ERROR: the service does not serve {args.procedure!r} — "
              f"wrong profile or wrong pool.", file=sys.stderr)
        return 1

    print(f"Asking {len(recs)} questions ({args.procedure}) "
          f"with {args.workers} workers...")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda r: ask(r, api_url), recs))
    wall = time.perf_counter() - t0
    rows.sort(key=lambda r: r["id"])   # concurrency scrambles completion order

    failed = [r for r in rows if r["our_error"]]
    payload = {
        "procedure": args.procedure,
        "generated": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "api_url": api_url,
        "profile": health.get("profile"),
        "model": health.get("model"),
        "rows": rows,
    }

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")

    toks = sum(r["our_usage"].get("completion_tokens", 0) for r in rows)
    print(f"Wrote {dest}  ({len(rows)} Q, {toks} tokens, {wall:.1f}s wall)")
    if failed:
        print(f"WARNING: {len(failed)} failed: {[r['id'] for r in failed]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
