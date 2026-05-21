"""HTTP client for the cpu-rag FastAPI service (Docker)."""

import json
import os
from dataclasses import dataclass
from typing import Iterator

import httpx

DEFAULT_API_URL = os.environ.get("RAG_API_URL", "http://localhost:8000")


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no override of already-set vars)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


@dataclass
class StreamEvent:
    event: str  # "chunk" | "done" | "error" | "message"
    data: dict


def get_health(api_url: str = DEFAULT_API_URL, timeout: float = 5.0) -> dict:
    with httpx.Client(timeout=timeout) as client:
        r = client.get(f"{api_url}/health")
        r.raise_for_status()
        return r.json()


def retrieve(
    question: str,
    procedure: str,
    *,
    top_k: int | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Call POST /retrieve and return {chunks, usage}.

    Retrieval-only — does not invoke the LLM. Useful for inspecting the
    retrieval stack without paying generation cost.
    """
    api_url = api_url or DEFAULT_API_URL
    api_key = api_key or os.environ.get("RAG_API_KEY")
    if not api_key:
        raise RuntimeError("RAG_API_KEY missing. Export it or add it to .env.")

    payload: dict = {"question": question, "procedure": procedure}
    if top_k is not None:
        payload["top_k"] = top_k

    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    with httpx.Client(timeout=timeout) as client:
        r = client.post(f"{api_url}/retrieve", json=payload, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
        return r.json()


def stream_query(
    question: str,
    procedure: str,
    *,
    api_url: str | None = None,
    api_key: str | None = None,
    session_id: str | None = None,
    timeout: float = 180.0,
) -> Iterator[StreamEvent]:
    """Stream a query through the /query SSE endpoint.

    Yields StreamEvent for each `event:`/`data:` block. Terminal events:
    `done` (success) or `error` (server-side failure).
    """
    api_url = api_url or DEFAULT_API_URL
    api_key = api_key or os.environ.get("RAG_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RAG_API_KEY missing. Export it or add it to .env."
        )

    payload: dict = {"question": question, "procedure": procedure}
    if session_id:
        payload["session_id"] = session_id

    headers = {
        "X-API-Key": api_key,
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }

    with httpx.Client(timeout=httpx.Timeout(timeout, read=timeout)) as client:
        with client.stream(
            "POST", f"{api_url}/query", json=payload, headers=headers,
        ) as resp:
            if resp.status_code != 200:
                body = resp.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {resp.status_code}: {body}")

            event_name: str | None = None
            data_lines: list[str] = []

            for raw in resp.iter_lines():
                line = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")

                if line == "":
                    if event_name is not None or data_lines:
                        data_str = "\n".join(data_lines)
                        try:
                            data = json.loads(data_str) if data_str else {}
                        except json.JSONDecodeError:
                            data = {"raw": data_str}
                        yield StreamEvent(event=event_name or "message", data=data)
                    event_name = None
                    data_lines = []
                    continue

                if line.startswith(":"):
                    continue  # SSE comment / keepalive
                if line.startswith("event:"):
                    event_name = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
                # other SSE fields (id:, retry:) are ignored
