"""Query endpoint with SSE streaming (fulldoc mode)."""

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from app.auth import verify_api_key
from app.config import settings
from app.prompt import get_system_prompt
from app.schemas import QueryRequest

router = APIRouter()
logger = logging.getLogger(__name__)


async def _stream_response(
    question: str, procedure: str, request_id: str
) -> AsyncGenerator[dict, None]:
    from app.main import app_state

    start_time = time.perf_counter()
    fulldoc_text = app_state.fulldoc_texts[procedure]

    system_prompt = get_system_prompt(procedure)
    # Must match warmup byte-for-byte so llama-cpp's KV cache prefix
    # covers the whole (system + fulldoc) payload.
    user_prompt = f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: {question}"

    completion_tokens = 0
    load_state_ms = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    first_token_t = None

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        loop = asyncio.get_event_loop()

        # One Llama, one live KV state. The lock serializes requests so
        # nobody overwrites the loaded snapshot mid-generation. Do NOT
        # save_state back — the post-generation state has the Q/A appended
        # and would poison the next request.
        async with app_state.gen_lock:

            def _prime_kv():
                """Put this procedure's warm prefix into the live KV, if we have one.

                "disk" unpickles it per request; "memory" already holds the
                object; "off" does nothing and lets create_chat_completion
                prefill, reusing whatever prefix llama-cpp still has cached —
                free when consecutive requests share a procedure, a full
                re-prefill when they alternate.
                """
                if settings.snapshot_mode == "off":
                    return
                if settings.snapshot_mode == "memory":
                    state = app_state.snapshot_states.get(procedure)
                    # Absent only if save_state failed at startup; falling
                    # through to a live prefill is correct, just slower.
                    if state is not None:
                        app_state.llm.load_state(state)
                    return

                from app.snapshot_cache import load_snapshot

                snapshot_path = app_state.snapshot_paths[procedure]
                state = load_snapshot(snapshot_path)
                if state is None:
                    raise RuntimeError(f"Snapshot pickle unreadable: {snapshot_path}")
                app_state.llm.load_state(state)

            load_start = time.perf_counter()
            await loop.run_in_executor(None, _prime_kv)
            load_state_ms = (time.perf_counter() - load_start) * 1000

            gen_start = time.perf_counter()

            def create_stream():
                return app_state.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=settings.max_tokens,
                    temperature=0.1,
                    stream=True,
                )

            stream = await loop.run_in_executor(None, create_stream)

            _SENTINEL = object()

            def _next_chunk():
                return next(stream, _SENTINEL)

            async def generate_tokens():
                nonlocal completion_tokens, prefill_ms, first_token_t
                while True:
                    chunk = await loop.run_in_executor(None, _next_chunk)
                    if chunk is _SENTINEL:
                        return
                    text = chunk["choices"][0]["delta"].get("content", "")
                    if text:
                        if first_token_t is None:
                            first_token_t = time.perf_counter()
                            prefill_ms = (first_token_t - gen_start) * 1000
                        completion_tokens += 1
                        yield text

            async with asyncio.timeout(settings.generation_timeout):
                async for token in generate_tokens():
                    yield {"event": "chunk", "data": json.dumps({"text": token})}

            if first_token_t is not None:
                decode_ms = (time.perf_counter() - first_token_t) * 1000

    except asyncio.TimeoutError:
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "code": "generation_timeout",
                    "detail": f"Generation exceeded {settings.generation_timeout}s timeout",
                }
            ),
        }
        return
    except Exception as e:
        logger.exception("Generation failed")
        yield {
            "event": "error",
            "data": json.dumps({"code": "model_error", "detail": str(e)}),
        }
        return

    total_ms = (time.perf_counter() - start_time) * 1000
    tok_s = (completion_tokens * 1000 / decode_ms) if decode_ms > 0 else 0.0

    # Single structured line per request — easy to grep + parse into CSV.
    logger.info(
        "REQ id=%s replica=%s proc=%s out_tok=%d "
        "load_state_ms=%.0f prefill_ms=%.0f decode_ms=%.0f "
        "decode_tok_s=%.2f total_ms=%.0f",
        request_id, settings.replica_id, procedure, completion_tokens,
        load_state_ms, prefill_ms, decode_ms, tok_s, total_ms,
    )

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "request_id": request_id,
                "replica_id": settings.replica_id,
                "usage": {
                    "completion_tokens": completion_tokens,
                    "load_state_ms": round(load_state_ms),
                    "prefill_ms": round(prefill_ms),
                    "decode_ms": round(decode_ms),
                    "decode_tok_s": round(tok_s, 2),
                    "total_ms": round(total_ms),
                },
            }
        ),
    }


@router.post("/query")
async def query(
    request: QueryRequest,
    http_request: Request,
    _api_key: str = Depends(verify_api_key),
):
    from app.main import app_state

    if request.procedure not in app_state.procedures:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unknown_procedure",
                "detail": f"Unknown procedure {request.procedure!r}",
                "available": sorted(app_state.procedures),
            },
        )

    # Honor a client-supplied request ID (sweep harness sets this for
    # row joining), otherwise generate one.
    request_id = http_request.headers.get("x-request-id") or uuid.uuid4().hex[:12]

    logger.info(
        "Query received id=%s replica=%s proc=%s q=%r",
        request_id, settings.replica_id, request.procedure,
        request.question[:60],
    )

    return EventSourceResponse(
        _stream_response(request.question, request.procedure, request_id)
    )
