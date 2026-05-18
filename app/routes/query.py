"""Query endpoint with SSE streaming (fulldoc mode)."""

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from app.auth import verify_api_key
from app.config import settings
from app.prompt import get_system_prompt
from app.schemas import QueryRequest

router = APIRouter()
logger = logging.getLogger(__name__)


async def _stream_response(
    question: str, procedure: str
) -> AsyncGenerator[dict, None]:
    from app.main import app_state

    start_time = time.perf_counter()
    fulldoc_text = app_state.fulldoc_texts[procedure]
    snapshot = app_state.snapshots[procedure]

    system_prompt = get_system_prompt(procedure)
    # Must match warmup byte-for-byte so llama-cpp's KV cache prefix
    # covers the whole (system + fulldoc) payload.
    user_prompt = f"INFORMACIÓN:\n{fulldoc_text}\n\nPREGUNTA: {question}"

    gen_start = time.perf_counter()
    completion_tokens = 0

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
            await loop.run_in_executor(
                None, app_state.llm.load_state, snapshot
            )

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
                nonlocal completion_tokens
                while True:
                    chunk = await loop.run_in_executor(None, _next_chunk)
                    if chunk is _SENTINEL:
                        return
                    text = chunk["choices"][0]["delta"].get("content", "")
                    if text:
                        completion_tokens += 1
                        yield text

            async with asyncio.timeout(settings.generation_timeout):
                async for token in generate_tokens():
                    yield {"event": "chunk", "data": json.dumps({"text": token})}

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

    gen_time = time.perf_counter() - gen_start
    total_time = time.perf_counter() - start_time

    yield {
        "event": "done",
        "data": json.dumps(
            {
                "usage": {
                    "completion_tokens": completion_tokens,
                    "generation_ms": round(gen_time * 1000),
                    "total_ms": round(total_time * 1000),
                },
            }
        ),
    }


@router.post("/query")
async def query(
    request: QueryRequest,
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

    logger.info(
        f"Query received: {request.question[:50]}... "
        f"[session={request.session_id}, procedure={request.procedure}]"
    )

    return EventSourceResponse(_stream_response(request.question, request.procedure))
