##########################################
# Service: cpu-rag-api
# Size target: <800MB (includes LLM deps)
##########################################

# Stage 1: Dependencies
FROM python:3.11-slim AS deps

WORKDIR /build

# Build dependencies for llama-cpp-python
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip

COPY pyproject.toml .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

# Security: non-root user (with a writable HOME for model caches)
RUN groupadd -r appuser \
    && useradd -r -g appuser -u 1001 -m -d /home/appuser appuser

# Runtime dependencies (curl for healthcheck)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Layer order matters: copy what changes least first so code edits don't bust
# the model layer (3GB). Order: model → src/ (rare changes) → app/ (frequent).

# Embed the default LLM so the image is self-contained for transfer
COPY --chown=appuser:appuser models/Ministral-3-3B-Q4_K_M.gguf /app/models/Ministral-3-3B-Q4_K_M.gguf

# Copy application code (src/ before app/ — app/ changes more often)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser corpus/ ./corpus/
COPY --chown=appuser:appuser app/ ./app/

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HOME=/home/appuser \
    XDG_CACHE_HOME=/home/appuser/.cache

# Metadata
LABEL service="cpu-rag-api" \
      version="1.0.0" \
      description="RAG API service for medical FAQ chatbot"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
