##########################################
# Service: cpu-rag-api
# Size target: <500MB (code + runtime only — model lives in a mounted volume)
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

# llama.cpp build flags. Default = portable AVX2+FMA baseline that runs on
# any modern x86_64 (laptop, HPC, EC2). Override per architecture via:
#   docker build --build-arg CMAKE_FLAGS='-DGGML_NATIVE=ON' -t cpu-rag-api:native .
# Common flavors:
#   portable: -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON
#   x86_64-v3: + -DGGML_F16C=ON -DGGML_BMI2=ON  (Haswell+, broadly portable)
#   native:    -DGGML_NATIVE=ON  (uses -march=native; HPC/EC2-specific image)
ARG CMAKE_FLAGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON"
ENV CMAKE_ARGS="${CMAKE_FLAGS}" \
    FORCE_CMAKE=1

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

# The model GGUF is NOT embedded — it ships in the mounted model-pack volume
# (see docker-compose.yml). Keeps the image small and lets us benchmark
# different models without rebuilding.

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
      version="1.2.0" \
      description="RAG API service for medical FAQ chatbot"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
