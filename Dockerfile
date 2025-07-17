# --- Base: شامل CUDA + ابزارهای بیلد ---
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Builder Stage ---
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    libgomp1 \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    libcuda1-535 \
    nvidia-driver-535 \
    && rm -rf /var/lib/apt/lists/*


# نصب llama-cpp-python با پشتیبانی GPU
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN .venv/bin/pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
RUN .venv/bin/pip install -r requirements.txt

# --- Final Stage ---
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04 AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser
USER appuser

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY config/ ./config/
COPY llm_server/ ./llm_server/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
