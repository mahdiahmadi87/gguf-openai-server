# syntax=docker/dockerfile:1.4

# Base stage for common setup
FROM python:3.12-slim AS base

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# --- Builder stage ---
FROM base AS builder

# Install build-time dependencies (including libgomp for llama_cpp compilation)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        libopenblas-dev \
        libstdc++6 \
        libgomp1 \
        git && \
        cmake \
        curl \
        wget \
        gnupg \
    rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDAToolkit_ROOT="/usr/local/cuda"

# Copy requirements and install Python deps
COPY requirements.txt ./
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    .venv/bin/pip install -r requirements.txt

RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install --no-cache-dir --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128
    
# --- Final stage ---
FROM python:3.12-slim AS final

# Install runtime dependencies (ensure libgomp present)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

# Copy venv and app code
COPY --from=builder /app/.venv /app/.venv
COPY config/ ./config/
COPY llm_server/ ./llm_server/
COPY models/ ./models/
# COPY requirements.txt .

EXPOSE 8000
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]