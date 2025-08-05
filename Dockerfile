# syntax=docker/dockerfile:1.4

# Base stage: use Debian-based CUDA image and install Python
FROM nvidia/cuda:12.2.0-base-debian12 AS base
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip wget gnupg2 ca-certificates apt-transport-https \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Builder stage: install build tools, CUDA Toolkit 12.8, and Python deps
FROM base AS builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential libgomp1 cmake \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repo and install CUDA Toolkit 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-8 && \
    rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb
ENV PATH="/usr/local/cuda/bin:${PATH}" CUDAToolkit_ROOT="/usr/local/cuda"

# Create virtual environment and install Python requirements + llama.cpp
COPY requirements.txt ./
RUN python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip setuptools wheel && \
    .venv/bin/pip install -r requirements.txt && \
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    .venv/bin/pip install --no-cache-dir --force-reinstall llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Final stage: reuse builder (with CUDA & venv) and add app code
FROM builder AS final
RUN useradd -m appuser
USER appuser
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --chown=appuser config/      ./config/
COPY --chown=appuser llm_server/  ./llm_server/
COPY --chown=appuser models/      ./models/
EXPOSE 8000
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
