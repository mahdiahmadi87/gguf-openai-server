# syntax=docker/dockerfile:1.4

# Base stage for common setup
FROM nvidia/cuda:12.2.0-base-ubuntu22.04 AS base
RUN apt-get update \
    && apt-get install -y python3 python3-venv python3-pip wget gnupg2 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Builder stage: install build deps, correct NVIDIA CUDA repo & create venv
FROM base AS builder
# Install compiler, CMake, libgomp
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         build-essential libgomp1 cmake \
    && rm -rf /var/lib/apt/lists/*

# Add NVIDIA CUDA repository for Ubuntu 22.04 and install toolkit 12.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-toolkit-12-8 \
    && rm -rf /var/lib/apt/lists/* cuda-keyring_1.1-1_all.deb

# Set CUDA environment for build
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDAToolkit_ROOT="/usr/local/cuda"

# Create virtual environment & install Python deps
COPY requirements.txt ./
RUN python3 -m venv .venv \
    && .venv/bin/pip install --upgrade pip setuptools wheel \
    && .venv/bin/pip install -r requirements.txt

# Install llama-cpp-python with CUDA support
RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    .venv/bin/pip install --no-cache-dir --force-reinstall llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122

# Final stage: copy venv & app code, drop to non-root
FROM base AS final
COPY --from=builder /app/.venv /app/.venv
RUN useradd -m appuser
USER appuser
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY config/      ./config/
COPY llm_server/  ./llm_server/
COPY models/      ./models/
EXPOSE 8000
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
