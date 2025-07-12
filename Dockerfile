# syntax=docker/dockerfile:1.4

########################################################################
# Builder stage: build llama-cpp-python with CUDA support and install deps
########################################################################
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS builder

# Install system build tools and Python
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        build-essential \
        cmake \
        git \
        libgomp1 \
        libopenblas-dev \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/venv
RUN python3.12 -m venv . && \
    . /opt/venv/bin/activate && \
    pip install --upgrade pip

# Install llama-cpp-python with CUDA cuBLAS support
RUN . /opt/venv/bin/activate && \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

# Copy and install Python requirements (excluding llama-cpp-python)
COPY requirements.txt /app/requirements.txt
RUN . /opt/venv/bin/activate && \
    pip install --no-cache-dir -r /app/requirements.txt

########################################################################
# Final stage: runtime image
########################################################################
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS final

# Install runtime dependencies
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

# Copy virtualenv and application code
COPY --from=builder /opt/venv /opt/venv
COPY config/ ./config/
COPY llm_server/ ./llm_server/
COPY models/ ./models/

EXPOSE 8000
ENTRYPOINT ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
