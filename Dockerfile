# syntax=docker/dockerfile:1.4

FROM python:3.12-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libopenblas-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Setup venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install llama-cpp-python with cuBLAS (GPU support)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir

# Install your project dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------
# Final runtime image
# ----------------------
FROM python:3.12-slim as runtime

# Install runtime libs (libgomp is needed by llama.cpp)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/opt/venv/bin:$PATH"

# Optional non-root user
RUN useradd -m appuser
USER appuser

COPY --from=builder /opt/venv /opt/venv
COPY config/ ./config/
COPY llm_server/ ./llm_server/
COPY models/ ./models/

EXPOSE 8000
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
