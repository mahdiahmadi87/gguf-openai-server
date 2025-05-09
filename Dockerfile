# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create app directory
WORKDIR /app

# --- Builder stage ---
FROM base AS builder

# Install system dependencies required for llama-cpp-python (OpenBLAS, etc.)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        libopenblas-dev \
        libstdc++6 \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements.txt first for better cache usage
COPY --link requirements.txt ./

# Create venv and install dependencies using pip cache
RUN python -m venv .venv && \
    .venv/bin/pip install --upgrade pip && \
    --mount=type=cache,target=/root/.cache/pip \
    .venv/bin/pip install -r requirements.txt

# --- Final stage ---
FROM base AS final

# Create a non-root user
RUN useradd -m appuser
USER appuser

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --link /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code (excluding .env, .git, etc. via .dockerignore)
COPY --link config/ ./config/
COPY --link llm_server/ ./llm_server/
COPY --link models/ ./models/
COPY --link requirements.txt ./
COPY --link README.md ./

# Expose the default port (as per settings.py and README)
EXPOSE 8000

# Entrypoint: use uvicorn to run the FastAPI app
CMD ["uvicorn", "llm_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
