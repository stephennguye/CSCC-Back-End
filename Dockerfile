# ============================================================
# AI Call Center Backend — Production Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies into a venv
#   Stage 2 (runtime): Copy venv + source; run as non-root user
#
# GPU variant:
#   Build with:  docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04 .
#   CPU default: python:3.12-slim
# ============================================================

ARG BASE_IMAGE=python:3.12-slim

# ────────────────────────────────────────────────
# Stage 1 — Builder (deps only)
# ────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ────────────────────────────────────────────────
# Stage 2 — Runtime image
# ────────────────────────────────────────────────
FROM ${BASE_IMAGE} AS runtime

# Install runtime system libraries (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy application source
COPY --chown=appuser:appgroup alembic/ ./alembic/
COPY --chown=appuser:appgroup alembic.ini ./alembic.ini
COPY --chown=appuser:appgroup src/ ./src/
COPY --chown=appuser:appgroup pyproject.toml ./pyproject.toml

# Create data directory for ChromaDB (local fallback)
RUN mkdir -p /app/data/chroma && chown -R appuser:appgroup /app/data

# Drop to non-root
USER appuser

# Expose application port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command — run via uvicorn with production workers
CMD ["uvicorn", "src.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--log-level", "info"]
