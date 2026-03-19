# ---------------------------------------------------------------------------
# Stage 1: builder — install Python deps into a virtual-env
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps needed to compile some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime — minimal image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy virtual-env from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source code
COPY src/ src/

# Ensure venv binaries take precedence
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Model configuration (override via environment / docker-compose)
ENV MODEL_PATH="" \
    MODEL_TYPE="xgboost" \
    PORT=8000

EXPOSE ${PORT}

# Non-root user for security
RUN useradd -m appuser
USER appuser

CMD ["sh", "-c", "uvicorn src.api.app:app --host 0.0.0.0 --port ${PORT}"]
