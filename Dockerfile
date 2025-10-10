# =========================
# Builder stage
# =========================
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt requirements-dev.txt ./

# Install dependencies into /install
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && pip install --no-cache-dir --prefix=/install -r requirements-dev.txt

# =========================
# Production stage
# =========================
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Set PATH
ENV PATH=/usr/local/bin:$PATH

# Expose port (Cloud Run uses $PORT env)
ENV PORT=8000
EXPOSE $PORT

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:$PORT/health || exit 1

# Run application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
