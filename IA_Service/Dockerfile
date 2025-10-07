# ---------- Runtime image ----------
FROM python:3.11-slim

# Fast, clean pip + HF caches in /tmp (writable in Cloud Run)
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user and tmp dirs
RUN useradd -m -u 10001 appuser \
 && mkdir -p /tmp/hf/transformers /tmp/hf/hub \
 && chown -R appuser:appuser /tmp /app

# Install Python deps (cache-friendly)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app code last to preserve layer cache
COPY . /app
RUN chown -R appuser:appuser /app

# Drop privileges
USER appuser

# Cloud Run injects $PORT; use tini as init, then uvicorn
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
