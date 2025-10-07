# ---------- Runtime image ----------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps (cache-friendly)
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

# Copy app code
COPY . /app

# Cloud Run provides $PORT
ENV PORT=8080

# Exec form (no shell); binds 0.0.0.0:$PORT
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
