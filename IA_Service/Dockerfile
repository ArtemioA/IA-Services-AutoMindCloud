# ============================================================
# Dockerfile — Cloud Run (Qwen2-VL CPU, FastBoot)
# ============================================================

# Base: lightweight Python with pip
FROM python:3.11-slim

# Disable Python buffering for better Cloud Run logs
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# ------------------------------------------------------------
# System deps: git, wget (needed by HF Hub)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install core dependencies
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    fastapi uvicorn pydantic \
    torch transformers==4.44.0 \
    accelerate safetensors \
    huggingface-hub

# Enable Hugging Face optimized file transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# ------------------------------------------------------------
# Copy application files
# ------------------------------------------------------------
COPY main.py /app/

# (Optional) if you have a start.sh script — else skip
# COPY start.sh /app/
# RUN chmod +x /app/start.sh

# ------------------------------------------------------------
# Cloud Run reads PORT from env
# ------------------------------------------------------------
ENV PORT=8080
EXPOSE 8080

# ------------------------------------------------------------
# Entry point (use python -m to ensure correct path)
# ------------------------------------------------------------
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
