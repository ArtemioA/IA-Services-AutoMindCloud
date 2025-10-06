# ---------- Build stage with model pre-download ----------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Copia requirements.txt e instala dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Pre-descarga del modelo durante el build
RUN python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('/app/models', exist_ok=True)
snapshot_download(
    repo_id='Qwen/Qwen2-VL-2B-Instruct',
    local_dir='/app/models/Qwen2-VL-2B-Instruct',
    local_dir_use_symlinks=False
)
print('Modelo descargado exitosamente')
"

# ---------- Runtime stage ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Paquetes mínimos del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copia dependencias y modelo pre-descargado
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models /app/models

# Copia tu aplicación
COPY main.py /app/

# Seguridad: usuario no root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Configuración para usar modelo local
ENV MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct \
    MODEL_DIR=/app/models/Qwen2-VL-2B-Instruct \
    ALLOW_DOWNLOAD=0 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PORT=8080

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
