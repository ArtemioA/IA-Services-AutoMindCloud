# ---------- Build stage: instalar dependencias y descargar el modelo ----------
FROM python:3.11-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Pip al día
RUN python -m pip install --upgrade pip

# Instala dependencias de Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# (Opcional) si el repo requiere token privado:
# ARG HUGGINGFACE_HUB_TOKEN
# ENV HUGGINGFACE_HUB_TOKEN=$HUGGINGFACE_HUB_TOKEN

# ↓↓↓ Hornear snapshot del modelo dentro de la imagen (no depende de CDN en runtime)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    "Qwen/Qwen2-VL-2B-Instruct",
    local_dir="/app/models/Qwen2-VL-2B-Instruct",
    local_dir_use_symlinks=False
)
PY


# ---------- Runtime stage: imagen limpia ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=8080 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct \
    MODEL_DIR=/app/models/Qwen2-VL-2B-Instruct \
    ALLOW_DOWNLOAD=0

WORKDIR /app

# Paquetes mínimos del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && python -m pip install --upgrade pip

# Copia dependencias ya instaladas
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia el snapshot horneado del modelo
COPY --from=builder /app/models /app/models

# Copia tu aplicación (si tienes más archivos, usa COPY . .)
COPY main.py /app/

# Seguridad: usuario no root
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Arranque del servicio
CMD ["sh", "-lc", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]

