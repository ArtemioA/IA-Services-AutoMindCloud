# Imagen base
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Dependencias del sistema (seguras y livianas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- 1) Instalar deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- 2) Descargar el modelo en build (horneado) ----------
# Config por ARG para que puedas cambiar repo/rev sin tocar Dockerfile
ARG MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REV=main

# Carpeta donde quedará el modelo horneado
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct
RUN mkdir -p "${MODEL_DIR}"

# Acelera descargas (opcional) y evita symlinks para que /models tenga archivos reales
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Descarga en tiempo de build (sin tocar $HOME)
# Nota: usamos local_dir y local_dir_use_symlinks=False
RUN python - << 'PY'
import os
from huggingface_hub import snapshot_download
repo = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
rev  = os.environ.get("MODEL_REV", "main")
local_dir = os.environ.get("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
snapshot_download(
    repo_id=repo,
    revision=rev,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print("✅ Modelo descargado en:", local_dir)
PY

# ---------- 3) Copiar app y configurar runtime ----------
COPY main.py .

# Caches (no se usarán si ya está horneado, pero son seguros en Cloud Run)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub

# Expone el puerto que Cloud Run inyecta en $PORT (por defecto 8080)
ENV PORT=8080
EXPOSE 8080

# Entrypoint (no uses --reload en producción)
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
