# ---------- Build stage: download model snapshot ----------
FROM python:3.11-slim AS downloader

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /tmp

# Certs para HTTPS en el build stage (por si tu base viene muy mínima)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir "huggingface_hub>=0.25"

# Token opcional en build-time (útil si el repo o los assets requieren auth o para evitar rate limits)
ARG HF_TOKEN=""
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

# Descarga el snapshot a una carpeta fija dentro del stage
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
repo_id = "Qwen/Qwen2-VL-2B-Instruct"
local_dir = "/opt/models/Qwen2-VL-2B-Instruct"
token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=token,
)
print(">> Snapshot downloaded to:", local_dir)
PY


# ---------- Runtime stage: clean image, no token, no internet needed ----------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Paquetes mínimos del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# Dependencias de la app (asegúrate de que coincidan con tu código)
# Ideal: fijar versiones en requirements.txt, p.ej.:
# transformers==4.56.2
# torch>=2.3.0
# huggingface_hub>=0.23.0
# fastapi
# uvicorn[standard]
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

# Tu aplicación
COPY main.py /app/

# Copia el modelo horneado desde el stage de descarga
COPY --from=downloader /opt/models /models

# Por seguridad, usa un usuario no root
RUN useradd -m appuser && chown -R appuser:appuser /app /models
USER appuser

# Entorno de la app — apunta al modelo horneado y desactiva descargas
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct \
    ALLOW_DOWNLOAD=0 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PORT=8080

EXPOSE 8080

# Comando de arranque
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
