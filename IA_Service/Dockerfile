# Dockerfile — CPU, modelo horneado
FROM python:3.11-slim

# 1) Dependencias mínimas del sistema
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 2) Caches SOLO durante build (para bajar el modelo una sola vez)
ENV HF_HOME=/models_cache \
    TRANSFORMERS_CACHE=/models_cache/transformers \
    HUGGINGFACE_HUB_CACHE=/models_cache/hub \
    PIP_NO_CACHE_DIR=1

# 3) Librerías (CPU)
RUN pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic==2.* \
    transformers==4.44.2 huggingface_hub==0.23.5 pillow==10.* \
    torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

# 4) (Opcional) token para evitar 429 durante el build
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 5) Descarga (horneado) del modelo a /models (queda dentro de la imagen)
RUN python - <<'PY'
from huggingface_hub import snapshot_download
import os
repo = "Qwen/Qwen2-VL-2B-Instruct"
snapshot_download(
    repo_id=repo,
    local_dir="/models/Qwen2-VL-2B-Instruct",
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None
)
print("✅ Modelo descargado en /models/Qwen2-VL-2B-Instruct")
PY

# 6) Copia tu app
WORKDIR /app
COPY main.py /app/main.py

# 7) En runtime, caches volátiles a /tmp (no descargará nada)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    MODEL_DIR=/models/Qwen2-VL-2B-Instruct \
    PORT=8080

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

