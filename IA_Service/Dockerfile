# Dockerfile — Qwen2-VL API (modelo horneado, CPU, Cloud Run)
FROM python:3.11-slim

# --- 1) Dependencias de sistema mínimas ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# --- 2) Ajustes base Python ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- 3) Librerías Python (pines probados en CPU) ---
# Nota: torch CPU desde el índice de PyTorch
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    pydantic==2.9.2 \
    requests==2.32.3 \
    numpy==1.26.4 \
    pillow==10.4.0 \
    torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
    transformers==4.43.3 \
    huggingface_hub==0.24.6 \
    accelerate==0.33.0 \
    tokenizers==0.19.1 \
    safetensors==0.4.3 \
    sentencepiece==0.2.0

# --- 4) (Opcional) Token HF para evitar límites durante el build ---
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# --- 5) Descargar/HORNEAR el modelo dentro de la imagen + verificar ---
# (Usamos una carpeta fija dentro de la imagen)
RUN mkdir -p /models/Qwen2-VL-2B-Instruct && \
    python - <<'PY'
from huggingface_hub import snapshot_download
import os, sys
repo = "Qwen/Qwen2-VL-2B-Instruct"
target = "/models/Qwen2-VL-2B-Instruct"
print(f"[build] Descargando {repo} -> {target}")
snapshot_download(
    repo_id=repo,
    local_dir=target,
    local_dir_use_symlinks=False,
    token=os.environ.get("HF_TOKEN") or None
)
# Archivos mínimos esperados
need = [
  "config.json", "generation_config.json", "special_tokens_map.json",
  "tokenizer.json", "tokenizer.model"
]
missing = [f for f in need if not os.path.exists(os.path.join(target, f))]
if missing:
    print("[build] ❌ Faltan archivos críticos:", missing)
    sys.exit(2)
print("[build] ✅ Modelo descargado OK")
PY
# Listado (útil en logs de build)
RUN ls -lah /models/Qwen2-VL-2B-Instruct | head -n 80

# --- 6) App FastAPI ---
WORKDIR /app
COPY main.py /app/main.py

# --- 7) Variables para el runtime ---
# En runtime usamos caches volátiles en /tmp para no persistir nada
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# --- 8) Entrypoint con verificación de modelo en runtime ---
COPY --chmod=755 <<'SH' /entry.sh
#!/usr/bin/env bash
set -euo pipefail
echo "[entry] PORT=${PORT:-8080}"
echo "[entry] MODEL_DIR=${MODEL_DIR:-/models/Qwen2-VL-2B-Instruct}"
if [ ! -d "${MODEL_DIR:-/models/Qwen2-VL-2B-Instruct}" ]; then
  echo "[entry] ❌ No existe ${MODEL_DIR:-/models/Qwen2-VL-2B-Instruct}"
  echo "[entry] Contenido de / y /models (debug):"
  ls -lah / || true
  ls -lah /models || true
  exit 10
fi
echo "[entry] ✅ Encontrado ${MODEL_DIR}"
ls -lah "${MODEL_DIR}" | head -n 80 || true
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"
SH

EXPOSE 8080
CMD ["/entry.sh"]

