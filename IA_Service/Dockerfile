cd ~/app-src/IA_Service || exit 1

# Backup del Dockerfile roto, por si acaso
cp -f Dockerfile Dockerfile.bak 2>/dev/null || true

# Crear Dockerfile nuevo y limpio (sin CRLF)
cat > Dockerfile <<'DOCKERFILE'
# ---------- Build stage: descarga snapshot ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# Sistema mínimo
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Herramientas para snapshot_download (tus versiones)
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      "huggingface-hub>=0.34.0,<1.0" \
      "hf-transfer==0.1.9"

# Permite sobreescribir el repo en build si quieres: --build-arg MODEL_REPO=...
ARG MODEL_REPO="Qwen/Qwen2-VL-2B-Instruct"
ENV MODEL_OUT="/app/models/Qwen2-VL-2B-Instruct"

# Descarga el snapshot al path final dentro de la imagen
RUN python - <<'PY'
import os, pathlib
from huggingface_hub import snapshot_download
repo = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
out  = os.environ.get("MODEL_OUT", "/app/models/Qwen2-VL-2B-Instruct")
pathlib.Path(out).mkdir(parents=True, exist_ok=True)
p = snapshot_download(repo_id=repo, local_dir=out)
print(">> Snapshot at:", p, flush=True)
PY

# Listado (informativo)
RUN echo ">> Model files (sample):" \
 && find "$MODEL_OUT" -maxdepth 2 -type f | head -n 40 || true \
 && echo ">> Disk usage:" && du -h -d1 /app/models || true


# ---------- Runtime image ----------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Sistema + tini
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

# Usuario no-root y cachés escribibles
RUN useradd -m -u 10001 appuser \
 && mkdir -p /tmp/hf/transformers /tmp/hf/hub \
 && chown -R appuser:appuser /tmp /app

# ==== Dependencias de RUNTIME (exactas como pediste) ====
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
      fastapi==0.118.0 \
      "uvicorn[standard]==0.37.0" \
      transformers \
      accelerate \
      Pillow \
      "safetensors>=0.4.3" \
      "huggingface-hub>=0.34.0,<1.0" \
      pydantic==2.11.9 \
      requests==2.32.4 \
      hf-transfer==0.1.9

# PyTorch CPU + torchvision (extra index oficial)
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1+cpu \
      torchvision==0.18.1+cpu

# Código de la app
COPY . /app

# Copia el snapshot horneado
COPY --from=builder /app/models /app/models

# Verificación visible
RUN ls -lah /app/models && du -h -d1 /app/models && test -d /app/models/Qwen2-VL-2B-Instruct

# Permisos y usuario
RUN chown -R appuser:appuser /app
USER appuser

# Entrypoint
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
DOCKERFILE

# Asegura fin de línea UNIX (por si se colaron CRLF)
command -v dos2unix >/dev/null 2>&1 || { sudo apt-get update -y && sudo apt-get install -y dos2unix; }
dos2unix Dockerfile >/dev/null 2>&1 || true

# Muestra las primeras 80 líneas para validar que no haya 'CAT', 'IMPORT', etc
nl -ba Dockerfile | sed -n '1,80p'
