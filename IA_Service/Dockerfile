# ---------- Build stage: descarga snapshot ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir "huggingface_hub>=0.23.0"

# Cambiable en build si lo necesitas: --build-arg MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REPO="Qwen/Qwen2-VL-2B-Instruct"
ENV MODEL_OUT="/app/models/Qwen2-VL-2B-Instruct"

# Descarga snapshot al directorio final dentro de la imagen
RUN python -c "from huggingface_hub import snapshot_download; import os, pathlib; repo=os.environ.get('MODEL_REPO','Qwen/Qwen2-VL-2B-Instruct'); out=os.environ.get('MODEL_OUT','/app/models/Qwen2-VL-2B-Instruct'); pathlib.Path(out).mkdir(parents=True, exist_ok=True); p=snapshot_download(repo_id=repo, local_dir=out, local_dir_use_symlinks=False); print('>> Snapshot at:', p)"

# Verificación dura (falla el build si no existe)
RUN test -d \"$MODEL_OUT\" && find \"$MODEL_OUT\" -maxdepth 2 -type f | head -n 20 && du -h -d1 /app/models

# ---------- Runtime image ----------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 10001 appuser \
 && mkdir -p /tmp/hf/transformers /tmp/hf/hub \
 && chown -R appuser:appuser /tmp /app

# Requisitos de ejecución
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Código app
COPY . /app

# Snapshot horneado desde el stage builder
COPY --from=builder /app/models /app/models

# Verificación visible
RUN ls -lah /app/models && du -h -d1 /app/models && test -d /app/models/Qwen2-VL-2B-Instruct

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
