# ====================== BUILD STAGE: descarga el snapshot ======================
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
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

# Puedes cambiar el repo al construir:  --build-arg MODEL_REPO=Qwen/Qwen2-VL-7B-Instruct
ARG MODEL_REPO="Qwen/Qwen2-VL-2B-Instruct"
ENV MODEL_OUT="/app/models/Qwen2-VL-2B-Instruct"

# Descarga del snapshot (sin here-docs Docker; una sola línea Python)
RUN python -c "from huggingface_hub import snapshot_download; import os, pathlib; \
repo=os.environ.get('MODEL_REPO'); out=os.environ.get('MODEL_OUT'); \
pathlib.Path(out).mkdir(parents=True, exist_ok=True); \
snapshot_download(repo_id=repo, local_dir=out); \
print('>> BAKED TO:', out)"

# Verificación estricta: falla si está vacío
RUN test -d "$MODEL_OUT" && [ -n \"$(ls -A \"$MODEL_OUT\" 2>/dev/null)\" ] \
 || (echo '❌ snapshot vacío en $MODEL_OUT' && exit 1)


# ====================== RUNTIME STAGE: app offline ======================
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
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

# ===== Dependencias RUNTIME (exactas que pediste) =====
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

# PyTorch CPU + torchvision (índice CPU oficial)
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1+cpu \
      torchvision==0.18.1+cpu

# Código de la app
COPY . /app

# Copia el snapshot horneado
COPY --from=builder /app/models /app/models

# Comprobación visible y permisos
RUN ls -la /app/models/Qwen2-VL-2B-Instruct | head -n 50 \
 && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
