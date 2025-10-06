# ---------- Base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl tini procps file dos2unix && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------- Deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import transformers, huggingface_hub, torch; print('✅ deps ok', transformers.__version__)"

# ---------- Args para elegir modelo/revisión en build ----------
ARG MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REV=main

# ---------- Descarga del modelo (horneado) ----------
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct
RUN mkdir -p "${MODEL_DIR}"

# Escribimos script de descarga (evita heredoc)
RUN printf '%s\n' \
"import os, sys" \
"from huggingface_hub import snapshot_download" \
"repo=os.environ.get('MODEL_REPO','Qwen/Qwen2-VL-2B-Instruct')" \
"rev=os.environ.get('MODEL_REV','main')" \
"dst=os.environ.get('MODEL_DIR','/models/Qwen2-VL-2B-Instruct')" \
"print('Descargando', repo, '@', rev, '->', dst)" \
"p=snapshot_download(repo_id=repo, revision=rev, local_dir=dst, local_dir_use_symlinks=False)" \
"print('✅ Snapshot en:', p)" \
> /app/download_model.py

RUN python /app/download_model.py && rm /app/download_model.py && \
    # sanity: debe existir al menos config.json o model_index.json
    (test -f "${MODEL_DIR}/config.json" || test -f "${MODEL_DIR}/model_index.json") && \
    echo "✅ Modelo horneado en ${MODEL_DIR}"

# ---------- App ----------
COPY main.py .
# Normaliza finales de línea por si llegaron en CRLF
RUN dos2unix -q /app/main.py || true

# Caches (inofensivos en runtime)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PORT=8080
EXPOSE 8080

# tini como init para señales limpias
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
