# ---------- Base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------- Deps ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Args para elegir modelo/revisión en build ----------
ARG MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REV=main

# ---------- Descarga del modelo (horneado) ----------
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct
RUN mkdir -p "${MODEL_DIR}"

# Escribimos un script pequeño para descargar el snapshot
RUN printf '%s\n' \
"import os" \
"from huggingface_hub import snapshot_download" \
"repo = os.environ.get('MODEL_REPO', 'Qwen/Qwen2-VL-2B-Instruct')" \
"rev  = os.environ.get('MODEL_REV', 'main')" \
"dst  = os.environ.get('MODEL_DIR', '/models/Qwen2-VL-2B-Instruct')" \
"snapshot_download(repo_id=repo, revision=rev, local_dir=dst, local_dir_use_symlinks=False)" \
"print('✅ Modelo descargado en:', dst)" \
> /app/download_model.py

RUN python /app/download_model.py && rm /app/download_model.py

# ---------- App ----------
COPY main.py .

# Caches (inofensivos en runtime)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub

ENV PORT=8080
EXPOSE 8080
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
