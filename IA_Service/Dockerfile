# ---------- Build stage: download model snapshot ----------
FROM python:3.11-slim AS downloader
WORKDIR /tmp
RUN python -m pip install --upgrade pip && pip install --no-cache-dir "huggingface_hub>=0.25"

# If you have a HF token in Secret Manager, you can pass it via --build-arg HF_TOKEN=...
ARG HF_TOKEN=""
ENV HUGGINGFACE_HUB_TOKEN=${HF_TOKEN}

RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Qwen/Qwen2-VL-2B-Instruct",
    local_dir="/opt/models/Qwen2-VL-2B-Instruct",
    local_dir_use_symlinks=False,
    token=None if "${HUGGINGFACE_HUB_TOKEN}"=="" else "${HUGGINGFACE_HUB_TOKEN}",
)
PY

# ---------- Runtime stage: clean image, no token, no internet needed ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app

# Minimal system deps and latest pip
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# App deps (make sure these match your code)
# If you already have requirements.txt, keep it; otherwise use the versions below
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Your app
COPY main.py /app/

# Copy the baked model from the first stage
COPY --from=downloader /opt/models /models

# App env — point to baked model and disable downloads
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct ALLOW_DOWNLOAD=0 PORT=8080
EXPOSE 8080

CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
