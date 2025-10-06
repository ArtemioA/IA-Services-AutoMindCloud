# ---------- Base ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tini procps dos2unix && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# ---------- Dependencias ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import transformers, huggingface_hub, torch
print("✅ deps ok:", transformers.__version__, huggingface_hub.__version__, torch.__version__)
PY

# ---------- Vars de modelo ----------
ARG MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REV=main
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct
RUN mkdir -p "${MODEL_DIR}"

# ---------- Hornear el modelo con la CLI ----------
# NOTA: --local-dir-use-symlinks False para copiar archivos reales (no symlinks)
RUN huggingface-cli download "${MODEL_REPO}" --revision "${MODEL_REV}" \
    --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False && \
    test -f "${MODEL_DIR}/config.json" && echo "✅ Modelo horneado en ${MODEL_DIR}" || (echo "❌ Falta config.json" && ls -la "${MODEL_DIR}" && exit 1)

# ---------- App ----------
COPY main.py .
RUN dos2unix -q /app/main.py || true

# Caches “inofensivos” en runtime (no se usarán si todo es local)
ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PORT=8080
EXPOSE 8080

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
