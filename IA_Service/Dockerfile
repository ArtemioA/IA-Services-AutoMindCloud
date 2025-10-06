FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git tini procps dos2unix && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import transformers, huggingface_hub, torch
print("✅ deps ok:", transformers.__version__, huggingface_hub.__version__, torch.__version__)
PY

# Modelo horneado
ARG MODEL_REPO=Qwen/Qwen2-VL-2B-Instruct
ARG MODEL_REV=main
ENV MODEL_DIR=/models/Qwen2-VL-2B-Instruct
RUN mkdir -p "${MODEL_DIR}"

# 🔥 Trae SIEMPRE pesos y configs (soporta ambos layouts: único archivo o shards)
RUN huggingface-cli download "${MODEL_REPO}" --revision "${MODEL_REV}" \
    --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False \
    --include "config.json" \
             "generation_config.json" \
             "model.safetensors" \
             "model-*.safetensors" \
             "pytorch_model.bin" \
             "pytorch_model-*.bin" \
             "tokenizer.json" \
             "tokenizer_config.json" \
             "special_tokens_map.json" \
             "vocab.json" \
             "merges.txt" \
             "image_processor/*" \
             "processor_config.json" \
  && ( test -f "${MODEL_DIR}/model.safetensors" \
    || ls "${MODEL_DIR}"/model-*.safetensors 1>/dev/null 2>&1 \
    || test -f "${MODEL_DIR}/pytorch_model.bin" \
    || ls "${MODEL_DIR}"/pytorch_model-*.bin 1>/dev/null 2>&1 ) \
  || (echo "❌ No se encontraron pesos del modelo en ${MODEL_DIR}" && ls -la "${MODEL_DIR}" && exit 1)

# App
COPY main.py .
RUN dos2unix -q /app/main.py || true

ENV HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
