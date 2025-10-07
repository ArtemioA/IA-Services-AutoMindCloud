# ---------- Runtime image ----------
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . /app

# (OPCIONAL) no necesitas fijar PORT aquí; Cloud Run lo inyecta.
# ENV PORT=8080

# ⬇️ Usa bash para expandir ${PORT}
CMD ["bash","-lc","python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
