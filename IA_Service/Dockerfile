FROM python:3.11-slim

WORKDIR /app

# Dependencias mínimas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código
COPY main.py .

# Entorno (Cloud Run friendly)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 8080

CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
