FROM python:3.11-slim

WORKDIR /app

# Dependencias básicas del sistema
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (mejor cacheo)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Entorno (Cloud Run-friendly)
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

EXPOSE 8080

# Arranque del servidor
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
