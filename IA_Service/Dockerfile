# Dockerfile — Cloud Run (Qwen2-VL CPU)
FROM python:3.11-slim

# Buenas prácticas y descargas HF más rápidas/robustas
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# Paquetes mínimos del sistema (certs + wget) y actualizar pip
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# Instala dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Código de la app
COPY main.py /app/

# Cloud Run inyecta PORT; no lo hardcodees
ENV PORT=8080
EXPOSE 8080

# Ejecuta uvicorn usando el PORT en runtime
CMD ["bash", "-lc", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
