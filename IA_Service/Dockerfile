# ---- Build base ----
FROM python:3.11-slim

# Variables de entorno para Cloud Run
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    HF_HOME=/tmp/hf \
    TRANSFORMERS_CACHE=/tmp/hf/transformers \
    HUGGINGFACE_HUB_CACHE=/tmp/hf/hub \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Crear directorio app
WORKDIR /app

# Copiar requirements y instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY . .

# Exponer puerto
EXPOSE 8080

# Comando de inicio
CMD ["bash", "-lc", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
