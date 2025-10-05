# Dockerfile — Cloud Run (Qwen2-VL CPU)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

# System tiny base (optional but handy)
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps from your requirements.txt
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# App code
COPY main.py /app/

# Cloud Run port
ENV PORT=8080
EXPOSE 8080

# Launch
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
