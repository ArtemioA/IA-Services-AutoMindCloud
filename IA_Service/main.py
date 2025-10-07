# main.py — FastAPI en Cloud Run con /healthz, /readyz y carga perezosa
import os
import threading
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ----------------- Config -----------------
ALLOW_DOWNLOAD = os.getenv("ALLOW_DOWNLOAD", "0") == "1"
FORCE_ONLINE   = os.getenv("FORCE_ONLINE", "0") == "1"
MODEL_REPO     = os.getenv("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR      = os.getenv("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")

# Sugerido para cache HF en /tmp (efímero pero RW):
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# ----------------- App -----------------
app = FastAPI()
_model = {"ready": False, "error": None, "pipe": None}
_lock = threading.Lock()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.2

def load_model_background():
    """Carga el modelo en un hilo aparte para no bloquear el arranque."""
    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        import torch

        # Si tienes CPU, dtype puede ser float32. Ajusta si usas quant.
        dtype = torch.float32

        # Modo offline si no se permite descargar
        local_files_only = not (ALLOW_DOWNLOAD or FORCE_ONLINE)

        # Preferir snapshot local si existe
        model_path = MODEL_DIR if os.path.exists(MODEL_DIR) else MODEL_REPO

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

        with _lock:
            _model["processor"] = processor
            _model["pipe"] = model
            _model["ready"] = True
            _model["error"] = None
    except Exception as e:
        with _lock:
            _model["ready"] = False
            _model["error"] = f"{type(e).__name__}: {e}"

# Arranca el hilo de carga apenas se inicia el proceso
threading.Thread(target=load_model_background, daemon=True).start()

# ----------------- Health -----------------
@app.get("/healthz")
def healthz():
    # Salud del proceso web (no del modelo)
    return {"ok": True}

@app.get("/readyz")
def readyz():
    # Listo solo si el modelo está cargado
    if _model["ready"]:
        return {"ready": True}
    return {"ready": False, "error": _model["error"]}, 503

@app.get("/_netcheck")
def netcheck():
    # Chequeo simple de red saliente (sin dependencias externas fuertes)
    return {"egress": "unknown-but-process-alive"}

# ----------------- Inference -----------------
@app.post("/generate")
def generate(req: GenRequest):
    if not _model["ready"]:
        # Expón el motivo si hubo error
        err = _model["error"]
        raise HTTPException(status_code=503, detail="Model not ready" + (f": {err}" if err else ""))

    # Aquí va una demo minimal de texto (adaptar a VL según lo que uses).
    # Para Qwen2-VL real, debes componer inputs con imágenes si corresponde.
    try:
        model = _model["pipe"]
        processor = _model["processor"]

        # Prompt -> tokens (texto puro)
        inputs = processor(text=req.prompt, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens or 64,
            do_sample=(req.temperature or 0) > 0,
            temperature=req.temperature or 0.0,
        )
        # Decodifica
        text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}: {e}")
