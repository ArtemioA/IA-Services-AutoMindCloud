# main.py — FastAPI en Cloud Run con /healthz, /status, /readyz y carga perezosa
import os
import threading
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ----------------- Config -----------------
ALLOW_DOWNLOAD = os.getenv("ALLOW_DOWNLOAD", "0") == "1"
FORCE_ONLINE   = os.getenv("FORCE_ONLINE", "0") == "1"
MODEL_REPO     = os.getenv("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR      = os.getenv("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")

# Caches HF en /tmp (RW en Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# ----------------- App & Estado -----------------
app = FastAPI(title="Qwen2-VL CPU Demo", version="1.0")

# Estado compartido del modelo
_model = {
    "ready": False,
    "error": None,
    "processor": None,
    "model": None,   # CausalLM
}
_lock = threading.Lock()

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.2

def _resolve_model_path() -> str:
    """Devuelve la ruta local si existe, si no el repo de HF."""
    try_path = MODEL_DIR
    if try_path and os.path.exists(try_path):
        return try_path
    return MODEL_REPO

def _load_model_background():
    """Carga el modelo en un hilo aparte para no bloquear el arranque."""
    try:
        # Imports dentro del hilo para evitar costo en arranque
        import torch
        from transformers import AutoProcessor, AutoModelForCausalLM

        # CPU-only settings
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "1")))
        dtype = torch.float32

        local_files_only = not (ALLOW_DOWNLOAD or FORCE_ONLINE)
        model_path = _resolve_model_path()

        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )

        with _lock:
            _model["processor"] = processor
            _model["model"] = model
            _model["ready"] = True
            _model["error"] = None
    except Exception as e:
        with _lock:
            _model["ready"] = False
            _model["error"] = f"{type(e).__name__}: {e}"

# Arranca la carga perezosa apenas inicia el proceso (una sola vez)
threading.Thread(target=_load_model_background, daemon=True).start()

# ----------------- Salud / Ready -----------------
@app.get("/healthz")
def healthz():
    # Salud del proceso web (no del modelo)
    return {"ok": True}

@app.get("/status")
def status():
    # Alias de salud (por si /healthz recibe tratamiento especial en el edge)
    return {"ok": True}

@app.get("/readyz")
def readyz():
    # Listo solo si el modelo está cargado
    with _lock:
        ready = bool(_model["ready"] and _model["processor"] and _model["model"])
        err = _model["error"]
    if ready:
        return {"ready": True, "error": None}
    else:
        return JSONResponse({"ready": False, "error": err}, status_code=503)

@app.get("/_netcheck")
def netcheck():
    # Señal mínima de que el proceso está vivo
    return {"egress": "process-alive"}

# ----------------- Inference -----------------
@app.post("/generate")
def generate(req: GenRequest):
    with _lock:
        ready = bool(_model["ready"] and _model["processor"] and _model["model"])
        err = _model["error"]
        processor = _model["processor"]
        model = _model["model"]

    if not ready:
        detail = "Model not ready"
        if err:
            detail += f": {err}"
        raise HTTPException(status_code=503, detail=detail)

    try:
        import torch
        # Para texto puro (sin imagen). Para VL real, compón inputs con imágenes también.
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            inputs = processor(text=req.prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
        else:
            # Fallback: intenta usar tokenizer del processor si expone método
            inputs = processor(text=req.prompt, return_tensors="pt")
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")

        gen_kwargs = {
            "max_new_tokens": req.max_new_tokens or 64,
        }
        # Sampling solo si temperature > 0
        if (req.temperature or 0) > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": float(req.temperature),
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
            })

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        # Decodificar (vía tokenizer del processor)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Processor does not expose a tokenizer for decoding.")
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"text": text}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}: {e}")
