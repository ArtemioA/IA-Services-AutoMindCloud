# main.py — ejemplo mínimo con carga local
import os, threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = FastAPI(title="Qwen2-VL (baked)")

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
PORT = int(os.environ.get("PORT", "8080"))

# Caches volátiles (no descarga nada si ya está horneado)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

_state = {"ready": False, "error": None, "device": "cpu", "processor": None, "model": None}
_lock = threading.Lock()

class Ask(BaseModel):
    prompt: str

def _lazy_load():
    with _lock:
        if _state["ready"] or _state["error"]:
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _state["device"] = device
            _state["processor"] = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
            _state["model"] = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            _state["model"].to("cpu")
            _state["ready"] = True
        except Exception as e:
            _state["error"] = str(e)

@app.get("/")
def root():
    if not _state["ready"] and not _state["error"]:
        threading.Thread(target=_lazy_load, daemon=True).start()
    return {"ok": True, "status": "starting" if not _state["ready"] else "ready",
            "device": _state["device"], "model_dir": MODEL_DIR, "error": _state["error"]}

@app.post("/generate")
def generate(q: Ask):
    if not _state["ready"]:
        if _state["error"]:
            raise HTTPException(500, f"Load error: {_state['error']}")
        _lazy_load()
        if not _state["ready"]:
            raise HTTPException(503, "Model loading, try again in a moment.")
    # Demo mínimo de texto (Qwen2-VL también acepta imágenes si amplías el payload)
    processor = _state["processor"]
    model = _state["model"]
    inputs = processor(text=q.prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"ok": True, "text": text}

# (Opcional) endpoint de diagnóstico DNS si quieres
@app.get("/_dns")
def dns_check():
    import socket
    hosts = ["huggingface.co", "cdn-lfs.huggingface.co", "google.com"]
    out = {}
    for h in hosts:
        try:
            out[h] = socket.getaddrinfo(h, 443)
        except Exception as e:
            out[h] = f"ERROR: {e}"
    return out
