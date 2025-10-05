# main.py — Qwen2-VL horneado (Cloud Run listo y probado)
import os, threading, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = FastAPI(title="Qwen2-VL (baked)")

# --- Configuración básica ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))

# Caches volátiles (no descarga nada en runtime)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# Limitar threads en CPU
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_state = {
    "ready": False,
    "error": None,
    "device": "cpu",
    "processor": None,
    "model": None
}
_lock = threading.Lock()

class Ask(BaseModel):
    prompt: str


# --- Carga perezosa del modelo local horneado ---
def _lazy_load():
    with _lock:
        if _state["ready"] or _state["error"]:
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _state["device"] = device

            # ✅ Cargar modelo desde ruta local absoluta
            model_path = os.path.abspath(MODEL_DIR)
            print(f"🔍 Loading local model from: {model_path}")
            if not os.path.isdir(model_path):
                raise RuntimeError(f"Model directory not found: {model_path}")

            proc = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            model.to("cpu").eval()
            _state["processor"] = proc
            _state["model"] = model
            _state["ready"] = True
            print("✅ Modelo cargado correctamente.")
        except Exception as e:
            _state["error"] = f"{type(e).__name__}: {e}"
            print(f"❌ Error al cargar modelo: {_state['error']}")


# --- Rutas FastAPI ---
@app.get("/")
def root():
    # Inicia carga si aún no está listo
    if not _state["ready"] and not _state["error"]:
        threading.Thread(target=_lazy_load, daemon=True).start()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("error" if _state["error"] else "starting"),
        "device": _state["device"],
        "model_dir": MODEL_DIR,
        "error": _state["error"]
    }


@app.get("/healthz")
def healthz():
    """Endpoint para verificar estado de carga"""
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}


@app.post("/generate")
def generate(q: Ask):
    """Genera texto a partir de un prompt"""
    if not _state["ready"]:
        if _state["error"]:
            raise HTTPException(500, f"Load error: {_state['error']}")
        _lazy_load()
        # espera breve (hasta ~5s) por la carga inicial
        for _ in range(50):
            if _state["ready"] or _state["error"]:
                break
            time.sleep(0.1)
        if not _state["ready"]:
            if _state["error"]:
                raise HTTPException(500, f"Load error: {_state['error']}")
            raise HTTPException(503, "Model loading, try again shortly.")

    proc = _state["processor"]
    model = _state["model"]

    # --- Generación simple ---
    inputs = proc(text=q.prompt, return_tensors="pt")
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    text = proc.batch_decode(tokens, skip_special_tokens=True)[0]

    return {"ok": True, "text": text}


# --- Diagnóstico DNS (opcional) ---
@app.get("/_dns")
def dns_check():
    import socket
    hosts = ["huggingface.co", "cdn-lfs.huggingface.co", "google.com"]
    out = {}
    for h in hosts:
        try:
            out[h] = [ai[4][0] for ai in socket.getaddrinfo(h, 443)]
        except Exception as e:
            out[h] = f"ERROR: {e}"
    return out


# --- Entrada principal (para pruebas locales) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
