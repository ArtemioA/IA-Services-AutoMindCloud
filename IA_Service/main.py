# main.py — Cloud Run (Qwen2-VL-2B-Instruct CPU) con readiness consistente
import os, socket, threading, time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ========= Estado global =========
model_ready = threading.Event()
last_error: Optional[str] = None
model = None
processor = None

# ========= Config por entorno =========
ALLOW_DOWNLOAD = os.getenv("ALLOW_DOWNLOAD", "0") == "1"
FORCE_ONLINE   = os.getenv("FORCE_ONLINE",   "0") == "1"
MODEL_REPO     = os.getenv("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR      = os.getenv("MODEL_DIR",  "/tmp/models/Qwen2-VL-2B-Instruct")
HF_HOME        = os.getenv("HF_HOME", "/tmp/hf")
PORT           = int(os.getenv("PORT", "8080"))

# ======== App =========
app = FastAPI(title="Qwen2-VL CPU demo", version="1.0")

# ======== Utilidades =========
def _set_error(msg: str):
    global last_error
    last_error = msg
    model_ready.clear()
    print(f"[LOAD][ERROR] {msg}", flush=True)

def _ok(msg: str):
    print(f"[LOAD][OK] {msg}", flush=True)

def _have_local_snapshot(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())

# ======== Carga del modelo (hilo de fondo) =========
def _load_model():
    global model, processor
    try:
        # Ruta 1: FORZAR online (descarga)
        if FORCE_ONLINE:
            if not ALLOW_DOWNLOAD:
                _set_error("FORCE_ONLINE=1 pero ALLOW_DOWNLOAD=0: impedida la descarga.")
                return
            _ok(f"FORCE_ONLINE=1: descargando {MODEL_REPO} en HF_HOME={HF_HOME}")
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
            _ok(f"Snapshot listo en {local_dir}")

        # Ruta 2: si no forzamos online, intentamos local y opcionalmente online si ALLOW_DOWNLOAD=1
        if not _have_local_snapshot(MODEL_DIR):
            if ALLOW_DOWNLOAD:
                _ok(f"No hay snapshot local en {MODEL_DIR}. Descargando {MODEL_REPO}...")
                from huggingface_hub import snapshot_download
                local_dir = snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
                _ok(f"Snapshot listo en {local_dir}")
            else:
                _set_error(f"Modelo no encontrado en {MODEL_DIR} y ALLOW_DOWNLOAD=0.")
                return
        else:
            _ok(f"Usando snapshot local en {MODEL_DIR}")

        # Carga en Transformers (CPU)
        from transformers import AutoProcessor, AutoModelForVision2Seq
        # Qwen2-VL requiere trust_remote_code=True
        processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_DIR, trust_remote_code=True
        )
        model.to("cpu")
        model.eval()

        model_ready.set()
        _ok("Modelo listo.")
    except Exception as e:
        _set_error(f"Load error: {e!s}")

# ======== Startup / Shutdown =========
@app.on_event("startup")
def on_startup():
    print("[BOOT] Vars:", {
        "ALLOW_DOWNLOAD": ALLOW_DOWNLOAD,
        "FORCE_ONLINE": FORCE_ONLINE,
        "MODEL_REPO": MODEL_REPO,
        "MODEL_DIR": MODEL_DIR,
        "HF_HOME": HF_HOME,
    }, flush=True)
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()

# ======== Schemas =========
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = 64

# ======== Endpoints =========
@app.get("/status")
def status():
    return {"ok": True}

@app.get("/readyz")
def readyz():
    return {"ready": model_ready.is_set(), "error": last_error}

@app.get("/_netcheck")
def netcheck():
    # Comprobación simple hacia Hugging Face
    import http.client
    try:
        conn = http.client.HTTPSConnection("huggingface.co", timeout=3)
        conn.request("HEAD", "/")
        r = conn.getresponse()
        return {"internet_ok": r.status < 500, "status": r.status}
    except Exception as e:
        return {"internet_ok": False, "error": str(e)}

@app.get("/env")
def env():
    return {
        "ALLOW_DOWNLOAD": ALLOW_DOWNLOAD,
        "FORCE_ONLINE": FORCE_ONLINE,
        "MODEL_REPO": MODEL_REPO,
        "MODEL_DIR": MODEL_DIR,
        "HF_HOME": HF_HOME,
    }

@app.post("/generate")
def generate(req: GenRequest):
    if not model_ready.is_set():
        # Devuelve la misma causa que /readyz
        raise HTTPException(status_code=503, detail=last_error or "Model not ready.")
    try:
        # Texto puro (sin imagen) — demostración mínima
        # Para prompts multimodales, habría que aceptar image_url / bytes y pasarlas al processor.
        inputs = processor(text=req.prompt, return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens or 64)
        # En Qwen2-VL, la decodificación puede variar con trust_remote_code; mantenerlo simple:
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e!s}")
