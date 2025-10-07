# main.py — Cloud Run (Qwen2-VL-2B-Instruct, CPU) con readiness consistente
import os
import socket
import threading
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

# ========= App =========
app = FastAPI(title="Qwen2-VL CPU demo", version="1.0")


# ========= Helpers =========
def _set_error(msg: str):
    """Setea error global y baja readiness."""
    global last_error
    last_error = msg
    model_ready.clear()
    print(f"[LOAD][ERROR] {msg}", flush=True)


def _ok(msg: str):
    print(f"[LOAD][OK] {msg}", flush=True)


def _have_local_snapshot(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())


# ========= Hilo de carga =========
def _load_model():
    """Descarga (si aplica) y carga Qwen2-VL usando la clase correcta."""
    global model, processor
    try:
        # 1) Resolver ruta del snapshot (preferir HF cache; usar local si ya existe y no forzamos online)
        snapshot_path: Optional[str] = None

        from huggingface_hub import snapshot_download

        if FORCE_ONLINE:
            if not ALLOW_DOWNLOAD:
                _set_error("FORCE_ONLINE=1 pero ALLOW_DOWNLOAD=0: impedida la descarga.")
                return
            _ok(f"FORCE_ONLINE=1: descargando {MODEL_REPO} -> HF cache (HF_HOME={HF_HOME})")
            snapshot_path = snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=None,                 # usar cache de HF
                local_dir_use_symlinks=False
            )
        else:
            if _have_local_snapshot(MODEL_DIR):
                snapshot_path = MODEL_DIR
                _ok(f"Usando snapshot LOCAL en {snapshot_path}")
            elif ALLOW_DOWNLOAD:
                _ok(f"No hay snapshot local en {MODEL_DIR}. Descargando {MODEL_REPO} -> HF cache")
                snapshot_path = snapshot_download(
                    repo_id=MODEL_REPO,
                    local_dir=None,
                    local_dir_use_symlinks=False
                )
            else:
                _set_error(f"Modelo no encontrado en {MODEL_DIR} y ALLOW_DOWNLOAD=0.")
                return

        if not snapshot_path or not Path(snapshot_path).exists():
            _set_error(f"Snapshot path inválido: {snapshot_path}")
            return

        _ok(f"Snapshot listo en {snapshot_path}")

        # 2) Cargar clases correctas (Qwen2-VL)
        #    Requiere transformers >= 4.42 aprox. Si no está, daremos error claro.
        try:
            from transformers import Qwen2VLForConditionalGeneration
        except Exception as e:
            _set_error(
                "Tu versión de 'transformers' no expone Qwen2VLForConditionalGeneration. "
                "Actualiza a transformers>=4.44.0 (recomendado). Detalle: " + str(e)
            )
            return

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            snapshot_path,
            trust_remote_code=True
        )
        # CPU
        model.to("cpu")
        model.eval()

        model_ready.set()
        _ok("Modelo listo.")
    except Exception as e:
        _set_error(f"Load error: {e!s}")


# ========= Eventos de app =========
@app.on_event("startup")
def on_startup():
    print("[BOOT] Vars:", {
        "ALLOW_DOWNLOAD": ALLOW_DOWNLOAD,
        "FORCE_ONLINE": FORCE_ONLINE,
        "MODEL_REPO": MODEL_REPO,
        "MODEL_DIR": MODEL_DIR,
        "HF_HOME": HF_HOME,
        "HOSTNAME": socket.gethostname()
    }, flush=True)
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()


# ========= Schemas =========
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int | None = 64
    # Futuro (multimodal): image_url: str | None = None


# ========= Endpoints =========
@app.get("/status")
def status():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ready": model_ready.is_set(), "error": last_error}


@app.get("/_netcheck")
def netcheck():
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
    # Readiness único para /readyz y /generate
    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail=last_error or "Model not ready.")

    try:
        # Texto-only mínimo (para multimodal, luego añadimos imágenes)
        import torch
        with torch.no_grad():
            inputs = processor(text=req.prompt, return_tensors="pt")
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens or 64
            )
        # Decodificar con el tokenizer del processor
        text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e!s}")


# ========= Main local =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
