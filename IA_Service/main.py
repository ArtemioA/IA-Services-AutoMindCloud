# main.py — Cloud Run (Qwen2-VL-2B-Instruct, CPU) con readiness consistente (OFFLINE-friendly)
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
OMP_THREADS    = int(os.getenv("OMP_NUM_THREADS", "1"))

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
    """Localiza snapshot y carga Qwen2-VL con la clase correcta."""
    global model, processor
    try:
        # 0) Configurar Torch threads para CPU
        try:
            import torch
            torch.set_num_threads(max(1, OMP_THREADS))
        except Exception:
            pass

        snapshot_path: Optional[str] = None

        # 1) OFFLINE estricto si ALLOW_DOWNLOAD=0
        if not ALLOW_DOWNLOAD:
            if _have_local_snapshot(MODEL_DIR):
                snapshot_path = MODEL_DIR
                _ok(f"Usando snapshot LOCAL en {snapshot_path} (offline).")
            else:
                _set_error(f"Modelo no encontrado en {MODEL_DIR} y ALLOW_DOWNLOAD=0 (offline).")
                return
        else:
            # 2) Modo con descarga habilitada (no es tu caso baked, pero lo dejamos correcto)
            if _have_local_snapshot(MODEL_DIR) and not FORCE_ONLINE:
                snapshot_path = MODEL_DIR
                _ok(f"Usando snapshot LOCAL en {snapshot_path}")
            else:
                from huggingface_hub import snapshot_download
                _ok(f"Descargando {MODEL_REPO} (HF_HOME={HF_HOME})...")
                snapshot_path = snapshot_download(repo_id=MODEL_REPO, local_dir=None)
                _ok(f"Snapshot listo en cache HF: {snapshot_path}")

        if not snapshot_path or not Path(snapshot_path).exists():
            _set_error(f"Snapshot path inválido: {snapshot_path}")
            return

        # 3) Cargar clases correctas (Qwen2-VL)
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except Exception as e:
            _set_error(
                "Tu versión de 'transformers' no expone Qwen2VLForConditionalGeneration. "
                "Actualiza a transformers>=4.44.0 (recomendado). Detalle: " + str(e)
            )
            return

        processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            snapshot_path,
            trust_remote_code=True
        )
        # CPU
        try:
            import torch
            model.to("cpu")
            model.eval()
        except Exception as e:
            _set_error(f"No pude mover el modelo a CPU: {e!s}")
            return

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
        "HOSTNAME": socket.gethostname(),
        "OMP_NUM_THREADS": OMP_THREADS,
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


@app.get("/disk")
def disk():
    # Para depurar espacio en /app/models y /tmp
    import os
    def _fs(path: str):
        try:
            s = os.statvfs(path)
            return {"total": s.f_blocks * s.f_frsize, "free": s.f_bavail * s.f_frsize}
        except Exception as e:
            return {"error": str(e)}
    return {
        "models_dir": str(MODEL_DIR),
        "models_dir_exists": Path(MODEL_DIR).exists(),
        "tmp_fs": _fs("/tmp"),
        "app_fs": _fs("/app"),
    }


@app.post("/generate")
def generate(req: GenRequest):
    # Readiness único para /readyz y /generate
    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail=last_error or "Model not ready.")

    try:
        import torch
        with torch.no_grad():
            inputs = processor(text=req.prompt, return_tensors="pt")
            # Evitar errores si falta pad/eos en el tokenizer (pasa en algunos snapshots)
            tokenizer = getattr(processor, "tokenizer", None)
            extra = {}
            if tokenizer is not None:
                if getattr(tokenizer, "eos_token_id", None) is None and tokenizer.eos_token:
                    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                if getattr(tokenizer, "pad_token_id", None) is None:
                    # fallback seguro: usa eos como pad si no existe pad
                    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token or tokenizer.unk_token
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                extra.update({
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                })

            out = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens or 64,
                **extra
            )

        # Decodificar con el tokenizer del processor (si existe)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            # fallback simple
            text = str(out[0].tolist())

        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e!s}")


# ========= Main local =========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
