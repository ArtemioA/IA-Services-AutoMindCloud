# main.py — Cloud Run (lazy download + lazy load, Qwen2-VL CPU) 
import os, socket, threading
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---- Config (por env) ----
MODEL_REPO  = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR   = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")
ALLOW_DL    = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT        = int(os.environ.get("PORT", "8080"))

# Caches en /tmp (Cloud Run los permite escribir)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---- Estado global ----
state = {
    "ready": False,
    "loading": False,
    "error": None,
    "model_repo": MODEL_REPO,
    "model_dir": MODEL_DIR,
}
_lock = threading.Lock()
_model = {"proc": None, "model": None, "tokenizer": None}

app = FastAPI(title="Qwen2-VL (baked)")

class Ask(BaseModel):
    prompt: str

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _download_if_needed():
    """
    Si MODEL_DIR no existe o está vacío y ALLOW_DOWNLOAD=1, descarga el repo de HF.
    """
    if os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR)):
        return  # ya hay archivos

    if not ALLOW_DL:
        raise RuntimeError(f"Model directory not found: {MODEL_DIR} (ALLOW_DOWNLOAD=0)")

    from huggingface_hub import snapshot_download
    _ensure_dir(MODEL_DIR)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        # si necesitas token privado: use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

def _load_model_locked():
    """
    Carga perezosa (idempotente y thread-safe).
    """
    if state["ready"] or state["loading"]:
        return

    state["loading"] = True
    state["error"] = None
    try:
        _download_if_needed()

        # Import tardío (acorta el boot)
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        device = "cpu"  # Cloud Run sin GPU

        proc = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, device_map=None
        )
        model.eval()

        _model["proc"] = proc
        _model["model"] = model

        state["ready"] = True
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        raise
    finally:
        state["loading"] = False

def _ensure_loaded():
    if state["ready"]:
        return
    with _lock:
        if state["ready"]:
            return
        _load_model_locked()

# ---------------- Endpoints ----------------

@app.get("/")
def root():
    return {
        "ok": True,
        "status": "ready" if state["ready"] else ("loading" if state["loading"] else "cold"),
        "device": "cpu",
        "model_dir": state["model_dir"],
        "model_repo": state["model_repo"],
        "error": state["error"],
    }

@app.get("/healthz", summary="Healthz", description="Endpoint para verificar estado de carga")
def healthz():
    # No forza descarga; solo indica si ya está listo
    return {"ok": True, "ready": state["ready"], "error": state["error"]}

@app.get("/_dns", summary="Dns Check")
def dns_check():
    try:
        host = socket.gethostbyname("huggingface.co")
        return {"ok": True, "host": host}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/generate", summary="Generate", description="Genera texto a partir de un prompt")
def generate(body: Ask):
    # Forzamos load perezoso (Descarga si hace falta y luego carga)
    try:
        _ensure_loaded()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

    proc = _model["proc"]
    model = _model["model"]

    # Para Qwen2-VL sin imagen: prompt textual simple
    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt vacío")

    try:
        # Formato mínimo para Qwen2-VL en modo texto
        # (usa AutoProcessor para tokenizar)
        inputs = proc(
            text=prompt,
            return_tensors="pt",
        )

        # Qwen2-VL usa generate en el modelo VL
        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                eos_token_id=proc.tokenizer.eos_token_id if hasattr(proc, "tokenizer") else None,
            )
        # Decodificar
        if hasattr(proc, "tokenizer"):
            text = proc.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        else:
            # respaldo por si el processor no expone tokenizer
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
            text = tok.decode(output_ids[0], skip_special_tokens=True)

        return {"ok": True, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")


# ------------- Run local (no usado en Cloud Run) -------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

