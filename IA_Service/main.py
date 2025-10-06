# main.py — Cloud Run (Qwen2-VL CPU) | offline/online modes with safe fallbacks
import os, socket, threading, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -------- Env config --------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")
ALLOW_DL   = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT       = int(os.environ.get("PORT", "8080"))

# Writable caches in Cloud Run
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# If hf_transfer package is missing, force-disable fast transfer
try:
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
except Exception:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

app = FastAPI(title="Qwen2-VL (CPU)")

# -------- Global state --------
_state = {
    "ready": False,
    "loading": False,
    "error": None,
    "model_repo": MODEL_REPO,
    "model_dir": MODEL_DIR,
    "device": "cpu",
    "mode": "online" if ALLOW_DL else "offline",
    "t0": time.time(),
}
_lock = threading.Lock()
_model = {"proc": None, "model": None}

class Ask(BaseModel):
    prompt: str
    max_new_tokens: int | None = 128
    temperature: float | None = 0.0  # greedy by default (CPU-friendly)

def _has_local_model() -> bool:
    return os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR))

def _load_model_locked():
    """
    Thread-safe lazy load.
    - If ALLOW_DL=False: require MODEL_DIR to exist, load with local_files_only=True (zero internet).
    - If ALLOW_DL=True: prefer MODEL_DIR if present, else pull from MODEL_REPO with local_files_only=False.
    """
    if _state["ready"] or _state["loading"]:
        return
    _state["loading"] = True
    _state["error"] = None
    try:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        local_ok = _has_local_model()
        if not ALLOW_DL and not local_ok:
            raise RuntimeError(
                f"Offline mode: MODEL_DIR not found or empty: {MODEL_DIR}. "
                f"Set ALLOW_DOWNLOAD=1 or bake/mount the model."
            )

        # Choose source and internet policy
        src = MODEL_DIR if local_ok else MODEL_REPO
        local_only = (not ALLOW_DL)  # offline forces local only

        proc = AutoProcessor.from_pretrained(
            src,
            trust_remote_code=True,
            local_files_only=local_only,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            src,
            trust_remote_code=True,
            local_files_only=local_only,
            torch_dtype=torch.float32,  # CPU friendly
        ).to("cpu").eval()

        _model["proc"] = proc
        _model["model"] = model
        _state["ready"] = True

    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
    finally:
        _state["loading"] = False

def _ensure_loaded_bg():
    if _state["ready"] or _state["loading"]:
        return
    def _bg():
        with _lock:
            if not _state["ready"]:
                _load_model_locked()
    threading.Thread(target=_bg, daemon=True).start()

def _ensure_loaded_blocking():
    if _state["ready"]:
        return
    with _lock:
        if not _state["ready"]:
            _load_model_locked()

# -------- Endpoints --------
@app.get("/")
def root():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "device": _state["device"],
        "mode": _state["mode"],
        "model_dir": _state["model_dir"],
        "model_repo": _state["model_repo"],
        "uptime_s": round(time.time() - _state["t0"], 1),
        "error": _state["error"],
        "allow_download": ALLOW_DL,
    }

@app.get("/status", summary="Status (warms up)")
def status():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "mode": _state["mode"],
        "error": _state["error"],
    }

@app.get("/healthz", summary="Health probe (no load)")
def healthz():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.get("/_dns", summary="DNS to huggingface.co (dev aid)")
def dns_check():
    try:
        host = socket.gethostbyname("huggingface.co")
        return {"ok": True, "host": host}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/generate", summary="Text-only generation")
def generate(body: Ask):
    try:
        _ensure_loaded_blocking()
    except Exception as e:
        # Clearer error messages for internet/offline cases
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. status="
                   f"{'loading' if _state['loading'] else 'cold'}; error={_state['error']}"
        )

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt vacío")

    try:
        import torch
        proc = _model["proc"]
        model = _model["model"]

        inputs = proc(text=[prompt], return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(body.max_new_tokens or 128),
                do_sample=(body.temperature or 0.0) > 0.0,
                temperature=float(body.temperature or 0.0),
            )

        if hasattr(proc, "batch_decode"):
            text = proc.batch_decode(output_ids, skip_special_tokens=True)[0]
        else:
            tok = getattr(proc, "tokenizer", None)
            if tok is None:
                from transformers import AutoTokenizer
                # Respect offline/online mode when fetching tokenizer
                src = MODEL_DIR if _has_local_model() or not ALLOW_DL else MODEL_REPO
                tok = AutoTokenizer.from_pretrained(
                    src, trust_remote_code=True, local_files_only=(not ALLOW_DL)
                )
            text = tok.decode(output_ids[0], skip_special_tokens=True)

        return {"ok": True, "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")

# -------- Local dev (not used on Cloud Run) --------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
