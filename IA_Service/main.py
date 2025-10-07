# main.py — Cloud Run (Qwen2-VL CPU) | online/offline with safe fallbacks + JSON health
import os, threading, time
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Env config ----------------
MODEL_REPO   = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR    = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")  # can be empty
ALLOW_DL     = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
FORCE_ONLINE = os.environ.get("FORCE_ONLINE", "0") == "1"  # ignore MODEL_DIR if 1
PORT         = int(os.environ.get("PORT", "8080"))

# Writable caches for Cloud Run
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

app = FastAPI(title="Qwen2-VL API (CPU)")

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
_model = {"proc": None, "tok": None, "model": None}
_lock = threading.Lock()


# ---------------- Helpers ----------------
def _has_local_snapshot(path: str) -> bool:
    """
    Valid snapshot if:
      - config.json exists
      - and at least ONE weights file exists: *.safetensors or pytorch_model.bin (any subfolder)
    """
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    for root, _dirs, files in os.walk(path):
        if "pytorch_model.bin" in files:
            return True
        if any(f.endswith(".safetensors") for f in files):
            return True
    return False


def _download_if_needed():
    """Download the repo to MODEL_DIR if allowed and snapshot not present."""
    if not ALLOW_DL or not MODEL_DIR or FORCE_ONLINE:
        return
    if _has_local_snapshot(MODEL_DIR):
        return
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )


def _choose_src():
    """
    Decide source:
      - FORCE_ONLINE=1 -> always use repo id (online)
      - If valid local snapshot -> use local folder
      - Else if ALLOW_DL -> try to download into MODEL_DIR, then use local if now valid
      - Else -> use repo id (but that will only work at runtime if egress allowed)
    """
    if FORCE_ONLINE:
        return MODEL_REPO, False

    use_local = _has_local_snapshot(MODEL_DIR)
    if not use_local and ALLOW_DL and MODEL_DIR:
        try:
            _download_if_needed()
            use_local = _has_local_snapshot(MODEL_DIR)
        except Exception:
            # Fall back to repo if download fails; health will reflect any load errors
            use_local = False

    return (MODEL_DIR, True) if use_local else (MODEL_REPO, False)


def _load_model_locked():
    """
    Lazy load (thread-safe).
    - Offline (ALLOW_DL=False): requires local snapshot; else it will attempt repo but with local_files_only=False ONLY if FORCE_ONLINE=1.
    - Online (ALLOW_DL=True): prefer local if present; else load from repo.
    """
    if _state["ready"] or _state["loading"]:
        return
    _state["loading"], _state["error"] = True, None
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        src, using_local = _choose_src()
        # ✅ KEY FIX: Only force local_files_only if we are actually loading from a local directory.
        # If src is a repo id, local_files_only must be False; otherwise loading will always fail.
        local_only = using_local and not FORCE_ONLINE
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

        proc = AutoProcessor.from_pretrained(
            src, trust_remote_code=True, local_files_only=local_only, token=token
        )
        tok = AutoTokenizer.from_pretrained(
            src, trust_remote_code=True, local_files_only=local_only, token=token
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            src,
            trust_remote_code=True,
            local_files_only=local_only,
            token=token,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        ).to("cpu").eval()

        _model["proc"], _model["tok"], _model["model"] = proc, tok, model
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


# ---------------- Schemas ----------------
class GenReq(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0


# ---------------- Endpoints ----------------
@app.get("/")
def root():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "device": _state["device"],
        "mode": _state["mode"],
        "model_repo": _state["model_repo"],
        "model_dir": _state["model_dir"],
        "uptime_s": round(time.time() - _state["t0"], 1),
        "error": _state["error"],
    }


@app.get("/healthz", summary="Health probe (no load)")
def healthz():
    # Do NOT block; just report current state
    return {"ready": _state["ready"], "loading": _state["loading"], "error": _state["error"]}


@app.get("/status", summary="Status (kick background load once)")
def status():
    _ensure_loaded_bg()
    return {"ready": _state["ready"], "loading": _state["loading"], "error": _state["error"]}


@app.post("/generate")
def generate(req: GenReq):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt vacío")
    if not _state["ready"]:
        _ensure_loaded_bg()
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        import torch
        tok = _model["tok"]
        model = _model["model"]

        # Text-only generation path (works with *Instruct variants*)
        inputs = tok(req.prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens or 128,
                do_sample=(req.temperature or 0.0) > 0.0,
                temperature=req.temperature or 0.0,
                pad_token_id=tok.eos_token_id,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")


@app.on_event("startup")
def _startup():
    # Kick background load promptly
    _ensure_loaded_bg()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
