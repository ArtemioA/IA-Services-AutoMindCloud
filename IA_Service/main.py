# main.py — Cloud Run (Qwen2-VL CPU) | online/offline with safe fallbacks
import os, socket, threading, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

print("🚀 Starting Qwen2-VL service initialization...")

# ---------------- Env config ----------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")  # puede ser vacío
ALLOW_DL   = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT       = int(os.environ.get("PORT", "8080"))

print(f"🌍 MODEL_REPO={MODEL_REPO}")
print(f"📂 MODEL_DIR={MODEL_DIR}")
print(f"⬇️  ALLOW_DOWNLOAD={ALLOW_DL}")
print(f"⚙️  PORT={PORT}")

# Writable caches (Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Disable hf_transfer if lib not present
try:
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        print("⚠️ hf_transfer not found, disabling HF_HUB_ENABLE_HF_TRANSFER")
except Exception as e:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    print(f"⚠️ Error checking hf_transfer: {e}")

# ---------------- App ----------------
app = FastAPI(title="Qwen2-VL (CPU)")
print("✅ FastAPI app created")

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
_model = {"proc": None, "tok": None, "model": None}

# ---------------- Helpers ----------------
def _has_local_snapshot(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    ok = os.path.exists(os.path.join(path, "config.json"))
    print(f"🔍 Checking local snapshot at {path} -> {'FOUND' if ok else 'NOT FOUND'}")
    return ok

def _download_if_needed():
    if not ALLOW_DL:
        print("⛔ Download disabled (ALLOW_DOWNLOAD=0)")
        return
    if not MODEL_DIR:
        print("⚠️ MODEL_DIR not set, skipping download")
        return
    if _has_local_snapshot(MODEL_DIR):
        print("✅ Snapshot already present, no download needed")
        return
    print("⬇️ Downloading model from Hugging Face Hub...")
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )
    print("✅ Download complete")

def _choose_src():
    print("🧭 Choosing model source...")
    use_local = _has_local_snapshot(MODEL_DIR)
    if not use_local and ALLOW_DL and MODEL_DIR:
        try:
            _download_if_needed()
            use_local = _has_local_snapshot(MODEL_DIR)
        except Exception as e:
            print(f"⚠️ Download failed, falling back to online: {e}")
    print(f"📦 Using {'LOCAL' if use_local else 'ONLINE'} source: {MODEL_DIR if use_local else MODEL_REPO}")
    return (MODEL_DIR if use_local else MODEL_REPO), use_local

# ---------------- Model loading ----------------
def _load_model_locked():
    if _state["ready"] or _state["loading"]:
        print("ℹ️ Model already ready or loading")
        return
    _state["loading"], _state["error"] = True, None
    print("🧠 Loading model...")
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        _state["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Device: {_state['device']}")

        src, using_local = _choose_src()
        local_only = (not ALLOW_DL)
        print(f"📥 Loading from: {src} | local_only={local_only}")

        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None

        proc = AutoProcessor.from_pretrained(src, trust_remote_code=True, local_files_only=local_only, token=token)
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, local_files_only=local_only, token=token)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            src,
            trust_remote_code=True,
            local_files_only=local_only,
            token=token,
            torch_dtype=torch.float32,
        ).to("cpu").eval()

        _model["proc"], _model["tok"], _model["model"] = proc, tok, model
        _state["ready"] = True
        print("✅ Model loaded successfully")

    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        print(f"❌ Model load failed: {_state['error']}")
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
    print("🧵 Background loading thread started")

def _ensure_loaded_blocking():
    if _state["ready"]:
        print("✅ Model already ready")
        return
    with _lock:
        if not _state["ready"]:
            _load_model_locked()

# ---------------- Schemas ----------------
class Ask(BaseModel):
    prompt: str
    max_new_tokens: int | None = 128
    temperature: float | None = 0.0

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    print("📡 GET / called — warming up...")
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

@app.get("/status")
def status():
    print("📡 GET /status")
    _ensure_loaded_bg()
    return {"ok": True, "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"), "mode": _state["mode"], "error": _state["error"]}

@app.get("/healthz")
def healthz():
    print("✅ GET /healthz")
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.get("/_dns")
def dns_check():
    print("🌐 GET /_dns")
    try:
        host = socket.gethostbyname("huggingface.co")
        print(f"🔎 huggingface.co -> {host}")
        return {"ok": True, "host": host}
    except Exception as e:
        print(f"❌ DNS error: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/_netcheck")
def netcheck():
    print("🌐 GET /_netcheck")
    import requests
    out = {}
    urls = [
        "https://huggingface.co",
        f"https://huggingface.co/api/models/{MODEL_REPO}",
        "https://cdn-lfs.huggingface.co/favicon.ico",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=6)
            out[url] = {"ok": True, "code": r.status_code}
            print(f"✅ {url} -> {r.status_code}")
        except Exception as e:
            out[url] = {"ok": False, "error": str(e)}
            print(f"❌ {url} -> {e}")
    return out

@app.post("/generate")
def generate(body: Ask):
    print(f"📝 POST /generate | prompt={body.prompt[:40]!r}...")
    try:
        _ensure_loaded_blocking()
    except Exception as e:
        print(f"❌ Load error: {e}")
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

    if not _state["ready"]:
        print("⚠️ Model not ready yet")
        raise HTTPException(status_code=503, detail=f"Model not ready: {_state['error']}")

    prompt = (body.prompt or "").strip()
    if not prompt:
        print("⚠️ Empty prompt")
        raise HTTPException(status_code=422, detail="prompt vacío")

    try:
        import torch
        tok = _model["tok"]
        model = _model["model"]

        inputs = tok(prompt, return_tensors="pt")
        print("🧩 Tokens prepared, generating...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(body.max_new_tokens or 128),
                do_sample=(body.temperature or 0.0) > 0.0,
                temperature=float(body.temperature or 0.0),
            )
        text = tok.decode(output_ids[0], skip_special_tokens=True)
        print("✅ Generation complete")
        return {"ok": True, "text": text}
    except Exception as e:
        print(f"❌ Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")

# ---------------- Local dev ----------------
if __name__ == "__main__":
    print(f"🏃 Running uvicorn on port {PORT}")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
