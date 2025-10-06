# main.py — Cloud Run (Qwen2-VL CPU) | online/offline with safe fallbacks
import os, socket, threading, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Env config ----------------
MODEL_REPO   = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR    = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")  # puede ser vacío
ALLOW_DL     = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
FORCE_ONLINE = os.environ.get("FORCE_ONLINE", "0") == "1"  # ignora MODEL_DIR si 1
PORT         = int(os.environ.get("PORT", "8080"))

# Writable caches (Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Deshabilita hf_transfer si no está presente
try:
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
except Exception:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ---------------- App ----------------
app = FastAPI(title="Qwen2-VL (CPU)")

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
    """
    Snapshot válido si:
      - existe config.json
      - y existe AL MENOS UN archivo de pesos: *.safetensors o pytorch_model.bin (en la carpeta o subcarpetas)
    """
    if not path or not os.path.isdir(path):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False

    for _root, _dirs, files in os.walk(path):
        if "pytorch_model.bin" in files:
            return True
        if any(f.endswith(".safetensors") for f in files):
            return True
    return False

def _download_if_needed():
    """Descarga el repo a MODEL_DIR si está permitido y aún no existe snapshot válido."""
    if not ALLOW_DL or not MODEL_DIR or FORCE_ONLINE:
        return
    if _has_local_snapshot(MODEL_DIR):
        return
    from huggingface_hub import snapshot_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,  # inofensivo aunque deprecado
        token=os.environ.get("HUGGINGFACE_HUB_TOKEN") or None,
    )

def _choose_src():
    """
    Decide la fuente de carga:
      - FORCE_ONLINE=1 -> siempre repo online.
      - Si hay snapshot local válido -> usar carpeta local.
      - Si no hay y ALLOW_DL y MODEL_DIR definido -> descargar y usar carpeta si queda válida.
      - Si no, usar repo online (requiere egress; token opcional).
    """
    if FORCE_ONLINE:
        return MODEL_REPO, False

    use_local = _has_local_snapshot(MODEL_DIR)
    if not use_local and ALLOW_DL and MODEL_DIR:
        try:
            _download_if_needed()
            use_local = _has_local_snapshot(MODEL_DIR)
        except Exception:
            use_local = False
    return (MODEL_DIR if use_local else MODEL_REPO), use_local

# ---------------- Model loading ----------------
def _load_model_locked():
    """
    Lazy load (thread-safe).
    - Offline (ALLOW_DL=False): exige snapshot local válido, local_files_only=True.
    - Online (ALLOW_DL=True): usa local si existe; si no, repo con local_files_only=False.
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
        local_only = (not ALLOW_DL) or (using_local and not FORCE_ONLINE)
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
            dtype=torch.float32,         # evita warning (antes torch_dtype)
            low_cpu_mem_usage=True,      # ayuda a reducir RAM pico en CPU
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

def _ensure_loaded_blocking():
    if _state["ready"]:
        return
    with _lock:
        if not _state["ready"]:
            _load_model_locked()

# ---------------- Schemas ----------------
class Ask(BaseModel):
    prompt: str
    max_new_tokens: int | None = 128
    temperature: float | None = 0.0  # greedy por defecto (CPU)

# ---------------- Endpoints ----------------
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
        "force_online": FORCE_ONLINE,
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

# Endpoints alternativos de health (por si /healthz choca con proxy)
@app.get("/health")
def health_plain():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.get("/_health")
def health_alt():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.get("/_dns", summary="DNS to huggingface.co (dev aid)")
def dns_check():
    try:
        host = socket.gethostbyname("huggingface.co")
        return {"ok": True, "host": host}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/_netcheck", summary="HTTP checks to HF and CDN")
def netcheck():
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
        except Exception as e:
            out[url] = {"ok": False, "error": str(e)}
    return out

# ---------- Generation: fixed to avoid empty outputs ----------
@app.post("/generate", summary="Text-only generation")
def generate(body: Ask):
    # Carga perezosa (bloqueante en la primera vez)
    try:
        _ensure_loaded_blocking()
    except Exception as e:
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
        tok = _model["tok"]
        model = _model["model"]

        # Qwen2-VL chat template (texto puro)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        chat_text = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tok(chat_text, return_tensors="pt")
        input_len = inputs["input_ids"].size(1)

        gen = model.generate(
            **inputs,
            max_new_tokens=int(body.max_new_tokens or 128),
            do_sample=(body.temperature or 0.0) > 0.0,
            temperature=float(body.temperature or 0.0),
            top_p=0.95 if (body.temperature or 0.0) > 0 else 1.0,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
            use_cache=True,
        )

        # Decodificar solo los tokens NUEVOS (no el prompt)
        gen_ids = gen[0][input_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Fallbacks por si queda vacío
        if not text:
            text = tok.decode(gen_ids, skip_special_tokens=False).strip()
        if not text:
            text = "(sin salida — intenta con un prompt más explícito o sube max_new_tokens)"

        return {"ok": True, "text": text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")

# ---------------- Local dev ----------------
if __name__ == "__main__":
    import uvicorn
    # Solo para ejecución local; en Cloud Run, el Dockerfile lanza uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
