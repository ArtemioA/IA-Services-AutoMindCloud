# server.py — Qwen2-VL API (bg load, offline/online modes, /tmp fallback, netcheck)
import os, threading, logging, torch, requests
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------- Config ----------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "/models/Qwen2-VL-2B-Instruct")

# Modo de operación:
#   OFFLINE (prod): USE_LOCAL_ONLY=1, DOWNLOAD_IF_MISSING=0  (modelo horneado)
#   ONLINE  (dev) : USE_LOCAL_ONLY=0, DOWNLOAD_IF_MISSING=1  (descarga al arrancar si hay egress)
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "0") == "1"
USE_LOCAL_ONLY      = os.environ.get("USE_LOCAL_ONLY", "1") == "1"

# Caches/vars recomendadas (Cloud Run solo escribe en /tmp):
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
# Forzar offline si pediste solo local:
if USE_LOCAL_ONLY:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwen2vl")

# ---------------- App & Globals ----------------
app = FastAPI(title="Qwen2-VL API (bg load)")
processor = None
model = None
model_ready = False
load_error = None
gen_lock = threading.Lock()  # serializa generate() para evitar OOM

# ---------------- Helpers ----------------
def _writable_dir_fallback(path: str) -> str:
    """Si no puedo crear 'path', caigo a /tmp/models/<basename>."""
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except PermissionError:
        tmp_path = f"/tmp/models/{os.path.basename(path.rstrip('/'))}"
        os.makedirs(tmp_path, exist_ok=True)
        log.warning("[model] No write perms on %s; falling back to %s", path, tmp_path)
        return tmp_path

def ensure_model_on_disk():
    """Verifica/descarga el modelo en disco si está permitido."""
    global MODEL_DIR, load_error
    MODEL_DIR = _writable_dir_fallback(MODEL_DIR)

    try:
        has_files = any(os.scandir(MODEL_DIR))
    except Exception as e:
        load_error = f"Cannot scan MODEL_DIR {MODEL_DIR}: {e!r}"
        log.error("[model] %s", load_error)
        return

    if has_files:
        log.info("[model] Found local model at %s", MODEL_DIR)
        return

    if not DOWNLOAD_IF_MISSING:
        load_error = f"Model dir {MODEL_DIR} is empty and DOWNLOAD_IF_MISSING=0"
        log.error("[model] %s", load_error)
        return

    # Descargar (requiere egress a Internet)
    try:
        from huggingface_hub import snapshot_download
        log.info("[model] Downloading %s to %s ...", MODEL_REPO, MODEL_DIR)
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        log.info("[model] Download complete")
    except Exception as e:
        load_error = f"snapshot_download failed: {repr(e)}"
        log.exception("[model] Download failed")

def load_model_into_memory():
    """Carga processor + model en RAM/VRAM (usa red solo si USE_LOCAL_ONLY=0)."""
    global processor, model, model_ready, load_error
    try:
        ensure_model_on_disk()
        if load_error:
            model_ready = False
            return

        kw = dict(trust_remote_code=True)
        if USE_LOCAL_ONLY:
            kw["local_files_only"] = True

        log.info("[model] Loading processor from %s (local_only=%s)", MODEL_DIR, USE_LOCAL_ONLY)
        proc = AutoProcessor.from_pretrained(MODEL_DIR, **kw)

        log.info("[model] Loading model from %s (device=%s, dtype=%s)", MODEL_DIR, device, dtype)
        mdl = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
            **kw,
        )
        mdl.to(device).eval()
        torch.set_grad_enabled(False)

        processor = proc
        model = mdl
        model_ready = True
        load_error = None
        log.info("[model] Ready (dir=%s)", MODEL_DIR)

    except Exception as e:
        load_error = repr(e)
        model_ready = False
        log.exception("[model] Load failed")

def kickoff_background_load():
    threading.Thread(target=load_model_into_memory, daemon=True).start()

# ---------------- Schemas ----------------
class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

class LegacyPayload(BaseModel):
    texto: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

# ---------------- Routes ----------------
@app.get("/")
def root():
    status = "ready" if model_ready else "starting"
    return {
        "ok": True,
        "status": status,
        "device": device,
        "model_dir": MODEL_DIR,
        "error": load_error,
        "msg": "qwen2-vl up",
    }

@app.get("/healthz")
def healthz():
    status = "ready" if model_ready else "starting"
    return {"status": status, "device": device, "model_dir": MODEL_DIR, "error": load_error}

@app.get("/netcheck")
def netcheck():
    """Diagnóstico simple de egress; útil si la descarga falla."""
    urls = [
        "https://huggingface.co/api/models/Qwen/Qwen2-VL-2B-Instruct",
        "https://cdn-lfs.huggingface.co",
    ]
    out = {}
    for u in urls:
        try:
            r = requests.get(u, timeout=5)
            out[u] = {"status": r.status_code}
        except Exception as e:
            out[u] = {"error": repr(e)}
    return out

def _run(req: EvalRequest) -> EvalResponse:
    msgs = [{"role": "user", "content": [{"type": "text", "text": req.input}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=req.max_new_tokens or 128)
    if req.temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = req.top_p

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_tokens = int(out[0].shape[-1]) - int(inputs["input_ids"].shape[-1])
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=decoded, tokens_generated=max(gen_tokens, 0))

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest, response: Response):
    if not model_ready:
        response.headers["Retry-After"] = "5"
        raise HTTPException(status_code=503, detail="Model is loading")
    with gen_lock:
        return _run(req)

# Compatibilidad con payload antiguo {"texto": "..."}
@app.post("/generate", response_model=EvalResponse)
def generate_legacy(data: LegacyPayload, response: Response):
    req = EvalRequest(
        input=data.texto,
        max_new_tokens=data.max_new_tokens,
        temperature=data.temperature,
        top_p=data.top_p,
    )
    return eval_endpoint(req, response)

# ---------------- Startup ----------------
@app.on_event("startup")
def on_startup():
    kickoff_background_load()

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
