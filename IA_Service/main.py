# server.py — Qwen2-VL API (bg load + /tmp fallback + raíz con estado)
import os, threading, logging, torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# -------- Config --------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "/models/Qwen2-VL-2B-Instruct")
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "0") == "1"  # 1: baja una vez al arrancar
USE_LOCAL_ONLY      = os.environ.get("USE_LOCAL_ONLY", "1") == "1"       # 1: solo archivos locales

# Forzar offline si es solo local y usar caches en /tmp (Cloud Run permite escribir en /tmp)
if USE_LOCAL_ONLY:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwen2vl")

app = FastAPI(title="Qwen2-VL API (bg load)")
processor = None
model = None
model_ready = False
load_error = None
gen_lock = threading.Lock()

def _writable_dir_fallback(path: str) -> str:
    """Si no puedo crear 'path', caer a /tmp/models/<basename>."""
    try:
        os.makedirs(path, exist_ok=True)
        return path
    except PermissionError:
        tmp_path = f"/tmp/models/{os.path.basename(path.rstrip('/'))}"
        os.makedirs(tmp_path, exist_ok=True)
        log.warning("[model] No write perms on %s; falling back to %s", path, tmp_path)
        return tmp_path

def ensure_model_on_disk():
    """Garantiza que el modelo exista en disco; si se permite, descarga una vez al arrancar."""
    global MODEL_DIR, load_error
    MODEL_DIR = _writable_dir_fallback(MODEL_DIR)

    try:
        has_files = any(os.scandir(MODEL_DIR))
    except Exception as e:
        load_error = f"Cannot scan MODEL_DIR {MODEL_DIR}: {e!r}"
        log.error("[model] %s", load_error)
        return

    if has_files:
        log.info(f"[model] Found local model at {MODEL_DIR}")
        return

    if not DOWNLOAD_IF_MISSING:
        load_error = f"Model dir {MODEL_DIR} is empty and DOWNLOAD_IF_MISSING=0"
        log.error("[model] %s", load_error)
        return

    try:
        log.info(f"[model] Downloading {MODEL_REPO} to {MODEL_DIR} ...")
        from huggingface_hub import snapshot_download
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
    """Carga processor+model en RAM/VRAM (sin red si USE_LOCAL_ONLY=1)."""
    global processor, model, model_ready, load_error
    try:
        ensure_model_on_disk()
        if load_error:
            model_ready = False
            return

        kw = dict(trust_remote_code=True)
        if USE_LOCAL_ONLY:
            kw["local_files_only"] = True

        log.info(f"[model] Loading processor from {MODEL_DIR} (local_only={USE_LOCAL_ONLY})")
        proc = AutoProcessor.from_pretrained(MODEL_DIR, **kw)

        log.info(f"[model] Loading model from {MODEL_DIR} (device={device}, dtype={dtype})")
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

# -------- Schemas --------
class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

# -------- Routes --------
@app.get("/")
def root():
    status = "ready" if model_ready else "starting"
    return {"ok": True, "status": status, "device": device, "model_dir": MODEL_DIR, "error": load_error, "msg": "qwen2-vl up"}

@app.get("/healthz")
def healthz():
    status = "ready" if model_ready else "starting"
    return {"status": status, "device": device, "model_dir": MODEL_DIR, "error": load_error}

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

# Compatibilidad {"texto": "..."}
class LegacyPayload(BaseModel):
    texto: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

@app.post("/generate", response_model=EvalResponse)
def generate_legacy(data: LegacyPayload, response: Response):
    req = EvalRequest(
        input=data.texto,
        max_new_tokens=data.max_new_tokens,
        temperature=data.temperature,
        top_p=data.top_p,
    )
    return eval_endpoint(req, response)

@app.on_event("startup")
def on_startup():
    kickoff_background_load()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

