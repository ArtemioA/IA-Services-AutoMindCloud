# server.py
import os, threading, logging, torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------- Config ----------------
# If you bake the model (Dockerfile below), MODEL_DIR points to a local folder.
# If you're NOT using Docker, you can set DOWNLOAD_IF_MISSING=1 to auto-download once at startup.
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "/models/Qwen2-VL-2B-Instruct")
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "0") == "1"
USE_LOCAL_ONLY = os.environ.get("USE_LOCAL_ONLY", "1") == "1"  # with baked image keep =1

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Reduce CPU thread usage a bit (helps on tiny instances)
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwen2vl")

# ---------------- Globals ----------------
app = FastAPI(title="Qwen2-VL API")
processor = None
model = None
model_ready = False
load_error = None
gen_lock = threading.Lock()  # simple serialize to avoid OOM thrash

# ---------------- Models ----------------
def ensure_model_on_disk():
    """Ensure MODEL_DIR has the model. If allowed, download once at startup."""
    global load_error
    if os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR)):
        log.info(f"[model] Found local model at {MODEL_DIR}")
        return
    if not DOWNLOAD_IF_MISSING:
        load_error = f"Model directory {MODEL_DIR} not found and DOWNLOAD_IF_MISSING=0"
        log.error("[model] %s", load_error)
        return
    log.info(f"[model] Local model not found; downloading from HF repo {MODEL_REPO} to {MODEL_DIR} ...")
    try:
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
    """Load processor+model to RAM/VRAM (no network if USE_LOCAL_ONLY=1)."""
    global processor, model, model_ready, load_error
    try:
        ensure_model_on_disk()
        if load_error:
            model_ready = False
            return

        kwargs_common = dict(trust_remote_code=True)
        if USE_LOCAL_ONLY:
            kwargs_common["local_files_only"] = True

        log.info(f"[model] Loading processor from {MODEL_DIR} (local_only={USE_LOCAL_ONLY})")
        processor = AutoProcessor.from_pretrained(MODEL_DIR, **kwargs_common)

        log.info(f"[model] Loading model from {MODEL_DIR} (device={device}, dtype={dtype})")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
            **kwargs_common,
        )
        model.to(device).eval()
        torch.set_grad_enabled(False)
        model_ready = True
        log.info("[model] Ready")
    except Exception as e:
        load_error = repr(e)
        model_ready = False
        log.exception("[model] Load failed")

# Load in background so the server can bind to $PORT immediately (Cloud Run requirement)
def kickoff_background_load():
    t = threading.Thread(target=load_model_into_memory, daemon=True)
    t.start()

# ---------------- Schemas ----------------
class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

# ---------------- Routes ----------------
@app.get("/healthz")
def health():
    status = "ready" if model_ready else "starting"
    return {
        "status": status,
        "device": device,
        "model_dir": MODEL_DIR,
        "error": load_error,
    }

def _run_generate(req: EvalRequest) -> EvalResponse:
    messages = [{"role": "user", "content": [{"type": "text", "text": req.input}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

    # Tokens generated = output minus input length
    gen_tokens = int(out[0].shape[-1]) - int(inputs["input_ids"].shape[-1])
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=decoded, tokens_generated=max(gen_tokens, 0))

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest, response: Response):
    if not model_ready:
        # help clients backoff politely
        response.headers["Retry-After"] = "5"
        raise HTTPException(status_code=503, detail="Model is loading")
    with gen_lock:
        return _run_generate(req)

# Legacy shim so your older client keep working
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

# ---------------- Startup ----------------
@app.on_event("startup")
def on_startup():
    kickoff_background_load()

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)
