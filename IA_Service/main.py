# server.py (lifespan sync-load)
import os, torch, logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---- Config ----
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "/models/Qwen2-VL-2B-Instruct")
DOWNLOAD_IF_MISSING = os.environ.get("DOWNLOAD_IF_MISSING", "0") == "1"
USE_LOCAL_ONLY = os.environ.get("USE_LOCAL_ONLY", "1") == "1"  # 1 si imagen ya horneada

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("qwen2vl")

# ---- Globals ----
processor = None
model = None

def ensure_model_on_disk():
    if os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR)):
        log.info(f"[model] Found local model at {MODEL_DIR}")
        return
    if not DOWNLOAD_IF_MISSING:
        raise RuntimeError(f"Model dir {MODEL_DIR} not found and DOWNLOAD_IF_MISSING=0")
    from huggingface_hub import snapshot_download
    log.info(f"[model] Downloading {MODEL_REPO} to {MODEL_DIR} ...")
    snapshot_download(repo_id=MODEL_REPO, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
    log.info("[model] Download complete")

def load_model():
    global processor, model
    ensure_model_on_disk()
    kw = dict(trust_remote_code=True)
    if USE_LOCAL_ONLY:
        kw["local_files_only"] = True
    log.info(f"[model] Loading processor from {MODEL_DIR} (local_only={USE_LOCAL_ONLY})")
    processor = AutoProcessor.from_pretrained(MODEL_DIR, **kw)
    log.info(f"[model] Loading model from {MODEL_DIR} (device={device}, dtype={dtype})")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=dtype, low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None, **kw
    )
    model.to(device).eval()
    torch.set_grad_enabled(False)
    log.info("[model] Ready")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga BLOQUEANTE: Cloud Run debe esperar con startupProbe
    load_model()
    yield
    # (opcional) liberar recursos

app = FastAPI(title="Qwen2-VL API", lifespan=lifespan)

class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

@app.get("/healthz")
def health():
    ok = processor is not None and model is not None
    return {"status": "ready" if ok else "starting", "device": device, "model_dir": MODEL_DIR}

def _run(req: EvalRequest) -> EvalResponse:
    msgs = [{"role":"user","content":[{"type":"text","text":req.input}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_kwargs = dict(max_new_tokens=req.max_new_tokens or 128)
    if req.temperature is not None: gen_kwargs.update(do_sample=True, temperature=req.temperature)
    if req.top_p is not None:      gen_kwargs.update(do_sample=True, top_p=req.top_p)
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    gen_tokens = int(out[0].shape[-1]) - int(inputs["input_ids"].shape[-1])
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=decoded, tokens_generated=max(gen_tokens, 0))

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest, response: Response):
    if processor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is loading")
    return _run(req)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

