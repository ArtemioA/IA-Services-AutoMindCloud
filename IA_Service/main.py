# main.py — Cloud Run: bind instantáneo + carga perezosa del modelo
import os, threading
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

# ---- App ligera (sin imports pesados al inicio) ----
app = FastAPI(title="Qwen2-VL (minimal)")

# Config mínima (ajústalas en Cloud Run si quieres)
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR",  "/tmp/qwen2vl")
ALLOW_DOWNLOAD = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"  # 0 si horneas el modelo
PORT = int(os.environ.get("PORT", "8080"))

# Caches en /tmp (escritura permitida en Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# Estado global (simple)
_state = {"ready": False, "error": None, "device": "cpu", "processor": None, "model": None}
_gen_lock = threading.Lock()
_load_once = threading.Lock()

# --------- Schemas ----------
class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

# --------- Utilidades ----------
def _has_files(path: str) -> bool:
    try:
        import os
        return os.path.exists(path) and any(os.scandir(path))
    except Exception:
        return False

def _background_load():
    """Descarga (si ALLOW_DOWNLOAD=1 y no existe) y carga el modelo en memoria."""
    with _load_once:
        if _state["ready"]:
            return
        try:
            # Imports pesados solo aquí
            import torch, os
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

            os.makedirs(MODEL_DIR, exist_ok=True)
            if not _has_files(MODEL_DIR):
                if not ALLOW_DOWNLOAD:
                    raise RuntimeError(f"Model not present at {MODEL_DIR} and ALLOW_DOWNLOAD=0")
                # Descarga anónima y reanudable
                snapshot_download(
                    repo_id=MODEL_REPO,
                    local_dir=MODEL_DIR,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=2,
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            torch.set_num_threads(1)

            proc = AutoProcessor.from_pretrained(
                MODEL_DIR, trust_remote_code=True, local_files_only=True
            )
            mdl = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_DIR,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True,
            )
            mdl.to(device).eval()

            _state.update({
                "ready": True, "error": None, "device": device,
                "processor": proc, "model": mdl
            })
        except Exception as e:
            _state.update({"ready": False, "error": repr(e)})

# --------- Startup ----------
@app.on_event("startup")
def on_startup():
    threading.Thread(target=_background_load, daemon=True).start()

# --------- Endpoints ----------
@app.get("/")
def root():
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else "starting",
        "device": _state["device"],
        "model_dir": MODEL_DIR,
        "error": _state["error"],
    }

@app.get("/healthz")
def healthz():
    return {"status": "ready" if _state["ready"] else "starting", "error": _state["error"]}

# Diagnóstico de egress a Internet
@app.get("/netcheck")
def netcheck():
    import requests
    urls = [
        "https://huggingface.co/api/models/Qwen/Qwen2-VL-2B-Instruct",
        "https://cdn-lfs.huggingface.co",
    ]
    out = {}
    for u in urls:
        try:
            r = requests.get(u, timeout=6)
            out[u] = {"status": r.status_code}
        except Exception as e:
            out[u] = {"error": repr(e)}
    return out

def _run(req: EvalRequest) -> EvalResponse:
    processor, model = _state["processor"], _state["model"]
    import torch
    msgs = [{"role": "user", "content": [{"type": "text", "text": req.input}]}]
    prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], return_tensors="pt")
    device = _state["device"]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=req.max_new_tokens or 128)
    gen_tokens = int(out[0].shape[-1]) - int(inputs["input_ids"].shape[-1])
    text = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=text, tokens_generated=max(gen_tokens, 0))

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest, response: Response):
    if not _state["ready"]:
        response.headers["Retry-After"] = "5"
        raise HTTPException(status_code=503, detail="Model is loading")
    with _gen_lock:
        return _run(req)

# --------- Entrypoint correcto ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)
