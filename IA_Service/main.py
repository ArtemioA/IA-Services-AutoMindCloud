# main.py — Cloud Run / baked Qwen2-VL
import os, torch
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# --- Config ---
MODEL_DIR = os.getenv("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
PORT = int(os.getenv("PORT", "8080"))
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

# --- Optimización CPU ---
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# --- Caches seguros ---
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

app = FastAPI(title="Qwen2-VL (baked)")
_model = None
_processor = None
_lock = Lock()


class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64


def _load_once():
    """Carga única desde disco; sin conexión a Internet."""
    global _model, _processor
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(f"MODEL_DIR inexistente: {MODEL_DIR}")
        _processor = AutoProcessor.from_pretrained(
            MODEL_DIR, trust_remote_code=True, local_files_only=True
        )
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            _model.to(device)
        _model.eval()
        print(f"✅ Modelo cargado desde {MODEL_DIR} ({device})")


@app.get("/healthz")
def healthz():
    try:
        _load_once()
        return {"ok": True, "device": device, "model_dir": MODEL_DIR}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


@app.post("/generate")
def generate(r: GenReq):
    try:
        _load_once()
        # Chat template mínimo (solo texto)
        messages = [{"role": "user", "content": [{"type": "text", "text": r.prompt}]}]
        prompt = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = _processor(text=[prompt], return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = _model.generate(**inputs, max_new_tokens=int(r.max_new_tokens))
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        text = _processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
        if text.startswith("assistant\n"):
            text = text[len("assistant\n"):].strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
