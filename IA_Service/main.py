# main.py — Qwen2-VL (baked) con diagnóstico
import os, sys, json, traceback, time, glob, torch
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_DIR = os.getenv("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
PORT = int(os.getenv("PORT", "8080"))
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

# CPU sanidad
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

app = FastAPI(title="Qwen2-VL (baked diag)")
_model = None
_processor = None
_lock = Lock()

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9

def _ls_model_dir(max_files: int = 50) -> Dict[str, Any]:
    if not os.path.isdir(MODEL_DIR):
        return {"exists": False, "path": MODEL_DIR}
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*")))
    head = files[:max_files]
    return {
        "exists": True,
        "path": MODEL_DIR,
        "count": len(files),
        "sample": [os.path.basename(p) for p in head],
        "has_config": os.path.isfile(os.path.join(MODEL_DIR, "config.json")),
    }

def _load_once():
    global _model, _processor
    if _model is not None and _processor is not None:
        return
    with _lock:
        if _model is not None and _processor is not None:
            return
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(f"MODEL_DIR inexistente: {MODEL_DIR}")

        try:
            # Comprobación previa de archivos clave
            if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
                raise RuntimeError(f"Falta config.json en {MODEL_DIR}")

            _processor = AutoProcessor.from_pretrained(
                MODEL_DIR, trust_remote_code=True, local_files_only=True
            )
        except Exception:
            print("❌ Error cargando AutoProcessor:", file=sys.stderr)
            traceback.print_exc()
            raise

        try:
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
        except Exception:
            print("❌ Error cargando Qwen2VLForConditionalGeneration:", file=sys.stderr)
            traceback.print_exc()
            raise
        print(f"✅ Modelo cargado desde {MODEL_DIR} ({device})", flush=True)

@app.get("/healthz")
def healthz():
    try:
        _load_once()
        return {"ok": True, "device": device, "model_dir": MODEL_DIR}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "model_dir": MODEL_DIR,
            "ls": _ls_model_dir(),
        }

@app.get("/_diag")
def diag():
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "device": device,
        "env": {
            "MODEL_DIR": MODEL_DIR,
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        },
        "model_dir": _ls_model_dir(),
        "loaded": _model is not None,
    }
    # Intento de carga (no falla duro; solo reporta)
    try:
        _load_once()
        info["load_ok"] = True
    except Exception as e:
        info["load_ok"] = False
        info["load_error"] = f"{type(e).__name__}: {e}"
        info["trace"] = traceback.format_exc(limit=3)
    return info

@app.post("/generate")
def generate(r: GenReq):
    try:
        _load_once()
        messages = [{"role": "user", "content": [{"type": "text", "text": r.prompt}]}]
        prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _processor(text=[prompt], return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(r.max_new_tokens)}
        if r.do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": float(r.temperature), "top_p": float(r.top_p)})

        with torch.inference_mode():
            out = _model.generate(**inputs, **gen_kwargs)

        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        text = _processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
        if text.startswith("assistant\n"):
            text = text[len("assistant\n"):].strip()
        return {"text": text}
    except Exception as e:
        print("❌ Error en /generate:", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
