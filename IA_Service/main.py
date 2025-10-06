import os
from threading import Lock
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Ruta del modelo horneado (debe existir dentro de la imagen)
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
PORT = int(os.environ.get("PORT", "8080"))

# Ahorro de CPU en contenedores
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Caches seguros (no deberían activarse si todo es local, pero no molestan)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype  = torch.float16 if device == "cuda" else torch.float32

app = FastAPI(title="Qwen2-VL (baked)")

# Singletons perezosos
_model = None
_processor = None
_lock = Lock()

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.9

def _load_once():
    global _model, _processor
    if _model is not None and _processor is not None:
        return
    with _lock:
        if _model is not None and _processor is not None:
            return
        if not os.path.isdir(MODEL_DIR):
            raise RuntimeError(f"MODEL_DIR inexistente: {MODEL_DIR}. Asegúrate de hornearlo en la imagen.")
        # Carga SOLO desde disco local
        _processor = AutoProcessor.from_pretrained(
            MODEL_DIR, trust_remote_code=True, local_files_only=True
        )
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            _model.to(device)
        _model.eval()

@app.get("/healthz")
def healthz():
    try:
        _load_once()
        return {"ok": True, "device": device, "model_dir": MODEL_DIR}
    except Exception as e:
        return {"ok": False, "error": str(e), "model_dir": MODEL_DIR}

@app.post("/generate")
def generate(r: GenReq):
    try:
        _load_once()
        # Chat template (texto solo para smoke test rápido)
        messages = [{"role": "user", "content": [{"type": "text", "text": r.prompt}]}]
        prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = _processor(text=[prompt], return_tensors="pt")
        if "input_ids" in inputs:
            inputs["input_ids"] = inputs["input_ids"].to(_model.device)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(_model.device)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(r.max_new_tokens)}
        if r.do_sample:
            gen_kwargs.update({"do_sample": True, "temperature": float(r.temperature), "top_p": float(r.top_p)})

        with torch.inference_mode():
            out_ids = _model.generate(**inputs, **gen_kwargs)

        # Mantener solo la continuación
        input_len = inputs["input_ids"].shape[-1]
        gen_ids = out_ids[0, input_len:]
        text = _processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()
        if text.startswith("assistant\n"):
            text = text[len("assistant\n"):].strip()
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
