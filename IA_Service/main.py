import os
from typing import Optional, Dict, Any
from threading import Lock

import torch
from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------- Config ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
REQUIRE_BEARER = os.environ.get("REQUIRE_BEARER", "false").lower() == "true"
API_TOKEN = os.environ.get("API_TOKEN", "")

# IMPORTANTE: no usar carpeta local hasta que realmente la hornees bien
MODEL_LOCAL_DIR = None  # fuerza a cargar desde HF y evitar rutas incompletas

# Limita threads en CPU (ahorra RAM/CPU)
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Qwen2-VL Text Generator (lazy, no safetensors)", version="1.0.0")

# ---------------- Estado global (lazy) ----------------
_model = None
_processor = None
_load_lock = Lock()

def _require_bearer(authorization: Optional[str]) -> None:
    if REQUIRE_BEARER:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

def _ensure_model_loaded():
    """Carga modelo y processor SOLO la primera vez que se llama."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return
    with _load_lock:
        if _model is not None and _processor is not None:
            return

        # Fuente: SIEMPRE del repo de HF (evita carpetas locales incompletas)
        source = MODEL_NAME

        # Clave: NO forzar safetensors → permite .bin
        common_kwargs: Dict[str, Any] = dict(trust_remote_code=True)
        model_kwargs: Dict[str, Any] = dict(
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto" if device == "cuda" else None,
            use_safetensors=False,  # <--- evita buscar *.safetensors
            trust_remote_code=True,
        )

        _processor = AutoProcessor.from_pretrained(source, **common_kwargs)
        _model = Qwen2VLForConditionalGeneration.from_pretrained(source, **model_kwargs)
        if device == "cpu":
            _model.to(device)

class Entrada(BaseModel):
    texto: str
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

def _generate_text(user_text: str, max_new_tokens: int,
                   do_sample: bool, temperature: float, top_p: float) -> str:
    _ensure_model_loaded()

    # Solo texto del usuario (sin prefijos)
    messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
    prompt = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = _processor(text=[prompt], return_tensors="pt")
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].to(_model.device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(_model.device)

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})

    with torch.inference_mode():
        out_ids = _model.generate(**inputs, **gen_kwargs)

    # Extraer solo lo generado (sin el prompt)
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = out_ids[0, input_len:]
    text = _processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

    if text.startswith("assistant\n"):
        text = text[len("assistant\n"):].strip()
    return text

@app.get("/healthz")
def healthz():
    return {"ok": True}

# Devuelve string plano (text/plain)
@app.post("/generate")
def generate(entrada: Entrada, authorization: Optional[str] = Header(default=None)):
    try:
        _require_bearer(authorization)
        out = _generate_text(
            user_text=entrada.texto,
            max_new_tokens=entrada.max_new_tokens or MAX_NEW_TOKENS,
            do_sample=bool(entrada.do_sample),
            temperature=float(entrada.temperature or 0.7),
            top_p=float(entrada.top_p or 0.9),
        )
        return Response(out, media_type="text/plain")
    except Exception as e:
        return Response(f"ERROR: {type(e).__name__}: {e}", media_type="text/plain", status_code=500)
