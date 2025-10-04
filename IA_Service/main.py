import os
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------- Config ----------------
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
REQUIRE_BEARER = os.environ.get("REQUIRE_BEARER", "false").lower() == "true"
API_TOKEN = os.environ.get("API_TOKEN", "")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32  # noqa: F841 (usado al mover tensores)

# ---------------- Carga modelo ----------------
# NOTA: sin imágenes; usamos solo texto.
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
if device == "cpu":
    model.to(device)

app = FastAPI(title="Qwen2-VL Text Generator", version="1.0.0")

# ---------------- Modelos de entrada ----------------
class Entrada(BaseModel):
    texto: str
    max_new_tokens: Optional[int] = None
    do_sample: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# ---------------- Utilidades ----------------
def require_bearer(authorization: Optional[str]) -> None:
    if REQUIRE_BEARER:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

def build_messages(user_text: str) -> Any:
    # Solo el texto del usuario, sin prefijos ni instrucciones añadidas.
    return [
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    ]

def generate_text(user_text: str, max_new_tokens: int, do_sample: bool,
                  temperature: float, top_p: float) -> str:
    messages = build_messages(user_text)
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Solo TEXTO (no pasamos images)
    inputs = processor(text=[prompt], return_tensors="pt")
    # Enviar tensores al dispositivo correcto
    if "input_ids" in inputs:
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(model.device)

    gen_kwargs: Dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if do_sample:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Extrae SOLO lo generado (quita el prompt)
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0, input_len:]
    text = processor.batch_decode(gen_ids.unsqueeze(0), skip_special_tokens=True)[0].strip()

    # A veces Qwen antepone "assistant\n"
    if text.startswith("assistant\n"):
        text = text[len("assistant\n"):].strip()
    return text

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {"message": "hello world", "model": MODEL_NAME, "device": device}

@app.post("/generate")
def generate(entrada: Entrada, authorization: Optional[str] = Header(default=None)):
    require_bearer(authorization)
    out = generate_text(
        user_text=entrada.texto,
        max_new_tokens=entrada.max_new_tokens or MAX_NEW_TOKENS,
        do_sample=bool(entrada.do_sample),
        temperature=float(entrada.temperature or 0.7),
        top_p=float(entrada.top_p or 0.9),
    )
    return {"text": out}
