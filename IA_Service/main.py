import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------- Config ----------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/flan-t5-small")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "256"))
REQUIRE_BEARER = os.environ.get("REQUIRE_BEARER", "false").lower() == "true"
API_TOKEN = os.environ.get("API_TOKEN", "")  # si activas auth

device = "cuda" if torch.cuda.is_available() else "cpu"

# En CPU el dtype queda en float32
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)

# ---------- App ----------
app = FastAPI(title="Explicador de Texto", version="1.0.0")

class Entrada(BaseModel):
    texto: str
    max_new_tokens: Optional[int] = None

def require_bearer(authorization: Optional[str]):
    if REQUIRE_BEARER:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != API_TOKEN:
            raise HTTPException(status_code=403, detail="Invalid token")

def construir_prompt(texto: str) -> str:
    # Prompt sencillo para “explicar”
    return f"Explica con claridad el siguiente mensaje en español:\n\n{texto}\n\nExplicación:"

@app.get("/")
def root():
    return {"message": "hello world", "model": MODEL_NAME}

@app.post("/explica")
def explica(entrada: Entrada, authorization: Optional[str] = Header(default=None)):
    require_bearer(authorization)

    prompt = construir_prompt(entrada.texto)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=entrada.max_new_tokens or MAX_NEW_TOKENS,
            # Para más creatividad, descomenta:
            # do_sample=True, temperature=0.7, top_p=0.9
        )
    explicacion = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    # El decode incluye el prompt en T5; extraemos solo la “Explicación” si aparece
    if "Explicación:" in explicacion:
        explicacion = explicacion.split("Explicación:", 1)[-1].strip()

    return {"explicacion": explicacion}
