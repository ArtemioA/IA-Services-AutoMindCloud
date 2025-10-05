# main.py — Qwen2-VL horneado (Cloud Run listo)
import os, threading, time, glob
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = FastAPI(title="Qwen2-VL (baked)")

# --- Config ---
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
MAX_NEW_TOKENS_DEFAULT = int(os.environ.get("MAX_NEW_TOKENS", "128"))
OMP = int(os.environ.get("OMP_NUM_THREADS", "1"))

# Volátiles (no descargamos en runtime)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# Threads CPU al mínimo para contención
torch.set_num_threads(OMP)
os.environ.setdefault("OMP_NUM_THREADS", str(OMP))
os.environ.setdefault("MKL_NUM_THREADS", "1")

_state = {
    "ready": False,
    "error": None,
    "device": "cpu",
    "processor": None,
    "model": None
}
_lock = threading.Lock()


# ---------- Models ----------
class Ask(BaseModel):
    prompt: str = Field(..., description="Texto del prompt")
    max_new_tokens: Optional[int] = Field(None, ge=1, le=1024)
    temperature: Optional[float] = Field(1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0)
    # (Opcional) URLs/base64 de imágenes, si más tarde quieres VL completo:
    images: Optional[List[str]] = None  # no se usan aún; queda para extender


def _lazy_load():
    with _lock:
        if _state["ready"] or _state["error"]:
            return
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _state["device"] = device

            model_path = os.path.abspath(MODEL_DIR)
            print(f"🔍 Loading local model from: {model_path}")
            if not os.path.isdir(model_path):
                raise RuntimeError(f"Model directory not found: {model_path}")

            proc = AutoProcessor.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).to("cpu").eval()

            _state["processor"] = proc
            _state["model"] = model
            _state["ready"] = True
            print("✅ Modelo cargado correctamente.")
        except Exception as e:
            _state["error"] = f"{type(e).__name__}: {e}"
            print(f"❌ Error al cargar modelo: {_state['error']}")


# Arranca hilo de carga DESPUÉS de bindear el puerto (ideal Cloud Run)
@app.on_event("startup")
def _kickoff_load():
    if not _state["ready"] and not _state["error"]:
        threading.Thread(target=_lazy_load, daemon=True).start()


# ---------- Routes ----------
@app.get("/")
def root():
    # si aún no arrancó el hilo por alguna razón, lánzalo
    if not _state["ready"] and not _state["error"]:
        threading.Thread(target=_lazy_load, daemon=True).start()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("error" if _state["error"] else "starting"),
        "device": _state["device"],
        "model_dir": MODEL_DIR,
        "error": _state["error"]
    }


@app.get("/healthz")
def healthz():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}


@app.get("/_ls_models")
def _ls_models():
    root = os.environ.get("MODEL_DIR", MODEL_DIR)
    exists = os.path.isdir(root)
    files = []
    if exists:
        for p in sorted(glob.glob(os.path.join(root, "*")))[:80]:
            try:
                s = os.stat(p)
                files.append({"name": os.path.basename(p), "size": s.st_size})
            except Exception:
                files.append({"name": os.path.basename(p), "size": None})
    return {"MODEL_DIR": root, "exists": exists, "files": files}


@app.post("/generate")
def generate(q: Ask):
    if not _state["ready"]:
        if _state["error"]:
            raise HTTPException(500, f"Load error: {_state['error']}")
        _lazy_load()
        # espera breve (hasta ~5s)
        for _ in range(50):
            if _state["ready"] or _state["error"]:
                break
            time.sleep(0.1)
        if not _state["ready"]:
            if _state["error"]:
                raise HTTPException(500, f"Load error: {_state['error']}")
            raise HTTPException(503, "Model loading, try again shortly.")

    proc = _state["processor"]
    model = _state["model"]

    max_new = q.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    max_new = int(max(1, min(max_new, 1024)))  # clamp defensivo

    # Para Qwen2-VL, los prompts funcionan mejor con chat_template:
    try:
        # Si el processor/tokenizer soporta .apply_chat_template
        prompt_txt = proc.apply_chat_template(
            [{"role": "user", "content": q.prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        # Fallback: usa texto tal cual
        prompt_txt = q.prompt

    inputs = proc(text=prompt_txt, return_tensors="pt")

    gen_kwargs = dict(
        max_new_tokens=max_new,
        do_sample=(q.temperature is not None and q.temperature > 0),
        temperature=float(q.temperature or 1.0),
        top_p=float(q.top_p or 0.9),
        pad_token_id=proc.tokenizer.eos_token_id if hasattr(proc, "tokenizer") else None,
        eos_token_id=proc.tokenizer.eos_token_id if hasattr(proc, "tokenizer") else None,
    )

    with torch.inference_mode():
        tokens = model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})

    text = proc.batch_decode(tokens, skip_special_tokens=True)[0]
    return {"ok": True, "text": text, "max_new_tokens": max_new}


@app.get("/_dns")
def dns_check():
    import socket
    hosts = ["huggingface.co", "cdn-lfs.huggingface.co", "google.com"]
    out = {}
    for h in hosts:
        try:
            out[h] = [ai[4][0] for ai in socket.getaddrinfo(h, 443)]
        except Exception as e:
            out[h] = f"ERROR: {e}"
    return out


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
