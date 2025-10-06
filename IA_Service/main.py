# main.py — Qwen2-VL (baked/offline) para Cloud Run
# - Carga SOLO desde MODEL_DIR (horneado en la imagen)
# - Sin descargas en runtime (local_files_only=True)
# - Endpoints: /healthz, /_diag, /generate

import os
import sys
import glob
import json
import logging
from threading import Lock
from typing import Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ========= Config =========
MODEL_DIR = os.getenv("MODEL_DIR", "/models/Qwen2-VL-2B-Instruct")
PORT = int(os.getenv("PORT", "8080"))

# Caches seguros (no deberían usarse en modo baked, pero no molestan)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")

# CPU friendly
torch.set_num_threads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# ========= Logging =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("qwen2vl-baked")
log.info("🔧 Config: MODEL_DIR=%s device=%s", MODEL_DIR, device)

# ========= App =========
app = FastAPI(title="Qwen2-VL (baked/offline)")

# Singletons
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
    """Lista rápida del contenido de MODEL_DIR para diagnóstico."""
    if not os.path.isdir(MODEL_DIR):
        return {"exists": False, "path": MODEL_DIR}
    files = sorted(glob.glob(os.path.join(MODEL_DIR, "*")))
    head = [os.path.basename(p) for p in files[:max_files]]
    has_any_weights = (
        os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors"))
        or len(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors"))) > 0
        or os.path.isfile(os.path.join(MODEL_DIR, "pytorch_model.bin"))
        or len(glob.glob(os.path.join(MODEL_DIR, "pytorch_model-*.bin"))) > 0
    )
    return {
        "exists": True,
        "path": MODEL_DIR,
        "count": len(files),
        "sample": head,
        "has_config": os.path.isfile(os.path.join(MODEL_DIR, "config.json")),
        "has_weights": has_any_weights,
    }


def _validate_baked_contents():
    """Falla temprano si faltan archivos esenciales del modelo."""
    info = _ls_model_dir()
    if not info.get("exists"):
        raise RuntimeError(f"MODEL_DIR inexistente: {MODEL_DIR}")
    if not info.get("has_config"):
        raise RuntimeError(f"Falta config.json en {MODEL_DIR}")
    if not info.get("has_weights"):
        raise RuntimeError(
            "No se encontraron pesos en "
            f"{MODEL_DIR} (model.safetensors | model-*.safetensors | pytorch_model(.bin)). "
            "Revisa el Dockerfile: incluye --include de pesos al descargar."
        )


def _load_once():
    """Carga única desde disco local (sin red)."""
    global _model, _processor
    if _model is not None and _processor is not None:
        return
    with _lock:
        if _model is not None and _processor is not None:
            return

        _validate_baked_contents()

        log.info("📥 Cargando processor local-only…")
        _processor = AutoProcessor.from_pretrained(
            MODEL_DIR, trust_remote_code=True, local_files_only=True
        )
        log.info("✅ Processor cargado")

        log.info("📥 Cargando modelo local-only… (device=%s)", device)
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
        log.info("✅ Modelo cargado desde %s", MODEL_DIR)


@app.get("/healthz")
def healthz():
    try:
        _load_once()
        return {"ok": True, "device": device, "model_dir": MODEL_DIR}
    except Exception as e:
        # No 500: devolvemos detalle para facilitar diagnóstico
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
    try:
        _load_once()
        info["load_ok"] = True
    except Exception as e:
        info["load_ok"] = False
        info["load_error"] = f"{type(e).__name__}: {e}"
    return info


@app.post("/generate")
def generate(r: GenReq):
    try:
        _load_once()
        # Chat-template mínimo (texto-solo)
        messages = [{"role": "user", "content": [{"type": "text", "text": r.prompt}]}]
        prompt = _processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = _processor(text=[prompt], return_tensors="pt")
        inputs = {k: v.to(_model.device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(r.max_new_tokens)}
        if r.do_sample:
            gen_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": float(r.temperature),
                    "top_p": float(r.top_p),
                }
            )

        with torch.inference_mode():
            out = _model.generate(**inputs, **gen_kwargs)

        # Solo la continuación (quita el prompt)
        in_len = inputs["input_ids"].shape[-1]
        gen_ids = out[0, in_len:]
        text = _processor.batch_decode(
            gen_ids.unsqueeze(0), skip_special_tokens=True
        )[0].strip()
        if text.startswith("assistant\n"):
            text = text[len("assistant\n") :].strip()
        return {"text": text}
    except Exception as e:
        log.exception("❌ Error en /generate")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
