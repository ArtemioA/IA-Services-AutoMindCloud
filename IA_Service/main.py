# main.py — Cloud Run (Qwen2-VL CPU) | descarga robusta + carga desde carpeta local
import os
import time
import threading
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Logging ----------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qwen2vl")

# ---------------- Env & defaults ----------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")
ALLOW_DOWNLOAD = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT = int(os.environ.get("PORT", "8080"))

# Cloud Run: caches deben ser escribibles (en /tmp)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Acelerador de descargas (ya tienes hf-transfer en requirements)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

# --------------- App state ----------------
app = FastAPI(title="Qwen2-VL (CPU)")
_state = {
    "ready": False,
    "loading": False,
    "error": None,
    "device": "cpu",
    "model_repo": MODEL_REPO,
    "mode": "online" if ALLOW_DOWNLOAD else "offline",
    "t0": time.time(),
}
_lock = threading.Lock()
_model = {"proc": None, "tok": None, "model": None}

# --------------- Schemas ------------------
class Ask(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.0


# --------------- Helpers ------------------
def _download_snapshot(repo: str, target_dir: str) -> str:
    """
    Descarga robusta del repo HF a 'target_dir' y devuelve la carpeta resuelta.
    Usa allow_patterns para evitar traer artefactos innecesarios.
    """
    from huggingface_hub import snapshot_download
    os.makedirs(target_dir, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN")

    logger.info(f"📥 Prefetch de snapshot HF: repo={repo} -> {target_dir}")
    resolved = snapshot_download(
        repo_id=repo,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
        # Archivos típicos necesarios para un modelo Transformers (incluye shards .safetensors)
        allow_patterns=[
            "*.safetensors", "*.bin", "*.json", "*.txt",
            "*.model", "*.py", "*.md", "tokenizer*",
            "config.json", "generation_config.json", "pytorch_model*"
        ],
        # confiamos en los reintentos del cliente; Cloud Run reinicia el contenedor si falla duro
    )
    logger.info(f"✅ Snapshot lista en: {resolved}")
    return resolved


def _load_model_locked():
    """Carga el modelo a RAM. Debe llamarse dentro de _lock."""
    if _state["ready"] or _state["loading"]:
        return
    _state["loading"], _state["error"] = True, None

    logger.info(
        f"🚀 Cargando modelo: repo={MODEL_REPO}, dir={MODEL_DIR}, "
        f"allow_download={ALLOW_DOWNLOAD}"
    )

    try:
        import torch
        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["device"] = device
        logger.info(f"⚙️  Dispositivo: {device}")

        # Descarga snapshot completa y consistente a disco (si ALLOW_DOWNLOAD)
        local_path = MODEL_DIR
        if ALLOW_DOWNLOAD:
            local_path = _download_snapshot(MODEL_REPO, MODEL_DIR)
        else:
            if not os.path.exists(local_path) or not os.listdir(local_path):
                raise RuntimeError(
                    f"Modo offline y carpeta vacía: {local_path}. "
                    f"Hornea el modelo en la imagen o habilita ALLOW_DOWNLOAD=1."
                )

        # Carga SIEMPRE desde carpeta local ya resuelta
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        logger.info("📦 Cargando processor/tokenizer/model desde carpeta local...")
        proc = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)

        # CPU-friendly: float32; para CPU a veces conviene float32 por compatibilidad
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            local_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cpu").eval()

        _model["proc"], _model["tok"], _model["model"] = proc, tok, model
        _state["ready"] = True
        logger.info("🎉 Modelo cargado y listo para inferencia")

    except Exception as e:
        import traceback
        msg = f"{type(e).__name__}: {e}"
        _state["error"] = msg
        logger.error(f"💥 Error cargando modelo: {msg}")
        logger.error("📋 Traceback:\n%s", traceback.format_exc())
    finally:
        # Si falló, ready será False y error tendrá detalle; loading vuelve a False
        _state["loading"] = False


def _ensure_loaded_bg():
    """Dispara la carga en background si aún no está lista."""
    if _state["ready"] or _state["loading"]:
        return

    def _bg():
        with _lock:
            if not _state["ready"]:
                _load_model_locked()

    threading.Thread(target=_bg, daemon=True).start()


def _ensure_loaded_blocking():
    """Bloquea hasta intentar cargar (usado por /generate)."""
    if _state["ready"]:
        return
    with _lock:
        if not _state["ready"]:
            _load_model_locked()


# --------------- Endpoints ----------------
@app.get("/")
def root():
    """Devuelve estado y dispara carga lazy si procede."""
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "device": _state["device"],
        "mode": _state["mode"],
        "model_repo": _state["model_repo"],
        "model_dir": MODEL_DIR,
        "uptime_s": round(time.time() - _state["t0"], 1),
        "error": _state["error"],
        "allow_download": ALLOW_DOWNLOAD,
    }


@app.get("/healthz")
def healthz():
    """Health check rápido (no bloquea)."""
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}


@app.post("/generate")
def generate(body: Ask):
    """Generación de texto (bloquea hasta intentar cargar el modelo)."""
    logger.info(f"🎯 /generate prompt={body.prompt[:100]!r}...")
    _ensure_loaded_blocking()

    if not _state["ready"]:
        # Devuelve 503 mientras está cargando o si hay error
        status = "loading" if _state["loading"] else "cold"
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready. status={status}; error={_state['error']}"
        )

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt vacío")

    try:
        import torch
        tok = _model["tok"]
        model = _model["model"]

        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(body.max_new_tokens or 128),
                do_sample=(float(body.temperature or 0.0) > 0.0),
                temperature=float(body.temperature or 0.0),
            )

        text = tok.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"✅ Generación OK ({len(text)} chars)")
        return {"ok": True, "text": text}

    except Exception as e:
        logger.error("💥 Error en inferencia: %s: %s", type(e).__name__, e)
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")


# --------------- Startup ----------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Servicio iniciando… carga lazy en background")
    _ensure_loaded_bg()


if __name__ == "__main__":
    import uvicorn
    logger.info(f"🏃 Ejecutando uvicorn en 0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
