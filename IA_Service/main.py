# main.py — Cloud Run (Qwen2-VL CPU) | Versión online
import os, socket, threading, time, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Setup logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------- Env config ----------------
MODEL_REPO = os.environ.get("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR  = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")
ALLOW_DL   = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT       = int(os.environ.get("PORT", "8080"))

logger.info(f"🔧 Configuración: MODEL_REPO={MODEL_REPO}, ALLOW_DL={ALLOW_DL}")

# Writable caches (Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------- App ----------------
app = FastAPI(title="Qwen2-VL (CPU)")

_state = {
    "ready": False,
    "loading": False,
    "error": None,
    "model_repo": MODEL_REPO,
    "device": "cpu",
    "mode": "online" if ALLOW_DL else "offline",
    "t0": time.time(),
}
_lock = threading.Lock()
_model = {"proc": None, "tok": None, "model": None}

# ---------------- Model loading ----------------
def _load_model_locked():
    """Carga el modelo desde Hugging Face Hub"""
    if _state["ready"] or _state["loading"]:
        return
    
    _state["loading"], _state["error"] = True, None
    logger.info("🚀 Iniciando carga del modelo desde Hugging Face Hub...")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["device"] = device
        logger.info(f"⚙️  Dispositivo: {device}")

        # Cargar desde Hugging Face Hub
        logger.info(f"📥 Descargando modelo: {MODEL_REPO}")
        
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        logger.info(f"🔑 Token HF: {'✅ Presente' if token else '❌ No presente (acceso público)'}")

        # Cargar componentes
        logger.info("📥 Cargando processor...")
        proc = AutoProcessor.from_pretrained(
            MODEL_REPO, 
            trust_remote_code=True, 
            token=token
        )
        logger.info("✅ Processor cargado")

        logger.info("📥 Cargando tokenizer...")
        tok = AutoTokenizer.from_pretrained(
            MODEL_REPO, 
            trust_remote_code=True, 
            token=token
        )
        logger.info("✅ Tokenizer cargado")

        logger.info("📥 Cargando modelo...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_REPO, 
            trust_remote_code=True, 
            token=token,
            torch_dtype=torch.float32,
        ).to("cpu").eval()
        logger.info("✅ Modelo cargado y en modo evaluación")

        _model["proc"], _model["tok"], _model["model"] = proc, tok, model
        _state["ready"] = True
        logger.info("🎉 Modelo completamente cargado y listo")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _state["error"] = error_msg
        logger.error(f"💥 Error cargando modelo: {error_msg}")
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")

def _ensure_loaded_bg():
    if _state["ready"] or _state["loading"]:
        return
    def _bg():
        with _lock:
            if not _state["ready"]:
                _load_model_locked()
    threading.Thread(target=_bg, daemon=True).start()

def _ensure_loaded_blocking():
    if _state["ready"]:
        return
    with _lock:
        if not _state["ready"]:
            _load_model_locked()

# ---------------- Schemas ----------------
class Ask(BaseModel):
    prompt: str
    max_new_tokens: int | None = 128
    temperature: float | None = 0.0

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "device": _state["device"],
        "mode": _state["mode"],
        "model_repo": _state["model_repo"],
        "uptime_s": round(time.time() - _state["t0"], 1),
        "error": _state["error"],
        "allow_download": ALLOW_DL,
    }

@app.get("/healthz")
def healthz():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.post("/generate")
def generate(body: Ask):
    logger.info(f"🎯 Solicitud generate recibida: prompt='{body.prompt[:100]}...'")
    
    _ensure_loaded_blocking()

    if not _state["ready"]:
        error_detail = f"Model not ready. status={'loading' if _state['loading'] else 'cold'}; error={_state['error']}"
        logger.error(f"💥 {error_detail}")
        raise HTTPException(status_code=503, detail=error_detail)

    prompt = (body.prompt or "").strip()
    if not prompt:
        logger.warning("⚠️  Prompt vacío recibido")
        raise HTTPException(status_code=422, detail="prompt vacío")

    try:
        import torch
        tok = _model["tok"]
        model = _model["model"]

        logger.info(f"⚡ Procesando generación: {len(prompt)} chars, max_tokens={body.max_new_tokens}")
        
        inputs = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=int(body.max_new_tokens or 128),
                do_sample=(body.temperature or 0.0) > 0.0,
                temperature=float(body.temperature or 0.0),
            )

        text = tok.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"✅ Generación completada: {len(text)} chars")
        return {"ok": True, "text": text}

    except Exception as e:
        logger.error(f"💥 Error en inferencia: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"inference error: {type(e).__name__}: {e}")

# ---------------- Startup event ----------------
@app.on_event("startup")
async def startup_event():
    """Inicia la carga del modelo en background al arrancar"""
    logger.info("🚀 Servicio iniciando...")
    _ensure_loaded_bg()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"🏃 Servidor local iniciando en puerto {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
