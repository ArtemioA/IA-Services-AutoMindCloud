# main.py — Cloud Run (Qwen2-VL CPU) | online/offline with safe fallbacks
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
MODEL_DIR  = os.environ.get("MODEL_DIR", "/tmp/models/Qwen2-VL-2B-Instruct")  # puede ser vacío
ALLOW_DL   = os.environ.get("ALLOW_DOWNLOAD", "1") == "1"
PORT       = int(os.environ.get("PORT", "8080"))

logger.info(f"🔧 Configuración: MODEL_REPO={MODEL_REPO}, MODEL_DIR={MODEL_DIR}, ALLOW_DL={ALLOW_DL}")

# Writable caches (Cloud Run)
os.environ.setdefault("HF_HOME", "/tmp/hf")
os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/hf/transformers")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/tmp/hf/hub")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Disable hf_transfer if lib not present
try:
    import importlib.util
    if importlib.util.find_spec("hf_transfer") is None:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        logger.info("🔧 hf_transfer no encontrado, deshabilitado")
    else:
        logger.info("✅ hf_transfer disponible")
except Exception as e:
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    logger.warning(f"🔧 Error verificando hf_transfer: {e}")

# ---------------- App ----------------
app = FastAPI(title="Qwen2-VL (CPU)")

_state = {
    "ready": False,
    "loading": False,
    "error": None,
    "model_repo": MODEL_REPO,
    "model_dir": MODEL_DIR,
    "device": "cpu",
    "mode": "online" if ALLOW_DL else "offline",
    "t0": time.time(),
}
_lock = threading.Lock()
_model = {"proc": None, "tok": None, "model": None}

# ---------------- Helpers ----------------
def _has_local_snapshot(path: str) -> bool:
    """Considera 'snapshot válido' solo si existe config.json (evita carpetas vacías)."""
    if not path or not os.path.isdir(path):
        logger.info(f"🔍 Snapshot local: path '{path}' no es directorio válido")
        return False
    
    config_path = os.path.join(path, "config.json")
    exists = os.path.exists(config_path)
    logger.info(f"🔍 Snapshot local: {path} -> {'✅' if exists else '❌'} (config.json: {exists})")
    return exists

def _download_if_needed():
    """Descarga el repo a MODEL_DIR si está permitido y aún no existe snapshot válido."""
    if not ALLOW_DL:
        logger.info("🔴 Descargas deshabilitadas por ALLOW_DOWNLOAD=0")
        return
    
    if not MODEL_DIR:
        logger.info("🔴 MODEL_DIR no definido, no se puede descargar")
        return
    
    if _has_local_snapshot(MODEL_DIR):
        logger.info("✅ Snapshot local encontrado, omitiendo descarga")
        return
    
    logger.info(f"⬇️  Iniciando descarga de {MODEL_REPO} a {MODEL_DIR}")
    try:
        from huggingface_hub import snapshot_download
        
        # Crear directorio si no existe
        os.makedirs(MODEL_DIR, exist_ok=True)
        logger.info(f"📁 Directorio creado: {MODEL_DIR}")
        
        # Verificar permisos de escritura
        test_file = os.path.join(MODEL_DIR, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            logger.info("✅ Permisos de escritura verificados")
        except Exception as e:
            logger.error(f"❌ Sin permisos de escritura en {MODEL_DIR}: {e}")
            raise
        
        # Descargar modelo
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
        logger.info(f"🔑 Token HF: {'✅ Presente' if token else '❌ No presente (acceso público)'}")
        
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True
        )
        logger.info("✅ Descarga completada exitosamente")
        
    except Exception as e:
        logger.error(f"🔴 Error en descarga: {type(e).__name__}: {e}")
        raise

def _choose_src():
    """
    Decide la fuente de carga:
      - Si hay snapshot local válido -> usar carpeta local.
      - Si no hay y ALLOW_DL y MODEL_DIR definido -> descargar y usar carpeta.
      - Si no, usar repo online (requiere egress; token opcional).
    """
    use_local = _has_local_snapshot(MODEL_DIR)
    
    if not use_local and ALLOW_DL and MODEL_DIR:
        logger.info("🔄 No hay snapshot local, intentando descarga...")
        try:
            _download_if_needed()
            use_local = _has_local_snapshot(MODEL_DIR)
            if use_local:
                logger.info("✅ Descarga exitosa, usando modelo local")
            else:
                logger.warning("❌ Descarga completada pero snapshot aún no válido")
        except Exception as e:
            logger.warning(f"⚠️  Descarga falló, usando repo online: {e}")
            # Si falla la descarga, caeremos a repo online.
            pass
    
    source = MODEL_DIR if use_local else MODEL_REPO
    logger.info(f"🎯 Fuente seleccionada: {source} ({'local' if use_local else 'remote'})")
    return source, use_local

# ---------------- Model loading ----------------
def _load_model_locked():
    """
    Lazy load (thread-safe).
    - Offline (ALLOW_DL=False): exige snapshot local válido, local_files_only=True.
    - Online (ALLOW_DL=True): usa local si existe; si no, repo con local_files_only=False.
    """
    if _state["ready"] or _state["loading"]:
        logger.info("🔄 Modelo ya listo o cargando, omitiendo")
        return
    
    _state["loading"], _state["error"] = True, None
    logger.info("🚀 Iniciando carga del modelo...")
    
    try:
        import torch
        from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["device"] = device
        logger.info(f"⚙️  Dispositivo: {device}")

        src, using_local = _choose_src()
        local_only = (not ALLOW_DL)  # offline fuerza local-only
        
        logger.info(f"📥 Cargando desde: {src}")
        logger.info(f"🔒 Modo local_only: {local_only}")

        # Cargas con trust_remote_code y respetando offline/online
        token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or None  # puede ser None
        logger.info(f"🔑 Token HF para carga: {'✅ Presente' if token else '❌ No presente'}")

        # Cargar processor
        logger.info("📥 Cargando processor...")
        proc = AutoProcessor.from_pretrained(
            src, 
            trust_remote_code=True, 
            local_files_only=local_only, 
            token=token
        )
        logger.info("✅ Processor cargado")

        # Cargar tokenizer  
        logger.info("📥 Cargando tokenizer...")
        tok = AutoTokenizer.from_pretrained(
            src, 
            trust_remote_code=True, 
            local_files_only=local_only, 
            token=token
        )
        logger.info("✅ Tokenizer cargado")

        # Cargar modelo
        logger.info("📥 Cargando modelo...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            src, 
            trust_remote_code=True, 
            local_files_only=local_only, 
            token=token,
            torch_dtype=torch.float32,  # CPU friendly
        ).to("cpu").eval()
        logger.info("✅ Modelo cargado y en modo evaluación")

        _model["proc"], _model["tok"], _model["model"] = proc, tok, model
        _state["ready"] = True
        logger.info("🎉 Modelo completamente cargado y listo")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        _state["error"] = error_msg
        logger.error(f"💥 Error cargando modelo: {error_msg}")
        # Log adicional para debugging
        import traceback
        logger.error(f"📋 Traceback: {traceback.format_exc()}")
    finally:
        _state["loading"] = False
        logger.info(f"🏁 Estado final: ready={_state['ready']}, error={_state['error']}")

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
    temperature: float | None = 0.0  # greedy por defecto (CPU)

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "device": _state["device"],
        "mode": _state["mode"],
        "model_dir": _state["model_dir"],
        "model_repo": _state["model_repo"],
        "uptime_s": round(time.time() - _state["t0"], 1),
        "error": _state["error"],
        "allow_download": ALLOW_DL,
    }

@app.get("/status", summary="Status (warms up)")
def status():
    _ensure_loaded_bg()
    return {
        "ok": True,
        "status": "ready" if _state["ready"] else ("loading" if _state["loading"] else "cold"),
        "mode": _state["mode"],
        "error": _state["error"],
    }

@app.get("/healthz", summary="Health probe (no load)")
def healthz():
    return {"ok": True, "ready": _state["ready"], "error": _state["error"]}

@app.get("/_dns", summary="DNS to huggingface.co (dev aid)")
def dns_check():
    try:
        host = socket.gethostbyname("huggingface.co")
        logger.info(f"🌐 DNS resuelto: huggingface.co -> {host}")
        return {"ok": True, "host": host}
    except Exception as e:
        logger.error(f"🌐 DNS error: {e}")
        return {"ok": False, "error": str(e)}

@app.get("/_netcheck", summary="HTTP checks to HF and CDN")
def netcheck():
    import requests
    out = {}
    urls = [
        "https://huggingface.co",
        f"https://huggingface.co/api/models/{MODEL_REPO}",
        "https://cdn-lfs.huggingface.co/favicon.ico",
    ]
    
    for url in urls:
        try:
            logger.info(f"🌐 Probando conectividad a: {url}")
            r = requests.get(url, timeout=10)
            out[url] = {"ok": True, "code": r.status_code}
            logger.info(f"✅ {url} -> HTTP {r.status_code}")
        except Exception as e:
            out[url] = {"ok": False, "error": str(e)}
            logger.error(f"❌ {url} -> Error: {e}")
    
    return out

@app.get("/_env", summary="Debug environment variables")
def env_debug():
    """Endpoint para debug de variables de entorno"""
    relevant_envs = {
        k: v for k, v in os.environ.items() 
        if any(prefix in k for prefix in ['HF_', 'TRANSFORMERS_', 'HUGGINGFACE_', 'MODEL_', 'ALLOW_'])
    }
    return {
        "environment": relevant_envs,
        "current_working_dir": os.getcwd(),
        "tmp_contents": os.listdir('/tmp') if os.path.exists('/tmp') else "No /tmp",
    }

@app.post("/generate", summary="Text-only generation")
def generate(body: Ask):
    logger.info(f"🎯 Solicitud generate recibida: prompt='{body.prompt[:100]}...'")
    
    # Carga perezosa (bloqueante en la primera vez)
    try:
        _ensure_loaded_blocking()
    except Exception as e:
        logger.error(f"💥 Error en ensure_loaded: {e}")
        raise HTTPException(status_code=500, detail=f"Load error: {e}")

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

# ---------------- Local dev ----------------
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🏃 Servidor local iniciando en puerto {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
