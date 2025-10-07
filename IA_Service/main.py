# main.py — Qwen2-VL-2B en Cloud Run (CPU, OFFLINE) con diagnóstico fuerte y carga “eager”
import os
import socket
import threading
from pathlib import Path
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ========= Estado global =========
model_ready = threading.Event()
last_error: Optional[str] = None
model = None
processor = None

# ========= Config por entorno =========
ALLOW_DOWNLOAD = os.getenv("ALLOW_DOWNLOAD", "0") == "1"
FORCE_ONLINE   = os.getenv("FORCE_ONLINE",   "0") == "1"
MODEL_REPO     = os.getenv("MODEL_REPO", "Qwen/Qwen2-VL-2B-Instruct")
MODEL_DIR      = os.getenv("MODEL_DIR",  "/tmp/models/Qwen2-VL-2B-Instruct")
HF_HOME        = os.getenv("HF_HOME", "/tmp/hf")
PORT           = int(os.getenv("PORT", "8080"))
OMP_THREADS    = int(os.getenv("OMP_NUM_THREADS", "1"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
REQUIRED_PATTERNS = ["*.safetensors", "*.bin"]  # al menos uno

app = FastAPI(title="Qwen2-VL CPU demo", version="1.0")


# ========= Helpers =========
def _set_error(msg: str):
    global last_error
    last_error = msg
    model_ready.clear()
    print(f"[LOAD][ERROR] {msg}", flush=True)


def _ok(msg: str):
    print(f"[LOAD][OK] {msg}", flush=True)


def _have_local_snapshot(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.iterdir())


def _list_some(path: str, limit: int = 80) -> List[str]:
    out: List[str] = []
    root = Path(path)
    if not root.exists():
        return out
    for i, f in enumerate(root.rglob("*")):
        try:
            rel = str(f.relative_to(root))
        except Exception:
            rel = str(f)
        out.append(rel)
        if i + 1 >= limit:
            break
    return out


def _validate_snapshot(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return f"Snapshot no existe: {path}"
    if not any(p.iterdir()):
        return f"Snapshot vacío en: {path}"
    missing = [f for f in REQUIRED_FILES if not (p / f).exists()]
    has_weights = any(next(p.glob(pat), None) for pat in REQUIRED_PATTERNS)
    problems = []
    if missing:
        problems.append(f"Faltan archivos: {', '.join(missing)}")
    if not has_weights:
        problems.append("No encuentro pesos (*.safetensors|*.bin)")
    if problems:
        return f"Snapshot incompleto en {path}: " + " ; ".join(problems)
    return None


# ========= Hilo de carga =========
def _load_model():
    global model, processor
    try:
        # 0) Torch threads para CPU
        try:
            import torch
            torch.set_num_threads(max(1, OMP_THREADS))
            _ok(f"Torch threads: {torch.get_num_threads()}")
        except Exception as e:
            _ok(f"Torch threads no ajustados: {e}")

        # 1) Resolver snapshot (OFFLINE por defecto)
        if not ALLOW_DOWNLOAD:
            if not _have_local_snapshot(MODEL_DIR):
                _set_error(f"Modelo no encontrado en {MODEL_DIR} y ALLOW_DOWNLOAD=0 (offline).")
                return
            err = _validate_snapshot(MODEL_DIR)
            if err:
                _set_error(err + f" | archivos_vistos={_list_some(MODEL_DIR, 30)}")
                return
            snapshot_path = MODEL_DIR
            _ok(f"Usando snapshot LOCAL: {snapshot_path}")
        else:
            from huggingface_hub import snapshot_download
            if _have_local_snapshot(MODEL_DIR) and not FORCE_ONLINE:
                snapshot_path = MODEL_DIR
                _ok(f"Usando snapshot LOCAL: {snapshot_path}")
            else:
                _ok(f"Descargando {MODEL_REPO} (HF_HOME={HF_HOME}) ...")
                snapshot_path = snapshot_download(repo_id=MODEL_REPO, local_dir=None)
                _ok(f"Snapshot cache HF: {snapshot_path}")
            err = _validate_snapshot(snapshot_path)
            if err:
                _set_error(err)
                return

        # 2) Cargar clases (transformers >= 4.44)
        _ok("Importando clases de transformers...")
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except Exception as e:
            _set_error(
                "Tu 'transformers' no expone Qwen2VLForConditionalGeneration. "
                "Usa transformers >= 4.44.x. Detalle: " + str(e)
            )
            return

        # 3) Cargar processor
        _ok("Cargando AutoProcessor...")
        processor_local = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
        _ok("AutoProcessor cargado.")

        # 4) Cargar modelo (amigable a CPU)
        _ok("Cargando modelo (low_cpu_mem_usage, attn=eager)...")
        import torch
        model_local = Qwen2VLForConditionalGeneration.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            device_map=None,                # CPU explícito
            low_cpu_mem_usage=True,         # reduce picos de RAM al cargar
            attn_implementation="eager",    # evita backends no disponibles
            torch_dtype=torch.float32       # CPU estable (float16 en CPU no aplica)
        )
        _ok("Pesos cargados; moviendo a CPU y eval()...")
        model_local.to("cpu")
        model_local.eval()

        # 5) Publicar en global
        global model, processor
        model = model_local
        processor = processor_local

        model_ready.set()
        _ok("Modelo listo.")
    except Exception as e:
        _set_error(f"Load error: {e!s}")


# ========= Eventos de app =========
@app.on_event("startup")
def on_startup():
    print("[BOOT] Vars:", {
        "ALLOW_DOWNLOAD": ALLOW_DOWNLOAD,
        "FORCE_ONLINE": FORCE_ONLINE,
        "MODEL_REPO": MODEL_REPO,
        "MODEL_DIR": MODEL_DIR,
        "HF_HOME": HF_HOME,
        "HOSTNAME": socket.gethostname(),
        "OMP_NUM_THREADS": OMP_THREADS,
    }, flush=True)
    t = threading.Thread(target=_load_model, daemon=True)
    t.start()


# ========= Schemas =========
class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 64


# ========= Endpoints =========
@app.get("/status")
def status():
    return {"ok": True}


@app.get("/readyz")
def readyz():
    return {"ready": model_ready.is_set(), "error": last_error}


@app.get("/_netcheck")
def netcheck():
    import http.client
    try:
        conn = http.client.HTTPSConnection("huggingface.co", timeout=3)
        conn.request("HEAD", "/")
        r = conn.getresponse()
        return {"internet_ok": r.status < 500, "status": r.status}
    except Exception as e:
        return {"internet_ok": False, "error": str(e)}


@app.get("/env")
def env():
    return {
        "ALLOW_DOWNLOAD": ALLOW_DOWNLOAD,
        "FORCE_ONLINE": FORCE_ONLINE,
        "MODEL_REPO": MODEL_REPO,
        "MODEL_DIR": MODEL_DIR,
        "HF_HOME": HF_HOME,
    }


@app.get("/disk")
def disk():
    def _fs(p: str) -> Dict[str, int]:
        import os
        try:
            s = os.statvfs(p)
            return {"total": s.f_blocks * s.f_frsize, "free": s.f_bavail * s.f_frsize}
        except Exception as e:
            return {"error": str(e)}  # type: ignore[return-value]
    return {
        "models_dir": MODEL_DIR,
        "models_dir_exists": Path(MODEL_DIR).exists(),
        "tmp_fs": _fs("/tmp"),
        "app_fs": _fs("/app"),
    }


@app.get("/models")
def models():
    return {
        "dir": MODEL_DIR,
        "exists": Path(MODEL_DIR).exists(),
        "some_files": _list_some(MODEL_DIR, 80),
    }


@app.post("/generate")
def generate(req: GenRequest):
    if not model_ready.is_set():
        raise HTTPException(status_code=503, detail=last_error or "Model not ready.")
    try:
        import torch
        with torch.no_grad():
            inputs = processor(text=req.prompt, return_tensors="pt")
            tok = getattr(processor, "tokenizer", None)
            extra = {}
            if tok is not None:
                if getattr(tok, "eos_token_id", None) is None and getattr(tok, "eos_token", None):
                    tok.eos_token_id = tok.convert_tokens_to_ids(tok.eos_token)
                if getattr(tok, "pad_token_id", None) is None:
                    tok.pad_token = tok.eos_token or tok.pad_token or tok.unk_token
                    tok.pad_token_id = tok.convert_tokens_to_ids(tok.pad_token)
                extra = {"eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id}
            out = model.generate(**inputs, max_new_tokens=req.max_new_tokens or 64, **extra)
        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            text = processor.tokenizer.decode(out[0], skip_special_tokens=True)
        else:
            text = str(out[0].tolist())
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference error: {e!s}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
