# main.py — Qwen2-VL-2B en Cloud Run (OFFLINE) con diagnóstico fuerte
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

# Archivos/patrones mínimos para considerar el snapshot válido
REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
REQUIRED_PATTERNS = ["*.safetensors", "*.bin"]  # al menos uno debe existir

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


def _list_some(path: str, limit: int = 60) -> List[str]:
    out: List[str] = []
    root = Path(path)
    if not root.exists():
        return out
    for i, f in enumerate(root.rglob("*")):
        rel = str(f.relative_to(root)) if f.exists() else str(f)
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
        # Torch threads (CPU)
        try:
            import torch
            torch.set_num_threads(max(1, OMP_THREADS))
        except Exception:
            pass

        # Política OFFLINE primero
        if not ALLOW_DOWNLOAD:
            if not _have_local_snapshot(MODEL_DIR):
                _set_error(f"Modelo no encontrado en {MODEL_DIR} y ALLOW_DOWNLOAD=0 (offline).")
                return
            err = _validate_snapshot(MODEL_DIR)
            if err:
                _set_error(err + f" | archivos_vistos={_list_some(MODEL_DIR, 30)}")
                return
            snapshot_path = MODEL_DIR
            _ok(f"Usando snapshot LOCAL en {snapshot_path} (offline)")
        else:
            # ONLINE (no recomendado para Cloud Run, pero lo dejamos correcto)
            from huggingface_hub import snapshot_download
            if _have_local_snapshot(MODEL_DIR) and not FORCE_ONLINE:
                snapshot_path = MODEL_DIR
                _ok(f"Usando snapshot LOCAL en {snapshot_path}")
            else:
                _ok(f"Descargando {MODEL_REPO} (HF_HOME={HF_HOME}) ...")
                snapshot_path = snapshot_download(repo_id=MODEL_REPO, local_dir=None)
                _ok(f"Snapshot en cache HF: {snapshot_path}")
            # Validación (por si local_dir apunta a carpeta propia)
            err = _validate_snapshot(snapshot_path)
            if err:
                _set_error(err)
                return

        # Carga de clases
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except Exception as e:
            _set_error(
                "Tu 'transformers' no expone Qwen2VLForConditionalGeneration. "
                "Usa transformers >= 4.44.x. Detalle: " + str(e)
            )
            return

        processor = AutoProcessor.from_pretrained(snapshot_path, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(snapshot_path, trust_remote_code=True)

        # CPU
        try:
            import torch
            model.to("cpu")
            model.eval()
        except Exception as e:
            _set_error(f"No pude mover el modelo a CPU: {e!s}")
            return

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
    # Espacio útil para depurar en /app y /tmp
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
            tokenizer = getattr(processor, "tokenizer", None)
            extra = {}
            if tokenizer is not None:
                # Establecer pad/eos si faltan (algunos snapshots)
                if getattr(tokenizer, "eos_token_id", None) is None and getattr(tokenizer, "eos_token", None):
                    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                if getattr(tokenizer, "pad_token_id", None) is None:
                    tokenizer.pad_token = tokenizer.eos_token or tokenizer.pad_token or tokenizer.unk_token
                    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                extra = {
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }

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
