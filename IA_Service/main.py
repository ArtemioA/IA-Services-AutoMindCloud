# server.py
import os, torch, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_PATH = os.getenv("MODEL_NAME", "/models/Qwen2-VL-2B-Instruct")  # ruta local horneada
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = None
model = None

class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

def load_model_from_local():
    global processor, model
    # Importante: TRANSFORMERS_OFFLINE=1 y ruta local evitán red
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
    )
    model.to(device).eval()
    torch.set_grad_enabled(False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Carga SINCRÓNICA antes de aceptar tráfico → no hay 503
    load_model_from_local()
    yield
    # (opcional) liberar recursos en shutdown

app = FastAPI(title="Qwen2-VL API (offline baked)", lifespan=lifespan)

@app.get("/healthz")
def health():
    ok = processor is not None and model is not None
    return {"status": "ready" if ok else "starting", "device": device, "model_path": MODEL_PATH}

def _run_generate(req: EvalRequest) -> EvalResponse:
    messages = [{"role":"user","content":[{"type":"text","text":req.input}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[prompt], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=req.max_new_tokens or 128)
    if req.temperature is not None:
        gen_kwargs["do_sample"] = True; gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["do_sample"] = True; gen_kwargs["top_p"] = req.top_p

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    gen_tokens = int(out[0].shape[-1]) - int(inputs["input_ids"].shape[-1])
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=decoded, tokens_generated=max(gen_tokens, 0))

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest, response: Response):
    if processor is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return _run_generate(req)

if __name__ == "__main__":
    import uvicorn, os
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

