# server.py
import os, torch, threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

app = FastAPI(title="Qwen2-VL API")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = None
model = None
model_ready = False

def load_model():
    global processor, model, model_ready
    try:
        processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        model.to(device).eval()
        torch.set_grad_enabled(False)
        model_ready = True
    except Exception as e:
        # Log the exception; healthz will show "starting" until you redeploy
        print("Model load failed:", repr(e))

@app.on_event("startup")
def kickoff():
    threading.Thread(target=load_model, daemon=True).start()

class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

@app.get("/healthz")
def health():
    return {"status": "ready" if model_ready else "starting"}

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is loading")
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
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]
    return EvalResponse(output=decoded, tokens_generated=int(out[0].shape[-1]))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, workers=1)

