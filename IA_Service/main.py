import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# ---------------- Config ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-2B-Instruct")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Optional: control HF cache (helps avoid re-downloading between runs on the same machine)
# os.environ["HF_HOME"] = "/root/.cache/huggingface"

# ---------------- App ----------------
app = FastAPI(title="Qwen2-VL-2B-Instruct Text API", version="1.0.0")

class EvalRequest(BaseModel):
    input: str
    max_new_tokens: int | None = 128
    temperature: float | None = None
    top_p: float | None = None

class EvalResponse(BaseModel):
    output: str
    tokens_generated: int

# Globals (populated at startup)
processor = None
model = None

def build_messages(user_text: str):
    # Qwen2-VL is chat-based and multimodal; here we only send pure text.
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": user_text}
        ]
    }]

@app.on_event("startup")
def load_model_once():
    global processor, model
    if processor is None or model is None:
        # Load once, keep in memory for all requests
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        # Put the text encoder/decoder on device if needed
        model.to(device)
        model.eval()
        torch.set_grad_enabled(False)

@app.post("/v1/eval", response_model=EvalResponse)
def eval_endpoint(req: EvalRequest):
    global processor, model
    if processor is None or model is None:
        load_model_once()

    messages = build_messages(req.input)
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Prepare inputs
    inputs = processor(text=[text_prompt], return_tensors="pt")
    # Move to device/dtype carefully (Qwen2-VL expects inputs on same device as model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=req.max_new_tokens or 128,
    )
    # Optional decoding settings
    if req.temperature is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        gen_kwargs["do_sample"] = True
        gen_kwargs["top_p"] = req.top_p

    # Generate
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    # Decode
    decoded = processor.batch_decode(out, skip_special_tokens=True)[0]

    # Some Qwen chat templates include the prompt + assistant text; we’ll try to trim after the last 'assistant' marker if present.
    # If the full text is fine for you, you can just return `decoded`.
    # Below is a minimal, safe approach that returns the full decoded string.
    return EvalResponse(output=decoded, tokens_generated=int(out[0].shape[-1]))
