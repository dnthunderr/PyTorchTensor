from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uuid
import time
from transformers import logging
logging.set_verbosity_error()

# Initialize FastAPI
app = FastAPI(title="Local Together AI Clone")

# CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL_NAME = "gpt2"  # Change to "mistralai/Mistral-7B-v0.1" for better quality
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else "cpu",
    ignore_mismatched_sizes=True  # For smaller models like gpt2
)

# Define request/response models
class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[dict]

# Routes
@app.get("/health")
def health():
    """Health check"""
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/api/generate")
async def generate(request: CompletionRequest):
    """Generate text (compatible with Together API format)"""
    try:
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4()}",
            model=MODEL_NAME,
            created=int(time.time()),
            choices=[{
                "text": generated_text,
                "finish_reason": "length",
                "index": 0
            }]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/completions")
async def completions_v1(request: CompletionRequest):
    """Alternative endpoint for OpenAI-compatible clients"""
    return await generate(request)