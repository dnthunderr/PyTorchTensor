# Local LLM Setup Guide (Alternative to api.together.xyz)

This guide documents how to set up local large language models for the Udacity Generative AI course instead of using paid API services like Together AI.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Option 1: Ollama (Easiest)](#option-1-ollama-easiest)
3. [Option 2: Hugging Face Transformers](#option-2-hugging-face-transformers)
4. [Option 3: LM Studio (GUI-based)](#option-3-lm-studio-gui-based)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements & Verification

### Windows + NVIDIA GPU Specs
- **OS**: Windows 10/11
- **GPU**: NVIDIA GPU (GeForce, RTX, Quadro, Tesla)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 20-50GB for models
- **CUDA**: 11.8+ or 12.x

### Verify Your GPU Setup

Open **PowerShell** and run:

```powershell
# Check NVIDIA driver
nvidia-smi

# You should see your GPU model and CUDA version
# Example output:
# NVIDIA-SMI 550.90    Driver Version: 550.90    CUDA Version: 12.4
```

If `nvidia-smi` doesn't work:
1. Download **NVIDIA Driver** from https://www.nvidia.com/Download/driverDetails.aspx
2. Update your GPU driver
3. Restart computer and verify again

---

## Step 1: NVIDIA GPU Setup

### Install NVIDIA CUDA Toolkit (if not installed)

1. **Check Current CUDA Version**
   ```powershell
   nvidia-smi
   # Look at "CUDA Version: X.X" in top right
   ```

2. **Download CUDA Toolkit** (if needed)
   - Visit: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → Windows 11 (or 10) → exe (local)
   - Download and install
   - Restart PowerShell after installation

3. **Verify Installation**
   ```powershell
   nvcc --version
   # Should show CUDA Compilation Tools version
   ```

### Install cuDNN (for optimal performance)

1. Register at: https://developer.nvidia.com/cudnn
2. Download cuDNN for your CUDA version
3. Extract and copy files to CUDA installation directory:
   ```powershell
   # Typically:
   # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x
   ```

---

## Step 2: Python Virtual Environment (.venv)

### Create Virtual Environment

Navigate to your project folder and create `.venv`:

```powershell
# Open PowerShell in your project directory
cd C:\GitHub\PyTorchTensor\Examples\generative-ai

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# You should see (.venv) in your terminal prompt
```

**If you get execution policy error:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Verify Python in .venv
```powershell
python --version
pip --version
# Both should show paths inside .venv folder
```

---

## Step 3: Install PyTorch with CUDA Support

### Install PyTorch with CUDA 12.1 (or your CUDA version)

In your activated `.venv`:

```powershell
# For CUDA 12.1 (most common)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Install Required Libraries

```powershell
# Core packages for LLMs
pip install transformers huggingface-hub accelerate bitsandbytes

# For API server
pip install fastapi uvicorn pydantic requests

# Additional tools
pip install flask flask-cors python-dotenv
```

### Verify GPU Access in Python

```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 4090 (or your GPU model)
```

---

## Step 4: Run Local Models

### Option A: Quick Test with Hugging Face Pipeline

Create file: `test_model.py`

```python
from transformers import pipeline
import torch

# Check GPU
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Load small model (fast)
print("Loading model...")
generator = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if torch.cuda.is_available() else -1  # 0=GPU, -1=CPU
)

# Generate text
prompt = "The future of AI is"
print(f"\nPrompt: {prompt}")
output = generator(prompt, max_length=50, num_return_sequences=1)
print(f"Output: {output[0]['generated_text']}")
```

Run it:
```powershell
python test_model.py
```

### Option B: Better Models (Mistral, Llama)

For better quality, use larger models:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "mistralai/Mistral-7B-v0.1"
# or: "meta-llama/Llama-2-7b-hf"
# or: "gpt2-medium"

print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    device_map="auto"  # Automatically use GPU
)

# Generate
prompt = "Explain machine learning:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Model Recommendations for Your GPU

| Model | Size | VRAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| gpt2 | 124M | 1GB | ⚡⚡⚡ | ⭐ |
| gpt2-medium | 355M | 2GB | ⚡⚡ | ⭐⭐ |
| distilgpt2 | 82M | 500MB | ⚡⚡⚡ | ⭐ |
| Mistral-7B | 7B | 16GB | ⚡ | ⭐⭐⭐⭐ |
| Llama-2-7b | 7B | 16GB | ⚡ | ⭐⭐⭐⭐ |

---

## Step 5: Local API Server (Together.xyz Compatible)

### Setup FastAPI Server to Replace Together API

Create file: `local_api_server.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uuid

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
    device_map="auto" if torch.cuda.is_available() else "cpu"
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
            created=int(torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))),
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

# Run with: uvicorn local_api_server:app --host 0.0.0.0 --port 5000 --reload
```

### Start the API Server

```powershell
# Make sure .venv is activated
.\.venv\Scripts\Activate.ps1

# Run the server
uvicorn local_api_server:app --host 0.0.0.0 --port 5000 --reload
```

**Output should show:**
```
INFO:     Started server process [12345]
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:5000
```

### Test the Server

Open another PowerShell window:

```powershell
# Test health endpoint
curl http://localhost:5000/health

# Test generation
$body = @{
    prompt = "Hello, how are you?"
    max_tokens = 50
} | ConvertTo-Json

curl -X POST `
  -H "Content-Type: application/json" `
  -Body $body `
  http://localhost:5000/api/generate
```

---

## Step 6: Update Your Code

### Before (Using Together API)

```python
import together

together.api_key = "your-paid-api-key"

response = together.Complete.create(
    prompt="Your prompt here",
    model="togethercomputer/llama-2-7b-chat",
    max_tokens=100
)

print(response["output"]["choices"][0]["text"])
```

### After (Using Local API Server)

```python
import requests
import json

# Use local API instead of together.xyz
LOCAL_API_URL = "http://localhost:5000/api/generate"

def generate_text(prompt, max_tokens=100):
    """Generate text using local API"""
    
    response = requests.post(
        LOCAL_API_URL,
        json={
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["text"]
    else:
        print(f"Error: {response.status_code}")
        return None

# Usage
output = generate_text("Explain quantum computing")
print(output)
```

### Create a Wrapper Class (Best Practice)

File: `local_llm.py`

```python
import requests
from typing import Optional

class LocalLLM:
    """Wrapper to replace Together API calls"""
    
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.endpoint = f"{api_url}/api/generate"
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate text similar to together.Complete.create()"""
        
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=120  # 2 minute timeout
            )
            
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            else:
                raise Exception(f"API Error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to local API. Is the server running?")

# Usage in your code
llm = LocalLLM()
output = llm.generate("What is machine learning?")
print(output)
```

### In Your Jupyter Notebooks

Replace Together imports:

```python
# OLD CODE (commented out)
# import together
# together.api_key = "your-api-key"

# NEW CODE
from local_llm import LocalLLM

llm = LocalLLM(api_url="http://localhost:5000")

# Use it
response = llm.generate(
    prompt="Your prompt",
    max_tokens=100
)
print(response)
```

---

## Full Workflow Example

### Complete Setup Script

Create: `setup_local_llm.ps1`

```powershell
# Setup Local LLM on Windows with NVIDIA GPU

Write-Host "1. Creating virtual environment..." -ForegroundColor Green
python -m venv .venv

Write-Host "2. Activating virtual environment..." -ForegroundColor Green
.\.venv\Scripts\Activate.ps1

Write-Host "3. Installing PyTorch with CUDA..." -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "4. Installing Transformers and dependencies..." -ForegroundColor Green
pip install transformers huggingface-hub accelerate bitsandbytes fastapi uvicorn pydantic requests

Write-Host "5. Verifying CUDA..." -ForegroundColor Green
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host "`nSetup complete! Next steps:" -ForegroundColor Green
Write-Host "1. Activate .venv: .\.venv\Scripts\Activate.ps1"
Write-Host "2. Run API server: uvicorn local_api_server:app --host 0.0.0.0 --port 5000"
Write-Host "3. In another terminal, test: curl http://localhost:5000/health"
```

Run it:
```powershell
.\setup_local_llm.ps1
```

---

## Troubleshooting

---

## Troubleshooting

### Issue: "nvidia-smi" Command Not Found

**Cause**: NVIDIA drivers not installed or not in PATH

**Solution**:
1. Check if installed: `C:\Program Files\NVIDIA Corporation\NVIDIA Driver`
2. If not found, download from: https://www.nvidia.com/Download/driverDetails.aspx
3. Restart PowerShell after installing
4. Restart computer if still not working

### Issue: CUDA Available: False (GPU not detected)

**Cause**: PyTorch installed without CUDA support

**Solution**:
```powershell
# Uninstall PyTorch
pip uninstall torch torchvision torchaudio -y

# Reinstall with correct CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then verify:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### Issue: Out of Memory (OOM) Error

**Cause**: Model too large for GPU VRAM

**Solution 1** - Use smaller model:
```python
# Instead of Mistral-7B, use:
model = "gpt2"  # 124MB
model = "gpt2-medium"  # 355MB
```

**Solution 2** - Use 8-bit quantization:
```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_8bit=True,
    device_map="auto"
)
```

**Solution 3** - Reduce batch size in API server:
```python
# In local_api_server.py, reduce max_new_tokens default
max_tokens: Optional[int] = 50  # Was 100
```

### Issue: API Server Crashes with "CUDA out of memory"

**Quick Fix 1** - Restart server and clear cache:
```powershell
# Stop the server (Ctrl+C)
# Then clear cache:
python -c "import torch; torch.cuda.empty_cache()"

# Restart server
uvicorn local_api_server:app --host 0.0.0.0 --port 5000 --reload
```

**Quick Fix 2** - Use CPU instead (slower but works):
```python
# In local_api_server.py, change model loading:
device_map="cpu"  # Instead of "auto"
# or
device = "cpu"
```

### Issue: Very Slow Text Generation

**Cause 1** - Running on CPU instead of GPU

**Check**:
```powershell
# During generation, open Task Manager > GPU
# If 0% GPU usage, you're on CPU
```

**Fix**:
```python
# Ensure device=0 or device_map="auto"
device=0 if torch.cuda.is_available() else -1
```

**Cause 2** - Model too large for your GPU

**Fix**: Use smaller model or enable int8 quantization

### Issue: "Cannot connect to local API" in Python Client

**Cause**: Server not running or wrong port

**Check**:
```powershell
# See if port 5000 is listening
netstat -ano | findstr :5000

# Manually test endpoint
curl http://localhost:5000/health
```

**Fix**:
```powershell
# Make sure server is running in another terminal
uvicorn local_api_server:app --host 0.0.0.0 --port 5000 --reload

# Check exact error in Python:
python -c "import requests; print(requests.get('http://localhost:5000/health').json())"
```

### Issue: Model Takes Forever to Download

**Cause**: Large model being downloaded from Hugging Face

**Info**: First download can take 10-30 minutes depending on internet speed

**Solution**: Be patient, or use smaller model:
```python
# Download smaller model for testing
model = "gpt2"  # Only 350MB, very fast
```

**Monitor download**:
```powershell
# In PowerShell, watch folder size growth
Get-ChildItem "$env:USERPROFILE\.cache\huggingface" -Recurse | Measure-Object -Sum Length
```

### Issue: "Expected Device cuda:0 but got cpu" Error

**Cause**: Model on GPU but input tensor on CPU (or vice versa)

**Fix**:
```python
# Ensure inputs are on same device as model
if torch.cuda.is_available():
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
```

### Issue: Execution Policy Error on PowerShell

**Error**: "cannot be loaded because running scripts is disabled"

**Fix**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Performance Optimization Tips

### 1. Use Smaller Models First
```python
# Test with small model
model = "gpt2"  # 124M, instant
# Then upgrade to larger if needed
model = "mistralai/Mistral-7B-v0.1"  # 7B, better quality
```

### 2. Enable Mixed Precision
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use 16-bit precision
    device_map="auto"
)
```

### 3. Use Quantization for Larger Models
```powershell
# Install BitsAndBytes for 8-bit/4-bit quantization
pip install bitsandbytes
```

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    load_in_8bit=True,  # 8-bit quantization
    device_map="auto"
)
```

### 4. Batch Multiple Requests
```python
# Instead of:
for prompt in prompts:
    generate(prompt)

# Do:
from transformers import pipeline
generator = pipeline("text-generation", model="gpt2", device=0)
outputs = generator(prompts, batch_size=4)  # Process 4 at once
```

### 5. Cache Models Locally
```python
# Download once, use offline
from transformers import AutoModel
model = AutoModel.from_pretrained("gpt2")
model.save_pretrained("./my_models/gpt2")

# Later, load from disk (much faster)
model = AutoModel.from_pretrained("./my_models/gpt2")
```

---

## Quick Reference: Common Commands

```powershell
# Activate environment
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check GPU status
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Start API server
uvicorn local_api_server:app --host 0.0.0.0 --port 5000 --reload

# Test API
curl http://localhost:5000/health

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# List CUDA info
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Summary

✅ **Windows NVIDIA GPU Setup Complete**

You now have:
- ✅ .venv virtual environment
- ✅ PyTorch with CUDA support
- ✅ Local API server (Together AI compatible)
- ✅ Python wrapper to replace api.together.xyz
- ✅ No credit card needed, free local inference

**Next Steps**:
1. Run `.\setup_local_llm.ps1` to automate setup
2. Start API server: `uvicorn local_api_server:app --host 0.0.0.0 --port 5000`
3. Update your notebooks to use `LocalLLM()` instead of Together API
4. No more paying for API credits!
