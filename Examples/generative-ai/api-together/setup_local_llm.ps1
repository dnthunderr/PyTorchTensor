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