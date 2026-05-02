# Setting up Hugging Face with PyTorch Environment

Since you already have a `.venv` environment set up with PyTorch, you generally do not need a separate environment. Hugging Face libraries (like `transformers`) are built to run on top of deep learning frameworks like PyTorch.

Here is a step-by-step guide to setting up your environment and registering the Jupyter kernel.

## Step 1: Activate your existing environment

Ensure your terminal is using the `.venv` where PyTorch is installed.

**For macOS/Linux:**
```bash
source .venv/bin/activate
```

**For Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**VS Code Terminal (Windows):**
1. Open a new PowerShell terminal: Terminal -> New Terminal
2. Activate:
```powershell
.\.venv\Scripts\Activate.ps1
```
If activation is blocked:
```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
.\.venv\Scripts\Activate.ps1
```

## Step 2: Install Hugging Face Libraries

With the environment active, install the core Hugging Face libraries:
*   `transformers`: The main library for models.
*   `datasets`: For loading and processing data.
*   `accelerate`: Often required for running large models efficiently in PyTorch.

```bash
pip install transformers datasets accelerate
```

### Optional: Enable Xet Storage support (hf_xet)

If you see a warning like "Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed," install the extra:
```bash
pip install "huggingface_hub[hf_xet]"
```
Or install just the package:
```bash
pip install hf_xet
```

## Step 3: Setup the Jupyter Kernel

To ensure this environment is registered as a selectable kernel in Jupyter Notebooks or VS Code:

1.  **Install ipykernel:**

    ```bash
    pip install ipykernel
    ```

2.  **Register the kernel:**
    You can give it a display name like "Python (Hugging Face)" so it is easy to identify.

    ```bash
    python -m ipykernel install --user --name=.venv --display-name "Python (Hugging Face)"
    ```

## Step 4: Verification

To verify that everything is connected correctly:

1.  Create a new Notebook (or Python script).
2.  Select the **"Python (Hugging Face)"** kernel.
3.  Run the following code to download a small model and perform a sentiment analysis task.

```python
import torch
from transformers import pipeline

# Diagnostic: Check PyTorch CUDA visibility
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

# Determine device: -1 for CPU, 0 for GPU (cuda:0)
if torch.cuda.is_available():
    device_id = 0
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device_id = -1
    print("Using CPU. (Ensure you installed PyTorch with CUDA support)")

# Initialize pipeline with explicit device
classifier = pipeline("sentiment-analysis", device=device_id)

# Verify model placement
print(f"Model is on device: {classifier.model.device}")

# Test the model
result = classifier("Setting up the environment was easier than I thought!")
print(result)
```

## Summary

1.  **Integration:** Added `transformers` to existing PyTorch `.venv`.
2.  **Kernel Registration:** Used `ipykernel` to make `.venv` visible as "Python (Hugging Face)".
3.  **Execution:** Confirmed Hugging Face access to PyTorch and GPU.

## Troubleshooting

### ModuleNotFoundError: No module named 'transformers'

If you see this error, your notebook is likely using the wrong kernel (e.g., "base" instead of ".venv").

1.  **Select Kernel:** Click the kernel name (top right of VS Code) and select **Python (Hugging Face)** or your `.venv`.
2.  **Install:** If the error persists, run `%pip install transformers` in a new cell.

### Kernel "Python (Hugging Face)" not visible in list

VS Code sometimes caches the list of kernels or prioritizes the Python Environment list over Jupyter kernels.

1.  **Reload Window:** Press `Ctrl+Shift+P` and select **"Developer: Reload Window"**.
2.  **Select by Path:** If the name still doesn't appear, click "Select Kernel" -> "Python Environments..." and look for your local folder path (e.g., `.venv\Scripts\python.exe` or `Python 3.x.x ('.venv')`). This is functionally identical to selecting the named kernel.
3.  **Verify Registration:** Run `python -m jupyter kernelspec list` in your terminal.
    *   **If you see `.venv`:** The kernel is registered. VS Code is just caching the list. Restart VS Code.
    *   **If you ONLY see `python3`:** The registration failed. Ensure you are in the `.venv` and re-run the registration command from Step 3.


### Troubleshoot Hugging Face - Cache Symbolic link
UserWarning: `huggingface_hub` cache-system uses symlinks by default to efficiently store duplicated files but your machine does not support them in C:\Users\dilip\.cache\huggingface\hub\models--bert-base-uncased. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator.

**Recommendation:** It is highly recommended to **activate Developer Mode** rather than running as administrator. Running as administrator poses a security risk by giving scripts full system access and can cause file permission conflicts.

In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development

This warning indicates that your system does not support symbolic links, which Hugging Face uses for efficient caching.
