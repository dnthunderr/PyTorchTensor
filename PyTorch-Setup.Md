## PyTorch setup (venv recommended)

This repository includes a short guide to install PyTorch, torchvision and torchaudio into a local virtual environment on Windows (PowerShell). The venv approach is recommended because it is self-contained and avoids modifying global or broken conda environments.

## 1) Choose your environment

- venv (recommended): Simple, reproducible, and isolated. Good for most users.
- conda: Useful if you rely on conda-managed CUDA toolchains or have existing conda workflows. If you use conda, use the official instructions at https://pytorch.org/get-started/locally/ to pick the right command.

If you are unsure, pick venv below.

## 2) Create and activate a venv (PowerShell)

Open a PowerShell terminal in the repository root (for example: `C:\GitHub\PyTorchTensor`). Then run:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

# Optional: On older PowerShell configurations you may need to enable script execution:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Notes:

- If `python` points to the wrong interpreter, use the full path to the desired Python executable (example: `C:\path\to\python.exe -m venv .venv`).
- To avoid ambiguity when installing packages, you can always call the venv Python directly: `.\.venv\Scripts\python.exe -m pip install ...`.

## 3) Upgrade pip and install PyTorch packages into the venv

Use the venv Python to upgrade pip and install the packages. The simple command below will install CPU or the best-matching prebuilt wheel for your environment.

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
```

If you need a specific CUDA-enabled build, visit the official selector at https://pytorch.org/get-started/locally/ and copy the pip command it generates. Example (this is an example for a specific CUDA version and may change):

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

## 4) Verify installation

Run a quick import and check the version/CUDA availability using the venv Python:

```powershell
.\.venv\Scripts\python.exe -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available())'
```

Expected output:

- A printed torch version (e.g. `2.x.x`).
- `cuda: True` or `cuda: False` depending on whether the installed build supports CUDA and if CUDA is available on the machine.

## 5) Make VS Code use the venv interpreter

1. Open Command Palette (Ctrl+Shift+P).
2. Choose `Python: Select Interpreter`.
3. Select the interpreter located at `.venv\Scripts\python.exe`.

This ensures running code and terminals in VS Code use the venv Python.

## 6) Troubleshooting

- If `pip install` fails with permission or environment errors, ensure you activated the venv or call the venv Python directly (see the commands above).
- If you see an error like `EnvironmentLocationNotFound` when using conda helper scripts, that indicates the selected conda env path is missing or invalid — use a venv instead or fix the conda interpreter in VS Code.
- If you need a GPU-enabled build and `torch.cuda.is_available()` is False, double-check your GPU driver and CUDA toolkit compatibility and install the correct wheel from the PyTorch website.

## 7) Quick checklist

- [ ] Create venv: `python -m venv .venv`
- [ ] Activate: `.\.venv\Scripts\Activate.ps1`
- [ ] Upgrade pip: `python -m pip install --upgrade pip`
- [ ] Install: `python -m pip install torch torchvision torchaudio`
- [ ] Verify: `python -c 'import torch; print(torch.__version__); print("cuda:", torch.cuda.is_available())'`

## References

- Official installer guide: https://pytorch.org/get-started/locally/

---

If you want, I can also add a short note with the exact `conda` commands for installing PyTorch (for example, creating a named conda env and using `conda install`), but the venv flow above is the recommended lightweight approach.
