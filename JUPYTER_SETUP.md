# Creating and Running Jupyter Notebooks in VS Code with PyTorch

This guide shows how to create a Jupyter notebook in your VS Code project and execute it using your local `.venv` PyTorch environment with GPU support.

## Prerequisites

- `.venv` created and PyTorch with CUDA installed (see `README.md`)
- VS Code with **Python** and **Jupyter** extensions installed

## Step 1: Install Jupyter in your venv

Activate your venv and install Jupyter:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install jupyter notebook jupyterlab
```

Or use the direct Python path:

```powershell
.\.venv\Scripts\python.exe -m pip install jupyter notebook jupyterlab
```

## Step 2: Create a new Jupyter notebook in VS Code

1. Open your repository folder in VS Code.
2. Press `Ctrl+Shift+P` to open Command Palette.
3. Type "Jupyter: Create New Blank Notebook".
4. VS Code will create an `Untitled-1.ipynb` file. Save it with a name, e.g., `test_pytorch.ipynb`.

## Step 3: Select the Python interpreter for the notebook

1. Open the newly created `.ipynb` file.
2. At the top right of the notebook editor, click on "Select Kernel" (or the Python version shown).
3. Choose "Python Environments...".
4. Select `.venv\Scripts\python.exe` (the one from your local virtual environment).

Once selected, VS Code will use that interpreter and its installed packages (torch, torchvision, etc.) for all cells.

## Step 4: Write and run cells

### Example cell 1: Import and check PyTorch/CUDA

In the first cell (markdown), type:

```markdown
# PyTorch GPU Test
Test PyTorch with CUDA on the local environment.
```

In the second cell (Python), type:

```python
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("Device count:", torch.cuda.device_count())
```

Press `Shift+Enter` or click the Run button to execute the cell. Output will appear below the cell.

### Example cell 2: Create a tensor on GPU

```python
# Create a tensor and move it to GPU
x = torch.randn(1000, 1000)
x_gpu = x.cuda()

print(f"Tensor shape: {x_gpu.shape}")
print(f"Tensor device: {x_gpu.device}")

# Simple matrix multiplication on GPU
y = torch.matmul(x_gpu, x_gpu)
print(f"Result shape: {y.shape}")
```

### Example cell 3: Performance comparison (CPU vs GPU)

```python
import time

size = 5000

# CPU timing
x_cpu = torch.randn(size, size)
y_cpu = torch.randn(size, size)

start = time.time()
result_cpu = torch.matmul(x_cpu, y_cpu)
cpu_time = time.time() - start

# GPU timing
x_gpu = torch.randn(size, size).cuda()
y_gpu = torch.randn(size, size).cuda()

start = time.time()
result_gpu = torch.matmul(x_gpu, y_gpu)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f}s")
print(f"GPU time: {gpu_time:.4f}s")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

## Step 5: Run the entire notebook

- **Run all cells**: Press `Ctrl+Alt+Enter` or click "Run All" button.
- **Run cell by cell**: Press `Shift+Enter` after each cell.
- **Clear outputs**: Right-click in the notebook and select "Clear All Outputs".

## Troubleshooting

### Kernel not found
- Ensure the venv is activated and Jupyter is installed: `.\.venv\Scripts\python.exe -m pip install jupyter`
- Restart VS Code after installing.

### Wrong Python interpreter selected
- Click "Select Kernel" at the top right and make sure `.venv\Scripts\python.exe` is chosen.

### CUDA not available in notebook
- Make sure you selected the correct interpreter (your `.venv` with PyTorch CUDA enabled).
- Run the first example cell to verify `torch.cuda.is_available()` returns `True`.

### Notebook takes time to start
- First notebook kernel start can be slow. Subsequent cells run faster. Be patient.

## Quick example: Complete notebook workflow

1. Create file `example_pytorch.ipynb`.
2. Select kernel `.venv\Scripts\python.exe`.
3. Add cells:
   - Cell 1 (markdown): `# My PyTorch Notebook`
   - Cell 2 (python): `import torch; print(torch.__version__)`
   - Cell 3 (python): `x = torch.tensor([1.0, 2.0]).cuda(); print(x)`
4. Run cells with `Shift+Enter`.
5. See output appear below each cell.

## Running Jupyter Lab (alternative)

If you prefer the full Jupyter Lab interface instead of VS Code's editor:

```powershell
# Activate your venv
.\.venv\Scripts\Activate.ps1

# Start Jupyter Lab
jupyter lab
```

This opens a browser window with Jupyter Lab. Create and edit notebooks there, and they will still use your `.venv` Python environment since you activated it before launching.

To stop Jupyter Lab, press `Ctrl+C` in the terminal.

## Summary

| Task | Command / Action |
|------|------------------|
| Install Jupyter | `python -m pip install jupyter` |
| Create notebook | Ctrl+Shift+P → "Jupyter: Create New Blank Notebook" |
| Select kernel | Click "Select Kernel" → choose `.venv\Scripts\python.exe` |
| Run cell | Shift+Enter |
| Run all | Ctrl+Alt+Enter |

You're all set! Your notebooks will use your local PyTorch environment with GPU support.
