# Understanding `app.function` and `src` Package

## Q1: What is `app.function`?

### `app.function` is a **Modal AI decorator**

**Modal** is a cloud platform that lets you run Python functions on remote GPUs without managing infrastructure.

### How It Works

```python
import modal

# 1. Create a Modal App
app = modal.App("my-gpu-app")

# 2. Define a function with @app.function decorator
@app.function(gpu="L40S", timeout=3600)
def run_on_gpu(data):
    import torch
    # This code runs on a cloud GPU!
    return torch.cuda.is_available()

# 3. Call it from your local machine
if __name__ == "__main__":
    with app.run():
        result = run_on_gpu.remote("test")  # .remote() runs on cloud
        print(result)  # True (because it ran on GPU)
```

### What Happens When You Call `.remote()`

```
Your Local Machine                     Modal AI Cloud
─────────────────                      ──────────────

1. Call function.remote()
        │
        ├─────────────────────────────→ 2. Spin up container
        │                                  • GPU: L40S/H100/A100
        │                                  • Image: nvidia/cuda:12.4.0
        │                                  • Python 3.10
        │                                  • Install packages
        │
        │                               3. Run function on GPU
        │                                  • Execute your code
        │                                  • Access GPU resources
        │
        │                               4. Return result
        ←─────────────────────────────────┤
        │
5. Receive result locally
```

### Key Parameters in `evaluate_with_modal.py`

```python
@app.function(
    image=image,        # Docker container definition (CUDA, PyTorch, etc.)
    timeout=3600,       # Maximum runtime: 1 hour
    gpu="L40S"          # GPU type: L40S, H100, A100, T4, etc.
)
def eval_single_sample_modal(ref_arch_src, custom_cuda, verbose, gpu_arch):
    # This entire function runs on Modal's cloud GPU!
    from src.eval import eval_kernel_against_ref
    return eval_kernel_against_ref(...)
```

### The `image` Parameter

The `image` defines the Docker container environment on Modal's cloud:

```python
image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .pip_install(
        "torch==2.5.0",
        "numpy",
        "datasets",
        "transformers",
        # ... more packages
    )
    .env({
        "PYTHONPATH": "/root/KernelBench"  # Add to Python path
    })
    .add_local_dir(
        KERNEL_BENCH_PATH,            # Local: ../external/KernelBench
        remote_path="/root/KernelBench"  # Remote: /root/KernelBench
    )
)
```

This means:
- ✅ Starts with NVIDIA CUDA base image
- ✅ Installs system packages (gcc, git, etc.)
- ✅ Installs Python packages (torch, numpy, etc.)
- ✅ Sets environment variables
- ✅ **Copies local `external/KernelBench` to cloud container**

---

## Q2: What is `src` Package?

### `src` is from the **external KernelBench repository**

It's NOT a PyPI package you install with pip. It's the source code from the upstream KernelBench project.

### Directory Structure

```
Trace-Bench/
│
├── KernelBench/                         ← Your wrapper code
│   ├── evaluate_with_modal.py           ← Uses 'src' package
│   ├── cuda_eval_server.py
│   ├── cuda_eval_client.py
│   ├── preprocess.py
│   ├── install.sh                       ← Run this to get 'src'!
│   └── .venv/
│
└── external/                             ← Created by install.sh
    └── KernelBench/                      ← Cloned from GitHub
        ├── src/                          ← THIS is the 'src' package
        │   ├── eval.py                   ← eval_kernel_against_ref()
        │   ├── utils.py                  ← set_gpu_arch()
        │   ├── prompt_constructor.py     ← prompt generation
        │   └── ...
        ├── scripts/
        │   └── eval_single_example.py    ← Used by server
        └── requirements.txt
```

### What's in `src/`?

The `src` package contains the core KernelBench evaluation logic:

| File | What It Does |
|------|--------------|
| `src/eval.py` | `eval_kernel_against_ref()` - Main evaluation function |
| `src/utils.py` | `set_gpu_arch()` - GPU architecture configuration |
| `src/prompt_constructor.py` | `prompt_generate_custom_cuda_from_prompt_template()` |
| `src/data.py` | Data structures and types |
| `src/constants.py` | Configuration constants |

### How `src` is Used in Modal

```python
# In evaluate_with_modal.py

# 1. SETUP: Copy external/KernelBench to cloud container
image = (
    modal.Image...
    .add_local_dir(
        KERNEL_BENCH_PATH,                 # ../external/KernelBench
        remote_path="/root/KernelBench"    # Copied to cloud
    )
    .env({
        "PYTHONPATH": "/root/KernelBench"  # Make it importable
    })
)

# 2. USE: Import from 'src' (runs on cloud GPU)
@app.function(image=image, gpu="L40S")
def eval_single_sample_modal(ref_arch_src, custom_cuda, verbose, gpu_arch):
    from src.eval import eval_kernel_against_ref  # ← This works on cloud
    from src.utils import set_gpu_arch

    set_gpu_arch(gpu_arch)
    return eval_kernel_against_ref(ref_arch_src, custom_cuda, ...)
```

**Key Point**: The `src` package is:
1. Stored locally in `external/KernelBench/src/`
2. Copied to cloud container via `.add_local_dir()`
3. Added to `PYTHONPATH` so Python can find it
4. Imported and used in the remote function

---

## Q3: Do You Have the `src` Package?

### **❌ NO - You need to run `install.sh` first!**

Check if you have it:

```bash
ls external/KernelBench/src/
```

If you see "No such file or directory", you need to install it.

---

## Installation: Getting the `src` Package

### Step 1: Run `install.sh`

```bash
cd KernelBench/
./install.sh
```

### What `install.sh` Does

```bash
#!/usr/bin/env sh

# 1. Clone KernelBench from GitHub
git clone https://github.com/ScalingIntelligence/KernelBench.git \
    ../external/KernelBench

# 2. Install Python dependencies
cd KernelBench/
uv sync  # Creates .venv with all dependencies

# 3. Install KernelBench package in editable mode
uv pip install -e ../external/KernelBench

# Result: You can now import from 'src'
```

### After Installation

```
external/
└── KernelBench/
    ├── src/                    ← ✅ Now you have this!
    │   ├── eval.py
    │   ├── utils.py
    │   ├── prompt_constructor.py
    │   └── ...
    ├── scripts/
    │   └── eval_single_example.py
    └── requirements.txt
```

### Verify Installation

```python
# Test if 'src' package is available
python3 -c "from src.eval import eval_kernel_against_ref; print('✓ src package found!')"
```

---

## How It All Works Together

### Local Development (Server/Client)

```python
# When using cuda_eval_server.py or cuda_eval_client.py

# Server calls subprocess:
subprocess.run([
    "python",
    "external/KernelBench/scripts/eval_single_example.py",  # ← Uses src internally
    "--custom-cuda", cuda_code,
    "--ref-arch-src", ref_arch_src
])

# The subprocess script does:
# from src.eval import eval_kernel_against_ref
# result = eval_kernel_against_ref(...)
```

### Cloud Development (Modal)

```python
# When using evaluate_with_modal.py

# 1. Local machine: Call function
result = eval_single_sample_modal.remote(ref_arch_src, cuda_code, ...)

# 2. Modal cloud: Spin up container
#    - Container has external/KernelBench copied to /root/KernelBench
#    - PYTHONPATH includes /root/KernelBench
#    - So 'from src.eval import ...' works!

# 3. Cloud GPU: Run evaluation
@app.function(image=image, gpu="L40S")
def eval_single_sample_modal(...):
    from src.eval import eval_kernel_against_ref  # ← Works because of setup
    return eval_kernel_against_ref(...)

# 4. Return result to local machine
```

---

## Quick Reference

### Do I need `src` package?

| Your Use Case | Need `src`? | How to Get It |
|---------------|-------------|---------------|
| **Modal evaluation** | ✅ Yes | Run `./install.sh` |
| **Server evaluation** | ✅ Yes | Run `./install.sh` |
| **Client-only** | ❌ No | Server handles it |
| **Just preprocessing** | ✅ Yes | For `prompt_constructor.py` |

### Import Paths

```python
# These imports require external/KernelBench to be installed:

from src.eval import eval_kernel_against_ref
from src.utils import set_gpu_arch
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
```

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# 1. Check if external/KernelBench exists
ls -la external/KernelBench/

# 2. If not, run install script
cd KernelBench/
./install.sh

# 3. Verify installation
python -c "from src.eval import eval_kernel_against_ref; print('OK')"
```

---

## Summary

### `app.function`
- ✅ Modal AI decorator for cloud GPU functions
- ✅ Runs your Python code on remote GPUs
- ✅ Use `.remote()` to trigger cloud execution
- ✅ No need to manage infrastructure

### `src` Package
- ✅ Source code from upstream KernelBench repo
- ✅ Located at: `external/KernelBench/src/`
- ✅ Contains evaluation logic (`eval_kernel_against_ref`)
- ✅ Get it by running: `./install.sh`

### Do You Have It?
- ❌ Not by default
- ✅ Run `./install.sh` to clone and install
- ✅ Required for both Modal and Server evaluation

### Installation Command
```bash
cd KernelBench/
./install.sh
```

That's it! Once installed, you can use both Modal evaluation and local server evaluation.
