#!/usr/bin/env python3
"""
Debug script to test eval_single_example.py directly
Run this on the remote machine to see what's going wrong
"""

import subprocess
import sys
import os

# Get paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EXTERNAL_DIR = os.path.join(REPO_ROOT, "external")
KERNELBENCH_DIR = os.path.join(EXTERNAL_DIR, "KernelBench")
EVAL_SCRIPT = os.path.join(SCRIPT_DIR, "eval_single_example.py")

print(f"Script dir: {SCRIPT_DIR}")
print(f"Repo root: {REPO_ROOT}")
print(f"External dir: {EXTERNAL_DIR}")
print(f"KernelBench dir: {KERNELBENCH_DIR}")
print(f"Eval script: {EVAL_SCRIPT}")
print()

# Check if paths exist
print("Checking paths:")
print(f"  eval_single_example.py exists: {os.path.exists(EVAL_SCRIPT)}")
print(f"  KernelBench dir exists: {os.path.exists(KERNELBENCH_DIR)}")
print(f"  KernelBench/src exists: {os.path.exists(os.path.join(KERNELBENCH_DIR, 'src'))}")
print()

# Simple test kernel
custom_cuda = """
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)
"""

ref_arch_src = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)

def get_init_inputs():
    return [10, 5]

def get_inputs():
    return [torch.randn(4, 10)]
"""

# Build command
args = [
    sys.executable,
    EVAL_SCRIPT,
    "--problem-id", "1",
    "--sample-id", "0",
    "--device", "cuda:0",
    "--custom-cuda", custom_cuda,
    "--ref-arch-src", ref_arch_src,
    "--verbose"
]

print("Running command:")
print(" ".join(args[:2]))
print(f"  (with custom-cuda and ref-arch-src args)")
print()

# Run the subprocess
print("=" * 80)
print("SUBPROCESS OUTPUT:")
print("=" * 80)

result = subprocess.run(
    args,
    capture_output=True,
    text=True,
    timeout=60,
    cwd=KERNELBENCH_DIR
)

print("STDOUT:")
print(result.stdout)
print()
print("STDERR:")
print(result.stderr)
print()
print(f"Return code: {result.returncode}")
print("=" * 80)

