#!/usr/bin/env sh
set -eu

# Determine repository root (parent of this script)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

# Ensure uv is installed
if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(command -v uv)"
else
  echo "uv not found; installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is available on PATH for this shell session
if ! command -v uv >/dev/null 2>&1; then
  if [ -f "$HOME/.local/bin/env" ]; then
    . "$HOME/.local/bin/env"
  elif [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
    export PATH
  fi
  # refresh shell command hash, if supported
  hash -r 2>/dev/null || true
fi

# Locate uv binary (absolute path fallback)
if command -v uv >/dev/null 2>&1; then
  UV_BIN=$(command -v uv)
elif [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
else
  echo "uv not found after installation; ensure $HOME/.local/bin is on PATH"
  exit 1
fi

# Prepare external directory and clone KernelBench there
command -v git >/dev/null 2>&1 || { echo "git is required"; exit 1; }

EXTERNAL_DIR=${EXTERNAL_DIR:-"$REPO_ROOT/external"}
KERNELBENCH_DIR=${KERNELBENCH_DIR:-"$EXTERNAL_DIR/KernelBench"}
KERNELBENCH_URL=${KERNELBENCH_URL:-"https://github.com/xuanfeiren/KernelBench.git"}

mkdir -p "$EXTERNAL_DIR"

if [ -d "$KERNELBENCH_DIR/.git" ]; then
  echo "Updating KernelBench in $KERNELBENCH_DIR"
  if ! (
    git -C "$KERNELBENCH_DIR" fetch --prune
    git -C "$KERNELBENCH_DIR" pull --ff-only
  ); then
    echo "KernelBench update failed; recloning..."
    rm -rf "$KERNELBENCH_DIR"
    git clone "$KERNELBENCH_URL" "$KERNELBENCH_DIR"
  fi
else
  echo "Cloning KernelBench into $KERNELBENCH_DIR"
  git clone "$KERNELBENCH_URL" "$KERNELBENCH_DIR"
fi

# Clone or update Trace repository (experimental branch)
TRACE_DIR=${TRACE_DIR:-"$REPO_ROOT/Trace"}
TRACE_URL=${TRACE_URL:-"https://github.com/xuanfeiren/Trace.git"}
TRACE_BRANCH=${TRACE_BRANCH:-"experimental"}

if [ -d "$TRACE_DIR/.git" ]; then
  echo "Updating Trace repository in $TRACE_DIR"
  if ! (
    git -C "$TRACE_DIR" fetch --prune
    git -C "$TRACE_DIR" checkout "$TRACE_BRANCH"
    git -C "$TRACE_DIR" pull --ff-only origin "$TRACE_BRANCH"
  ); then
    echo "Trace update failed; recloning..."
    rm -rf "$TRACE_DIR"
    git clone -b "$TRACE_BRANCH" "$TRACE_URL" "$TRACE_DIR"
  fi
else
  echo "Cloning Trace repository (branch: $TRACE_BRANCH) into $TRACE_DIR"
  git clone -b "$TRACE_BRANCH" "$TRACE_URL" "$TRACE_DIR"
fi

# Clone or update DSPy repository
DSPY_DIR=${DSPY_DIR:-"$EXTERNAL_DIR/dspy-repo"}
DSPY_URL=${DSPY_URL:-"https://github.com/xuanfeiren/dspy-repo.git"}

if [ -d "$DSPY_DIR/.git" ]; then
  echo "Updating DSPy repository in $DSPY_DIR"
  if ! (
    git -C "$DSPY_DIR" fetch --prune
    git -C "$DSPY_DIR" pull --ff-only
  ); then
    echo "DSPy update failed; recloning..."
    rm -rf "$DSPY_DIR"
    git clone "$DSPY_URL" "$DSPY_DIR"
  fi
else
  echo "Cloning DSPy repository into $DSPY_DIR"
  git clone "$DSPY_URL" "$DSPY_DIR"
fi

# Clone or update GEPA repository
GEPA_DIR=${GEPA_DIR:-"$EXTERNAL_DIR/gepa-repo"}
GEPA_URL=${GEPA_URL:-"https://github.com/xuanfeiren/gepa-repo.git"}

if [ -d "$GEPA_DIR/.git" ]; then
  echo "Updating GEPA repository in $GEPA_DIR"
  if ! (
    git -C "$GEPA_DIR" fetch --prune
    git -C "$GEPA_DIR" pull --ff-only
  ); then
    echo "GEPA update failed; recloning..."
    rm -rf "$GEPA_DIR"
    git clone "$GEPA_URL" "$GEPA_DIR"
  fi
else
  echo "Cloning GEPA repository into $GEPA_DIR"
  git clone "$GEPA_URL" "$GEPA_DIR"
fi

# Clone or update OpenEvolve repository
OPENEVOLVE_DIR=${OPENEVOLVE_DIR:-"$EXTERNAL_DIR/openevolve"}
OPENEVOLVE_URL=${OPENEVOLVE_URL:-"https://github.com/xuanfeiren/openevolve.git"}

if [ -d "$OPENEVOLVE_DIR/.git" ]; then
  echo "Updating OpenEvolve repository in $OPENEVOLVE_DIR"
  if ! (
    git -C "$OPENEVOLVE_DIR" fetch --prune
    git -C "$OPENEVOLVE_DIR" pull --ff-only
  ); then
    echo "OpenEvolve update failed; recloning..."
    rm -rf "$OPENEVOLVE_DIR"
    git clone "$OPENEVOLVE_URL" "$OPENEVOLVE_DIR"
  fi
else
  echo "Cloning OpenEvolve repository into $OPENEVOLVE_DIR"
  git clone "$OPENEVOLVE_URL" "$OPENEVOLVE_DIR"
fi

# Install Python dependencies for this task using uv
echo "Installing Python dependencies with uv in $SCRIPT_DIR"
(cd "$SCRIPT_DIR" && "$UV_BIN" sync)
echo "Activating the Python environment at $SCRIPT_DIR/.venv"
. "$SCRIPT_DIR/.venv/bin/activate"

# Use uv to install upstream KernelBench as an editable package if it provides pyproject.toml or setup.py
if [ -f "$KERNELBENCH_DIR/pyproject.toml" ] || [ -f "$KERNELBENCH_DIR/setup.py" ]; then
  echo "Installing/updating upstream KernelBench package in editable mode with uv"
  "$UV_BIN" pip install -e "$KERNELBENCH_DIR"
fi

# Install DSPy repository as editable package
if [ -f "$DSPY_DIR/pyproject.toml" ] || [ -f "$DSPY_DIR/setup.py" ]; then
  echo "Installing/updating DSPy package in editable mode with uv"
  "$UV_BIN" pip install -e "$DSPY_DIR"
fi

# Install GEPA repository as editable package
if [ -f "$GEPA_DIR/pyproject.toml" ] || [ -f "$GEPA_DIR/setup.py" ]; then
  echo "Installing/updating GEPA package in editable mode with uv"
  "$UV_BIN" pip install -e "$GEPA_DIR"
fi

# Install OpenEvolve repository as editable package
if [ -f "$OPENEVOLVE_DIR/pyproject.toml" ] || [ -f "$OPENEVOLVE_DIR/setup.py" ]; then
  echo "Installing/updating OpenEvolve package in editable mode with uv"
  "$UV_BIN" pip install -e "$OPENEVOLVE_DIR"
fi

# If KernelBench has a requirements.txt, install those requirements explicitly.
if [ -f "$KERNELBENCH_DIR/requirements.txt" ]; then
  echo "Installing KernelBench requirements.txt into the uv environment"
  
  # On macOS, filter out NVIDIA CUDA packages that are Linux-only
  if [ "$(uname)" = "Darwin" ]; then
    echo "Detected macOS - filtering out CUDA/GPU packages (Linux-only)"
    # Create a filtered requirements file excluding CUDA-related packages
    # Filter out: nvidia-*, triton, cupy-cuda*, tilelang, torch (torch is in pyproject.toml with macOS-compatible version)
    # Note: torch, einops are already in pyproject.toml and will be installed via uv sync
    grep -vE "^(nvidia-|triton|torch|cupy-cuda|tilelang)" "$KERNELBENCH_DIR/requirements.txt" > "$SCRIPT_DIR/.requirements_filtered.txt" || true
    if [ -s "$SCRIPT_DIR/.requirements_filtered.txt" ]; then
      "$UV_BIN" pip install -r "$SCRIPT_DIR/.requirements_filtered.txt"
      rm -f "$SCRIPT_DIR/.requirements_filtered.txt"
    else
      echo "No non-CUDA requirements to install"
    fi
  else
    # On Linux, install all requirements including NVIDIA packages
    "$UV_BIN" pip install -r "$KERNELBENCH_DIR/requirements.txt"
  fi
fi

echo "KernelBench setup complete."

ACTIVATE_CMD="source \"$SCRIPT_DIR/.venv/bin/activate\""
echo "To activate this environment later, run: $ACTIVATE_CMD"
