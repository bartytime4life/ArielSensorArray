# ------------------------------------------------------------------------------
# SpectraMind V50 — Dockerfile (GPU-ready dev/runtime)
# Targets: training, inference, diagnostics, and CI parity
# ------------------------------------------------------------------------------

# Base: CUDA + cuDNN runtime on Ubuntu 22.04 (adjust to your driver/toolkit)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Noninteractive to avoid tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    git curl ca-certificates tini build-essential \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set a workspace
WORKDIR /workspace

# Install Poetry (system-wide, no venvs inside venvs)
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

# Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml poetry.lock* ./

# Install Python deps into the container env (no-root installs project later)
RUN poetry install --no-interaction --no-root

# Copy project files
COPY . .

# (Optional) If you have extra system wheels or private deps, handle them here
# RUN pip install --no-cache-dir <extras>

# Cache Hugging Face models (optional; otherwise they download on first use)
# ENV TRANSFORMERS_CACHE=/workspace/.hf_cache
# RUN mkdir -p ${TRANSFORMERS_CACHE}

# Basic health / version info
RUN python3 - <<'PY'
import sys, platform, subprocess
print("python:", sys.version.replace("\n"," "))
print("platform:", platform.platform())
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch not installed or import failed:", e)
PY

# Ensure the CLI is resolvable if project uses poetry scripts/entrypoints
# For Typer entrypoint (spectramind.py), create a console script shim if needed:
# RUN ln -sf /workspace/spectramind.py /usr/local/bin/spectramind && chmod +x /usr/local/bin/spectramind

# Default: drop into a bash shell (use `docker run --gpus all -it --rm -v $PWD:/workspace <img> bash`)
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]