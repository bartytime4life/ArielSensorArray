# syntax=docker/dockerfile:1.6
# ------------------------------------------------------------------------------
# SpectraMind V50 — Dockerfile
# GPU-ready, reproducible, CI/Compose-friendly, Poetry-based, multi-stage
#
# Quick switches:
#   • GPU (default): nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   • CPU:           --build-arg BASE_IMAGE=python:3.11-slim
#
# Build examples:
#   docker build -t spectramind:gpu .
#   docker build -t spectramind:cpu --build-arg BASE_IMAGE=python:3.11-slim .
#
# Run examples:
#   # GPU
#   docker run --rm -it --gpus all spectramind:gpu spectramind --help
#   # CPU
#   docker run --rm -it spectramind:cpu spectramind --help
# ------------------------------------------------------------------------------

# ===== 1) Base (CUDA or pure Python) =====
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# OCI labels
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="SpectraMind V50" \
      org.opencontainers.image.description="Neuro-symbolic, physics-informed AI pipeline for NeurIPS 2025 Ariel Data Challenge" \
      org.opencontainers.image.vendor="SpectraMind" \
      org.opencontainers.image.source="https://github.com/bartytime4life/SpectraMindV50" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="MIT"

# Common env
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    MPLBACKEND=Agg \
    HYDRA_FULL_ERROR=1

# Optional: set PyTorch extra index for CUDA wheels automatically when on CUDA base.
# For CPU builds (e.g., python:3.11-slim), leave empty or override during build/run.
ARG TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu121
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHL_INDEX}

# Versions / args
ARG POETRY_VERSION=1.8.3
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Detect apt-based image; install practical system deps:
#  - python & headers (CUDA bases might not ship these)
#  - build-essential/pkg-config: compile native wheels
#  - git & git-lfs: repos + large files (DVC/HF)
#  - curl, ca-certificates: secure downloads
#  - tini: sane PID 1
#  - libgl1, libglib2.0-0: common runtime deps for cv/plot libs
#  - graphviz, libgraphviz-dev: for diagrams/plots
#  - ffmpeg: rendering videos/gifs if needed in diagnostics
RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        build-essential pkg-config \
        git git-lfs curl ca-certificates \
        tini \
        libgl1 libglib2.0-0 \
        graphviz libgraphviz-dev \
        ffmpeg \
      && rm -rf /var/lib/apt/lists/*; \
      update-alternatives --install /usr/bin/python python /usr/bin/python3 1; \
    else \
      echo "Assuming Python base already contains python & pip"; \
    fi

# Poetry (global)
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION} \
  && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

# Non-root user (volume ownership parity with host)
RUN groupadd -g ${GID} ${USERNAME} \
  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

# ===== 2) Deps stage (cache-friendly, BuildKit-aware) =====
FROM base AS deps

# Copy only dependency manifests to leverage Docker layer cache
COPY pyproject.toml poetry.lock* ./

# (Optional) freeze bootstrap tools for reproducibility
RUN python -m pip install --upgrade pip setuptools wheel

# Install project dependencies (no source yet, no-root to retain cache)
# BuildKit cache mounts (safe no-op when not using buildx)
RUN --mount=type=cache,target=/root/.cache/pip \
    poetry config virtualenvs.create false \
 && poetry install --no-interaction --no-ansi --no-root

# ===== 3) Builder stage (install project for CLI) =====
FROM base AS build

# Bring in resolved deps/site-packages and console scripts
COPY --from=deps /usr/local /usr/local

# Copy source AFTER deps for better caching
COPY . /workspace

# Install project so entry points (e.g., `spectramind`) are available
# If pyproject defines console_scripts, Poetry will expose them to /usr/local/bin
RUN poetry install --no-interaction --no-ansi

# Quick smoke: can we import & show help?
RUN spectramind --help >/dev/null

# ===== 4) Runtime stage (slim, non-root, tini) =====
FROM base AS runtime

# Copy the fully installed Python env + scripts from build
COPY --from=build /usr/local /usr/local
# Copy the working tree for runtime (needed for configs/assets; code already installed)
COPY --chown=${UID}:${GID} . /workspace

# Recommended caches and envs
ENV HF_HOME=/home/${USERNAME}/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONHASHSEED=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Writable dirs for logs / outputs / data (bind-mount friendly)
RUN mkdir -p /workspace/outputs /workspace/logs /workspace/data \
 && chown -R ${UID}:${GID} /workspace /home/${USERNAME}

# Switch to non-root
USER ${USERNAME}

# Healthcheck: verify CLI import and help
HEALTHCHECK --interval=30s --timeout=8s --start-period=45s --retries=3 \
  CMD python -c "import importlib; importlib.import_module('spectramind')" >/dev/null 2>&1 && spectramind --help >/dev/null 2>&1 || exit 1

# Tini as entrypoint for proper signal handling & zombie reaping
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["spectramind", "--help"]

# ------------------------------------------------------------------------------
# Notes
# ------------------------------------------------------------------------------
# 1) CUDA vs CPU:
#    - Default GPU base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#      Build: docker build -t spectramind:gpu .
#      Run:   docker run --rm -it --gpus all spectramind:gpu spectramind --help
#    - CPU base: --build-arg BASE_IMAGE=python:3.11-slim
#      Build: docker build -t spectramind:cpu --build-arg BASE_IMAGE=python:3.11-slim .
#      Run:   docker run --rm -it spectramind:cpu spectramind --help
#
# 2) PyTorch wheels:
#    - For CUDA builds, PIP_EXTRA_INDEX_URL defaults to cu121 wheel index.
#      Override if you target a different CUDA minor:
#        --build-arg TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu124
#    - For CPU builds, set empty to avoid wrong wheel resolution:
#        --build-arg TORCH_WHL_INDEX=
#
# 3) Optional runtime extras:
#    - DVC remotes: add in pyproject (recommended) or install ad-hoc:
#        pip install "dvc[s3]" "dvc[gdrive]"
#    - Torch Geometric matching your torch+cuda combo (example for torch 2.4 + cu121):
#        pip install torch-scatter     -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-sparse      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-cluster     -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#
# 4) Layer caching:
#    - Keep dependency install isolated in the "deps" stage with only pyproject/lock copied in.
#    - Copy source later to avoid invalidating dependency layer on code edits.
#    - BuildKit cache mounts are enabled for pip to speed up iterative builds.
#
# 5) Compose (GPU):
#    version: "3.9"
#    services:
#      smv50:
#        image: spectramind:gpu
#        deploy:
#          resources:
#            reservations:
#              devices:
#                - capabilities: ["gpu"]
#        volumes:
#          - ./:/workspace
#
# 6) User mapping:
#    - UID/GID args ensure container writes are owned by your host user when volume-mounting.
#
# 7) Minimal base footprint:
#    - We install only essential libs (graphviz, ffmpeg, libgl1) commonly needed by diagnostics.
#    - Remove or add system packages according to your pipeline requirements.
#
# 8) Entrypoint:
#    - Defaults to `spectramind --help`. Override with subcommands, e.g.:
#      docker run --rm --gpus all -v $PWD:/workspace spectramind:gpu spectramind train --config-name=config_v50.yaml
# ------------------------------------------------------------------------------
