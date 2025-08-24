# ------------------------------------------------------------------------------
# SpectraMind V50 — Dockerfile
# GPU-ready, reproducible, CI/Compose-friendly, Poetry-based
#
# Quick switches:
#   • GPU (default): nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#   • CPU:          --build-arg BASE_IMAGE=python:3.11-slim
#
# Build examples:
#   docker build -t spectramind:gpu .
#   docker build -t spectramind:cpu --build-arg BASE_IMAGE=python:3.11-slim .
#
# Run examples:
#   # GPU
#   docker run --rm -it --gpus all spectramind:gpu bash
#   # CPU
#   docker run --rm -it spectramind:cpu bash
# ------------------------------------------------------------------------------

# ===== 1) Base (CUDA or pure Python) =====
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# OCI labels
LABEL org.opencontainers.image.title="SpectraMind V50" \
      org.opencontainers.image.description="Neuro‑symbolic, physics‑informed AI pipeline for NeurIPS 2025 Ariel Data Challenge" \
      org.opencontainers.image.vendor="SpectraMind" \
      org.opencontainers.image.licenses="Apache-2.0"

# Common env
ENV DEBIAN_FRONTEND=noninteractive \
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

# Detect if we're on Debian/Ubuntu vs slim Python base
# Install minimal, practical system deps:
#  - python3, pip, dev headers (if CUDA base may not ship Python)
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

# Install Poetry (global)
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION} \
  && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

# Create non-root user (matching host UID/GID simplifies volume ownership)
RUN groupadd -g ${GID} ${USERNAME} \
  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

# ===== 2) Deps stage (cache-friendly) =====
FROM base AS deps

# Copy only dependency manifests to leverage Docker layer cache
COPY pyproject.toml poetry.lock* ./

# Optional: pin pip/setuptools/wheel for reproducible builds
RUN python -m pip install --upgrade pip setuptools wheel

# Install project dependencies (no source yet, no-root to retain cache)
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root

# ===== 3) Runtime stage =====
FROM base AS runtime

# Copy installed site-packages and scripts from deps
# (Poetry installed into /usr/local for system Python)
COPY --from=deps /usr/local /usr/local

# Workdir and user
WORKDIR /workspace
USER ${USERNAME}

# Copy repository AFTER deps for better caching
# Ensure .dockerignore excludes data/, outputs/, .venv/, .dvc/cache, etc.
COPY --chown=${UID}:${GID} . .

# Optionally install the project as a package to expose entry points (CLIs)
# Uncomment if you define console_scripts in pyproject.toml
# RUN poetry install --no-interaction --no-ansi

# Healthcheck: verify Python can import the package
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import importlib; importlib.import_module('spectramind')" >/dev/null 2>&1 || exit 1

# Tini as entrypoint for proper signal handling & zombie reaping
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

# ------------------------------------------------------------------------------
# Notes
# ------------------------------------------------------------------------------
# 1) CUDA vs CPU:
#    - Default GPU base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#      Build: docker build -t spectramind:gpu .
#      Run:   docker run --rm -it --gpus all spectramind:gpu bash
#    - CPU base: --build-arg BASE_IMAGE=python:3.11-slim
#      Build: docker build -t spectramind:cpu --build-arg BASE_IMAGE=python:3.11-slim .
#
# 2) PyTorch wheels:
#    - For CUDA builds, PIP_EXTRA_INDEX_URL defaults to cu121 wheel index.
#      Override if you target a different CUDA minor:
#        --build-arg TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu124
#    - For CPU builds, set empty to avoid wrong wheel resolution:
#        --build-arg TORCH_WHL_INDEX=
#
# 3) Torch Geometric (if installed by Poetry):
#    - Make sure versions in pyproject.toml match your torch+cuda combo.
#    - If needed, you can install wheels at runtime:
#        pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-sparse  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#        pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
#
# 4) Caching:
#    - Keep dependency install isolated in the "deps" stage with only pyproject/lock copied in.
#    - Copy source later to avoid invalidating dependency layer on code edits.
#    - Use BuildKit cache mounts for pip/poetry if desired (docker buildx).
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
# 8) Healthcheck:
#    - Adjust the module name if your top-level package is not 'spectramind'.
# ------------------------------------------------------------------------------
