# ------------------------------------------------------------------------------
# SpectraMind V50 — Dockerfile (GPU-ready, reproducible, CI/Compose friendly)
# - CUDA base by default; switch to CPU base with: --build-arg BASE_IMAGE=python:3.10-slim
# - Multi-stage: deps (poetry) -> runtime
# - Non-root user, sane defaults, Tini init, clean layers
# ------------------------------------------------------------------------------

# ========== 1) Base (can be CUDA or pure Python) ==========
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# OCI labels (fill in as you like)
LABEL org.opencontainers.image.title="SpectraMind V50" \
      org.opencontainers.image.description="Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge" \
      org.opencontainers.image.vendor="SpectraMind" \
      org.opencontainers.image.source="https://github.com/bartytime4life/ArielSensorArray"

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Versions / args
ARG POETRY_VERSION=1.8.3
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

# System deps (minimal but practical)
# - build-essential & python3-dev: compile wheels if needed
# - curl, ca-certificates: fetch/install tools
# - git: resolve vcs deps
# - tini: proper PID 1
# - libgl1, libglib2.0-0: common runtime deps for cv/plot libs
# - pkg-config: help some native builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential pkg-config \
    git curl ca-certificates \
    tini \
    libgl1 libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Set up Python alias (on CUDA images python3 is present; ensure 'python' exists)
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install Poetry globally
ENV POETRY_HOME=/opt/poetry
RUN curl -sSL https://install.python-poetry.org | python - --version ${POETRY_VERSION} \
  && ln -s ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

# Create non-root user (matching host UID/GID helps with bind mounts)
RUN groupadd -g ${GID} ${USERNAME} \
  && useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

# ========== 2) Deps stage (leverage Docker layer caching) ==========
FROM base AS deps

# Copy only dependency manifests to maximize cache hits
COPY pyproject.toml poetry.lock* ./

# Configure Poetry and install project deps (no editable install yet, no root package)
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi --no-root

# ========== 3) Runtime stage ==========
FROM base AS runtime

# Copy installed site-packages / scripts from deps
COPY --from=deps /usr/local /usr/local

# Workdir and user
WORKDIR /workspace
USER ${USERNAME}

# Copy the repository (after deps to preserve caching)
# Use .dockerignore to exclude data/, outputs/, .venv/, etc.
COPY --chown=${UID}:${GID} . .

# Optional: verify CLI is available (won't fail build if not wired yet)
# RUN python -m spectramind --version || true

# Healthcheck (lightweight; adjust to your needs)
# Here we just verify Python can import the package
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import importlib; importlib.import_module('spectramind')" >/dev/null 2>&1 || exit 1

# Default shell; Tini as entrypoint for signal handling & zombie reaping
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]

# ---------- Notes ----------
# 1) CUDA vs CPU:
#    - GPU build (default): BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
#      Run with: docker run --gpus all ...
#    - CPU build: --build-arg BASE_IMAGE=python:3.10-slim
#
# 2) Poetry:
#    - Project deps are installed in the deps stage via poetry install --no-root.
#    - If you want to install the project itself as a package (entrypoints, etc.):
#        RUN poetry install --no-interaction --no-ansi
#      (Move that line into the deps stage and keep copying /usr/local forward.)
#
# 3) Caching:
#    - Keep pyproject.toml / poetry.lock caching high; only copy source after deps.
#    - Use a .dockerignore to avoid sending large data/outputs directories to the daemon.
#
# 4) Compose:
#    - Use device requests for NVIDIA in docker-compose.yml:
#        deploy.resources.reservations.devices:
#          - capabilities: ["gpu"]
#      Or with Compose v2: device_requests:
#          - driver: "nvidia"
#            count: -1
#            capabilities: ["gpu"]
#
# 5) User mapping:
#    - UID/GID args let container writes be owned by your host user when volume-mounting.