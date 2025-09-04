syntax=docker/dockerfile:1.6

——————————————————————————

SpectraMind V50 — Dockerfile (Upgraded)

GPU-ready, reproducible, CI/Compose-friendly, Poetry-based, multi-stage



Highlights

• CPU/GPU switch via BASE_IMAGE

• BuildKit caches for apt/pip/poetry → fast, repeatable builds

• Non-root user, tini (PID1), healthcheck, deterministic threads/locale

• Optional Torch/DVC install via build-args (CUDA/CPU wheel indexes)

• Installs package exposing spectramind CLI



Quick switches:

• GPU (default): nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

• CPU:           –build-arg BASE_IMAGE=python:3.11-slim



Build examples:

docker build -t spectramind:gpu .

docker build -t spectramind:cpu –build-arg BASE_IMAGE=python:3.11-slim .



Run examples:

docker run –rm -it –gpus all -v $PWD:/workspace spectramind:gpu spectramind –help

docker run –rm -it -v $PWD:/workspace spectramind:cpu spectramind –help

——————————————————————————

===== 1) Base (CUDA or pure Python) =====

ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

—– OCI labels (override via –build-arg) —–

ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title=“SpectraMind V50” 
org.opencontainers.image.description=“Neuro-symbolic, physics-informed AI pipeline for NeurIPS 2025 Ariel Data Challenge” 
org.opencontainers.image.vendor=“SpectraMind” 
org.opencontainers.image.source=“https://github.com/bartytime4life/SpectraMindV50” 
org.opencontainers.image.revision=”${VCS_REF}” 
org.opencontainers.image.created=”${BUILD_DATE}” 
org.opencontainers.image.licenses=“MIT”

Common env

ENV DEBIAN_FRONTEND=noninteractive 
TZ=UTC 
LANG=C.UTF-8 
LC_ALL=C.UTF-8 
PYTHONDONTWRITEBYTECODE=1 
PYTHONUNBUFFERED=1 
PIP_DISABLE_PIP_VERSION_CHECK=1 
PIP_NO_CACHE_DIR=1 
MPLBACKEND=Agg 
HYDRA_FULL_ERROR=1

Torch wheel index (auto for CUDA; empty for CPU to avoid mismatch)

Override at build: –build-arg TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu124 or “” for CPU

ARG TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu121
ENV PIP_EXTRA_INDEX_URL=${TORCH_WHL_INDEX}

Versions / args

ARG POETRY_VERSION=1.8.3
ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

SHELL [”/bin/bash”, “-o”, “pipefail”, “-c”]

System deps (with BuildKit apt cache)

RUN set -eux; 
if command -v apt-get >/dev/null 2>&1; then 
–mount=type=cache,target=/var/cache/apt,sharing=locked 
–mount=type=cache,target=/var/lib/apt,sharing=locked 
apt-get update && apt-get install -y –no-install-recommends 
python3 python3-venv python3-pip python3-dev 
build-essential pkg-config 
git git-lfs curl ca-certificates 
tini 
libgl1 libglib2.0-0 
graphviz libgraphviz-dev 
ffmpeg 
&& rm -rf /var/lib/apt/lists/* && 
update-alternatives –install /usr/bin/python python /usr/bin/python3 1 && 
git lfs install –system; 
else 
echo “Non-apt base detected; assuming Python toolchain is present.”; 
fi

Poetry (global)

ENV POETRY_HOME=/opt/poetry 
POETRY_NO_INTERACTION=1 
POETRY_VIRTUALENVS_CREATE=false
RUN –mount=type=cache,target=/root/.cache,sharing=locked 
curl -sSL https://install.python-poetry.org | python - –version ${POETRY_VERSION} 
&& ln -sf ${POETRY_HOME}/bin/poetry /usr/local/bin/poetry

Non-root user (volume ownership parity with host)

RUN groupadd -g ${GID} ${USERNAME} 
&& useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

WORKDIR /workspace

===== 2) Deps stage (cache-friendly, BuildKit-aware) =====

FROM base AS deps

Copy only dependency manifests to leverage Docker layer cache

COPY pyproject.toml poetry.lock* ./

Optional: bootstrap tools (pinned) for reproducibility

RUN python -m pip install –upgrade pip setuptools wheel

Install project dependencies (no source yet)

RUN –mount=type=cache,target=/root/.cache/pip,sharing=locked 
poetry install –no-interaction –no-ansi –no-root

===== 3) Builder stage (install project for CLI) =====

FROM base AS build

Bring in resolved deps/site-packages and console scripts

COPY –from=deps /usr/local /usr/local

Copy source AFTER deps for better caching

COPY . /workspace

Optional Torch install (GPU/CPU) — controlled by build args

• GPU (CUDA):  –build-arg INSTALL_TORCH=true –build-arg TORCH_WHL_INDEX=https://download.pytorch.org/whl/cu124

• CPU:         –build-arg INSTALL_TORCH=true –build-arg TORCH_WHL_INDEX=

ARG INSTALL_TORCH=false
ARG TORCH_VERSION=
ARG TORCHVISION_VERSION=
ARG TORCHAUDIO_VERSION=
RUN –mount=type=cache,target=/root/.cache/pip,sharing=locked 
if [ “${INSTALL_TORCH}” = “true” ]; then 
TORCH_PKG=”${TORCH_VERSION:+torch==${TORCH_VERSION}}”; 
TORCH_PKG=”${TORCH_PKG:-torch}”; 
VISION_PKG=”${TORCHVISION_VERSION:+torchvision==${TORCHVISION_VERSION}}”; 
VISION_PKG=”${VISION_PKG:-torchvision}”; 
AUDIO_PKG=”${TORCHAUDIO_VERSION:+torchaudio==${TORCHAUDIO_VERSION}}”; 
AUDIO_PKG=”${AUDIO_PKG:-torchaudio}”; 
pip install –no-cache-dir ${TORCH_PKG} ${VISION_PKG} ${AUDIO_PKG}; 
fi

Optional DVC remotes (e.g., s3,gdrive,azure,ssh,gs,webdav)

ARG INSTALL_DVC=false
ARG DVC_EXTRAS=“s3”
RUN –mount=type=cache,target=/root/.cache/pip,sharing=locked 
if [ “${INSTALL_DVC}” = “true” ]; then 
pip install –no-cache-dir “dvc[${DVC_EXTRAS}]”; 
fi

Install project so entry points (e.g., spectramind) are available

RUN poetry install –no-interaction –no-ansi

Quick smoke: can we import & show help?

RUN spectramind –help >/dev/null

===== 4) Runtime stage (slim, non-root, tini) =====

FROM base AS runtime

Copy the fully installed Python env + scripts from build

COPY –from=build /usr/local /usr/local

Copy the working tree for runtime (configs/assets; code already installed)

COPY –chown=${UID}:${GID} . /workspace

Runtime envs (determinism & headless)

ENV HF_HOME=/home/${USERNAME}/.cache/huggingface 
HF_HUB_DISABLE_TELEMETRY=1 
TRANSFORMERS_OFFLINE=1 
PYTHONHASHSEED=0 
OMP_NUM_THREADS=1 
MKL_NUM_THREADS=1 
OPENBLAS_NUM_THREADS=1

Writable dirs for logs / outputs / data (bind-mount friendly)

RUN mkdir -p /workspace/outputs /workspace/logs /workspace/data 
&& chown -R ${UID}:${GID} /workspace /home/${USERNAME}

Optional CUDA probe (non-fatal)

RUN bash -lc ‘ldconfig -p | grep -i cuda || true’

Switch to non-root

USER ${USERNAME}

Healthcheck: verify CLI import and help

HEALTHCHECK –interval=30s –timeout=8s –start-period=45s –retries=3 
CMD python -c “import importlib; importlib.import_module(‘spectramind’)” >/dev/null 2>&1 && spectramind –help >/dev/null 2>&1 || exit 1

Volumes (hint to orchestrators)

VOLUME [”/workspace/outputs”, “/workspace/logs”, “/workspace/data”]

Tini as entrypoint for proper signal handling & zombie reaping

ENTRYPOINT [”/usr/bin/tini”, “–”]
CMD [“spectramind”, “–help”]

——————————————————————————

Notes / Tips

——————————————————————————

1) Torch Geometric (example for Torch 2.4 + cu121):

pip install torch-scatter     -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install torch-sparse      -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install torch-cluster     -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html



2) Compose (GPU):

version: “3.9”

services:

smv50:

image: spectramind:gpu

deploy:

resources:

reservations:

devices:

- capabilities: [“gpu”]

volumes:

- ./:/workspace



3) UID/GID mapping:

Build with: –build-arg UID=$(id -u) –build-arg GID=$(id -g)

Ensures host-mounted files are owned by your user.



4) Cache hygiene:

Keep dependency install isolated in the “deps” stage and copy source later.

BuildKit cache mounts speed up apt, pip, and poetry during rebuilds.