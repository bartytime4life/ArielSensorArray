# ==============================================================================
# SpectraMind V50 — Master Makefile (Ultimate, Upgraded)
#
# Neuro-Symbolic, Physics-Informed AI Pipeline
# Reproducibility • CLI-first • Hydra-safe • DVC • CI • Kaggle • Docker
# ==============================================================================

# ========= Shell & Make hygiene =========
SHELL                  := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS            := -Eeuo pipefail -c
MAKEFLAGS             += --warn-undefined-variables --no-builtin-rules --no-print-directory
.SUFFIXES:

# ========= Optional .env =========
ifneq (,$(wildcard .env))
include .env
export
endif

# ========= Tooling =========
PYTHON       ?= python3
POETRY       ?= poetry
CLI          ?= $(POETRY) run spectramind
NODE         ?= node
NPM          ?= npm
KAGGLE       ?= kaggle
PIP          ?= $(PYTHON) -m pip
DVC          ?= dvc
GIT          ?= git
DOCKER       ?= docker
COMPOSE      ?= docker compose

# ========= Determinism / Seeds =========
export PYTHONHASHSEED := 0
SEED         ?= 42

# ========= Run identity =========
GIT_SHA      := $(shell $(GIT) rev-parse --short HEAD 2>/dev/null || echo "nogit")
RUN_TS       := $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_ID       := $(RUN_TS)-$(GIT_SHA)

# ========= Defaults =========
DEVICE       ?= cpu
EPOCHS       ?= 1
TS           := $(shell date +%Y%m%d_%H%M%S)

OUT_DIR        ?= outputs
LOGS_DIR       ?= logs
DIAG_DIR       ?= $(OUT_DIR)/diagnostics
PRED_DIR       ?= $(OUT_DIR)/predictions
SUBMIT_DIR     ?= $(OUT_DIR)/submission
SUBMIT_ZIP     ?= $(SUBMIT_DIR)/bundle.zip
MANIFEST_DIR   ?= $(OUT_DIR)/manifests
RUN_HASH_FILE  ?= run_hash_summary_v50.json
KAGGLE_COMP    ?= neurips-2025-ariel

REQ_CORE     ?= requirements.txt
REQ_DEV      ?= requirements-dev.txt
REQ_KAGGLE   ?= requirements-kaggle.txt
REQ_MIN      ?= requirements-min.txt
REQ_FREEZE   ?= requirements.freeze.txt

DIAGRAMS_SRC_DIR  ?= assets/diagrams
DIAGRAMS_OUT_DIR  ?= $(DIAGRAMS_SRC_DIR)/outputs
MMD_MAIN          ?= $(DIAGRAMS_SRC_DIR)/architecture_stack.mmd
MMDC_BIN          ?= node_modules/.bin/mmdc

# ========= Docker =========
IMAGE_NAME          ?= spectramind-v50
IMAGE_TAG           ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE_GPU           := $(IMAGE_NAME):$(IMAGE_TAG)-gpu
IMAGE_CPU           := $(IMAGE_NAME):$(IMAGE_TAG)-cpu
BUILD_CACHE_DIR     ?= .docker-build-cache
WORKDIR_MOUNT       := -v $(PWD):/workspace -w /workspace
CACHE_BASE          ?= $(PWD)/.cache
HF_CACHE_MNT        := -v $(CACHE_BASE)/hf:/cache/hf
WANDB_CACHE_MNT     := -v $(CACHE_BASE)/wandb:/cache/wandb
PIP_CACHE_MNT       := -v $(CACHE_BASE)/pip:/root/.cache/pip
BASE_ENV            := -e PYTHONUNBUFFERED=1 -e HF_HOME=/cache/hf -e TRANSFORMERS_CACHE=/cache/hf -e WANDB_DIR=/cache/wandb -e WANDB_MODE=offline
HAS_NVIDIA          := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)
DOCKER_GPU_FLAG?=
ifeq ($(HAS_NVIDIA),1)
DOCKER_GPU_FLAG := --gpus all
endif

# ========= Colors =========
BOLD := \033[1m
DIM  := \033[2m
RED  := \033[31m
GRN  := \033[32m
YLW  := \033[33m
CYN  := \033[36m
RST  := \033[0m

# ========= PHONY =========
.PHONY: help list init env doctor quickstart \
        train predict diagnose submit \
        diagrams diagrams-clean \
        dvc-pull dvc-push dvc-status \
        docker-build-gpu docker-build-cpu \
        docker-run-gpu docker-run-cpu \
        cli cli-gpu cli-cpu \
        clean cache-clean realclean distclean

# ========= Default =========
.DEFAULT_GOAL := help

## Show top-level help with common tasks
help:
	@echo ""
	@echo "$(BOLD)SpectraMind V50 — Make targets$(RST)"
	@echo "  $(CYN)quickstart$(RST)     : install deps, init dirs, run doctor"
	@echo "  $(CYN)train$(RST)          : train (epochs=$(EPOCHS), device=$(DEVICE))"
	@echo "  $(CYN)predict$(RST)        : inference → $(PRED_DIR)/submission.csv"
	@echo "  $(CYN)diagnose$(RST)       : diagnostics (smoothness/dashboard)"
	@echo "  $(CYN)submit$(RST)         : package submission zip"
	@echo "  $(CYN)diagrams$(RST)       : render Mermaid diagrams"
	@echo "  $(CYN)docker-*(RST)        : dockerized workflow (GPU autodetect)"
	@echo "  $(CYN)list$(RST)           : list all targets with descriptions"
	@echo ""

## List all available targets with descriptions (reads '## ' lines above targets)
list:
	@echo "$(BOLD)Available targets$(RST) (auto-extracted):"
	@awk '\
	  BEGIN{FS=":"} \
	  /^## /{desc=$$0; sub(/^## /,"",desc); next} \
	  /^[a-zA-Z0-9_.-]+:([^=]|$$)/{ \
	    tgt=$$1; \
	    if (!seen[tgt]++) { \
	      printf "  %-22s %s\n", tgt, (desc!=""?desc:""); \
	    } \
	    desc=""; \
	  }' $(lastword $(MAKEFILE_LIST)) | sort -k1,1

# ========= Init / Env =========
## Create run/output directories
init: env

## Ensure output/log/diag/submit/diagram dirs exist
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)" "$(DIAGRAMS_OUT_DIR)" "$(MANIFEST_DIR)"

## Install deps (Poetry) and run doctor
quickstart: env
	$(POETRY) install --no-root
	$(MAKE) doctor

## Check core tooling presence and torch import
doctor:
	@ok=1; command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)Missing python3$(RST)"; ok=0; }; \
	command -v $(POETRY) >/dev/null 2>&1 || { echo "$(YLW)Poetry missing$(RST)"; ok=0; }; \
	$(PYTHON) - <<'PY' || ok=0; \
import sys; \
try: \
 import torch; print("torch:", getattr(torch, "__version__", "n/a")); \
except Exception as e: \
 print("torch: (missing)", e); \
PY
	test $$ok -eq 1

# ========= Core pipeline =========
## Train model (uses CLI; override EPOCHS/DEVICE)
train: init
	$(CLI) train +training.epochs=$(EPOCHS) +training.seed=$(SEED) --device $(DEVICE)

## Run inference and write submission.csv
predict: init
	mkdir -p "$(PRED_DIR)"
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv"

## Run diagnostics and export dashboard
diagnose: init
	$(CLI) diagnose dashboard --outdir "$(DIAG_DIR)" || true

## Package submission zip bundle
submit: init
	mkdir -p "$(SUBMIT_DIR)"
	$(CLI) submit --zip-out "$(SUBMIT_ZIP)"

# ========= Diagrams =========
## Render all Mermaid diagrams (uses bin/render-diagrams.sh)
diagrams:
	@echo ">>> Rendering diagrams"
	@npm ci
	@bash bin/render-diagrams.sh $(DIAGRAMS_SRC_DIR)

## Clean rendered diagram outputs
diagrams-clean:
	rm -rf "$(DIAGRAMS_OUT_DIR)"

# ========= DVC =========
## DVC pull artifacts (best-effort)
dvc-pull: ; $(DVC) pull || true

## DVC push artifacts (best-effort)
dvc-push: ; $(DVC) push || true

## DVC status (best-effort)
dvc-status: ; $(DVC) status || true

# ========= Docker =========
## Build GPU image (expects Dockerfile target: runtime-gpu)
docker-build-gpu:
	$(DOCKER) build --target runtime-gpu -t $(IMAGE_GPU) --progress=plain .

## Build CPU image (expects Dockerfile target: runtime-cpu)
docker-build-cpu:
	$(DOCKER) build --target runtime-cpu -t $(IMAGE_CPU) --progress=plain .

## Launch interactive GPU shell in container
docker-run-gpu:
	$(DOCKER) run --rm -it --gpus all $(WORKDIR_MOUNT) $(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) $(BASE_ENV) $(IMAGE_GPU) bash

## Launch interactive CPU shell in container
docker-run-cpu:
	$(DOCKER) run --rm -it $(WORKDIR_MOUNT) $(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) $(BASE_ENV) $(IMAGE_CPU) bash

## Run an arbitrary CLI command in best available image
cli:
	@if [ "$(HAS_NVIDIA)" = "1" ]; then $(MAKE) -s cli-gpu CMD="$(CMD)"; else $(MAKE) -s cli-cpu CMD="$(CMD)"; fi

## Run in GPU image: make cli CMD='spectramind --version'
cli-gpu:
	$(DOCKER) run --rm -t --gpus all $(WORKDIR_MOUNT) $(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) $(BASE_ENV) $(IMAGE_GPU) bash -lc '$(CMD)'

## Run in CPU image: make cli-cpu CMD='spectramind diagnose ...'
cli-cpu:
	$(DOCKER) run --rm -t $(WORKDIR_MOUNT) $(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) $(BASE_ENV) $(IMAGE_CPU) bash -lc '$(CMD)'

# ========= Cleanup =========
## Remove diagnostics, predictions, submission artifacts
clean:
	rm -rf "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

## Remove caches (pytest/ruff/mypy/DVC temp)
cache-clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .dvc/tmp || true

## Remove outputs and caches
realclean: clean cache-clean
	rm -rf "$(OUT_DIR)"

## Full reset (incl. venv & local pip caches)
distclean: realclean
	rm -rf .venv ~/.cache/pypoetry ~/.cache/pip || true