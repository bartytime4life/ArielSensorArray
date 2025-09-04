==============================================================================

SpectraMind V50 — Master Makefile (Ultimate, Upgraded)

Neuro-Symbolic, Physics-Informed AI Pipeline

Reproducibility • CLI-first • Hydra-safe • DVC • CI • Kaggle • Docker

==============================================================================

This Makefile is the single pane-of-glass for local dev, CI parity, Docker,

DVC, diagnostics, diagrams, Kaggle helpers, GUI demos, and reproducibility

manifests. Every target is designed to be:

• Deterministic (strict bash flags, PYTHONHASHSEED=0)

• CLI-first (delegates all analytics to spectramind …)

• Evidence-producing (logs, manifests, artifacts)

• Safe to call in CI (non-interactive, exits on error)



Conventions:

- Variables are overridable at invocation, e.g.:

make train DEVICE=gpu EPOCHS=3 OVERRIDES=”+data=ariel_nominal +training.seed=1337”

- OVERRIDES are Hydra-style “+key=value” tokens passed through to CLI.

- EXTRA_ARGS pass raw flags to CLI subcommands.

========= Shell & Make hygiene =========

SHELL                  := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS            := -Eeuo pipefail -c
MAKEFLAGS             += –warn-undefined-variables –no-builtin-rules –no-print-directory
.SUFFIXES:

Load optional .env (does not fail if missing)

ifneq (,$(wildcard .env))
include .env
export
endif

========= Tooling =========

PYTHON       ?= python3
POETRY       ?= poetry
CLI          ?= $(POETRY) run spectramind

NODE         ?= node
NPM          ?= npm
KAGGLE       ?= kaggle
PIP          ?= $(PYTHON) -m pip
DVC          ?= dvc
GIT          ?= git
JQ           ?= jq

Optional tools (best-effort checks in targets)

SBOM_SYFT    ?= syft
SBOM_GRYPE   ?= grype
PIP_AUDIT    ?= pip-audit
TRIVY        ?= trivy

GUI tools

STREAMLIT    ?= streamlit
UVICORN      ?= uvicorn

========= Determinism / Seeds =========

export PYTHONHASHSEED := 0
SEED         ?= 42

========= Repro run identity =========

GIT_SHA      := $(shell $(GIT) rev-parse –short HEAD 2>/dev/null || echo “nogit”)
RUN_TS       := $(shell date -u +%Y%m%dT%H%M%SZ)
RUN_ID       := $(RUN_TS)-$(GIT_SHA)

========= Defaults (override at CLI) =========

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

Kaggle competition handle

KAGGLE_COMP  ?= neurips-2025-ariel

Requirements files

REQ_CORE     ?= requirements.txt
REQ_EXTRAS   ?= requirements-extras.txt
REQ_DEV      ?= requirements-dev.txt
REQ_KAGGLE   ?= requirements-kaggle.txt
REQ_MIN      ?= requirements-min.txt
REQ_FREEZE   ?= requirements.freeze.txt

Mermaid / diagrams

DIAGRAMS_SRC_DIR  ?= diagrams
DIAGRAMS_OUT_DIR  ?= outputs/diagrams
MMD_MAIN          ?= $(DIAGRAMS_SRC_DIR)/main.mmd
MMDC_BIN          ?= mmdc

Hydra overrides / passthrough

OVERRIDES    ?=
EXTRA_ARGS   ?=

Docs export (pandoc)

DOC_MD    ?= assets/AI_Design_and_Modeling.md
DOC_HTML  ?= assets/AI_Design_and_Modeling.html
DOC_PDF   ?= assets/AI_Design_and_Modeling.pdf
DOC_TITLE ?= AI Design and Modeling — SpectraMind V50
DOC_CSS   ?= https://cdn.jsdelivr.net/npm/water.css@2/out/water.css

========= GUI demo suite locations =========

GUI_DIR            ?= docs/gui/demo_suite
STREAMLIT_APP      ?= $(GUI_DIR)/streamlit_demo.py
FASTAPI_APP        ?= $(GUI_DIR)/fastapi_backend.py
QT_APP             ?= $(GUI_DIR)/qt_diag_demo.py
FASTAPI_PORT       ?= 8089

========= Docker (upgraded, lean context, GPU/CPU multi-target) =========

DOCKER              ?= docker
export DOCKER_BUILDKIT = 1

image meta

IMAGE_NAME          ?= spectramind-v50
IMAGE_TAG           ?= $(shell git rev-parse –short HEAD 2>/dev/null || echo dev)
IMAGE_GPU           := $(IMAGE_NAME):$(IMAGE_TAG)-gpu
IMAGE_CPU           := $(IMAGE_NAME):$(IMAGE_TAG)-cpu

build cache dir (local)

BUILD_CACHE_DIR     ?= .docker-build-cache

repo mount → /workspace

WORKDIR_MOUNT       := -v $(PWD):/workspace -w /workspace

cache mounts for runtime (host-side)

CACHE_BASE          ?= $(PWD)/.cache
HF_CACHE_MNT        := -v $(CACHE_BASE)/hf:/cache/hf
WANDB_CACHE_MNT     := -v $(CACHE_BASE)/wandb:/cache/wandb
PIP_CACHE_MNT       := -v $(CACHE_BASE)/pip:/root/.cache/pip

base env (no secrets baked)

BASE_ENV            := 
-e PYTHONUNBUFFERED=1 
-e HF_HOME=/cache/hf 
-e TRANSFORMERS_CACHE=/cache/hf 
-e WANDB_DIR=/cache/wandb 
-e WANDB_MODE=offline

optional env-file (e.g., ENV_FILE=.env.local)

ENV_FILE            ?=
ENVFILE_FLAG        := $(if $(strip $(ENV_FILE)),–env-file $(ENV_FILE),)

pass-through build args if needed (avoid secrets)

EXTRA_BUILD_ARGS    ?=

detect NVIDIA runtime

HAS_NVIDIA          := $(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

Generic docker variables for legacy helpers

DOCKERFILE     ?= Dockerfile
DOCKER_FULL    ?= $(IMAGE_CPU)
DOCKER_GPU_FLAG?=
ifeq ($(HAS_NVIDIA),1)
DOCKER_GPU_FLAG := –gpus all
endif

Docker Compose

COMPOSE        ?= docker compose
COMPOSE_FILE   ?= docker-compose.yml

========= Colors =========

BOLD := \033[1m
DIM  := \033[2m
RED  := \033[31m
GRN  := \033[32m
YLW  := \033[33m
CYN  := \033[36m
RST  := \033[0m

========= PHONY =========

.PHONY: help init env info doctor versions guards 
fmt lint mypy test pre-commit 
selftest selftest-deep validate-env 
calibrate calibrate-temp corel-train train predict predict-e2e diagnose open-report open-latest 
submit submission-bin repair verify-submission bundle-zip 
ablate ablate-light ablate-heavy ablate-grid ablate-optuna 
analyze-log analyze-log-short check-cli-map 
dvc-pull dvc-push dvc-status dvc-check dvc-repro 
bench-selftest benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean 
benchmark-bin diagnostics-bin ci-smoke 
kaggle-verify kaggle-run kaggle-submit kaggle-dataset-create kaggle-dataset-push 
node-info node-ci mmd-version diagrams diagrams-png diagrams-svg diagrams-watch diagrams-lint diagrams-format diagrams-clean 
ci ci-fast ci-calibration ci-docs quickstart clean realclean distclean cache-clean 
export-reqs export-reqs-dev export-kaggle-reqs export-freeze 
install-core install-extras install-dev install-kaggle 
deps deps-min deps-lock verify-deps 
env-capture hash-config git-clean-check git-status release-tag 
pip-audit audit sbom sbom-scan docs docs-html docs-pdf docs-open docs-clean docs-serve docs-build 
pyg-install kaggle-pyg-index 
docker-help docker-print docker-build docker-buildx docker-run docker-shell docker-test docker-clean 
compose-ps compose-logs compose-up-gpu compose-up-cpu compose-up-api compose-up-web compose-up-docs compose-up-viz compose-up-lab compose-up-llm compose-up-ci compose-down compose-recreate compose-rebuild 
repro-start repro-snapshot repro-verify repro-manifest 
ensure-exec ensure-bin 
gui-demo gui-backend gui-backend-stop gui-qt gui-help

========= Default Goal =========

.DEFAULT_GOAL := help

========= Help =========

help:
@echo “”
@echo “$(BOLD)SpectraMind V50 — Make targets$(RST)”
@echo “  $(CYN)quickstart$(RST)          : install deps (poetry), init dirs, print info”
@echo “  $(CYN)doctor$(RST)              : dependency checks (python/poetry/node/npm/cli)”
@echo “  $(CYN)selftest$(RST)            : fast integrity checks (CLI + files)”
@echo “  $(CYN)train$(RST)               : run training (EPOCHS=$(EPOCHS), DEVICE=$(DEVICE))”
@echo “  $(CYN)predict$(RST)             : run inference → $(PRED_DIR)/submission.csv”
@echo “  $(CYN)predict-e2e$(RST)         : smoke test asserting submission exists”
@echo “  $(CYN)diagnose$(RST)            : build diagnostics (smoothness + dashboard)”
@echo “  $(CYN)open-report$(RST)         : open newest diagnostics HTML via system opener”
@echo “  $(CYN)submit$(RST)              : package submission ZIP ($(SUBMIT_ZIP))”
@echo “  $(CYN)submission-bin$(RST)      : run ./bin/make-submission.sh (dry-run unless ARGS overrides)”
@echo “  $(CYN)repair$(RST)              : run ./bin/repair_and_push.sh MSG="…"”
@echo “  $(CYN)ablate*$(RST)             : ablation sweeps (light/heavy/grid/optuna)”
@echo “  $(CYN)analyze-log$(RST)         : parse logs → $(OUT_DIR)/log_table.{md,csv}”
@echo “  $(CYN)benchmark$(RST)           : CLI-based benchmark pipeline (kept for parity)”
@echo “  $(CYN)benchmark-bin$(RST)       : run ./bin/benchmark.sh (preferred, feature-rich)”
@echo “  $(CYN)diagnostics-bin$(RST)     : run ./bin/diagnostics.sh (rich overlays + options)”
@echo “  $(CYN)ci-smoke$(RST)            : local smoke of benchmark+diagnostics (CPU)”
@echo “  $(CYN)repro-$(RST)             : run snapshot & manifest (config/data hashing)”
@echo “  $(CYN)dvc-$(RST)               : DVC pull/push/status/repro & sanity checks”
@echo “  $(CYN)kaggle-$(RST)            : Kaggle run/submit/dataset publish”
@echo “  $(CYN)diagrams$(RST)           : render Mermaid diagrams with mmdc”
@echo “  $(CYN)docker-build/run/shell$(RST) : Dockerized workflow (GPU autodetect)”
@echo “  $(CYN)compose-up-…$(RST)        : Start services by profile (gpu/cpu/api/web/docs/viz/lab/llm/ci)”
@echo “  $(CYN)gui-help$(RST)            : list GUI demo targets (Streamlit/FastAPI/Qt)”
@echo “”

========= Init / Env =========

init: env

env:
mkdir -p “$(OUT_DIR)” “$(LOGS_DIR)” “$(DIAG_DIR)” “$(PRED_DIR)” “$(SUBMIT_DIR)” “$(DIAGRAMS_OUT_DIR)” “$(MANIFEST_DIR)”

versions:
@echo “$(BOLD)Versions$(RST)”
@echo “python : $$($(PYTHON) –version 2>&1 || true)”
@echo “poetry : $$($(POETRY) –version 2>&1 || true)”
@echo “node   : $$($(NODE) –version 2>&1 || true)”
@echo “npm    : $$($(NPM) –version 2>&1 || true)”
@echo “kaggle : $$($(KAGGLE) –version 2>&1 || true)”
@echo “cli    : $(CLI)”

info: versions
@echo “device : $(DEVICE)”
@echo “OUT_DIR: $(OUT_DIR)”
@echo “RUN_ID : $(RUN_ID)”
@echo “RUN_HASH_FILE: $(RUN_HASH_FILE)”

doctor:
@ok=1; 
command -v $(PYTHON) >/dev/null 2>&1 || { echo “$(RED)Missing python3$(RST)”; ok=0; }; 
command -v $(POETRY) >/dev/null 2>&1 || { echo “$(YLW)Poetry not found — install via pipx/pip$(RST)”; ok=0; }; 
command -v $(NODE)   >/dev/null 2>&1 || { echo “$(YLW)Node not found (needed for mermaid-cli)$(RST)”; }; 
command -v $(NPM)    >/dev/null 2>&1 || { echo “$(YLW)npm not found (needed for mermaid-cli)$(RST)”; }; 
{ $(CLI) –version >/dev/null 2>&1 && echo “$(GRN)CLI OK$(RST)”; } || { echo “$(YLW)CLI not yet installed or venv not active$(RST)”; }; 
$(PYTHON) - <<‘PY’ || ok=0
import sys
try:
import torch
print(“torch:”, getattr(torch, “version”, “n/a”))
print(“cuda :”, getattr(getattr(torch, “version”, None), “cuda”, “n/a”))
except Exception:
print(“torch: (missing)”)
PY
test $$ok -eq 1

quickstart: env info
@echo “$(CYN)Installing project deps via Poetry (no-root)…$(RST)”
@$(POETRY) install –no-root
@echo “$(GRN)Done.$(RST)”
@$(MAKE) doctor

======== Guards ========

guards:
@command -v $(POETRY) >/dev/null 2>&1 || { echo “$(RED)Poetry missing$(RST)”; exit 1; }
@$(POETRY) run spectramind –version >/dev/null 2>&1 || { echo “$(RED)spectramind CLI not runnable$(RST)”; exit 1; }

========= Dev / Quality =========

fmt:
$(POETRY) run isort .
$(POETRY) run black .

lint:
$(POETRY) run ruff check .

mypy:
$(POETRY) run mypy –strict src || true

test: init
$(POETRY) run pytest -q || $(POETRY) run pytest -q -x

pre-commit:
$(POETRY) run pre-commit install
$(POETRY) run pre-commit run –all-files || true

========= Env validation =========

validate-env:
@if [ -x scripts/validate_env.py ] || [ -f scripts/validate_env.py ]; then 
echo “>>> Validating .env schema”; 
$(PYTHON) scripts/validate_env.py || exit 1; 
else 
echo “>>> Skipping validate-env (scripts/validate_env.py not found)”; 
fi

========= Pipeline =========

selftest: guards init
$(CLI) selftest

selftest-deep: guards init
$(CLI) selftest –deep

calibrate: guards init
$(CLI) calibrate $(OVERRIDES) $(EXTRA_ARGS)

calibrate-temp: guards init
$(CLI) calibrate-temp $(OVERRIDES) $(EXTRA_ARGS)

corel-train: guards init
$(CLI) corel-train $(OVERRIDES) $(EXTRA_ARGS)

train: guards init
$(CLI) train +training.epochs=$(EPOCHS) +training.seed=$(SEED) $(OVERRIDES) –device $(DEVICE) $(EXTRA_ARGS)

predict: guards init
mkdir -p “$(PRED_DIR)”
$(CLI) predict –out-csv “$(PRED_DIR)/submission.csv” $(OVERRIDES) $(EXTRA_ARGS)

verify-submission:
@test -s “$(PRED_DIR)/submission.csv” || { echo “::error::$(PRED_DIR)/submission.csv missing or empty”; exit 1; }
@head -n 1 “$(PRED_DIR)/submission.csv” | grep -Eiq ‘id|planet’ || { echo “::warning::Header not matched (id/planet) — continuing”; true; }

predict-e2e: predict verify-submission
@echo “$(GRN)OK: $(PRED_DIR)/submission.csv$(RST)”

diagnose: guards init
$(CLI) diagnose smoothness –outdir “$(DIAG_DIR)” $(EXTRA_ARGS)
$(CLI) diagnose dashboard –no-umap –no-tsne –outdir “$(DIAG_DIR)” $(EXTRA_ARGS) || 
$(CLI) diagnose dashboard –outdir “$(DIAG_DIR)” $(EXTRA_ARGS) || true

open-report:
@latest=$$(ls -t $(DIAG_DIR)/*.html 2>/dev/null | head -n1 || true); 
if [ -n “$$latest” ]; then 
echo “Opening $$latest”; 
if command -v xdg-open >/dev/null 2>&1; then xdg-open “$$latest” || true; 
elif command -v open >/dev/null 2>&1; then open “$$latest” || true; 
else echo “No opener found (CI/headless)”; fi; 
else echo “No diagnostics HTML found.”; fi

alias

open-latest: open-report

submit: guards init
mkdir -p “$(SUBMIT_DIR)”
$(CLI) submit –zip-out “$(SUBMIT_ZIP)” $(EXTRA_ARGS)

bundle-zip: repro-snapshot
@echo “>>> Creating reproducible bundle ZIP (manifests + predictions + diagnostics)”
@mkdir -p “$(OUT_DIR)/bundles”
@tar -czf “$(OUT_DIR)/bundles/spectramind_ci_bundle_$(RUN_ID).tar.gz” 
–exclude=’.pt’ –exclude=’.pth’ –exclude=’*.onnx’ 
“$(MANIFEST_DIR)” “$(DIAG_DIR)” “$(PRED_DIR)” “$(OUT_DIR)/log_table.md” “$(OUT_DIR)/log_table.csv” 2>/dev/null || true
@echo “Bundle: $(OUT_DIR)/bundles/spectramind_ci_bundle_$(RUN_ID).tar.gz”

========= bin/ wrappers =========

ensure-exec:
@mkdir -p “$(LOGS_DIR)”
@find bin -maxdepth 1 -type f -name “*.sh” -print0 2>/dev/null | xargs -0 chmod +x 2>/dev/null || true

ensure-bin: ensure-exec
@for f in bin/make-submission.sh bin/repair_and_push.sh bin/benchmark.sh bin/diagnostics.sh; do 
if [ ! -f “$$f” ]; then echo “::warning::missing $$f”; fi; 
done

Use with: make submission-bin ARGS=”–tag v50.0.1 –open”

submission-bin: ensure-exec
@echo “[bin] ./bin/make-submission.sh $(ARGS)”
@./bin/make-submission.sh $(ARGS)

Use with: make repair MSG=“Fix hashes”

repair: ensure-exec
@if [ -z “$(MSG)” ]; then echo “Usage: make repair MSG="Commit message"”; exit 2; fi
@echo “[bin] ./bin/repair_and_push.sh ‘$(MSG)’”
@./bin/repair_and_push.sh “$(MSG)”

New: run bin/benchmark.sh (feature-rich wrapper)

benchmark-bin: ensure-exec
@echo “[bin] ./bin/benchmark.sh $(ARGS)”
@./bin/benchmark.sh $(ARGS)

New: run bin/diagnostics.sh (rich overlays/options)

diagnostics-bin: ensure-exec
@echo “[bin] ./bin/diagnostics.sh $(ARGS)”
@./bin/diagnostics.sh $(ARGS)

Local smoke to mirror CI (CPU-only)

ci-smoke: ensure-exec
@./bin/benchmark.sh –profile cpu –epochs 1 –tag ci-smoke –manifest
@./bin/diagnostics.sh –no-umap –no-tsne –manifest

========= Ablation =========

ablate: guards init
$(CLI) ablate $(OVERRIDES) $(EXTRA_ARGS)

ablate-light: guards init
$(CLI) ablate ablation=ablation_light $(EXTRA_ARGS)

ablate-heavy: guards init
$(CLI) ablate ablation=ablation_heavy $(EXTRA_ARGS)

ablate-grid: guards init
$(CLI) ablate -m ablate.sweeper=basic +ablate.search=v50_fast_grid ablation=ablation_light $(EXTRA_ARGS)

ablate-optuna: guards init
$(CLI) ablate -m ablate.sweeper=optuna +ablate.search=v50_symbolic_core ablation=ablation_heavy $(EXTRA_ARGS)

========= Log analysis =========

analyze-log: guards init
$(CLI) analyze-log –md “$(OUT_DIR)/log_table.md” –csv “$(OUT_DIR)/log_table.csv” $(EXTRA_ARGS)

analyze-log-short: guards init
@if [ ! -f “$(OUT_DIR)/log_table.csv” ]; then 
echo “>>> Generating log CSV via analyze-log”; 
$(CLI) analyze-log –md “$(OUT_DIR)/log_table.md” –csv “$(OUT_DIR)/log_table.csv” $(EXTRA_ARGS); 
fi; 
if [ -f “$(OUT_DIR)/log_table.csv” ]; then 
echo “=== Last 5 CLI invocations ===”; 
tail -n +2 “$(OUT_DIR)/log_table.csv” | tail -n 5 | 
awk -F’,’ ‘BEGIN{OFS=” | “} {print “time=”$${1}, “cmd=”$${2}, “git_sha=”$${3}, “cfg=”$${4}}’; 
else 
echo “::warning::No log_table.csv to summarize”; 
fi

check-cli-map: guards
$(CLI) check-cli-map

========= DVC =========

dvc-pull:
$(DVC) pull || true

dvc-push:
$(DVC) push || true

dvc-status:
$(DVC) status || true

dvc-repro:
$(DVC) repro || true

dvc-check:
@echo “>>> DVC sanity”
@$(DVC) status -c || true
@$(DVC) doctor || true

========= Benchmarks (CLI-native pipeline kept for parity) =========

bench-selftest:
$(CLI) selftest

benchmark: bench-selftest
@$(MAKE) benchmark-run DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OVERRIDES=’$(OVERRIDES)’ EXTRA_ARGS=’$(EXTRA_ARGS)’

benchmark-cpu: bench-selftest
@$(MAKE) benchmark-run DEVICE=cpu EPOCHS=$(EPOCHS) OVERRIDES=’$(OVERRIDES)’ EXTRA_ARGS=’$(EXTRA_ARGS)’

benchmark-gpu: bench-selftest
@$(MAKE) benchmark-run DEVICE=gpu EPOCHS=$(EPOCHS) OVERRIDES=’$(OVERRIDES)’ EXTRA_ARGS=’$(EXTRA_ARGS)’

benchmark-run:
@OUTDIR=“benchmarks/$(TS)_$(DEVICE)”; 
mkdir -p “$$OUTDIR”; 
$(CLI) train +training.epochs=$(EPOCHS) +training.seed=$(SEED) $(OVERRIDES) –device $(DEVICE) –outdir “$$OUTDIR” $(EXTRA_ARGS); 
$(CLI) diagnose smoothness –outdir “$$OUTDIR” $(EXTRA_ARGS); 
$(CLI) diagnose dashboard –no-umap –no-tsne –outdir “$$OUTDIR” $(EXTRA_ARGS) || 
$(CLI) diagnose dashboard –outdir “$$OUTDIR” $(EXTRA_ARGS) || true; 
{ 
echo “Benchmark summary”; 
date; 
echo “python   : $$($(PYTHON) –version 2>&1)”; 
echo “poetry   : $$($(POETRY) –version 2>&1 || true)”; 
echo “cli      : $(CLI)”; 
echo “device   : $(DEVICE)”; 
echo “epochs   : $(EPOCHS)”; 
echo “seed     : $(SEED)”; 
echo “overrides: $(OVERRIDES)”; 
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true; 
echo “”; 
echo “Artifacts in $$OUTDIR:”; 
ls -lh “$$OUTDIR” || true; 
} > “$$OUTDIR/summary.txt”; 
echo “>>> Benchmark complete → $$OUTDIR/summary.txt”

benchmark-report:
mkdir -p aggregated
{ 
echo “# SpectraMind V50 Benchmark Report”; 
echo “”; 
for f in $$(find benchmarks -type f -name summary.txt | sort); do 
echo “## $$f”; echo “”; cat “$$f”; echo “”; 
done; 
} > aggregated/report.md
@echo “>>> Aggregated → aggregated/report.md”

benchmark-clean:
rm -rf benchmarks aggregated

========= Kaggle Helpers =========

kaggle-verify:
@command -v $(KAGGLE) >/dev/null 2>&1 || { echo “$(RED)Kaggle CLI missing$(RST)”; exit 1; }
@$(KAGGLE) competitions list >/dev/null 2>&1 || { echo “$(RED)Kaggle CLI not logged in$(RST)”; exit 1; }
@echo “$(GRN)Kaggle CLI OK$(RST)”

kaggle-run: guards init
@echo “>>> Running single-epoch GPU run (Kaggle-like)”
$(CLI) selftest
$(CLI) train +training.epochs=1 +training.seed=$(SEED) –device gpu –outdir “$(OUT_DIR)”
$(CLI) predict –out-csv “$(PRED_DIR)/submission.csv”

kaggle-submit: kaggle-verify kaggle-run
@echo “>>> Submitting to Kaggle competition $(KAGGLE_COMP)”
$(KAGGLE) competitions submit -c “$(KAGGLE_COMP)” -f “$(PRED_DIR)/submission.csv” -m “SpectraMind V50 auto-submit ($(RUN_ID))”

(Optional) publish artifacts as Kaggle dataset

kaggle-dataset-create:
@echo “>>> Creating Kaggle dataset placeholder (id: $$USER/spectramind-v50-$(RUN_TS))”
@$(KAGGLE) datasets create -p “$(OUT_DIR)” -u || true

kaggle-dataset-push:
@echo “>>> Pushing latest outputs as dataset”
@$(KAGGLE) datasets version -p “$(OUT_DIR)” -m “SpectraMind V50 artifacts $(RUN_ID)” -r zip -d

========= Requirements export / install =========

export-reqs:
@echo “>>> Exporting Poetry deps → $(REQ_CORE)”
$(POETRY) export -f requirements.txt –without-hashes -o $(REQ_CORE)

export-reqs-dev:
@echo “>>> Exporting Poetry deps (incl. dev) → $(REQ_DEV)”
$(POETRY) export -f requirements.txt –with dev –without-hashes -o $(REQ_DEV)

export-kaggle-reqs:
@echo “>>> Exporting Kaggle-friendly requirements → $(REQ_KAGGLE)”
$(POETRY) export -f requirements.txt –without-hashes | 
grep -vE ‘^(torch|torchvision|torchaudio|torch-geometric)(==|>=)’ > $(REQ_KAGGLE)

export-freeze:
@echo “>>> Freezing active env → $(REQ_FREEZE)”
$(PIP) freeze -q > $(REQ_FREEZE)
@echo “>>> Wrote $(REQ_FREEZE)”

install-core:
$(PIP) install -r $(REQ_CORE)

install-extras:
@if [ -f “$(REQ_EXTRAS)” ]; then $(PIP) install -r $(REQ_EXTRAS); else echo “::warning::$(REQ_EXTRAS) not found”; fi

install-dev:
$(PIP) install -r $(REQ_DEV)

install-kaggle:
$(PIP) install -r $(REQ_KAGGLE)

–– Unified dependency workflows ––

deps:
@echo “>>> Upgrading pip/setuptools/wheel”
$(PIP) install -U pip setuptools wheel
@echo “>>> Installing full dev/CI stack from $(REQ_CORE)”
@test -f “$(REQ_CORE)” || { echo “$(RED)$(REQ_CORE) not found$(RST)”; exit 1; }
$(PIP) install -r $(REQ_CORE)
@$(MAKE) verify-deps

deps-min:
@echo “>>> Upgrading pip/setuptools/wheel”
$(PIP) install -U pip setuptools wheel
@echo “>>> Installing minimal Kaggle runtime from $(REQ_MIN)”
@test -f “$(REQ_MIN)” || { echo “$(RED)$(REQ_MIN) not found$(RST)”; exit 1; }
$(PIP) install -r $(REQ_MIN) || true
@$(MAKE) verify-deps

deps-lock:
@echo “>>> Lock (Poetry), then export pinned requirements + freeze”
$(POETRY) lock –no-update
@$(MAKE) export-reqs
@$(MAKE) export-reqs-dev
@$(MAKE) export-freeze
@echo “$(GRN)Locked and exported.$(RST)”

verify-deps:
@echo “>>> Key package versions”
@$(PYTHON) - << ‘PY’
import importlib
pkgs = [“torch”,“torchvision”,“torchaudio”,“numpy”,“scipy”,“pandas”,“sklearn”,“matplotlib”,“umap”,“shap”,“typer”,“hydra”,“omegaconf”]
for name in pkgs:
mod = “sklearn” if name==“sklearn” else name
try:
m = importlib.import_module(mod)
print(f”{name:>14}: {getattr(m,’version’,‘n/a’)}”)
except Exception:
print(f”{name:>14}: (missing)”)
PY

========= CLI utilities (reproducibility) =========

env-capture:
$(CLI) env-capture

hash-config:
$(CLI) hash-config

git-clean-check:
@dirty=$$($(GIT) status –porcelain); 
if [ -n “$$dirty” ]; then echo “::warning::Git working tree dirty”; echo “$$dirty”; else echo “$(GRN)Git clean$(RST)”; fi

git-status:
$(GIT) status –short –branch

release-tag:
@if [ -z “$(TAG)” ]; then echo “Usage: make release-tag TAG=v0.50.0-best”; exit 2; fi
$(GIT) tag -a “$(TAG)” -m “Release $(TAG)”
$(GIT) push origin “$(TAG)”

========= Security / SBOM / Docs =========

pip-audit:
@echo “>>> pip-audit (CVE scan)”
@if ! command -v $(PIP_AUDIT) >/dev/null 2>&1; then $(PIP) install pip-audit; fi
pip-audit -r $(REQ_CORE) || true

sbom:
@echo “>>> SBOM via syft (if installed)”
@if command -v $(SBOM_SYFT) >/dev/null 2>&1; then 
$(SBOM_SYFT) packages dir:. -o cyclonedx-json > $(OUT_DIR)/sbom.json || true; 
echo “Wrote $(OUT_DIR)/sbom.json”; 
else echo “::warning::syft not found”; fi

sbom-scan:
@echo “>>> Vulnerability scan via grype/trivy (if installed)”
@if command -v $(SBOM_GRYPE) >/dev/null 2>&1; then 
$(SBOM_GRYPE) dir:. –add-cpes-if-none –fail-on high || true; 
else echo “::warning::grype not found”; fi
@if command -v $(TRIVY) >/dev/null 2>&1; then 
$(TRIVY) fs –severity HIGH,CRITICAL –exit-code 0 . || true; 
else echo “::warning::trivy not found”; fi

audit: pip-audit sbom sbom-scan

docs: docs-html docs-pdf

docs-html:
@command -v pandoc >/dev/null || { echo “pandoc not found. Install pandoc (and TeX for PDF).”; exit 1; }
@test -f “$(DOC_MD)” || { echo “Missing $(DOC_MD).”; exit 1; }
@mkdir -p assets
pandoc “$(DOC_MD)” -f markdown+smart -t html5 -s –metadata title=”$(DOC_TITLE)” -c “$(DOC_CSS)” -o “$(DOC_HTML)”
@echo “Wrote $(DOC_HTML)”

docs-pdf:
@command -v pandoc >/dev/null || { echo “pandoc not found. Install pandoc + TeX (texlive).”; exit 1; }
@test -f “$(DOC_MD)” || { echo “Missing $(DOC_MD).”; exit 1; }
pandoc “$(DOC_MD)” -f markdown+smart -V geometry:margin=1in -V linkcolor:blue -V fontsize=11pt -o “$(DOC_PDF)”
@echo “Wrote $(DOC_PDF)”

docs-open:
@if [ -f “$(DOC_HTML)” ]; then 
if command -v xdg-open >/dev/null 2>&1; then xdg-open “$(DOC_HTML)”; 
elif command -v open >/dev/null 2>&1; then open “$(DOC_HTML)”; 
else echo “Open $(DOC_HTML) manually”; fi; 
else echo “No HTML found. Run ‘make docs’ first.”; fi

Optional local docs serve/build (mkdocs or simple http)

docs-serve:
@if command -v mkdocs >/dev/null 2>&1; then 
mkdocs serve -a 0.0.0.0:8000; 
else 
echo “::notice::mkdocs not found; falling back to python -m http.server on docs/”; 
cd docs && $(PYTHON) -m http.server 8000; 
fi

docs-build:
@if command -v mkdocs >/dev/null 2>&1; then mkdocs build; else echo “::notice::mkdocs not found”; fi

docs-clean:
rm -f “$(DOC_HTML)” “$(DOC_PDF)”
@echo “Cleaned $(DOC_HTML) and $(DOC_PDF)”

========= CI convenience =========

ci: validate-env selftest train diagnose analyze-log-short

ci-fast: selftest train analyze-log-short

ci-calibration: selftest calibrate diagnose analyze-log-short

ci-docs: docs

========= PyTorch Geometric helper =========

kaggle-pyg-index:
@$(PYTHON) - << ‘PY’
import torch
ver = torch.version.split(’+’)[0]
cu  = (torch.version.cuda or ‘cpu’).replace(’.’,’’)
base = f”https://data.pyg.org/whl/torch-{ver}+{‘cu’+cu if torch.version.cuda else ‘cpu’}.html”
print(base)
PY

pyg-install:
@echo “>>> Installing torch-geometric matching the current torch/CUDA”
@PYG_INDEX=”$$( $(MAKE) –no-print-directory kaggle-pyg-index )”; 
echo “Using index: $$PYG_INDEX”; 
$(PIP) install torch-geometric==2.5.3 -f “$$PYG_INDEX”

========= Docker helpers =========

.PHONY: docker-help
docker-help:
@echo “Docker targets”
@echo “  make docker-build-gpu      # build GPU image  → $(IMAGE_GPU)”
@echo “  make docker-build-cpu      # build CPU image  → $(IMAGE_CPU)”
@echo “  make docker-run-gpu        # bash shell (GPU), mounts repo + caches”
@echo “  make docker-run-cpu        # bash shell (CPU), mounts repo + caches”
@echo “  make cli CMD=‘spectramind –version’        # auto GPU/CPU”
@echo “  make cli-gpu CMD=‘spectramind train …’    # run in GPU image”
@echo “  make cli-cpu CMD=‘spectramind diagnose …’ # run in CPU image”
@echo “  make docker-context-check  # quick context size sanity”
@echo “  make docker-cache-clean    # remove local build cache dir”

Build GPU (expects Dockerfile stage: runtime-gpu)

docker-build-gpu:
@mkdir -p “$(BUILD_CACHE_DIR)”
@echo “>> Building GPU image: $(IMAGE_GPU)”
$(DOCKER) build 
–target runtime-gpu 
-t $(IMAGE_GPU) 
–progress=plain 
–cache-from type=local,src=$(BUILD_CACHE_DIR) 
–cache-to   type=local,dest=$(BUILD_CACHE_DIR),mode=max 
$(EXTRA_BUILD_ARGS) 
.

Build CPU (expects Dockerfile stage: runtime-cpu)

docker-build-cpu:
@mkdir -p “$(BUILD_CACHE_DIR)”
@echo “>> Building CPU image: $(IMAGE_CPU)”
$(DOCKER) build 
–target runtime-cpu 
-t $(IMAGE_CPU) 
–progress=plain 
–cache-from type=local,src=$(BUILD_CACHE_DIR) 
–cache-to   type=local,dest=$(BUILD_CACHE_DIR),mode=max 
$(EXTRA_BUILD_ARGS) 
.

Interactive shells

docker-run-gpu:
@echo “>> Running GPU shell: $(IMAGE_GPU)”
$(DOCKER) run –rm -it –name spectramind-gpu 
–gpus all 
$(WORKDIR_MOUNT) 
$(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) 
$(BASE_ENV) $(ENVFILE_FLAG) 
$(IMAGE_GPU) bash

docker-run-cpu:
@echo “>> Running CPU shell: $(IMAGE_CPU)”
$(DOCKER) run –rm -it –name spectramind-cpu 
$(WORKDIR_MOUNT) 
$(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) 
$(BASE_ENV) $(ENVFILE_FLAG) 
$(IMAGE_CPU) bash

CLI runners (auto-select GPU if available)

CMD ?= spectramind –help

cli:
@echo “>> Detecting NVIDIA runtime…”
@if [ “$(HAS_NVIDIA)” = “1” ]; then 
echo ‘>> NVIDIA found — using GPU image’; 
$(MAKE) -s cli-gpu CMD=”$(CMD)”; 
else 
echo ‘>> NVIDIA not found — using CPU image’; 
$(MAKE) -s cli-cpu CMD=”$(CMD)”; 
fi

cli-gpu:
@echo “>> [GPU] $(CMD)”
$(DOCKER) run –rm -t –name spectramind-cli-gpu 
–gpus all 
$(WORKDIR_MOUNT) 
$(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) 
$(BASE_ENV) $(ENVFILE_FLAG) 
$(IMAGE_GPU) bash -lc ‘$(CMD)’

cli-cpu:
@echo “>> [CPU] $(CMD)”
$(DOCKER) run –rm -t –name spectramind-cli-cpu 
$(WORKDIR_MOUNT) 
$(HF_CACHE_MNT) $(WANDB_CACHE_MNT) $(PIP_CACHE_MNT) 
$(BASE_ENV) $(ENVFILE_FLAG) 
$(IMAGE_CPU) bash -lc ‘$(CMD)’

docker-context-check:
@echo “>> Docker build context (top offenders):”
@du -h -d1 . | sort -hr | head -n 30

docker-cache-clean:
@rm -rf “$(BUILD_CACHE_DIR)”
@echo “>> Removed $(BUILD_CACHE_DIR)”

docker-print:
@echo “Image : $(DOCKER_FULL)”
@echo “GPU   : $(HAS_NVIDIA)”
@echo “File  : $(DOCKERFILE)”
@echo “Args  : $(DOCKER_BUILD_ARGS)”

DOCKER_BUILD_ARGS ?=

docker-build: docker-print
$(DOCKER) build 
–build-arg BUILDKIT_INLINE_CACHE=1 
-f $(DOCKERFILE) 
-t $(DOCKER_FULL) 
$(DOCKER_BUILD_ARGS) 
.

docker-buildx: docker-print
$(DOCKER) buildx build –load 
–build-arg BUILDKIT_INLINE_CACHE=1 
-f $(DOCKERFILE) 
-t $(DOCKER_FULL) 
$(DOCKER_BUILD_ARGS) 
.

docker-run: init
$(DOCKER) run –rm -it $(DOCKER_GPU_FLAG) 
-v “$$(pwd):/workspace” 
-w /workspace 
-e DEVICE=$(DEVICE) 
-e EPOCHS=$(EPOCHS) 
-e PYTHONHASHSEED=$(PYTHONHASHSEED) 
$(DOCKER_FULL) 
bash -lc ‘make ci || true’

docker-shell: init
$(DOCKER) run –rm -it $(DOCKER_GPU_FLAG) 
-v “$$(pwd):/workspace” 
-w /workspace 
-e PYTHONHASHSEED=$(PYTHONHASHSEED) 
$(DOCKER_FULL) 
bash

docker-test: init
$(DOCKER) run –rm $(DOCKER_GPU_FLAG) 
-v “$$(pwd):/workspace” 
-w /workspace 
-e PYTHONHASHSEED=$(PYTHONHASHSEED) 
$(DOCKER_FULL) 
bash -lc ‘make selftest && make test’

docker-clean:
-$(DOCKER) image rm $(DOCKER_FULL) 2>/dev/null || true

========= Docker Compose (profiles from docker-compose.yml) =========

compose-ps:
$(COMPOSE) -f $(COMPOSE_FILE) ps

compose-logs:
$(COMPOSE) -f $(COMPOSE_FILE) logs -f –tail=200

compose-up-gpu:
$(COMPOSE) -f $(COMPOSE_FILE) –profile gpu up -d spectramind-gpu

compose-up-cpu:
$(COMPOSE) -f $(COMPOSE_FILE) –profile cpu up -d spectramind-cpu

compose-up-api:
$(COMPOSE) -f $(COMPOSE_FILE) –profile api up -d api

compose-up-web:
$(COMPOSE) -f $(COMPOSE_FILE) –profile web up -d web

compose-up-docs:
$(COMPOSE) -f $(COMPOSE_FILE) –profile docs up docs

compose-up-viz:
$(COMPOSE) -f $(COMPOSE_FILE) –profile viz up tensorboard

compose-up-lab:
$(COMPOSE) -f $(COMPOSE_FILE) –profile lab up jupyter

compose-up-llm:
$(COMPOSE) -f $(COMPOSE_FILE) –profile llm up -d ollama

compose-up-ci:
$(COMPOSE) -f $(COMPOSE_FILE) –profile ci up –abort-on-container-exit ci

compose-down:
$(COMPOSE) -f $(COMPOSE_FILE) down

compose-recreate:
$(COMPOSE) -f $(COMPOSE_FILE) up -d –force-recreate

compose-rebuild:
$(COMPOSE) -f $(COMPOSE_FILE) build –no-cache

========= Mermaid / Diagrams =========

node-info:
@echo “node : $$($(NODE) –version 2>/dev/null || echo ‘missing’)”
@echo “npm  : $$($(NPM) –version 2>/dev/null || echo ‘missing’)”
@echo “mmdc : $$($(MMDC_BIN) -V 2>/dev/null || echo ‘missing’)”

mmd-version:
@$(MMDC_BIN) -V 2>/dev/null || echo “::warning::mmdc not installed”

node-ci:
@command -v $(NPM) >/dev/null 2>&1 || { echo “$(YLW)npm missing — skip mmdc install$(RST)”; exit 0; }
@command -v $(MMDC_BIN) >/dev/null 2>&1 || { echo “>>> Installing @mermaid-js/mermaid-cli globally”; $(NPM) i -g @mermaid-js/mermaid-cli; }

diagrams: diagrams-png

diagrams-png:
@mkdir -p “$(DIAGRAMS_OUT_DIR)”
@if [ -f “$(MMD_MAIN)” ]; then 
echo “>>> Rendering PNG from $(MMD_MAIN)”; 
$(MMDC_BIN) -i “$(MMD_MAIN)” -o “$(DIAGRAMS_OUT_DIR)/main.png” -b light || true; 
else echo “::warning::No $(MMD_MAIN) found”; fi

diagrams-svg:
@mkdir -p “$(DIAGRAMS_OUT_DIR)”
@if [ -f “$(MMD_MAIN)” ]; then 
echo “>>> Rendering SVG from $(MMD_MAIN)”; 
$(MMDC_BIN) -i “$(MMD_MAIN)” -o “$(DIAGRAMS_OUT_DIR)/main.svg” -b light || true; 
else echo “::warning::No $(MMD_MAIN) found”; fi

diagrams-watch:
@command -v entr >/dev/null 2>&1 || { echo “::warning::entr not installed; watch disabled”; exit 0; }
@ls $(DIAGRAMS_SRC_DIR)/*.mmd | entr -r make diagrams

diagrams-lint:
@echo “>>> Lint diagrams — ensure .mmd files compile”
@for f in $(DIAGRAMS_SRC_DIR)/*.mmd; do 
[ -f “$$f” ] || continue; 
$(MMDC_BIN) -i “$$f” -o /dev/null >/dev/null 2>&1 || echo “::warning::Failed to render $$f”; 
done

diagrams-format:
@echo “>>> (Optional) apply prettier to Mermaid if configured”

diagrams-clean:
rm -rf “$(DIAGRAMS_OUT_DIR)”

========= Reproducibility snapshots (config+data hashing, manifest) =========

repro-start: init
@echo “>>> Starting reproducible run $(RUN_ID)”
@echo “$(RUN_ID)” > “$(LOGS_DIR)/current_run_id.txt”

repro-snapshot: guards init
@echo “>>> Capturing environment & config”
$(CLI) env-capture || true
$(CLI) hash-config || true
@echo “>>> Capturing DVC status”
$(DVC) status -c > “$(MANIFEST_DIR)/dvc_status_$(RUN_ID).txt” || true
@echo “>>> Writing run manifest JSON”
@$(PYTHON) - <<‘PY’
import json, os, subprocess, time, pathlib
run_id = os.environ.get(“RUN_ID”,“unknown”)
outdir = os.environ.get(“MANIFEST_DIR”,“outputs/manifests”)
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
def sh(cmd):
try:
return subprocess.check_output(cmd, shell=True, text=True).strip()
except Exception:
return “”
manifest = {
“run_id”: run_id,
“ts_utc”: time.strftime(”%Y-%m-%dT%H:%M:%SZ”, time.gmtime()),
“git”: {
“commit”: sh(“git rev-parse –short HEAD 2>/dev/null || echo ‘nogit’”),
“status”: sh(“git status –porcelain || true”)
},
“hydra_config_hash”: sh(“spectramind hash-config 2>/dev/null || echo ‘’”),
“device”: os.environ.get(“DEVICE”,””),
“epochs”: os.environ.get(“EPOCHS”,””),
“seed”: os.environ.get(“SEED”,””),
}
fp = os.path.join(outdir, f”run_manifest_{run_id}.json”)
with open(fp, “w”) as f:
json.dump(manifest, f, indent=2)
print(“Wrote manifest:”, fp)
PY

repro-verify:
@echo “>>> Manifest files:”
@ls -lh “$(MANIFEST_DIR)” || true
@echo “>>> Show last manifest:”
@ls -t “$(MANIFEST_DIR)”/run_manifest_*.json 2>/dev/null | head -n1 | xargs -I{} cat {}

repro-manifest: repro-start repro-snapshot

========= GUI demo suite targets =========

gui-help:
@echo “$(BOLD)GUI Demo Targets$(RST)”
@echo “  $(CYN)gui-demo$(RST)           : Run Streamlit demo (thin wrapper around CLI & artifacts)”
@echo “  $(CYN)gui-backend$(RST)        : Run FastAPI backend (contracts for React dashboard) on port $(FASTAPI_PORT)”
@echo “  $(CYN)gui-backend-stop$(RST)   : Try to stop FastAPI dev server (best-effort)”
@echo “  $(CYN)gui-qt$(RST)             : Run PyQt demo (QProcess + embedded diagnostics HTML)”
@echo “”
@echo “Notes: demos live under $(GUI_DIR). They do not mutate the pipeline and only call $(CLI) + render artifacts.”

gui-demo: init
@command -v $(STREAMLIT) >/dev/null 2>&1 || { echo “$(YLW)streamlit not found — install via Poetry/pip$(RST)”; exit 1; }
@test -f “$(STREAMLIT_APP)” || { echo “$(RED)Missing $(STREAMLIT_APP)$(RST)”; exit 1; }
$(STREAMLIT) run “$(STREAMLIT_APP)”

gui-backend: init
@command -v $(UVICORN) >/dev/null 2>&1 || { echo “$(YLW)uvicorn not found — install via Poetry/pip$(RST)”; exit 1; }
@test -f “$(FASTAPI_APP)” || { echo “$(RED)Missing $(FASTAPI_APP)$(RST)”; exit 1; }
$(UVICORN) “$(FASTAPI_APP:.py=):app” –reload –port $(FASTAPI_PORT)

gui-backend-stop:
@pgrep -f “uvicorn.$(FASTAPI_APP:.py=):app” >/dev/null 2>&1 && 
kill $$(pgrep -f “uvicorn.$(FASTAPI_APP:.py=):app”) || 
echo “::warning::No uvicorn process found”

gui-qt: init
@test -f “$(QT_APP)” || { echo “$(RED)Missing $(QT_APP)$(RST)”; exit 1; }
$(PYTHON) “$(QT_APP)”

========= Cleanup =========

clean:
rm -rf “$(DIAG_DIR)” “$(PRED_DIR)” “$(SUBMIT_DIR)”

cache-clean:
@echo “>>> Cleaning caches and logs”
find . -type d -name “pycache” -prune -exec rm -rf {} + || true
rm -rf .pytest_cache .ruff_cache .mypy_cache .dvc/tmp || true
find $(LOGS_DIR) -type f -name “*.log” -delete 2>/dev/null || true

realclean: clean cache-clean
rm -rf “$(OUT_DIR)”
rm -rf .dvc/cache

distclean: realclean
@echo “>>> Removing Poetry caches and local venv (full reset)”
rm -rf .venv
rm -rf ~/.cache/pypoetry || true
rm -rf ~/.cache/pip || true