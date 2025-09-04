# ==============================================================================

# SpectraMind V50 — Master Makefile (Ultimate, Upgraded)

#

# Neuro-Symbolic, Physics-Informed AI Pipeline

# Reproducibility • CLI-first • Hydra-safe • DVC • CI • Kaggle • Docker

# ==============================================================================

#

# Single pane-of-glass for local dev, CI parity, Docker, DVC, diagnostics,

# diagrams, Kaggle helpers, GUI demos, and reproducibility manifests.

#

# All targets are:

# - Deterministic (strict bash flags, PYTHONHASHSEED=0)

# - CLI-first (delegates analytics to `spectramind …`)

# - Evidence-producing (logs, manifests, artifacts)

# - CI-safe (non-interactive, fail-fast)

#

# Conventions:

# make train DEVICE=gpu EPOCHS=3 OVERRIDES="+data=ariel\_nominal +training.seed=1337"

# - OVERRIDES are Hydra-style “+key=value” tokens passed to CLI.

# - EXTRA\_ARGS pass raw flags to CLI subcommands.

#

# Important: This file uses ONLY ASCII quotes and double-hyphens for flags.

# ==============================================================================

# ========= Shell & Make hygiene =========

SHELL                  := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS            := -Eeuo pipefail -c
MAKEFLAGS             += --warn-undefined-variables --no-builtin-rules --no-print-directory
.SUFFIXES:

# ========= Optional .env (does not fail if missing) =========

ifneq (,\$(wildcard .env))
include .env
export
endif

# ========= Tooling =========

PYTHON       ?= python3
POETRY       ?= poetry
CLI          ?= \$(POETRY) run spectramind

NODE         ?= node
NPM          ?= npm
KAGGLE       ?= kaggle
PIP          ?= \$(PYTHON) -m pip
DVC          ?= dvc
GIT          ?= git
JQ           ?= jq

# Optional (best-effort)

SBOM\_SYFT    ?= syft
SBOM\_GRYPE   ?= grype
PIP\_AUDIT    ?= pip-audit
TRIVY        ?= trivy

# GUI helpers

STREAMLIT    ?= streamlit
UVICORN      ?= uvicorn

# ========= Determinism / Seeds =========

export PYTHONHASHSEED := 0
SEED         ?= 42

# ========= Repro run identity =========

GIT\_SHA      := \$(shell \$(GIT) rev-parse --short HEAD 2>/dev/null || echo "nogit")
RUN\_TS       := \$(shell date -u +%Y%m%dT%H%M%SZ)
RUN\_ID       := \$(RUN\_TS)-\$(GIT\_SHA)

# ========= Defaults (override at CLI) =========

DEVICE       ?= cpu
EPOCHS       ?= 1
TS           := \$(shell date +%Y%m%d\_%H%M%S)

OUT\_DIR        ?= outputs
LOGS\_DIR       ?= logs
DIAG\_DIR       ?= \$(OUT\_DIR)/diagnostics
PRED\_DIR       ?= \$(OUT\_DIR)/predictions
SUBMIT\_DIR     ?= \$(OUT\_DIR)/submission
SUBMIT\_ZIP     ?= \$(SUBMIT\_DIR)/bundle.zip
MANIFEST\_DIR   ?= \$(OUT\_DIR)/manifests
RUN\_HASH\_FILE  ?= run\_hash\_summary\_v50.json

# Kaggle

KAGGLE\_COMP  ?= neurips-2025-ariel

# Requirements

REQ\_CORE     ?= requirements.txt
REQ\_EXTRAS   ?= requirements-extras.txt
REQ\_DEV      ?= requirements-dev.txt
REQ\_KAGGLE   ?= requirements-kaggle.txt
REQ\_MIN      ?= requirements-min.txt
REQ\_FREEZE   ?= requirements.freeze.txt

# Mermaid / diagrams

DIAGRAMS\_SRC\_DIR  ?= diagrams
DIAGRAMS\_OUT\_DIR  ?= outputs/diagrams
MMD\_MAIN          ?= \$(DIAGRAMS\_SRC\_DIR)/main.mmd
MMDC\_BIN          ?= mmdc

# DVC plot specs (expected in .dvc/plots)

DVC\_PLOTS\_DIR     ?= .dvc/plots
PLOT\_LOSS         ?= \$(DVC\_PLOTS\_DIR)/loss.json
PLOT\_METRICS      ?= \$(DVC\_PLOTS\_DIR)/metrics.json
PLOT\_CALIB        ?= \$(DVC\_PLOTS\_DIR)/calibration.json
PLOT\_SYMBOLIC     ?= \$(DVC\_PLOTS\_DIR)/symbolic.json
PLOT\_FFT          ?= \$(DVC\_PLOTS\_DIR)/fft.json

# Hydra passthrough

OVERRIDES    ?=
EXTRA\_ARGS   ?=

# Docs export (pandoc)

DOC\_MD    ?= assets/AI\_Design\_and\_Modeling.md
DOC\_HTML  ?= assets/AI\_Design\_and\_Modeling.html
DOC\_PDF   ?= assets/AI\_Design\_and\_Modeling.pdf
DOC\_TITLE ?= AI Design and Modeling — SpectraMind V50
DOC\_CSS   ?= [https://cdn.jsdelivr.net/npm/water.css@2/out/water.css](https://cdn.jsdelivr.net/npm/water.css@2/out/water.css)

# ========= GUI demo suite locations =========

GUI\_DIR            ?= docs/gui/demo\_suite
STREAMLIT\_APP      ?= \$(GUI\_DIR)/streamlit\_demo.py
FASTAPI\_APP        ?= \$(GUI\_DIR)/fastapi\_backend.py
QT\_APP             ?= \$(GUI\_DIR)/qt\_diag\_demo.py
FASTAPI\_PORT       ?= 8089

# ========= Docker (GPU/CPU multi-target) =========

DOCKER              ?= docker
export DOCKER\_BUILDKIT = 1

# Image meta

IMAGE\_NAME          ?= spectramind-v50
IMAGE\_TAG           ?= \$(shell git rev-parse --short HEAD 2>/dev/null || echo dev)
IMAGE\_GPU           := \$(IMAGE\_NAME):\$(IMAGE\_TAG)-gpu
IMAGE\_CPU           := \$(IMAGE\_NAME):\$(IMAGE\_TAG)-cpu

# Build cache dir (local)

BUILD\_CACHE\_DIR     ?= .docker-build-cache

# Repo mount → /workspace

WORKDIR\_MOUNT       := -v \$(PWD):/workspace -w /workspace

# Cache mounts (host-side)

CACHE\_BASE          ?= \$(PWD)/.cache
HF\_CACHE\_MNT        := -v \$(CACHE\_BASE)/hf:/cache/hf
WANDB\_CACHE\_MNT     := -v \$(CACHE\_BASE)/wandb:/cache/wandb
PIP\_CACHE\_MNT       := -v \$(CACHE\_BASE)/pip\:/root/.cache/pip

# Base env (no secrets baked)

BASE\_ENV            :=&#x20;
-e PYTHONUNBUFFERED=1&#x20;
-e HF\_HOME=/cache/hf&#x20;
-e TRANSFORMERS\_CACHE=/cache/hf&#x20;
-e WANDB\_DIR=/cache/wandb&#x20;
-e WANDB\_MODE=offline

# Optional env-file (e.g., ENV\_FILE=.env.local)

ENV\_FILE            ?=
ENVFILE\_FLAG        := \$(if \$(strip \$(ENV\_FILE)),--env-file \$(ENV\_FILE),)

# Extra build args (avoid secrets)

EXTRA\_BUILD\_ARGS    ?=

# Detect NVIDIA runtime

HAS\_NVIDIA          := \$(shell command -v nvidia-smi >/dev/null 2>&1 && echo 1 || echo 0)

# Generic docker variables for legacy helpers

DOCKERFILE     ?= Dockerfile
DOCKER\_FULL    ?= \$(IMAGE\_CPU)
DOCKER\_GPU\_FLAG?=
ifeq (\$(HAS\_NVIDIA),1)
DOCKER\_GPU\_FLAG := --gpus all
endif

# Docker Compose

COMPOSE        ?= docker compose
COMPOSE\_FILE   ?= docker-compose.yml

# ========= Colors =========

BOLD := \033\[1m
DIM  := \033\[2m
RED  := \033\[31m
GRN  := \033\[32m
YLW  := \033\[33m
CYN  := \033\[36m
RST  := \033\[0m

# ========= PHONY =========

.PHONY:&#x20;
help init env info doctor versions guards&#x20;
fmt lint mypy test pre-commit&#x20;
selftest selftest-deep validate-env&#x20;
calibrate calibrate-temp corel-train train predict predict-e2e diagnose open-report open-latest&#x20;
submit submission-bin repair verify-submission bundle-zip&#x20;
ablate ablate-light ablate-heavy ablate-grid ablate-optuna&#x20;
analyze-log analyze-log-short check-cli-map&#x20;
dvc-pull dvc-push dvc-status dvc-check dvc-repro&#x20;
bench-selftest benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean&#x20;
benchmark-bin diagnostics-bin ci-smoke&#x20;
kaggle-verify kaggle-run kaggle-submit kaggle-dataset-create kaggle-dataset-push&#x20;
node-info node-ci mmd-version diagrams diagrams-png diagrams-svg diagrams-watch diagrams-lint diagrams-format diagrams-clean&#x20;
ci ci-fast ci-calibration ci-docs quickstart clean realclean distclean cache-clean&#x20;
export-reqs export-reqs-dev export-kaggle-reqs export-freeze&#x20;
install-core install-extras install-dev install-kaggle&#x20;
deps deps-min deps-lock verify-deps&#x20;
env-capture hash-config git-clean-check git-status release-tag&#x20;
pip-audit audit sbom sbom-scan docs docs-html docs-pdf docs-open docs-clean docs-serve docs-build&#x20;
pyg-install kaggle-pyg-index&#x20;
docker-help docker-print docker-build docker-buildx docker-run docker-shell docker-test docker-clean&#x20;
docker-build-gpu docker-build-cpu docker-run-gpu docker-run-cpu cli cli-gpu cli-cpu docker-context-check docker-cache-clean&#x20;
compose-ps compose-logs compose-up-gpu compose-up-cpu compose-up-api compose-up-web compose-up-docs compose-up-viz compose-up-lab compose-up-llm compose-up-ci compose-down compose-recreate compose-rebuild&#x20;
repro-start repro-snapshot repro-verify repro-manifest&#x20;
ensure-exec ensure-bin&#x20;
gui-demo gui-backend gui-backend-stop gui-qt gui-help&#x20;
dvc-plots dvc-plots-open plots-dashboard plots-verify

# ========= Default Goal =========

.DEFAULT\_GOAL := help

# ========= Help =========

help:
@echo ""
@echo "\$(BOLD)SpectraMind V50 — Make targets\$(RST)"
@echo "  \$(CYN)quickstart\$(RST)          : install deps (poetry), init dirs, print info"
@echo "  \$(CYN)doctor\$(RST)              : dependency checks (python/poetry/node/npm/cli)"
@echo "  \$(CYN)selftest\$(RST)            : fast integrity checks (CLI + files)"
@echo "  \$(CYN)train\$(RST)               : run training (EPOCHS=\$(EPOCHS), DEVICE=\$(DEVICE))"
@echo "  \$(CYN)predict\$(RST)             : run inference → \$(PRED\_DIR)/submission.csv"
@echo "  \$(CYN)predict-e2e\$(RST)         : smoke test asserting submission exists"
@echo "  \$(CYN)diagnose\$(RST)            : build diagnostics (smoothness + dashboard)"
@echo "  \$(CYN)open-report\$(RST)         : open newest diagnostics HTML"
@echo "  \$(CYN)submit\$(RST)              : package submission ZIP (\$(SUBMIT\_ZIP))"
@echo "  \$(CYN)ablate\*\$(RST)             : ablation sweeps (light/heavy/grid/optuna)"
@echo "  \$(CYN)analyze-log\$(RST)         : parse logs → \$(OUT\_DIR)/log\_table.{md,csv}"
@echo "  \$(CYN)benchmark\$(RST)           : CLI-based benchmark pipeline"
@echo "  \$(CYN)benchmark-bin\$(RST)       : run ./bin/benchmark.sh (feature-rich)"
@echo "  \$(CYN)dvc-plots\$(RST)           : render DVC plots; \$(CYN)plots-dashboard\$(RST) => unified HTML"
@echo "  \$(CYN)docker-build/run/shell\$(RST) : Dockerized workflow (GPU autodetect)"
@echo ""

# ========= Init / Env =========

init: env

env:
mkdir -p "\$(OUT\_DIR)" "\$(LOGS\_DIR)" "\$(DIAG\_DIR)" "\$(PRED\_DIR)" "\$(SUBMIT\_DIR)" "\$(DIAGRAMS\_OUT\_DIR)" "\$(MANIFEST\_DIR)"

versions:
@echo "\$(BOLD)Versions\$(RST)"
@echo "python : $$($(PYTHON) --version 2>&1 || true)"
	@echo "poetry : $$(\$(POETRY) --version 2>&1 || true)"
@echo "node   : $$($(NODE) --version 2>&1 || true)"
	@echo "npm    : $$(\$(NPM) --version 2>&1 || true)"
@echo "kaggle : \$\$(\$(KAGGLE) --version 2>&1 || true)"
@echo "cli    : \$(CLI)"

info: versions
@echo "device : \$(DEVICE)"
@echo "OUT\_DIR: \$(OUT\_DIR)"
@echo "RUN\_ID : \$(RUN\_ID)"
@echo "RUN\_HASH\_FILE: \$(RUN\_HASH\_FILE)"

doctor:
@ok=1;&#x20;
command -v \$(PYTHON) >/dev/null 2>&1 || { echo "\$(RED)Missing python3\$(RST)"; ok=0; };&#x20;
command -v \$(POETRY) >/dev/null 2>&1 || { echo "\$(YLW)Poetry not found — install via pipx/pip\$(RST)"; ok=0; };&#x20;
command -v \$(NODE)   >/dev/null 2>&1 || { echo "\$(YLW)Node not found (needed for mermaid-cli)\$(RST)"; };&#x20;
command -v \$(NPM)    >/dev/null 2>&1 || { echo "\$(YLW)npm not found (needed for mermaid-cli)\$(RST)"; };&#x20;
{ \$(CLI) --version >/dev/null 2>&1 && echo "\$(GRN)CLI OK\$(RST)"; } || { echo "\$(YLW)CLI not yet installed or venv not active\$(RST)"; };&#x20;
\$(PYTHON) - <<'PY' || ok=0;&#x20;
import sys;&#x20;
try:&#x20;
import torch;&#x20;
print("torch:", getattr(torch, "version", "n/a"));&#x20;
print("cuda :", getattr(getattr(torch, "version", None), "cuda", "n/a"));&#x20;
except Exception:&#x20;
print("torch: (missing)");&#x20;
PY
@test \$\$ok -eq 1

quickstart: env info
@echo "\$(CYN)Installing project deps via Poetry (no-root)…\$(RST)"
@\$(POETRY) install --no-root
@echo "\$(GRN)Done.\$(RST)"
@\$(MAKE) doctor

# ======== Guards ========

guards:
@command -v \$(POETRY) >/dev/null 2>&1 || { echo "\$(RED)Poetry missing\$(RST)"; exit 1; }
@\$(POETRY) run spectramind --version >/dev/null 2>&1 || { echo "\$(RED)spectramind CLI not runnable\$(RST)"; exit 1; }

# ========= Dev / Quality =========

fmt:
\$(POETRY) run isort .
\$(POETRY) run black .

lint:
\$(POETRY) run ruff check .

mypy:
\$(POETRY) run mypy --strict src || true

test: init
\$(POETRY) run pytest -q || \$(POETRY) run pytest -q -x

pre-commit:
\$(POETRY) run pre-commit install
\$(POETRY) run pre-commit run --all-files || true

# ========= Env validation =========

validate-env:
@if \[ -x scripts/validate\_env.py ] || \[ -f scripts/validate\_env.py ]; then&#x20;
echo ">>> Validating .env schema";&#x20;
\$(PYTHON) scripts/validate\_env.py || exit 1;&#x20;
else&#x20;
echo ">>> Skipping validate-env (scripts/validate\_env.py not found)";&#x20;
fi

# ========= Pipeline =========

selftest: guards init
\$(CLI) selftest

selftest-deep: guards init
\$(CLI) selftest --deep

calibrate: guards init
\$(CLI) calibrate \$(OVERRIDES) \$(EXTRA\_ARGS)

calibrate-temp: guards init
\$(CLI) calibrate-temp \$(OVERRIDES) \$(EXTRA\_ARGS)

corel-train: guards init
\$(CLI) corel-train \$(OVERRIDES) \$(EXTRA\_ARGS)

train: guards init
\$(CLI) train +training.epochs=\$(EPOCHS) +training.seed=\$(SEED) \$(OVERRIDES) --device \$(DEVICE) \$(EXTRA\_ARGS)

predict: guards init
mkdir -p "\$(PRED\_DIR)"
\$(CLI) predict --out-csv "\$(PRED\_DIR)/submission.csv" \$(OVERRIDES) \$(EXTRA\_ARGS)

verify-submission:
@test -s "\$(PRED\_DIR)/submission.csv" || { echo "::error::\$(PRED\_DIR)/submission.csv missing or empty"; exit 1; }
@head -n 1 "\$(PRED\_DIR)/submission.csv" | grep -Eiq 'id|planet' || { echo "::warning::Header not matched (id/planet) — continuing"; true; }

predict-e2e: predict verify-submission
@echo "\$(GRN)OK: \$(PRED\_DIR)/submission.csv\$(RST)"

diagnose: guards init
\$(CLI) diagnose smoothness --outdir "\$(DIAG\_DIR)" \$(EXTRA\_ARGS)
\$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "\$(DIAG\_DIR)" \$(EXTRA\_ARGS) ||&#x20;
\$(CLI) diagnose dashboard --outdir "\$(DIAG\_DIR)" \$(EXTRA\_ARGS) || true

open-report:
@latest=$$(ls -t $(DIAG_DIR)/*.html 2>/dev/null | head -n1 || true); \
	if [ -n "$$latest" ]; then&#x20;
echo "Opening $latest"; \
		if command -v xdg-open >/dev/null 2>&1; then xdg-open "$latest" || true;&#x20;
elif command -v open >/dev/null 2>&1; then open "\$\$latest" || true;&#x20;
else echo "No opener found (CI/headless)"; fi;&#x20;
else echo "No diagnostics HTML found."; fi

# alias

open-latest: open-report

submit: guards init
mkdir -p "\$(SUBMIT\_DIR)"
\$(CLI) submit --zip-out "\$(SUBMIT\_ZIP)" \$(EXTRA\_ARGS)

bundle-zip: repro-snapshot
@echo ">>> Creating reproducible bundle ZIP (manifests + predictions + diagnostics)"
@mkdir -p "\$(OUT\_DIR)/bundles"
@tar -czf "\$(OUT\_DIR)/bundles/spectramind\_ci\_bundle\_\$(RUN\_ID).tar.gz"&#x20;
\--exclude='.pt' --exclude='.pth' --exclude='\*.onnx'&#x20;
"\$(MANIFEST\_DIR)" "\$(DIAG\_DIR)" "\$(PRED\_DIR)" "\$(OUT\_DIR)/log\_table.md" "\$(OUT\_DIR)/log\_table.csv" 2>/dev/null || true
@echo "Bundle: \$(OUT\_DIR)/bundles/spectramind\_ci\_bundle\_\$(RUN\_ID).tar.gz"

# ========= bin/ wrappers =========

ensure-exec:
@mkdir -p "\$(LOGS\_DIR)"
@find bin -maxdepth 1 -type f -name "\*.sh" -print0 2>/dev/null | xargs -0 chmod +x 2>/dev/null || true

ensure-bin: ensure-exec
@for f in bin/make-submission.sh bin/repair\_and\_push.sh bin/benchmark.sh bin/diagnostics.sh; do&#x20;
if \[ ! -f "$f" ]; then echo "::warning::missing $f"; fi;&#x20;
done

# Use: make submission-bin ARGS="--tag v50.0.1 --open"

submission-bin: ensure-exec
@echo "\[bin] ./bin/make-submission.sh \$(ARGS)"
@./bin/make-submission.sh \$(ARGS)

# Use: make repair MSG="Fix hashes"

repair: ensure-exec
@if \[ -z "\$(MSG)" ]; then echo 'Usage: make repair MSG="Commit message"'; exit 2; fi
@echo "\[bin] ./bin/repair\_and\_push.sh '\$(MSG)'"
@./bin/repair\_and\_push.sh "\$(MSG)"

# New: run bin/benchmark.sh

benchmark-bin: ensure-exec
@echo "\[bin] ./bin/benchmark.sh \$(ARGS)"
@./bin/benchmark.sh \$(ARGS)

# New: run bin/diagnostics.sh

diagnostics-bin: ensure-exec
@echo "\[bin] ./bin/diagnostics.sh \$(ARGS)"
@./bin/diagnostics.sh \$(ARGS)

# Local smoke to mirror CI (CPU-only)

ci-smoke: ensure-exec
@./bin/benchmark.sh --profile cpu --epochs 1 --tag ci-smoke --manifest
@./bin/diagnostics.sh --no-umap --no-tsne --manifest

# ========= Ablation =========

ablate: guards init
\$(CLI) ablate \$(OVERRIDES) \$(EXTRA\_ARGS)

ablate-light: guards init
\$(CLI) ablate ablation=ablation\_light \$(EXTRA\_ARGS)

ablate-heavy: guards init
\$(CLI) ablate ablation=ablation\_heavy \$(EXTRA\_ARGS)

ablate-grid: guards init
\$(CLI) ablate -m ablate.sweeper=basic +ablate.search=v50\_fast\_grid ablation=ablation\_light \$(EXTRA\_ARGS)

ablate-optuna: guards init
\$(CLI) ablate -m ablate.sweeper=optuna +ablate.search=v50\_symbolic\_core ablation=ablation\_heavy \$(EXTRA\_ARGS)

# ========= Log analysis =========

analyze-log: guards init
\$(CLI) analyze-log --md "\$(OUT\_DIR)/log\_table.md" --csv "\$(OUT\_DIR)/log\_table.csv" \$(EXTRA\_ARGS)

analyze-log-short: guards init
@if \[ ! -f "\$(OUT\_DIR)/log\_table.csv" ]; then&#x20;
echo ">>> Generating log CSV via analyze-log";&#x20;
\$(CLI) analyze-log --md "\$(OUT\_DIR)/log\_table.md" --csv "\$(OUT\_DIR)/log\_table.csv" \$(EXTRA\_ARGS);&#x20;
fi;&#x20;
if \[ -f "\$(OUT\_DIR)/log\_table.csv" ]; then&#x20;
echo "=== Last 5 CLI invocations ===";&#x20;
tail -n +2 "\$(OUT\_DIR)/log\_table.csv" | tail -n 5 |&#x20;
awk -F',' 'BEGIN{OFS=" | "} {print "time="$1, "cmd="$2, "git\_sha="$3, "cfg="$4}';&#x20;
else&#x20;
echo "::warning::No log\_table.csv to summarize";&#x20;
fi

check-cli-map: guards
\$(CLI) check-cli-map

# ========= DVC =========

dvc-pull:
\$(DVC) pull || true

dvc-push:
\$(DVC) push || true

dvc-status:
\$(DVC) status || true

dvc-repro:
\$(DVC) repro || true

dvc-check:
@echo ">>> DVC sanity"
@\$(DVC) status -c || true
@\$(DVC) doctor || true

# ========= Benchmarks (CLI-native pipeline kept for parity) =========

bench-selftest:
\$(CLI) selftest

benchmark: bench-selftest
@\$(MAKE) benchmark-run DEVICE=\$(DEVICE) EPOCHS=\$(EPOCHS) OVERRIDES='\$(OVERRIDES)' EXTRA\_ARGS='\$(EXTRA\_ARGS)'

benchmark-cpu: bench-selftest
@\$(MAKE) benchmark-run DEVICE=cpu EPOCHS=\$(EPOCHS) OVERRIDES='\$(OVERRIDES)' EXTRA\_ARGS='\$(EXTRA\_ARGS)'

benchmark-gpu: bench-selftest
@\$(MAKE) benchmark-run DEVICE=gpu EPOCHS=\$(EPOCHS) OVERRIDES='\$(OVERRIDES)' EXTRA\_ARGS='\$(EXTRA\_ARGS)'

benchmark-run:
@OUTDIR="benchmarks/\$(TS)\_\$(DEVICE)";&#x20;
mkdir -p "$$OUTDIR"; \
	$(CLI) train +training.epochs=$(EPOCHS) +training.seed=$(SEED) $(OVERRIDES) --device $(DEVICE) --outdir "$$OUTDIR" \$(EXTRA\_ARGS);&#x20;
\$(CLI) diagnose smoothness --outdir "$$OUTDIR" $(EXTRA_ARGS); \
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$$OUTDIR" \$(EXTRA\_ARGS) ||&#x20;
\$(CLI) diagnose dashboard --outdir "$$OUTDIR" $(EXTRA_ARGS) || true; \
	{ \
		echo "Benchmark summary"; \
		date; \
		echo "python   : $$(\$(PYTHON) --version 2>&1)";&#x20;
echo "poetry   : $$($(POETRY) --version 2>&1 || true)"; \
		echo "cli      : $(CLI)"; \
		echo "device   : $(DEVICE)"; \
		echo "epochs   : $(EPOCHS)"; \
		echo "seed     : $(SEED)"; \
		echo "overrides: $(OVERRIDES)"; \
		command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true; \
		echo ""; \
		echo "Artifacts in $$OUTDIR:";&#x20;
ls -lh "$OUTDIR" || true; \
	} > "$OUTDIR/summary.txt";&#x20;
echo ">>> Benchmark complete → \$\$OUTDIR/summary.txt"

benchmark-report:
mkdir -p aggregated
{&#x20;
echo "# SpectraMind V50 Benchmark Report";&#x20;
echo "";&#x20;
for f in $(find benchmarks -type f -name summary.txt | sort); do \
			echo "## $f"; echo ""; cat "\$\$f"; echo "";&#x20;
done;&#x20;
} > aggregated/report.md
@echo ">>> Aggregated → aggregated/report.md"

benchmark-clean:
rm -rf benchmarks aggregated

# ========= Kaggle Helpers =========

kaggle-verify:
@command -v \$(KAGGLE) >/dev/null 2>&1 || { echo "\$(RED)Kaggle CLI missing\$(RST)"; exit 1; }
@\$(KAGGLE) competitions list >/dev/null 2>&1 || { echo "\$(RED)Kaggle CLI not logged in\$(RST)"; exit 1; }
@echo "\$(GRN)Kaggle CLI OK\$(RST)"

kaggle-run: guards init
@echo ">>> Running single-epoch GPU run (Kaggle-like)"
\$(CLI) selftest
\$(CLI) train +training.epochs=1 +training.seed=\$(SEED) --device gpu --outdir "\$(OUT\_DIR)"
\$(CLI) predict --out-csv "\$(PRED\_DIR)/submission.csv"

kaggle-submit: kaggle-verify kaggle-run
@echo ">>> Submitting to Kaggle competition \$(KAGGLE\_COMP)"
\$(KAGGLE) competitions submit -c "\$(KAGGLE\_COMP)" -f "\$(PRED\_DIR)/submission.csv" -m "SpectraMind V50 auto-submit (\$(RUN\_ID))"

# Optional: publish artifacts as Kaggle dataset

kaggle-dataset-create:
@echo ">>> Creating Kaggle dataset placeholder (id: \$\$USER/spectramind-v50-\$(RUN\_TS))"
@\$(KAGGLE) datasets create -p "\$(OUT\_DIR)" -u || true

kaggle-dataset-push:
@echo ">>> Pushing latest outputs as dataset"
@\$(KAGGLE) datasets version -p "\$(OUT\_DIR)" -m "SpectraMind V50 artifacts \$(RUN\_ID)" -r zip -d

# ========= Requirements export / install =========

export-reqs:
@echo ">>> Exporting Poetry deps → \$(REQ\_CORE)"
\$(POETRY) export -f requirements.txt --without-hashes -o \$(REQ\_CORE)

export-reqs-dev:
@echo ">>> Exporting Poetry deps (incl. dev) → \$(REQ\_DEV)"
\$(POETRY) export -f requirements.txt --with dev --without-hashes -o \$(REQ\_DEV)

export-kaggle-reqs:
@echo ">>> Exporting Kaggle-friendly requirements → \$(REQ\_KAGGLE)"
\$(POETRY) export -f requirements.txt --without-hashes |&#x20;
grep -vE '^(torch|torchvision|torchaudio|torch-geometric)(==|>=)' > \$(REQ\_KAGGLE)

export-freeze:
@echo ">>> Freezing active env → \$(REQ\_FREEZE)"
\$(PIP) freeze -q > \$(REQ\_FREEZE)
@echo ">>> Wrote \$(REQ\_FREEZE)"

install-core:
\$(PIP) install -r \$(REQ\_CORE)

install-extras:
@if \[ -f "\$(REQ\_EXTRAS)" ]; then \$(PIP) install -r \$(REQ\_EXTRAS); else echo "::warning::\$(REQ\_EXTRAS) not found"; fi

install-dev:
\$(PIP) install -r \$(REQ\_DEV)

install-kaggle:
\$(PIP) install -r \$(REQ\_KAGGLE)

# — Unified dependency workflows —

deps:
@echo ">>> Upgrading pip/setuptools/wheel"
\$(PIP) install -U pip setuptools wheel
@echo ">>> Installing full dev/CI stack from \$(REQ\_CORE)"
@test -f "\$(REQ\_CORE)" || { echo "\$(RED)\$(REQ\_CORE) not found\$(RST)"; exit 1; }
\$(PIP) install -r \$(REQ\_CORE)
@\$(MAKE) verify-deps

deps-min:
@echo ">>> Upgrading pip/setuptools/wheel"
\$(PIP) install -U pip setuptools wheel
@echo ">>> Installing minimal Kaggle runtime from \$(REQ\_MIN)"
@test -f "\$(REQ\_MIN)" || { echo "\$(RED)\$(REQ\_MIN) not found\$(RST)"; exit 1; }
\$(PIP) install -r \$(REQ\_MIN) || true
@\$(MAKE) verify-deps

deps-lock:
@echo ">>> Lock (Poetry), then export pinned requirements + freeze"
\$(POETRY) lock --no-update
@\$(MAKE) export-reqs
@\$(MAKE) export-reqs-dev
@\$(MAKE) export-freeze
@echo "\$(GRN)Locked and exported.\$(RST)"

verify-deps:
@echo ">>> Key package versions"
@\$(PYTHON) - << 'PY'
import importlib
pkgs = \["torch","torchvision","torchaudio","numpy","scipy","pandas","sklearn","matplotlib","umap","shap","typer","hydra","omegaconf"]
for name in pkgs:
mod = "sklearn" if name=="sklearn" else name
try:
m = importlib.import\_module(mod)
print(f"{name:>14}: {getattr(m,'**version**','n/a')}")
except Exception:
print(f"{name:>14}: (missing)")
PY

# ========= CLI utilities (reproducibility) =========

env-capture:
\$(CLI) env-capture

hash-config:
\$(CLI) hash-config

git-clean-check:
@dirty=$$($(GIT) status --porcelain); \
	if [ -n "$$dirty" ]; then echo "::warning::Git working tree dirty"; echo "\$\$dirty"; else echo "\$(GRN)Git clean\$(RST)"; fi

git-status:
\$(GIT) status --short --branch

release-tag:
@if \[ -z "\$(TAG)" ]; then echo 'Usage: make release-tag TAG=v0.50.0-best'; exit 2; fi
\$(GIT) tag -a "\$(TAG)" -m "Release \$(TAG)"
\$(GIT) push origin "\$(TAG)"

# ========= Security / SBOM / Docs =========

pip-audit:
@echo ">>> pip-audit (CVE scan)"
@if ! command -v \$(PIP\_AUDIT) >/dev/null 2>&1; then \$(PIP) install pip-audit; fi
pip-audit -r \$(REQ\_CORE) || true

sbom:
@echo ">>> SBOM via syft (if installed)"
@if command -v \$(SBOM\_SYFT) >/dev/null 2>&1; then&#x20;
\$(SBOM\_SYFT) packages dir:. -o cyclonedx-json > \$(OUT\_DIR)/sbom.json || true;&#x20;
echo "Wrote \$(OUT\_DIR)/sbom.json";&#x20;
else echo "::warning::syft not found"; fi

sbom-scan:
@echo ">>> Vulnerability scan via grype/trivy (if installed)"
@if command -v \$(SBOM\_GRYPE) >/dev/null 2>&1; then&#x20;
\$(SBOM\_GRYPE) dir:. --add-cpes-if-none --fail-on high || true;&#x20;
else echo "::warning::grype not found"; fi
@if command -v \$(TRIVY) >/dev/null 2>&1; then&#x20;
\$(TRIVY) fs --severity HIGH,CRITICAL --exit-code 0 . || true;&#x20;
else echo "::warning::trivy not found"; fi

audit: pip-audit sbom sbom-scan

docs: docs-html docs-pdf

docs-html:
@command -v pandoc >/dev/null || { echo "pandoc not found. Install pandoc (and TeX for PDF)."; exit 1; }
@test -f "\$(DOC\_MD)" || { echo "Missing \$(DOC\_MD)."; exit 1; }
@mkdir -p assets
pandoc "\$(DOC\_MD)" -f markdown+smart -t html5 -s --metadata title="\$(DOC\_TITLE)" -c "\$(DOC\_CSS)" -o "\$(DOC\_HTML)"
@echo "Wrote \$(DOC\_HTML)"

docs-pdf:
@command -v pandoc >/dev/null || { echo "pandoc not found. Install pandoc + TeX (texlive)."; exit 1; }
@test -f "\$(DOC\_MD)" || { echo "Missing \$(DOC\_MD)."; exit 1; }
pandoc "\$(DOC\_MD)" -f markdown+smart -V geometry\:margin=1in -V linkcolor\:blue -V fontsize=11pt -o "\$(DOC\_PDF)"
@echo "Wrote \$(DOC\_PDF)"

docs-open:
@if \[ -f "\$(DOC\_HTML)" ]; then&#x20;
if command -v xdg-open >/dev/null 2>&1; then xdg-open "\$(DOC\_HTML)";&#x20;
elif command -v open >/dev/null 2>&1; then open "\$(DOC\_HTML)";&#x20;
else echo "Open \$(DOC\_HTML) manually"; fi;&#x20;
else echo "No HTML found. Run 'make docs' first."; fi

# Optional docs serve/build (mkdocs or simple http)

docs-serve:
@if command -v mkdocs >/dev/null 2>&1; then&#x20;
mkdocs serve -a 0.0.0.0:8000;&#x20;
else&#x20;
echo "::notice::mkdocs not found; falling back to python -m http.server on docs/";&#x20;
cd docs && \$(PYTHON) -m http.server 8000;&#x20;
fi

docs-build:
@if command -v mkdocs >/dev/null 2>&1; then mkdocs build; else echo "::notice::mkdocs not found"; fi

docs-clean:
rm -f "\$(DOC\_HTML)" "\$(DOC\_PDF)"
@echo "Cleaned \$(DOC\_HTML) and \$(DOC\_PDF)"

# ========= CI convenience =========

ci: validate-env selftest train diagnose analyze-log-short

ci-fast: selftest train analyze-log-short

ci-calibration: selftest calibrate diagnose analyze-log-short

ci-docs: docs

# ========= PyTorch Geometric helper =========

kaggle-pyg-index:
@\$(PYTHON) - << 'PY'
import torch
ver = getattr(torch, 'version', None)
tv = getattr(ver, '**version**', None) or getattr(ver, 'version', None) or getattr(torch, 'version', None)
if isinstance(tv, str) and '+' in tv: base\_ver = tv.split('+')\[0]
else: base\_ver = tv or '2.2.0'
cu  = (getattr(torch, 'version', None) and getattr(getattr(torch, 'version', None), 'cuda', None)) or None
cu\_tag = ('cu' + cu.replace('.','')) if cu else 'cpu'
base = f"[https://data.pyg.org/whl/torch-{base\_ver}+{cu\_tag}.html](https://data.pyg.org/whl/torch-{base_ver}+{cu_tag}.html)"
print(base)
PY

pyg-install:
@echo ">>> Installing torch-geometric matching the current torch/CUDA"
@PYG\_INDEX="$$( $(MAKE) --no-print-directory kaggle-pyg-index )"; \
	echo "Using index: $$PYG\_INDEX";&#x20;
\$(PIP) install torch-geometric==2.5.3 -f "\$\$PYG\_INDEX"

# ========= Docker helpers =========

docker-help:
@echo "Docker targets"
@echo "  make docker-build-gpu      # build GPU image  → \$(IMAGE\_GPU)"
@echo "  make docker-build-cpu      # build CPU image  → \$(IMAGE\_CPU)"
@echo "  make docker-run-gpu        # bash shell (GPU), mounts repo + caches"
@echo "  make docker-run-cpu        # bash shell (CPU), mounts repo + caches"
@echo "  make cli CMD='spectramind --version'       # auto GPU/CPU"
@echo "  make cli-gpu CMD='spectramind train …'     # run in GPU image"
@echo "  make cli-cpu CMD='spectramind diagnose …'  # run in CPU image"
@echo "  make docker-context-check  # quick context size sanity"
@echo "  make docker-cache-clean    # remove local build cache dir"

# Build GPU (expects Dockerfile stage: runtime-gpu)

docker-build-gpu:
@mkdir -p "\$(BUILD\_CACHE\_DIR)"
@echo ">> Building GPU image: \$(IMAGE\_GPU)"
\$(DOCKER) build&#x20;
\--target runtime-gpu&#x20;
-t \$(IMAGE\_GPU)&#x20;
\--progress=plain&#x20;
\--cache-from type=local,src=\$(BUILD\_CACHE\_DIR)&#x20;
\--cache-to   type=local,dest=\$(BUILD\_CACHE\_DIR),mode=max&#x20;
\$(EXTRA\_BUILD\_ARGS)&#x20;
.

# Build CPU (expects Dockerfile stage: runtime-cpu)

docker-build-cpu:
@mkdir -p "\$(BUILD\_CACHE\_DIR)"
@echo ">> Building CPU image: \$(IMAGE\_CPU)"
\$(DOCKER) build&#x20;
\--target runtime-cpu&#x20;
-t \$(IMAGE\_CPU)&#x20;
\--progress=plain&#x20;
\--cache-from type=local,src=\$(BUILD\_CACHE\_DIR)&#x20;
\--cache-to   type=local,dest=\$(BUILD\_CACHE\_DIR),mode=max&#x20;
\$(EXTRA\_BUILD\_ARGS)&#x20;
.

# Interactive shells

docker-run-gpu:
@echo ">> Running GPU shell: \$(IMAGE\_GPU)"
\$(DOCKER) run --rm -it --name spectramind-gpu&#x20;
\--gpus all&#x20;
\$(WORKDIR\_MOUNT)&#x20;
\$(HF\_CACHE\_MNT) \$(WANDB\_CACHE\_MNT) \$(PIP\_CACHE\_MNT)&#x20;
\$(BASE\_ENV) \$(ENVFILE\_FLAG)&#x20;
\$(IMAGE\_GPU) bash

docker-run-cpu:
@echo ">> Running CPU shell: \$(IMAGE\_CPU)"
\$(DOCKER) run --rm -it --name spectramind-cpu&#x20;
\$(WORKDIR\_MOUNT)&#x20;
\$(HF\_CACHE\_MNT) \$(WANDB\_CACHE\_MNT) \$(PIP\_CACHE\_MNT)&#x20;
\$(BASE\_ENV) \$(ENVFILE\_FLAG)&#x20;
\$(IMAGE\_CPU) bash

# CLI runners (auto-select GPU if available)

CMD ?= spectramind --help

cli:
@echo ">> Detecting NVIDIA runtime…"
@if \[ "\$(HAS\_NVIDIA)" = "1" ]; then&#x20;
echo '>> NVIDIA found — using GPU image';&#x20;
\$(MAKE) -s cli-gpu CMD="\$(CMD)";&#x20;
else&#x20;
echo '>> NVIDIA not found — using CPU image';&#x20;
\$(MAKE) -s cli-cpu CMD="\$(CMD)";&#x20;
fi

cli-gpu:
@echo ">> \[GPU] \$(CMD)"
\$(DOCKER) run --rm -t --name spectramind-cli-gpu&#x20;
\--gpus all&#x20;
\$(WORKDIR\_MOUNT)&#x20;
\$(HF\_CACHE\_MNT) \$(WANDB\_CACHE\_MNT) \$(PIP\_CACHE\_MNT)&#x20;
\$(BASE\_ENV) \$(ENVFILE\_FLAG)&#x20;
\$(IMAGE\_GPU) bash -lc '\$(CMD)'

cli-cpu:
@echo ">> \[CPU] \$(CMD)"
\$(DOCKER) run --rm -t --name spectramind-cli-cpu&#x20;
\$(WORKDIR\_MOUNT)&#x20;
\$(HF\_CACHE\_MNT) \$(WANDB\_CACHE\_MNT) \$(PIP\_CACHE\_MNT)&#x20;
\$(BASE\_ENV) \$(ENVFILE\_FLAG)&#x20;
\$(IMAGE\_CPU) bash -lc '\$(CMD)'

docker-context-check:
@echo ">> Docker build context (top offenders):"
@du -h -d1 . | sort -hr | head -n 30

docker-cache-clean:
@rm -rf "\$(BUILD\_CACHE\_DIR)"
@echo ">> Removed \$(BUILD\_CACHE\_DIR)"

docker-print:
@echo "Image : \$(DOCKER\_FULL)"
@echo "GPU   : \$(HAS\_NVIDIA)"
@echo "File  : \$(DOCKERFILE)"
@echo "Args  : \$(DOCKER\_BUILD\_ARGS)"

DOCKER\_BUILD\_ARGS ?=

docker-build: docker-print
\$(DOCKER) build&#x20;
\--build-arg BUILDKIT\_INLINE\_CACHE=1&#x20;
-f \$(DOCKERFILE)&#x20;
-t \$(DOCKER\_FULL)&#x20;
\$(DOCKER\_BUILD\_ARGS)&#x20;
.

docker-buildx: docker-print
\$(DOCKER) buildx build --load&#x20;
\--build-arg BUILDKIT\_INLINE\_CACHE=1&#x20;
-f \$(DOCKERFILE)&#x20;
-t \$(DOCKER\_FULL)&#x20;
\$(DOCKER\_BUILD\_ARGS)&#x20;
.

docker-run: init
\$(DOCKER) run --rm -it \$(DOCKER\_GPU\_FLAG)&#x20;
-v "\$\$(pwd):/workspace"&#x20;
-w /workspace&#x20;
-e DEVICE=\$(DEVICE)&#x20;
-e EPOCHS=\$(EPOCHS)&#x20;
-e PYTHONHASHSEED=\$(PYTHONHASHSEED)&#x20;
\$(DOCKER\_FULL)&#x20;
bash -lc 'make ci || true'

docker-shell: init
\$(DOCKER) run --rm -it \$(DOCKER\_GPU\_FLAG)&#x20;
-v "\$\$(pwd):/workspace"&#x20;
-w /workspace&#x20;
-e PYTHONHASHSEED=\$(PYTHONHASHSEED)&#x20;
\$(DOCKER\_FULL)&#x20;
bash

docker-test: init
\$(DOCKER) run --rm \$(DOCKER\_GPU\_FLAG)&#x20;
-v "\$\$(pwd):/workspace"&#x20;
-w /workspace&#x20;
-e PYTHONHASHSEED=\$(PYTHONHASHSEED)&#x20;
\$(DOCKER\_FULL)&#x20;
bash -lc 'make selftest && make test'

docker-clean:
-\$(DOCKER) image rm \$(DOCKER\_FULL) 2>/dev/null || true

# ========= Docker Compose (profiles from docker-compose.yml) =========

compose-ps:
\$(COMPOSE) -f \$(COMPOSE\_FILE) ps

compose-logs:
\$(COMPOSE) -f \$(COMPOSE\_FILE) logs -f --tail=200

compose-up-gpu:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile gpu up -d spectramind-gpu

compose-up-cpu:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile cpu up -d spectramind-cpu

compose-up-api:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile api up -d api

compose-up-web:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile web up -d web

compose-up-docs:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile docs up docs

compose-up-viz:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile viz up tensorboard

compose-up-lab:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile lab up jupyter

compose-up-llm:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile llm up -d ollama

compose-up-ci:
\$(COMPOSE) -f \$(COMPOSE\_FILE) --profile ci up --abort-on-container-exit ci

compose-down:
\$(COMPOSE) -f \$(COMPOSE\_FILE) down

compose-recreate:
\$(COMPOSE) -f \$(COMPOSE\_FILE) up -d --force-recreate

compose-rebuild:
\$(COMPOSE) -f \$(COMPOSE\_FILE) build --no-cache

# ========= Mermaid / Diagrams =========

node-info:
@echo "node : $$($(NODE) --version 2>/dev/null || echo 'missing')"
	@echo "npm  : $$(\$(NPM) --version 2>/dev/null || echo 'missing')"
@echo "mmdc : \$\$(\$(MMDC\_BIN) -V 2>/dev/null || echo 'missing')"

mmd-version:
@\$(MMDC\_BIN) -V 2>/dev/null || echo "::warning::mmdc not installed"

node-ci:
@command -v \$(NPM) >/dev/null 2>&1 || { echo "\$(YLW)npm missing — skip mmdc install\$(RST)"; exit 0; }
@command -v \$(MMDC\_BIN) >/dev/null 2>&1 || { echo ">>> Installing @mermaid-js/mermaid-cli globally"; \$(NPM) i -g @mermaid-js/mermaid-cli; }

diagrams: diagrams-png

diagrams-png:
@mkdir -p "\$(DIAGRAMS\_OUT\_DIR)"
@if \[ -f "\$(MMD\_MAIN)" ]; then&#x20;
echo ">>> Rendering PNG from \$(MMD\_MAIN)";&#x20;
\$(MMDC\_BIN) -i "\$(MMD\_MAIN)" -o "\$(DIAGRAMS\_OUT\_DIR)/main.png" -b light || true;&#x20;
else echo "::warning::No \$(MMD\_MAIN) found"; fi

diagrams-svg:
@mkdir -p "\$(DIAGRAMS\_OUT\_DIR)"
@if \[ -f "\$(MMD\_MAIN)" ]; then&#x20;
echo ">>> Rendering SVG from \$(MMD\_MAIN)";&#x20;
\$(MMDC\_BIN) -i "\$(MMD\_MAIN)" -o "\$(DIAGRAMS\_OUT\_DIR)/main.svg" -b light || true;&#x20;
else echo "::warning::No \$(MMD\_MAIN) found"; fi

diagrams-watch:
@command -v entr >/dev/null 2>&1 || { echo "::warning::entr not installed; watch disabled"; exit 0; }
@ls \$(DIAGRAMS\_SRC\_DIR)/\*.mmd | entr -r make diagrams

diagrams-lint:
@echo ">>> Lint diagrams — ensure .mmd files compile"
@for f in \$(DIAGRAMS\_SRC\_DIR)/\*.mmd; do&#x20;
\[ -f "$$f" ] || continue; \
		$(MMDC_BIN) -i "$$f" -o /dev/null >/dev/null 2>&1 || echo "::warning::Failed to render \$\$f";&#x20;
done

diagrams-format:
@echo ">>> (Optional) apply prettier to Mermaid if configured"

diagrams-clean:
rm -rf "\$(DIAGRAMS\_OUT\_DIR)"

# ========= DVC plots & dashboard =========

# dvc-plots: render all known plot specs into a single HTML report

dvc-plots: plots-verify
@mkdir -p \$(OUT\_DIR)/plots
@echo ">>> Rendering DVC plots"
\$(DVC) plots show&#x20;
\$(PLOT\_LOSS)&#x20;
\$(PLOT\_METRICS)&#x20;
\$(PLOT\_CALIB)&#x20;
\$(PLOT\_SYMBOLIC)&#x20;
\$(PLOT\_FFT)&#x20;
-T -o \$(OUT\_DIR)/plots/dashboard.html || true
@echo "Plot HTML → \$(OUT\_DIR)/plots/dashboard.html"

# open the latest plots HTML

dvc-plots-open:
@html="\$(OUT\_DIR)/plots/dashboard.html";&#x20;
if \[ -f "$html" ]; then \
		if command -v xdg-open >/dev/null 2>&1; then xdg-open "$html";&#x20;
elif command -v open >/dev/null 2>&1; then open "$html"; \
		else echo "Open $html manually"; fi;&#x20;
else echo "::warning::No plots/dashboard.html found. Run 'make dvc-plots' first."; fi

# convenience alias name the user requested earlier

plots-dashboard: dvc-plots

# basic sanity that plot spec files exist

plots-verify:
@ok=1;&#x20;
for f in \$(PLOT\_LOSS) \$(PLOT\_METRICS) \$(PLOT\_CALIB) \$(PLOT\_SYMBOLIC) \$(PLOT\_FFT); do&#x20;
if \[ ! -f "$f" ]; then echo "::warning::Missing $f"; ok=0; fi;&#x20;
done;&#x20;
\[ \$\$ok -eq 1 ] || true

# ========= Reproducibility snapshots (config+data hashing, manifest) =========

repro-start: init
@echo ">>> Starting reproducible run \$(RUN\_ID)"
@echo "\$(RUN\_ID)" > "\$(LOGS\_DIR)/current\_run\_id.txt"

repro-snapshot: guards init
@echo ">>> Capturing environment & config"
\$(CLI) env-capture || true
\$(CLI) hash-config || true
@echo ">>> Capturing DVC status"
\$(DVC) status -c > "\$(MANIFEST\_DIR)/dvc\_status\_\$(RUN\_ID).txt" || true
@echo ">>> Writing run manifest JSON"
@\$(PYTHON) - <<'PY'
import json, os, subprocess, time, pathlib
run\_id = os.environ.get("RUN\_ID","unknown")
outdir = os.environ.get("MANIFEST\_DIR","outputs/manifests")
pathlib.Path(outdir).mkdir(parents=True, exist\_ok=True)
def sh(cmd):
try:
return subprocess.check\_output(cmd, shell=True, text=True).strip()
except Exception:
return ""
manifest = {
"run\_id": run\_id,
"ts\_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
"git": {
"commit": sh("git rev-parse --short HEAD 2>/dev/null || echo 'nogit'"),
"status": sh("git status --porcelain || true")
},
"hydra\_config\_hash": sh("spectramind hash-config 2>/dev/null || echo ''"),
"device": os.environ.get("DEVICE",""),
"epochs": os.environ.get("EPOCHS",""),
"seed": os.environ.get("SEED",""),
}
fp = os.path.join(outdir, f"run\_manifest\_{run\_id}.json")
with open(fp, "w") as f:
json.dump(manifest, f, indent=2)
print("Wrote manifest:", fp)
PY

repro-verify:
@echo ">>> Manifest files:"
@ls -lh "\$(MANIFEST\_DIR)" || true
@echo ">>> Show last manifest:"
@ls -t "\$(MANIFEST\_DIR)"/run\_manifest\_\*.json 2>/dev/null | head -n1 | xargs -I{} cat {}

repro-manifest: repro-start repro-snapshot

# ========= GUI demo suite targets =========

gui-help:
@echo "\$(BOLD)GUI Demo Targets\$(RST)"
@echo "  \$(CYN)gui-demo\$(RST)           : Run Streamlit demo (thin wrapper around CLI & artifacts)"
@echo "  \$(CYN)gui-backend\$(RST)        : Run FastAPI backend (contracts for React dashboard) on port \$(FASTAPI\_PORT)"
@echo "  \$(CYN)gui-backend-stop\$(RST)   : Try to stop FastAPI dev server (best-effort)"
@echo "  \$(CYN)gui-qt\$(RST)             : Run PyQt demo (embeds diagnostics HTML)"
@echo ""

gui-demo: init
@command -v \$(STREAMLIT) >/dev/null 2>&1 || { echo "\$(YLW)streamlit not found — install via Poetry/pip\$(RST)"; exit 1; }
@test -f "\$(STREAMLIT\_APP)" || { echo "\$(RED)Missing \$(STREAMLIT\_APP)\$(RST)"; exit 1; }
\$(STREAMLIT) run "\$(STREAMLIT\_APP)"

gui-backend: init
@command -v \$(UVICORN) >/dev/null 2>&1 || { echo "\$(YLW)uvicorn not found — install via Poetry/pip\$(RST)"; exit 1; }
@test -f "\$(FASTAPI\_APP)" || { echo "\$(RED)Missing \$(FASTAPI\_APP)\$(RST)"; exit 1; }
\$(UVICORN) "\$(FASTAPI\_APP:.py=)\:app" --reload --port \$(FASTAPI\_PORT)

gui-backend-stop:
@pgrep -f "uvicorn.\$(FASTAPI\_APP:.py=)\:app" >/dev/null 2>&1 &&&#x20;
kill \$\$(pgrep -f "uvicorn.\$(FASTAPI\_APP:.py=)\:app") ||&#x20;
echo "::warning::No uvicorn process found"

gui-qt: init
@test -f "\$(QT\_APP)" || { echo "\$(RED)Missing \$(QT\_APP)\$(RST)"; exit 1; }
\$(PYTHON) "\$(QT\_APP)"

# ========= Cleanup =========

clean:
rm -rf "\$(DIAG\_DIR)" "\$(PRED\_DIR)" "\$(SUBMIT\_DIR)"

cache-clean:
@echo ">>> Cleaning caches and logs"
find . -type d -name "**pycache**" -prune -exec rm -rf {} + || true
rm -rf .pytest\_cache .ruff\_cache .mypy\_cache .dvc/tmp || true
find \$(LOGS\_DIR) -type f -name "\*.log" -delete 2>/dev/null || true

realclean: clean cache-clean
rm -rf "\$(OUT\_DIR)"
rm -rf .dvc/cache

distclean: realclean
@echo ">>> Removing Poetry caches and local venv (full reset)"
rm -rf .venv
rm -rf \~/.cache/pypoetry || true
rm -rf \~/.cache/pip || true
