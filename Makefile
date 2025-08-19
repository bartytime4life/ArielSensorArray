# SpectraMind V50 — Makefile

# -----------------------------------------------------------------------------

# Convenience targets to keep the repo reproducible and developer-friendly.

# All heavy work is delegated to the unified Typer CLI: `python -m spectramind`.

# Variables can be overridden on the command line, e.g.:

# make train OVERRIDES="data=toy training=default model=v50"

# -----------------------------------------------------------------------------

# Python entrypoint (can be changed to `spectramind` if installed as a console script)

PY ?= python
CLI ?= -m spectramind

# General settings

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eo pipefail -c
.DEFAULT\_GOAL := help

# Colored echo

YELLOW := \033\[1;33m
GREEN  := \033\[1;32m
BLUE   := \033\[1;34m
RED    := \033\[1;31m
BOLD   := \033\[1m
DIM    := \033\[2m
RESET  := \033\[0m

# User-overridable Hydra overrides for train/predict/etc.

# Example: make train OVERRIDES="data=kaggle training=fast model=v50 +training.seed=1337"

OVERRIDES ?=

# Common paths

OUTDIR   := outputs
LOGDIR   := logs
DIAGDIR  := \$(OUTDIR)/diagnostics
SUBMISSION := \$(OUTDIR)/submission.csv
REPORT\_HTML := \$(DIAGDIR)/report\_v1.html

# Docker (optional)

IMAGE ?= spectramindv50\:dev

# -----------------------------------------------------------------------------

# Internal helpers

# -----------------------------------------------------------------------------

# Ensure required directories exist

REQUIRED\_DIRS := \$(OUTDIR) \$(LOGDIR) \$(DIAGDIR)

\$(REQUIRED\_DIRS):
@mkdir -p \$@
@echo -e "\$(DIM)\[MKDIR] Ensured: \$@\$(RESET)"

# Print a divider

define \_divider
@printf "\n\$(DIM)%0.s─" \$\$(seq 1 78); printf "\$(RESET)\n"
endef

# -----------------------------------------------------------------------------

# Meta

# -----------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
@echo -e "\$(BLUE)SpectraMind V50 — Makefile targets\$(RESET)"
@echo
@grep -E '^\[a-zA-Z0-9\_-]+:.\*?## .+$$' $(MAKEFILE_LIST) \ 	| sed -E 's/:.*?## /: /' \ 	| sort \ 	| awk -F': ' '{ printf "  \033[1m%-20s\033[0m %s\n", $$1, \$\$2 }'
@echo
@echo -e "\$(DIM)Examples:\$(RESET)"
@echo -e "  make train OVERRIDES="data=toy training=fast model=v50 +training.seed=1337""
@echo -e "  make predict OVERRIDES="+predict.checkpoints=last""

.PHONY: init
init: \$(REQUIRED\_DIRS) ## Initialize repo: create dirs, install pre-commit hooks (if present)
@if command -v pre-commit >/dev/null 2>&1; then&#x20;
echo -e "\$(YELLOW)Installing pre-commit hooks...\$(RESET)";&#x20;
pre-commit install -t pre-commit -t pre-push;&#x20;
else&#x20;
echo -e "\$(YELLOW)pre-commit not found; skipping hook install\$(RESET)";&#x20;
fi
@echo -e "\$(GREEN)Init done.\$(RESET)"

# -----------------------------------------------------------------------------

# Checks & QA

# -----------------------------------------------------------------------------

.PHONY: selftest
selftest: \$(REQUIRED\_DIRS) ## Run CLI selftest (wiring & integrity)
\$(PY) \$(CLI) selftest

.PHONY: selftest-deep
selftest-deep: \$(REQUIRED\_DIRS) ## Run CLI selftest with Hydra compose & deep checks
\$(PY) \$(CLI) selftest --deep

.PHONY: lint
lint: ## Run ruff + black --check + isort --check (if installed)
@if command -v ruff >/dev/null 2>&1; then ruff check .; else echo "ruff not installed"; fi
@if command -v black >/dev/null 2>&1; then black --check .; else echo "black not installed"; fi
@if command -v isort >/dev/null 2>&1; then isort --check-only .; else echo "isort not installed"; fi

.PHONY: fmt
fmt: ## Auto-format with black + isort (if installed)
@if command -v isort >/dev/null 2>&1; then isort .; else echo "isort not installed"; fi
@if command -v black >/dev/null 2>&1; then black .; else echo "black not installed"; fi

.PHONY: test
test: ## Run pytest (quiet)
@if command -v pytest >/dev/null 2>&1; then pytest -q; else echo "pytest not installed"; fi

# -----------------------------------------------------------------------------

# Pipeline shortcuts (thin wrappers around the CLI)

# -----------------------------------------------------------------------------

.PHONY: calibrate
calibrate: \$(REQUIRED\_DIRS) ## Run calibration kill chain → outputs/calibrated
\$(PY) \$(CLI) calibrate \$(OVERRIDES)

.PHONY: train
train: \$(REQUIRED\_DIRS) ## Train model (uses Hydra overrides via OVERRIDES)
\$(PY) \$(CLI) train \$(OVERRIDES)

.PHONY: predict
predict: \$(REQUIRED\_DIRS) ## Predict μ/σ and write submission CSV
\$(PY) \$(CLI) predict \$(OVERRIDES) --out-csv \$(SUBMISSION)

.PHONY: temp-scale
temp-scale: \$(REQUIRED\_DIRS) ## Temperature scaling for uncertainty calibration
\$(PY) \$(CLI) calibrate-temp \$(OVERRIDES)

.PHONY: corel-train
corel-train: \$(REQUIRED\_DIRS) ## Train COREL conformal model
\$(PY) \$(CLI) corel-train \$(OVERRIDES)

.PHONY: diagnose
diagnose: \$(REQUIRED\_DIRS) ## Build diagnostics dashboard HTML
\$(PY) \$(CLI) diagnose dashboard \$(OVERRIDES) --html-out \$(REPORT\_HTML)

.PHONY: submit
submit: selftest \$(REQUIRED\_DIRS) ## Create submission bundle ZIP from latest artifacts (runs selftest first)
\$(PY) \$(CLI) submit --bundle --validate \$(OVERRIDES)

.PHONY: e2e
e2e: \$(REQUIRED\_DIRS) ## End-to-end: selftest → train → predict → (optional) temp-scale → diagnose
@echo -e "\$(BOLD)\$(BLUE)\[E2E]\$(RESET) Starting full pipeline"
\$(PY) \$(CLI) selftest --fast
\$(\_divider)
\$(PY) \$(CLI) train \$(OVERRIDES)
\$(\_divider)
\$(PY) \$(CLI) predict \$(OVERRIDES) --out-csv \$(SUBMISSION)
\$(\_divider)
@if \$(PY) -c 'import sys; sys.exit(0)'; then&#x20;
echo -e "\$(DIM)\[E2E] Optional temp-scale step (skip if not configured)\$(RESET)";&#x20;
\$(PY) \$(CLI) calibrate-temp \$(OVERRIDES) || true;&#x20;
fi
\$(\_divider)
\$(PY) \$(CLI) diagnose dashboard \$(OVERRIDES) --html-out \$(REPORT\_HTML)
@echo -e "\$(GREEN)\[E2E] Complete. Artifacts in \$(OUTDIR)/\$(RESET)"

.PHONY: open-report
open-report: ## Open latest diagnostics HTML report, if present
@if \[ -f "\$(REPORT\_HTML)" ]; then&#x20;
echo -e "\$(BLUE)\[OPEN]\$(RESET) \$(REPORT\_HTML)";&#x20;
(xdg-open "\$(REPORT\_HTML)" || open "\$(REPORT\_HTML)" || python -m webbrowser "\$(REPORT\_HTML)" || true) >/dev/null 2>&1;&#x20;
else&#x20;
echo -e "\$(RED)\[ERROR]\$(RESET) \$(REPORT\_HTML) not found. Run 'make diagnose' first.";&#x20;
exit 1;&#x20;
fi

.PHONY: analyze-log
analyze-log: \$(REQUIRED\_DIRS) ## Parse logs/v50\_debug\_log.md → markdown & csv tables
\$(PY) \$(CLI) analyze-log --md "\$(LOGDIR)/log\_table.md" --csv "\$(LOGDIR)/log\_table.csv"
@echo -e "\$(GREEN)\[ANALYZE]\$(RESET) Wrote \$(LOGDIR)/log\_table.{md,csv}"

# -----------------------------------------------------------------------------

# CI entrypoint (lightweight pipeline)

# -----------------------------------------------------------------------------

.PHONY: ci
ci: \$(REQUIRED\_DIRS) ## CI smoke test: fast selftest + dry-run train/predict
@echo -e "\$(BLUE)\[CI]\$(RESET) Smoke test"
\$(PY) \$(CLI) selftest --fast
\$(PY) \$(CLI) train --dry-run \$(OVERRIDES)
\$(PY) \$(CLI) predict --dry-run \$(OVERRIDES)

# -----------------------------------------------------------------------------

# DVC helpers (optional; no-op if dvc not installed)

# -----------------------------------------------------------------------------

.PHONY: dvc-init
dvc-init: ## Initialize DVC and create default remote (set URL via REMOTE=...)
@if command -v dvc >/dev/null 2>&1; then&#x20;
dvc init -q || true;&#x20;
if \[ -n "\$(REMOTE)" ]; then dvc remote add -d storage "\$(REMOTE)" || true; fi;&#x20;
else echo "dvc not installed"; fi

.PHONY: dvc-pull
dvc-pull: ## Pull tracked data/artifacts
@if command -v dvc >/dev/null 2>&1; then dvc pull; else echo "dvc not installed"; fi

.PHONY: dvc-push
dvc-push: ## Push artifacts to remote
@if command -v dvc >/dev/null 2>&1; then dvc push; else echo "dvc not installed"; fi

# -----------------------------------------------------------------------------

# Docker helpers (optional)

# -----------------------------------------------------------------------------

.PHONY: docker-build
docker-build: ## Build dev Docker image
@echo -e "\$(BLUE)\[DOCKER]\$(RESET) Building image \$(IMAGE)"
docker build -t \$(IMAGE) .

.PHONY: docker-run
docker-run: ## Run dev shell (GPU if available)
@echo -e "\$(BLUE)\[DOCKER]\$(RESET) Running container \$(IMAGE)"
docker run --gpus all -it --rm -v "\$\$PWD:/workspace" -w /workspace \$(IMAGE) bash

# -----------------------------------------------------------------------------

# Hygiene

# -----------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove transient outputs (keeps logs and checkpoints)
@echo -e "\$(YELLOW)Cleaning outputs (except checkpoints/logs)...\$(RESET)"
@find "\$(OUTDIR)" -maxdepth 1 -type f -name "*.csv" -delete || true
@find "\$(OUTDIR)" -maxdepth 1 -type f -name "*.zip" -delete || true
@find "\$(OUTDIR)" -maxdepth 1 -type f -name "*.json" -delete || true
@find "\$(OUTDIR)" -maxdepth 1 -type f -name "*.html" -delete || true
@rm -rf "\$(DIAGDIR)" || true
@rm -f "\$(OUTDIR)/temp\_scaling.json" || true
@echo -e "\$(GREEN)Clean complete.\$(RESET)"

.PHONY: distclean
distclean: ## Thorough cleanup (outputs/, logs/ diagnostics/); keeps data/
@echo -e "\$(YELLOW)Deep cleaning outputs and logs...\$(RESET)"
@rm -rf "\$(OUTDIR)" || true
@rm -rf "\$(LOGDIR)" || true
@mkdir -p "\$(OUTDIR)" "\$(LOGDIR)"
@echo -e "\$(GREEN)Distclean complete.\$(RESET)"
