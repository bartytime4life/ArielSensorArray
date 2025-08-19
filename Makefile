# SpectraMind V50 — Makefile
# -----------------------------------------------------------------------------
# Convenience targets to keep the repo reproducible and developer-friendly.
# All heavy work is delegated to the unified Typer CLI: `python -m spectramind`.
# Variables can be overridden on the command line, e.g.:
#   make train OVERRIDES="data=toy training=default model=v50"
# -----------------------------------------------------------------------------

# Python entrypoint (can be changed to `spectramind` if installed as a console script)
PY ?= python
CLI ?= -m spectramind

# General settings
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Colored echo
YELLOW := \033[1;33m
GREEN  := \033[1;32m
BLUE   := \033[1;34m
RESET  := \033[0m

# User-overridable Hydra overrides for train/predict/etc.
# Example: make train OVERRIDES="data=kaggle training=fast model=v50 +training.seed=1337"
OVERRIDES ?=

# Common paths
OUTDIR := outputs
LOGDIR := logs
DIAGDIR := $(OUTDIR)/diagnostics

# -----------------------------------------------------------------------------
# Meta
# -----------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@echo -e "$(BLUE)SpectraMind V50 — Makefile targets$(RESET)"
	@echo
	@grep -E '^[a-zA-Z0-9_\-]+:.*?## .+$$' $(MAKEFILE_LIST) \
	| sed -E 's/:.*?## /: /' \
	| sort \
	| awk -F': ' '{ printf "  \033[1m%-18s\033[0m %s\n", $$1, $$2 }'

.PHONY: init
init: ## Initialize repo: create dirs, install pre-commit hooks (if present)
	@mkdir -p $(OUTDIR) $(LOGDIR) $(DIAGDIR)
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo -e "$(YELLOW)Installing pre-commit hooks...$(RESET)"; \
		pre-commit install; \
	else \
		echo -e "$(YELLOW)pre-commit not found; skipping hook install$(RESET)"; \
	fi
	@echo -e "$(GREEN)Init done.$(RESET)"

.PHONY: clean
clean: ## Remove transient outputs (keeps logs and checkpoints)
	@echo -e "$(YELLOW)Cleaning outputs (except checkpoints/logs)...$(RESET)"
	@find $(OUTDIR) -maxdepth 1 -type f -name "*.csv" -delete || true
	@rm -rf $(DIAGDIR) || true
	@rm -f $(OUTDIR)/temp_scaling.json || true
	@echo -e "$(GREEN)Clean complete.$(RESET)"

.PHONY: distclean
distclean: ## Thorough cleanup (outputs/, logs/ diagnostics/); keeps data/
	@echo -e "$(YELLOW)Deep cleaning outputs and logs...$(RESET)"
	@rm -rf $(OUTDIR) || true
	@rm -rf $(LOGDIR) || true
	@mkdir -p $(OUTDIR) $(LOGDIR)
	@echo -e "$(GREEN)Distclean complete.$(RESET)"

# -----------------------------------------------------------------------------
# Checks & QA
# -----------------------------------------------------------------------------

.PHONY: selftest
selftest: ## Run CLI selftest (wiring & integrity)
	$(PY) $(CLI) selftest

.PHONY: selftest-deep
selftest-deep: ## Run CLI selftest with Hydra compose smoke check
	$(PY) $(CLI) selftest --deep

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
calibrate: ## Run calibration kill chain → outputs/calibrated
	$(PY) $(CLI) calibrate $(OVERRIDES)

.PHONY: train
train: ## Train model (uses Hydra overrides via OVERRIDES)
	$(PY) $(CLI) train $(OVERRIDES)

.PHONY: predict
predict: ## Predict μ/σ and write submission CSV
	$(PY) $(CLI) predict $(OVERRIDES) --out-csv $(OUTDIR)/submission.csv

.PHONY: temp-scale
temp-scale: ## Temperature scaling for uncertainty calibration
	$(PY) $(CLI) calibrate-temp $(OVERRIDES)

.PHONY: corel-train
corel-train: ## Train COREL conformal model
	$(PY) $(CLI) corel-train $(OVERRIDES)

.PHONY: diagnose
diagnose: ## Build diagnostics dashboard HTML
	$(PY) $(CLI) diagnose dashboard $(OVERRIDES) --html-out $(DIAGDIR)/report_v1.html

.PHONY: submit
submit: ## Create submission bundle ZIP from latest artifacts
	$(PY) $(CLI) submit --zip-out $(OUTDIR)/submission_bundle.zip

.PHONY: analyze-log
analyze-log: ## Parse logs/v50_debug_log.md → markdown & csv tables
	$(PY) $(CLI) analyze-log --md $(OUTDIR)/logs/log_table.md --csv $(OUTDIR)/logs/log_table.csv

# -----------------------------------------------------------------------------
# DVC helpers (optional; no-op if dvc not installed)
# -----------------------------------------------------------------------------

.PHONY: dvc-init
dvc-init: ## Initialize DVC and create default remote (set URL via REMOTE=...)
	@if command -v dvc >/dev/null 2>&1; then \
		dvc init -q || true; \
		if [ -n "$(REMOTE)" ]; then dvc remote add -d storage "$(REMOTE)" || true; fi; \
	else echo "dvc not installed"; fi

.PHONY: dvc-pull
dvc-pull: ## Pull tracked data/artifacts
	@if command -v dvc >/dev/null 2>&1; then dvc pull; else echo "dvc not installed"; fi

.PHONY: dvc-push
dvc-push: ## Push artifacts to remote
	@if command -v dvc >/dev/null 2>&1; then dvc push; else echo "dvc not installed"; fi
```
