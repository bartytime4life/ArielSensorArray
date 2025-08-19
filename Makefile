```makefile
# SpectraMind V50 — Makefile (root)
# CLI-first wrappers for common tasks (Poetry + Typer + Hydra + DVC)
# Usage:
#   make help
#   make init
#   make selftest
#   make train DATA=toy CFG="model=v50 training=fast"
#   make predict OUT=outputs/submission.csv
#   make diagnose HTML=outputs/diagnostics/report.html
#   make submit ZIP=outputs/submission_bundle.zip
#   make docs
#   make clean

# ---------------------------------------------------------------------
# Variables (override on CLI, e.g. `make train DATA=kaggle`)
# ---------------------------------------------------------------------
PY        := poetry run
PYTHON    := poetry run python
SPECTRA   := $(PYTHON) -m spectramind

DATA      ?= toy
CFG       ?=
OUT       ?= outputs/submission.csv
HTML      ?= outputs/diagnostics/report.html
ZIP       ?= outputs/submission_bundle.zip
N_JOBS    ?= 1

# ---------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------
.PHONY: help init lock env-update lint format test coverage \
        selftest calibrate train predict calibrate-temp corel-train \
        diagnose submit ablate analyze-log check-cli-map \
        docs dvc-init dvc-repro clean deepclean

# ---------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------
help:
	@echo ""
	@echo "SpectraMind V50 — Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  init                 Install Poetry deps, pre-commit hooks, and DVC (optional)."
	@echo "  lock                 Refresh poetry.lock from pyproject.toml."
	@echo "  env-update           Reinstall deps using Poetry (no-root)."
	@echo "  lint                 Run ruff + black --check + isort --check."
	@echo "  format               Auto-format with black + isort; fix lint with ruff."
	@echo "  test                 Run pytest."
	@echo "  coverage             Run pytest with coverage report."
	@echo "  selftest             Run spectramind selftest (deep)."
	@echo "  calibrate            Run calibration kill chain (DATA=$(DATA))."
	@echo "  train                Train model (DATA=$(DATA) CFG='$(CFG)')."
	@echo "  predict              Run inference to CSV (OUT=$(OUT))."
	@echo "  calibrate-temp       Temperature scaling for σ calibration."
	@echo "  corel-train          Train COREL conformal/graph calibration."
	@echo "  diagnose             Build diagnostics dashboard (HTML=$(HTML))."
	@echo "  submit               End-to-end submission bundle (ZIP=$(ZIP))."
	@echo "  ablate               Run ablation suite (Hydra multirun supported)."
	@echo "  analyze-log          Summarize CLI calls from v50_debug_log.md."
	@echo "  check-cli-map        Show command-to-file mapping integrity."
	@echo "  docs                 Build docs site (MkDocs) if configured."
	@echo "  dvc-init             Initialize DVC (remote optional)."
	@echo "  dvc-repro            Reproduce DVC pipeline end-to-end."
	@echo "  clean                Remove temp outputs and caches."
	@echo "  deepclean            Remove venv caches, outputs, and DVC cache."
	@echo ""

# ---------------------------------------------------------------------
# Install / Environment
# ---------------------------------------------------------------------
init:
	@echo ">> Installing Poetry dependencies..."
	poetry install --no-root
	@echo ">> Installing pre-commit hooks..."
	poetry run pre-commit install
	@echo ">> (Optional) Initialize DVC if needed: make dvc-init"
	@echo ">> Done."

lock:
	@echo ">> Updating poetry.lock..."
	poetry lock --no-update
	@echo ">> Done."

env-update:
	@echo ">> Reinstalling env from lockfile..."
	poetry install --no-root
	@echo ">> Done."

# ---------------------------------------------------------------------
# Quality: lint / format / tests / coverage
# ---------------------------------------------------------------------
lint:
	@echo ">> Lint: ruff + black --check + isort --check..."
	$(PY) ruff check .
	$(PY) black --check .
	$(PY) isort --check-only .
	@echo ">> Lint OK."

format:
	@echo ">> Formatting with black + isort; autofix lint with ruff..."
	$(PY) isort .
	$(PY) black .
	$(PY) ruff check . --fix
	@echo ">> Format OK."

test:
	@echo ">> Running pytest..."
	$(PY) pytest -q
	@echo ">> Tests OK."

coverage:
	@echo ">> Running tests with coverage..."
	$(PY) pytest --cov=src --cov-report=term-missing --cov-report=xml
	@echo ">> Coverage report generated (coverage.xml)."

# ---------------------------------------------------------------------
# CLI Wrappers
# ---------------------------------------------------------------------
selftest:
	@echo ">> Running SpectraMind selftest (deep)..."
	$(SPECTRA) selftest --deep

calibrate:
	@echo ">> Calibrating dataset: $(DATA)"
	$(SPECTRA) calibrate data=$(DATA) calibration.cache=true

train:
	@echo ">> Training on dataset: $(DATA) with overrides: $(CFG)"
	$(SPECTRA) train data=$(DATA) $(CFG)

predict:
	@echo ">> Predicting μ/σ → $(OUT)"
	mkdir -p $(dir $(OUT))
	$(SPECTRA) predict data=$(DATA) --out-csv $(OUT)

calibrate-temp:
	@echo ">> Temperature scaling for σ calibration..."
	$(SPECTRA) calibrate-temp data=$(DATA) $(CFG)

corel-train:
	@echo ">> Training COREL conformal/graph calibration..."
	$(SPECTRA) corel-train data=$(DATA) $(CFG)

diagnose:
	@echo ">> Building diagnostics dashboard → $(HTML)"
	mkdir -p $(dir $(HTML))
	$(SPECTRA) diagnose dashboard $(CFG) --html-out $(HTML)

submit:
	@echo ">> End-to-end submission → $(ZIP)"
	mkdir -p $(dir $(ZIP))
	$(SPECTRA) submit $(CFG) --zip-out $(ZIP)

ablate:
	@echo ">> Running ablation suite (Hydra multirun enabled)..."
	# Example: make ablate CFG="-m training.lr=1e-4,2e-4 uq.epistemic=ensemble,mc_dropout"
	$(SPECTRA) ablate $(CFG)

analyze-log:
	@echo ">> Analyzing CLI logs..."
	$(SPECTRA) analyze-log $(CFG)

check-cli-map:
	@echo ">> Checking command-to-file mapping..."
	$(SPECTRA) check-cli-map $(CFG)

# ---------------------------------------------------------------------
# Docs / MkDocs
# ---------------------------------------------------------------------
docs:
	@echo ">> Building docs site (MkDocs)..."
	@if command -v mkdocs >/dev/null 2>&1; then \
	  mkdocs build --clean; \
	else \
	  echo "MkDocs not installed. Install with: poetry run pip install mkdocs mkdocs-material"; \
	fi

# ---------------------------------------------------------------------
# DVC helpers
# ---------------------------------------------------------------------
dvc-init:
	@echo ">> Initializing DVC..."
	@if command -v dvc >/dev/null 2>&1; then \
	  dvc init -q || true; \
	  echo "DVC initialized. Configure remote with: dvc remote add -d storage <url>"; \
	else \
	  echo "DVC not installed (pip install dvc)."; \
	fi

dvc-repro:
	@echo ">> Reproducing pipeline via DVC..."
	@if command -v dvc >/dev/null 2>&1; then \
	  dvc repro -j $(N_JOBS); \
	else \
	  echo "DVC not installed."; \
	fi

# ---------------------------------------------------------------------
# Cleaners
# ---------------------------------------------------------------------
clean:
	@echo ">> Cleaning temporary artifacts..."
	rm -rf outputs/diagnostics/*.tmp 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache 2>/dev/null || true
	find . -type d -name "__pycache__" -prune -exec rm -rf {} \; 2>/dev/null || true
	@echo ">> Clean OK."

deepclean: clean
	@echo ">> Deep cleaning (artifacts + caches + DVC cache)..."
	rm -rf outputs/* 2>/dev/null || true
	rm -rf logs/*.jsonl 2>/dev/null || true
	rm -rf .dvc/tmp .dvc/cache 2>/dev/null || true
	@echo ">> Deep clean OK."
```
