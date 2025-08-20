# ==============================================================================
# SpectraMind V50 — Master Makefile (Dev/Local, CI‑Safe, Kaggle‑Ready)
# Neuro‑Symbolic, Physics‑Informed AI Pipeline
# ==============================================================================

# ========= Shell =========
SHELL        := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS  := -euo pipefail -c

# ========= Tooling =========
PYTHON       ?= python3
POETRY       ?= poetry
CLI          ?= $(POETRY) run spectramind

NODE         ?= node
NPM          ?= npm
KAGGLE       ?= kaggle
PIP          ?= $(PYTHON) -m pip

# ========= Defaults (override at CLI) =========
DEVICE       ?= cpu
EPOCHS       ?= 1
TS           := $(shell date +%Y%m%d_%H%M%S)

OUT_DIR      ?= outputs
LOGS_DIR     ?= logs
DIAG_DIR     ?= $(OUT_DIR)/diagnostics
PRED_DIR     ?= $(OUT_DIR)/predictions
SUBMIT_DIR   ?= $(OUT_DIR)/submission
SUBMIT_ZIP   ?= $(SUBMIT_DIR)/bundle.zip

RUN_HASH_FILE ?= run_hash_summary_v50.json

# Requirements files
REQ_CORE          ?= requirements.txt
REQ_EXTRAS        ?= requirements-extras.txt
REQ_DEV           ?= requirements-dev.txt
REQ_KAGGLE        ?= requirements-kaggle.txt
REQ_FREEZE        ?= requirements.freeze.txt

# Mermaid export defaults
DIAGRAMS_DIR       ?= docs/diagrams
MERMAID_FILES      ?= ARCHITECTURE.md README.md
MERMAID_THEME      ?=
MERMAID_EXPORT_PNG ?= 0   # 1 to export PNG alongside SVG

# Hydra overrides and passthrough args for the CLI
OVERRIDES    ?=
EXTRA_ARGS   ?=

# ========= Colors =========
BOLD := \033[1m
DIM  := \033[2m
RED  := \033[31m
GRN  := \033[32m
YLW  := \033[33m
CYN  := \033[36m
RST  := \033[0m

# ========= PHONY =========
.PHONY: help init env info doctor versions \
        fmt lint mypy test pre-commit hooks \
        selftest selftest-deep validate-env \
        calibrate calibrate-temp corel-train \
        train predict predict-e2e diagnose submit \
        ablate ablate-light ablate-heavy ablate-grid ablate-optuna \
        analyze-log analyze-log-short check-cli-map open-report \
        dvc-pull dvc-push \
        bench-selftest benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean \
        kaggle-run kaggle-submit \
        mermaid-init diagrams diagrams-png mermaid-export mermaid-clean \
        ci quickstart clean realclean distclean cache-clean \
        export-reqs export-reqs-dev export-kaggle-reqs export-freeze \
        install-core install-extras install-dev install-kaggle \
        env-capture hash-config \
        pip-audit audit docs-serve docs-build \
        pyg-install kaggle-pyg-index

# ========= Default Goal =========
.DEFAULT_GOAL := help

# ========= Help =========
help:
	@echo ""
	@echo "$(BOLD)SpectraMind V50 — Make targets$(RST)"
	@echo "  $(CYN)quickstart$(RST)       : install deps (poetry), init dirs, print info"
	@echo "  $(CYN)doctor$(RST)           : dependency checks (python/poetry/node/npm/cli)"
	@echo "  $(CYN)selftest$(RST)         : fast integrity checks (CLI + files)"
	@echo "  $(CYN)train$(RST)            : run training (EPOCHS=$(EPOCHS), DEVICE=$(DEVICE))"
	@echo "  $(CYN)predict$(RST)          : run inference → $(PRED_DIR)/submission.csv"
	@echo "  $(CYN)predict-e2e$(RST)      : smoke test asserting submission exists"
	@echo "  $(CYN)diagnose$(RST)         : build diagnostics (smoothness + dashboard)"
	@echo "  $(CYN)submit$(RST)           : package submission ZIP ($(SUBMIT_ZIP))"
	@echo "  $(CYN)ablate*$(RST)          : ablation sweeps (light/heavy/grid/optuna)"
	@echo "  $(CYN)analyze-log$(RST)      : parse logs → $(OUT_DIR)/log_table.{md,csv}"
	@echo "  $(CYN)diagrams$(RST)         : render Mermaid in $(MERMAID_FILES) → $(DIAGRAMS_DIR)"
	@echo "  $(CYN)benchmark-*$(RST)      : small benchmark runs (cpu/gpu)"
	@echo "  $(CYN)kaggle-*$(RST)         : Kaggle run+submit (requires Kaggle CLI login)"
	@echo "  $(CYN)fmt | lint | mypy | test | pre-commit$(RST) : code quality"
	@echo "  $(CYN)env-capture | hash-config$(RST) : reproducibility utilities"
	@echo "  $(CYN)export-reqs* | install-*(RST)  : requirements export/install helpers"
	@echo "  $(CYN)docs-serve | docs-build$(RST)  : MkDocs docs helpers (optional)"
	@echo "  $(CYN)pip-audit$(RST)                : CVE scan for installed packages"
	@echo "  $(CYN)cache-clean$(RST)              : wipe caches/__pycache__/logs"
	@echo "  $(CYN)clean | realclean | distclean$(RST)"
	@echo ""
	@echo "$(DIM)Vars: DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OUT_DIR=$(OUT_DIR) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'$(RST)"
	@echo "$(DIM)Mermaid: MERMAID_FILES='$(MERMAID_FILES)' DIAGRAMS_DIR='$(DIAGRAMS_DIR)' THEME='$(MERMAID_THEME)' PNG=$(MERMAID_EXPORT_PNG)$(RST)"
	@echo ""

# ========= Init / Env =========
init: env
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

versions:
	@echo "$(BOLD)Versions$(RST)"
	@echo "python : $$($(PYTHON) --version 2>&1 || true)"
	@echo "poetry : $$($(POETRY) --version 2>&1 || true)"
	@echo "node   : $$($(NODE) --version 2>&1 || true)"
	@echo "npm    : $$($(NPM) --version 2>&1 || true)"
	@echo "kaggle : $$($(KAGGLE) --version 2>&1 || true)"
	@echo "cli    : $(CLI)"

info: versions
	@echo "device : $(DEVICE)"
	@echo "OUT_DIR: $(OUT_DIR)"
	@echo "RUN_HASH_FILE: $(RUN_HASH_FILE)"

doctor:
	@ok=1; \
	command -v $(PYTHON) >/dev/null 2>&1 || { echo "$(RED)Missing python3$(RST)"; ok=0; }; \
	command -v $(POETRY) >/dev/null 2>&1 || { echo "$(YLW)Poetry not found — install via pipx/pip$(RST)"; ok=0; }; \
	command -v $(NODE)   >/dev/null 2>&1 || { echo "$(YLW)Node not found (needed for mermaid-cli)$(RST)"; }; \
	command -v $(NPM)    >/dev/null 2>&1 || { echo "$(YLW)npm not found (needed for mermaid-cli)$(RST)"; }; \
	{ $(CLI) --version >/dev/null 2>&1 && echo "$(GRN)CLI OK$(RST)"; } || { echo "$(YLW)CLI not yet installed or venv not active$(RST)"; }; \
	test $$ok -eq 1

quickstart: env info
	@echo "$(CYN)Installing project deps via Poetry (no-root)…$(RST)"
	@$(POETRY) install --no-root
	@echo "$(GRN)Done.$(RST)"
	@$(MAKE) doctor

# ========= Dev / Quality =========
fmt:
	$(POETRY) run isort .
	$(POETRY) run black .

lint:
	$(POETRY) run ruff check .

mypy:
	$(POETRY) run mypy --strict src || true

test: init
	$(POETRY) run pytest -q || $(POETRY) run pytest -q -x

pre-commit:
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit run --all-files || true

# ========= Env validation (safe no-op if script missing) =========
validate-env:
	@if [ -x scripts/validate_env.py ] || [ -f scripts/validate_env.py ]; then \
	  echo ">>> Validating .env schema"; \
	  $(PYTHON) scripts/validate_env.py || exit 1; \
	else \
	  echo ">>> Skipping validate-env (scripts/validate_env.py not found)"; \
	fi

# ========= Pipeline =========
selftest: init
	$(CLI) selftest

selftest-deep: init
	$(CLI) selftest --deep

calibrate: init
	$(CLI) calibrate $(OVERRIDES) $(EXTRA_ARGS)

calibrate-temp: init
	$(CLI) calibrate-temp $(OVERRIDES) $(EXTRA_ARGS)

corel-train: init
	$(CLI) corel-train $(OVERRIDES) $(EXTRA_ARGS)

train: init
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) $(EXTRA_ARGS)

predict: init
	mkdir -p "$(PRED_DIR)"
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv" $(OVERRIDES) $(EXTRA_ARGS)

predict-e2e: predict
	@test -f "$(PRED_DIR)/submission.csv" && echo "$(GRN)OK: $(PRED_DIR)/submission.csv$(RST)" || (echo "$(RED)Missing submission.csv$(RST)"; exit 1)

diagnose: init
	$(CLI) diagnose smoothness --outdir "$(DIAG_DIR)" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || true

open-report:
	@latest=$$(ls -t $(DIAG_DIR)/*.html 2>/dev/null | head -n1 || true); \
	if [ -n "$$latest" ]; then \
	  echo "Opening $$latest"; \
	  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$$latest" || true; \
	  elif command -v open >/dev/null 2>&1; then open "$$latest" || true; \
	  else echo "No opener found (CI/headless)"; fi; \
	else echo "No diagnostics HTML found."; fi

submit: init
	mkdir -p "$(SUBMIT_DIR)"
	$(CLI) submit --zip-out "$(SUBMIT_ZIP)" $(EXTRA_ARGS)

# ========= Ablation (profiles & sweep styles) =========
ablate: init
	$(CLI) ablate $(OVERRIDES) $(EXTRA_ARGS)
	@$(PYTHON) tools/ablation_post.py --csv outputs/ablate/leaderboard.csv --metric gll --ascending --top-n 5 --outdir outputs/ablate --html-template tools/leaderboard_template.html || true

ablate-light: init
	$(CLI) ablate ablation=ablation_light $(EXTRA_ARGS)
	@$(PYTHON) tools/ablation_post.py --csv outputs/ablate/leaderboard.csv --metric gll --ascending --top-n 3 --outdir outputs/ablate_light --html-template tools/leaderboard_template.html || true

ablate-heavy: init
	$(CLI) ablate ablation=ablation_heavy $(EXTRA_ARGS)
	@$(PYTHON) tools/ablation_post.py --csv outputs/ablate/leaderboard.csv --metric gll --ascending --top-n 10 --outdir outputs/ablate_heavy --html-template tools/leaderboard_template.html || true

ablate-grid: init
	$(CLI) ablate -m ablate.sweeper=basic +ablate.search=v50_fast_grid ablation=ablation_light $(EXTRA_ARGS)
	@$(PYTHON) tools/ablation_post.py --csv outputs/ablate/leaderboard.csv --metric gll --ascending --top-n 5 --outdir outputs/ablate --html-template tools/leaderboard_template.html || true

ablate-optuna: init
	$(CLI) ablate -m ablate.sweeper=optuna +ablate.search=v50_symbolic_core ablation=ablation_heavy $(EXTRA_ARGS)
	@$(PYTHON) tools/ablation_post.py --csv outputs/ablate/leaderboard.csv --metric gll --ascending --top-n 10 --outdir outputs/ablate --html-template tools/leaderboard_template.html || true

# ========= Log analysis =========
analyze-log: init
	$(CLI) analyze-log --md "$(OUT_DIR)/log_table.md" --csv "$(OUT_DIR)/log_table.csv" $(EXTRA_ARGS)

analyze-log-short: init
	@if [ ! -f "$(OUT_DIR)/log_table.csv" ]; then \
	  echo ">>> Generating log CSV via analyze-log"; \
	  $(CLI) analyze-log --md "$(OUT_DIR)/log_table.md" --csv "$(OUT_DIR)/log_table.csv" $(EXTRA_ARGS); \
	fi; \
	if [ -f "$(OUT_DIR)/log_table.csv" ]; then \
	  echo "=== Last 5 CLI invocations ==="; \
	  tail -n +2 "$(OUT_DIR)/log_table.csv" | tail -n 5 | \
	    awk -F',' 'BEGIN{OFS=" | "} {print "time="$${1}, "cmd="$${2}, "git_sha="$${3}, "cfg="$${4}}'; \
	else \
	  echo "::warning::No log_table.csv to summarize"; \
	fi

check-cli-map:
	$(CLI) check-cli-map

# ========= Mermaid / Diagrams =========
mermaid-init:
	@if ! command -v $(NPM) >/dev/null 2>&1; then \
	  echo "$(RED)ERROR: npm not found. Please install Node.js/npm.$(RST)"; exit 1; \
	fi
	$(NPM) ci
	mkdir -p "$(DIAGRAMS_DIR)"

diagrams: mermaid-init
	@echo ">>> Rendering Mermaid diagrams (SVG; PNG=$(MERMAID_EXPORT_PNG))"
	EXPORT_PNG=$(MERMAID_EXPORT_PNG) THEME="$(MERMAID_THEME)" \
	$(PYTHON) scripts/export_mermaid.py $(MERMAID_FILES)
	@echo ">>> Output → $(DIAGRAMS_DIR)"

diagrams-png:
	@$(MAKE) --no-print-directory diagrams MERMAID_EXPORT_PNG=1

mermaid-export:
	@$(MAKE) --no-print-directory diagrams

mermaid-clean:
	rm -rf "$(DIAGRAMS_DIR)" .mermaid_tmp

# ========= CI convenience (dev/local reuse) =========
ci: validate-env selftest train diagnose analyze-log-short

# ========= DVC =========
dvc-pull:
	dvc pull || true

dvc-push:
	dvc push || true

# ========= Benchmarks =========
bench-selftest:
	$(CLI) selftest

benchmark: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-cpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=cpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-gpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=gpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-run:
	OUTDIR="benchmarks/$(TS)_$(DEVICE)"; \
	mkdir -p "$$OUTDIR"; \
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) --outdir "$$OUTDIR" $(EXTRA_ARGS); \
	$(CLI) diagnose smoothness --outdir "$$OUTDIR" $(EXTRA_ARGS); \
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$$OUTDIR" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$$OUTDIR" $(EXTRA_ARGS) || true; \
	{ \
	  echo "Benchmark summary"; \
	  date; \
	  echo "python   : $$($(PYTHON) --version 2>&1)"; \
	  echo "poetry   : $$($(POETRY) --version 2>&1 || true)"; \
	  echo "cli      : $(CLI)"; \
	  echo "device   : $(DEVICE)"; \
	  echo "epochs   : $(EPOCHS)"; \
	  echo "overrides: $(OVERRIDES)"; \
	  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true; \
	  echo ""; \
	  echo "Artifacts in $$OUTDIR:"; \
	  ls -lh "$$OUTDIR" || true; \
	} > "$$OUTDIR/summary.txt"; \
	echo ">>> Benchmark complete → $$OUTDIR/summary.txt"

benchmark-report:
	mkdir -p aggregated
	{ \
	  echo "# SpectraMind V50 Benchmark Report"; \
	  echo ""; \
	  for f in $$(find benchmarks -type f -name summary.txt | sort); do \
	    echo "## $$f"; echo ""; cat "$$f"; echo ""; \
	  done; \
	} > aggregated/report.md
	@echo ">>> Aggregated → aggregated/report.md"

benchmark-clean:
	rm -rf benchmarks aggregated

# ========= Kaggle Helpers =========
kaggle-run: init
	@echo ">>> Running single-epoch GPU run (Kaggle-like)"
	$(CLI) selftest
	$(CLI) train +training.epochs=1 --device gpu --outdir "$(OUT_DIR)"
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv"

kaggle-submit: kaggle-run
	@echo ">>> Submitting to Kaggle competition"
	$(KAGGLE) competitions submit -c neurips-2025-ariel -f "$(PRED_DIR)/submission.csv" -m "Spectramind V50 auto-submit"

# ========= Requirements export / install =========
export-reqs:
	@echo ">>> Exporting Poetry deps → $(REQ_CORE)"
	$(POETRY) export -f requirements.txt --without-hashes -o $(REQ_CORE)
	@echo ">>> Done."

export-reqs-dev:
	@echo ">>> Exporting Poetry deps (incl. dev) → $(REQ_DEV)"
	$(POETRY) export -f requirements.txt --with dev --without-hashes -o $(REQ_DEV)
	@echo ">>> Done."

export-kaggle-reqs:
	@echo ">>> Exporting Kaggle-friendly requirements → $(REQ_KAGGLE)"
	$(POETRY) export -f requirements.txt --without-hashes | \
		grep -vE '^(torch|torchvision|torchaudio|torch-geometric)(==|>=)' > $(REQ_KAGGLE)
	@echo ">>> Done."

export-freeze:
	@echo ">>> Freezing active env → $(REQ_FREEZE)"
	$(PIP) freeze -q > $(REQ_FREEZE)
	@echo ">>> Wrote $(REQ_FREEZE)"

install-core:
	$(PIP) install -r $(REQ_CORE)

install-extras:
	@if [ -f "$(REQ_EXTRAS)" ]; then $(PIP) install -r $(REQ_EXTRAS); else echo "::warning::$(REQ_EXTRAS) not found"; fi

install-dev:
	$(PIP) install -r $(REQ_DEV)

install-kaggle:
	$(PIP) install -r $(REQ_KAGGLE)

# ========= CLI utilities (reproducibility) =========
env-capture:
	$(CLI) env-capture

hash-config:
	$(CLI) hash-config

# ========= Security / Docs (optional) =========
pip-audit:
	@echo ">>> pip-audit (CVE scan)"
	@if ! command -v pip-audit >/dev/null 2>&1; then $(PIP) install pip-audit; fi
	pip-audit -r $(REQ_CORE) || true

audit: pip-audit

docs-serve:
	@if ! command -v mkdocs >/dev/null 2>&1; then echo "$(YLW)MkDocs not installed (pip install mkdocs mkdocs-material)$(RST)"; exit 1; fi
	mkdocs serve

docs-build:
	@if ! command -v mkdocs >/dev/null 2>&1; then echo "$(YLW)MkDocs not installed (pip install mkdocs mkdocs-material)$(RST)"; exit 1; fi
	mkdocs build

# ========= PyTorch Geometric helper (Kaggle/Local) =========
kaggle-pyg-index:
	@$(PYTHON) - << 'PY'
import torch
ver = torch.__version__.split('+')[0]
cu  = (torch.version.cuda or 'cpu').replace('.','')
base = f"https://data.pyg.org/whl/torch-{ver}+{'cu'+cu if torch.version.cuda else 'cpu'}.html"
print(base)
PY

pyg-install:
	@echo ">>> Installing torch-geometric matching the current torch/CUDA"
	@PYG_INDEX="$$( $(MAKE) --no-print-directory kaggle-pyg-index )"; \
	echo "Using index: $$PYG_INDEX"; \
	$(PIP) install torch-geometric==2.5.3 -f "$$PYG_INDEX"

# ========= Cleanup =========
clean:
	rm -rf "$(OUT_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

cache-clean:
	@echo ">>> Cleaning caches and logs"
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + || true
	rm -rf .pytest_cache .ruff_cache .mypy_cache .dvc/tmp || true
	find $(LOGS_DIR) -type f -name "*.log" -delete 2>/dev/null || true

realclean: clean cache-clean
	rm -rf .dvc/cache

distclean: realclean
	@echo ">>> Removing Poetry caches and local venv (this is a full reset)"
	rm -rf .venv
	rm -rf ~/.cache/pypoetry || true
	rm -rf ~/.cache/pip || true