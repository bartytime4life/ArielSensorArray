# ==============================================================================
# SpectraMind V50 — Master Makefile (Dev/Local)
# Neuro‑Symbolic, Physics‑Informed AI Pipeline
# ==============================================================================

# ========= Shell =========
SHELL       := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c

# ========= Tooling =========
PYTHON      ?= python3
POETRY      ?= poetry
CLI         ?= $(POETRY) run spectramind

# Node (for mermaid-cli); keep overridable for CI/local differences
NODE        ?= node
NPM         ?= npm

# ========= Defaults (override at CLI) =========
DEVICE      ?= cpu
EPOCHS      ?= 1
TS          := $(shell date +%Y%m%d_%H%M%S)

OUT_DIR     ?= outputs
LOGS_DIR    ?= logs
DIAG_DIR    ?= $(OUT_DIR)/diagnostics
PRED_DIR    ?= $(OUT_DIR)/predictions
SUBMIT_DIR  ?= $(OUT_DIR)/submission
SUBMIT_ZIP  ?= $(SUBMIT_DIR)/bundle.zip

# Mermaid export defaults
DIAGRAMS_DIR      ?= docs/diagrams
MERMAID_FILES     ?= ARCHITECTURE.md README.md
MERMAID_THEME     ?=
MERMAID_EXPORT_PNG?= 0   # 1 to export PNG alongside SVG

OVERRIDES   ?=
EXTRA_ARGS  ?=

# ========= PHONY =========
.PHONY: help init env info \
        fmt lint test \
        selftest selftest-deep validate-env \
        calibrate calibrate-temp corel-train \
        train predict predict-e2e diagnose submit \
        ablate ablate-light ablate-heavy ablate-grid ablate-optuna \
        analyze-log analyze-log-short check-cli-map open-report \
        dvc-pull dvc-push \
        bench-selftest benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean \
        kaggle-run kaggle-submit \
        mermaid-init diagrams diagrams-png mermaid-export mermaid-clean \
        ci clean realclean distclean

# ========= Help =========
help:
	@echo "SpectraMind V50 — Make targets"
	@echo "  selftest | calibrate | train | predict | diagnose | submit"
	@echo "  ablate(-light|-heavy|-grid|-optuna)"
	@echo "  analyze-log | analyze-log-short | open-report"
	@echo "  diagrams | diagrams-png | mermaid-init | mermaid-clean"
	@echo "  benchmark | kaggle-run | kaggle-submit | ci"
	@echo "  fmt | lint | test | validate-env"
	@echo "  clean | realclean | distclean"
	@echo "Vars: DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OUT_DIR=$(OUT_DIR)"
	@echo "      OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'"
	@echo "      MERMAID_FILES='$(MERMAID_FILES)' DIAGRAMS_DIR='$(DIAGRAMS_DIR)'"
	@echo "      MERMAID_THEME='$(MERMAID_THEME)' MERMAID_EXPORT_PNG=$(MERMAID_EXPORT_PNG)"

# ========= Init =========
init: env
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

info:
	@echo "python : $$($(PYTHON) --version 2>&1)"
	@echo "poetry : $$($(POETRY) --version 2>&1 || true)"
	@echo "cli    : $(CLI)"
	@echo "device : $(DEVICE)"
	@echo "node   : $$($(NODE) --version 2>&1 || true)"
	@echo "npm    : $$($(NPM) --version 2>&1 || true)"

# ========= Dev / Quality =========
fmt:
	$(POETRY) run isort .
	$(POETRY) run black .

lint:
	$(POETRY) run ruff check .

test: init
	$(POETRY) run pytest -q || $(POETRY) run pytest -q -x

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

# ensure CSV exists (e2e smoke)
predict-e2e: predict
	@test -f "$(PRED_DIR)/submission.csv" && echo "OK: $(PRED_DIR)/submission.csv" || (echo "Missing submission.csv"; exit 1)

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

# Short CI-friendly summary: last 5 entries from CSV (auto-runs analyze-log if CSV missing)
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
# Install @mermaid-js/mermaid-cli via npm ci (using package.json at repo root)
mermaid-init:
	@if ! command -v $(NPM) >/dev/null 2>&1; then \
	  echo "ERROR: npm not found. Please install Node.js/npm."; exit 1; \
	fi
	$(NPM) ci
	mkdir -p "$(DIAGRAMS_DIR)"

# Render SVGs (and optionally PNGs with MERMAID_EXPORT_PNG=1)
diagrams: mermaid-init
	@echo ">>> Rendering Mermaid diagrams (SVG; PNG=$(MERMAID_EXPORT_PNG))"
	EXPORT_PNG=$(MERMAID_EXPORT_PNG) THEME="$(MERMAID_THEME)" \
	$(PYTHON) scripts/export_mermaid.py $(MERMAID_FILES)
	@echo ">>> Output → $(DIAGRAMS_DIR)"

# Convenience target: force PNG export alongside SVG
diagrams-png:
	@$(MAKE) --no-print-directory diagrams MERMAID_EXPORT_PNG=1

# Full export with explicit file list (override MERMAID_FILES on CLI)
mermaid-export:
	@$(MAKE) --no-print-directory diagrams

# Clean generated diagrams and temp
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
	$(CLI) selftest --fast

benchmark: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-cpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=cpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-gpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=gpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-run:
	OUTDIR="benchmarks/$(TS)_$(DEVICE)"
	mkdir -p "$$OUTDIR"
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) --outdir "$$OUTDIR" $(EXTRA_ARGS)
	$(CLI) diagnose smoothness --outdir "$$OUTDIR" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$$OUTDIR" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$$OUTDIR" $(EXTRA_ARGS) || true
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
	} > "$$OUTDIR/summary.txt"
	@echo ">>> Benchmark complete → $$OUTDIR/summary.txt"

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
	$(CLI) selftest --fast
	$(CLI) train +training.epochs=1 --device gpu --outdir "$(OUT_DIR)"
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv"

kaggle-submit: kaggle-run
	@echo ">>> Submitting to Kaggle competition"
	kaggle competitions submit -c neurips-2025-ariel -f "$(PRED_DIR)/submission.csv" -m "Spectramind V50 auto-submit"

# ========= Cleanup =========
clean:
	rm -rf "$(OUT_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

realclean: clean
	rm -rf .pytest_cache .ruff_cache .mypy_cache .dvc/tmp .dvc/cache

# Full reset: artifacts + caches + Poetry envs (LOCAL USE ONLY)
distclean: realclean
	@echo ">>> Removing Poetry caches and local venv (this is a full reset)"
	rm -rf .venv
	rm -rf ~/.cache/pypoetry || true
	rm -rf ~/.cache/pip || true