# ==============================================================================
# SpectraMind V50 — Master Makefile (Dev/Local)
# Neuro‑Symbolic, Physics‑Informed AI Pipeline
# ==============================================================================
# Philosophy:
#   • CLI-first → all tasks through Typer CLI (`spectramind`)
#   • Hydra-backed overrides via OVERRIDES
#   • Parity helpers → local benchmark + Kaggle shims
#   • Dev ergonomics → fmt / lint / test / analyze-log
#
# Quickstart:
#   make selftest
#   make calibrate
#   make train DEVICE=gpu EPOCHS=2
#   make predict
#   make diagnose
#   make submit
#   make benchmark DEVICE=gpu
# ==============================================================================

# ========= Shell =========
SHELL       := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -euo pipefail -c

# ========= Tooling =========
PYTHON      ?= python3
POETRY      ?= poetry
CLI         ?= $(POETRY) run spectramind

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

OVERRIDES   ?=
EXTRA_ARGS  ?=

# ========= PHONY =========
.PHONY: help init env info \
        fmt lint test \
        selftest selftest-deep validate-env \
        calibrate calibrate-temp corel-train \
        train predict predict-e2e diagnose ablate submit \
        analyze-log check-cli-map open-report \
        dvc-pull dvc-push \
        benchmark bench-selftest benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean \
        kaggle-run kaggle-submit \
        ablate-light ablate-heavy ablate-grid ablate-optuna \
        clean realclean

# ========= Help =========
help:
	@echo "SpectraMind V50 — Make targets"
	@echo "  selftest | calibrate | train | predict | diagnose | submit"
	@echo "  benchmark | kaggle-run | kaggle-submit"
	@echo "  fmt | lint | test | analyze-log | validate-env | open-report"
	@echo "  ablate-light | ablate-heavy | ablate-grid | ablate-optuna"
	@echo "Vars: DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OUT_DIR=$(OUT_DIR) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'"

# ========= Init =========
init: env
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

info:
	@echo "python : $$($(PYTHON) --version 2>&1)"
	@echo "poetry : $$($(POETRY) --version 2>&1 || true)"
	@echo "cli    : $(CLI)"
	@echo "device : $(DEVICE)"

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

# convenience: ensure we actually produced a csv (CI smoke)
predict-e2e: predict
	@test -f "$(PRED_DIR)/submission.csv" && echo "OK: $(PRED_DIR)/submission.csv" || (echo "Missing submission.csv"; exit 1)

diagnose: init
	$(CLI) diagnose smoothness --outdir "$(DIAG_DIR)" $(EXTRA_ARGS)
	# try lightweight dashboard first (no UMAP/TSNE), fall back to full
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || true

# tries to open latest diagnostics report (no-op on headless CI)
open-report:
	@latest=$$(ls -t $(DIAG_DIR)/*.html 2>/dev/null | head -n1 || true); \
	if [ -n "$$latest" ]; then \
	  echo "Opening $$latest"; \
	  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$$latest" || true; \
	  elif command -v open >/dev/null 2>&1; then open "$$latest" || true; \
	  else echo "No opener found (CI/headless)"; fi; \
	else echo "No diagnostics HTML found."; fi

ablate: init
	$(CLI) ablate $(OVERRIDES) $(EXTRA_ARGS)

# ========= Ablation shortcuts (profiles & sweep styles) =========
# Profiles assume you’ve added:
#   configs/ablation/ablation_light.yaml
#   configs/ablation/ablation_heavy.yaml
# Sweep styles assume upgraded ablation.yml with search_spaces.

ablate-light: init
	$(CLI) ablate ablation=ablation_light $(EXTRA_ARGS)

ablate-heavy: init
	$(CLI) ablate ablation=ablation_heavy $(EXTRA_ARGS)

# quick grid across v50_fast_grid, using 'light' profile for speed
ablate-grid: init
	$(CLI) ablate -m ablate.sweeper=basic +ablate.search=v50_fast_grid ablation=ablation_light $(EXTRA_ARGS)

# optuna TPE across v50_symbolic_core (deeper search), heavy profile
ablate-optuna: init
	$(CLI) ablate -m ablate.sweeper=optuna +ablate.search=v50_symbolic_core ablation=ablation_heavy $(EXTRA_ARGS)

submit: init
	mkdir -p "$(SUBMIT_DIR)"
	$(CLI) submit --zip-out "$(SUBMIT_ZIP)" $(EXTRA_ARGS)

analyze-log: init
	$(CLI) analyze-log --md "$(OUT_DIR)/log_table.md" --csv "$(OUT_DIR)/log_table.csv" $(EXTRA_ARGS)

check-cli-map:
	$(CLI) check-cli-map

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
