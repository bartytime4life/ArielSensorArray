# ==============================================================================
# SpectraMind V50 — Master Makefile
# Neuro-Symbolic, Physics-Informed AI Pipeline
# NeurIPS 2025 Ariel Data Challenge
#
# Philosophy:
#   • CLI-first → all tasks routed through Typer CLI (`spectramind`)
#   • Hydra-backed configs → reproducibility, override via OVERRIDES
#   • CI/Kaggle parity → same commands locally, in CI, and on Kaggle
#   • Scientific rigor → calibration, training, diagnostics, ablations
#
# Quickstart:
#   make selftest            # verify wiring
#   make calibrate           # raw → calibrated
#   make train               # train model
#   make predict             # run inference
#   make diagnose            # build diagnostics dashboard
#   make submit              # bundle outputs
#   make benchmark           # local Kaggle-like run
#
# Variables override examples:
#   make train DEVICE=gpu EPOCHS=2 OVERRIDES='+data.split=toy'
#   make diagnose EXTRA_ARGS='--open' OVERRIDES='+diagnostics.light=true'
# ==============================================================================

# ========= Global =========
SHELL         := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS   := -euo pipefail -c

# ========= Tooling =========
PYTHON        ?= python3
POETRY        ?= poetry
CLI           ?= $(POETRY) run spectramind

# ========= Defaults (override at CLI) =========
DEVICE        ?= cpu
EPOCHS        ?= 1
TS            := $(shell date +%Y%m%d_%H%M%S)
OUT_DIR       ?= outputs
LOGS_DIR      ?= logs
DIAG_DIR      ?= $(OUT_DIR)/diagnostics
PRED_DIR      ?= $(OUT_DIR)/predictions
SUBMIT_DIR    ?= $(OUT_DIR)/submission
SUBMIT_ZIP    ?= $(SUBMIT_DIR)/bundle.zip
OVERRIDES     ?=
EXTRA_ARGS    ?=

# ========= PHONY =========
.PHONY: help init env info \
        fmt lint test \
        selftest selftest-deep \
        calibrate calibrate-temp corel-train \
        train predict diagnose ablate submit \
        analyze-log check-cli-map \
        dvc-pull dvc-push \
        benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean bench-selftest \
        kaggle-run kaggle-submit \
        clean realclean

# ========= Help =========
help:
	@echo "SpectraMind V50 — CLI targets"
	@echo "make selftest          # fast integrity check"
	@echo "make train             # train model"
	@echo "make predict           # inference → $(PRED_DIR)"
	@echo "make diagnose          # diagnostics dashboard"
	@echo "make submit            # package submission"
	@echo "make benchmark         # simulate Kaggle run"
	@echo "make kaggle-run        # run inside Kaggle env"
	@echo "make kaggle-submit     # push to Kaggle competition"
	@echo "make dvc-pull/push     # sync DVC artifacts"
	@echo "make clean/realclean   # remove artifacts/caches"

# ========= Init =========
init: env
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

info:
	@echo "python  : $$($(PYTHON) --version 2>&1)"
	@echo "poetry  : $$($(POETRY) --version 2>&1 || true)"
	@echo "cli     : $(CLI)"
	@echo "device  : $(DEVICE)"

# ========= Dev / Quality =========
fmt:
	$(POETRY) run isort .
	$(POETRY) run black .

lint:
	$(POETRY) run ruff check .

test: init
	$(POETRY) run pytest -q || $(POETRY) run pytest -q -x

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
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv" $(OVERRIDES) $(EXTRA_ARGS)

diagnose: init
	$(CLI) diagnose smoothness --outdir "$(DIAG_DIR)" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --outdir "$(DIAG_DIR)" $(EXTRA_ARGS)

ablate: init
	$(CLI) ablate $(OVERRIDES) $(EXTRA_ARGS)

submit: init
	$(CLI) submit --zip-out "$(SUBMIT_ZIP)" $(EXTRA_ARGS)

analyze-log: init
	$(CLI) analyze-log --md "$(OUT_DIR)/log_table.md" --csv "$(OUT_DIR)/log_table.csv" $(EXTRA_ARGS)

check-cli-map:
	$(CLI) check-cli-map

# ========= Data via DVC =========
dvc-pull:
	dvc pull || true

dvc-push:
	dvc push || true

# ========= Benchmarks =========
bench-selftest:
	$(CLI) selftest --fast

benchmark: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=$(DEVICE)

benchmark-cpu:
	@$(MAKE) --no-print-directory benchmark-run DEVICE=cpu

benchmark-gpu:
	@$(MAKE) --no-print-directory benchmark-run DEVICE=gpu

benchmark-run:
	OUTDIR="benchmarks/$(TS)_$(DEVICE)"
	mkdir -p "$$OUTDIR"
	$(CLI) train +training.epochs=$(EPOCHS) --device $(DEVICE) --outdir "$$OUTDIR" $(OVERRIDES)
	$(CLI) diagnose smoothness --outdir "$$OUTDIR"
	$(CLI) diagnose dashboard --outdir "$$OUTDIR" || true
	@echo "Benchmark complete → $$OUTDIR/summary.txt"

benchmark-report:
	mkdir -p aggregated
	{ \
	  echo "# Benchmark Report"; \
	  for f in $$(find benchmarks -name summary.txt | sort); do \
	    echo "## $$f"; echo ""; cat "$$f"; echo ""; \
	  done; \
	} > aggregated/report.md

benchmark-clean:
	rm -rf benchmarks aggregated

# ========= Kaggle Helpers =========
kaggle-run: init
	@echo ">>> Running inside Kaggle notebook"
	$(CLI) selftest --fast
	$(CLI) train +training.epochs=1 --device gpu --outdir "$(OUT_DIR)"
	$(CLI) predict --out-csv "$(PRED_DIR)/submission.csv"

kaggle-submit: kaggle-run
	kaggle competitions submit -c neurips-2025-ariel -f "$(PRED_DIR)/submission.csv" -m "SpectraMind V50 auto-submit"

# ========= Cleanup =========
clean:
	rm -rf "$(OUT_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

realclean: clean
	rm -rf .pytest_cache .ruff_cache .mypy_cache .dvc/cache