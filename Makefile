# ------------------------------------------------------------------------------
# SpectraMind V50 — Makefile (CLI-first, Hydra-backed, CI-parity)
# NeurIPS 2025 Ariel Data Challenge
#
# Quick usage:
#   make help
#   make selftest             # fast integrity check
#   make train                # train with Hydra overrides
#   make predict              # run inference and write artifacts
#   make diagnose             # generate diagnostics dashboard + smoothness checks
#   make submit               # package submission bundle
#   make benchmark            # local benchmark (CPU by default)
#
# Override examples:
#   make train DEVICE=gpu EPOCHS=2 OVERRIDES='+data.split=toy'
#   make diagnose EXTRA_ARGS='--open' OVERRIDES='+diagnostics.light=true'
#
# Notes:
# - All commands route through the Typer CLI: `spectramind`
# - Hydra overrides can be appended via OVERRIDES (quoted)
# - Parity with CI is maintained for selftest, minimal train, diagnose, and artifact paths
# ------------------------------------------------------------------------------

# ========= Global settings =========
SHELL         := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS   := -euo pipefail -c

# ========= Tooling =========
PYTHON        ?= python3
POETRY        ?= poetry
CLI           ?= $(POETRY) run spectramind

# ========= Default knobs (override with make VAR=value) =========
DEVICE        ?= cpu               # cpu|gpu (or pass auto if your CLI supports it)
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

# ========= PHONY targets =========
.PHONY: help init env info \
        fmt lint test \
        selftest selftest-deep \
        calibrate calibrate-temp corel-train \
        train predict diagnose ablate submit \
        analyze-log check-cli-map \
        dvc-pull dvc-push \
        benchmark benchmark-cpu benchmark-gpu benchmark-run benchmark-report benchmark-clean bench-selftest \
        clean realclean

# ========= Help =========
help:
	@echo "SpectraMind V50 — Make targets"
	@echo ""
	@echo "Dev / quality:"
	@echo "  make fmt                # run Black/isort (format code)"
	@echo "  make lint               # run Ruff lint"
	@echo "  make test               # run pytest (logs -> $(LOGS_DIR)/test.log)"
	@echo ""
	@echo "Pipeline (CLI-first):"
	@echo "  make selftest           # fast integrity & wiring check"
	@echo "  make selftest-deep      # deep selftest (CUDA/DVC checks if available)"
	@echo "  make calibrate          # run calibration kill chain on raw data"
	@echo "  make calibrate-temp     # temperature scaling"
	@echo "  make corel-train        # COREL conformal training"
	@echo "  make train              # train model (Hydra overrides via OVERRIDES)"
	@echo "  make predict            # inference -> $(PRED_DIR)"
	@echo "  make diagnose           # diagnostics (smoothness + dashboard)"
	@echo "  make ablate             # run ablations"
	@echo "  make submit             # package submission -> $(SUBMIT_ZIP)"
	@echo "  make analyze-log        # summarize CLI debug log"
	@echo "  make check-cli-map      # validate command-to-file mapping"
	@echo ""
	@echo "Benchmarks (local parity with CI):"
	@echo "  make benchmark          # CPU by default (see DEVICE)"
	@echo "  make benchmark-cpu      # force CPU"
	@echo "  make benchmark-gpu      # force GPU"
	@echo "  make benchmark-report   # aggregate summaries -> aggregated/report.md"
	@echo "  make benchmark-clean    # remove local benchmark artifacts"
	@echo ""
	@echo "Data:"
	@echo "  make dvc-pull           # pull DVC-tracked artifacts"
	@echo "  make dvc-push           # push DVC-tracked artifacts"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              # remove outputs/diagnostics artifacts"
	@echo "  make realclean          # plus cache dirs (.pytest_cache, .ruff_cache, etc.)"
	@echo ""
	@echo "Variables (override via CLI):"
	@echo "  DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OUT_DIR=$(OUT_DIR)"
	@echo "  OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'"

# ========= Environment bootstrap =========
init: env
env:
	mkdir -p "$(OUT_DIR)" "$(LOGS_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

info:
	@echo "python : $$($(PYTHON) --version 2>&1)"
	@echo "poetry : $$($(POETRY) --version 2>&1 || true)"
	@echo "cli    : $(CLI)"
	@echo "device : $(DEVICE)"

# ========= Dev / quality =========
fmt:
	@echo ">>> Formatting with isort + black"
	$(POETRY) run isort .
	$(POETRY) run black .

lint:
	@echo ">>> Linting with ruff"
	$(POETRY) run ruff check .

test: init
	@echo ">>> Running pytest"
	$(POETRY) run pytest -q || $(POETRY) run pytest -q -x
	@echo ">>> Tests complete"

# ========= Pipeline (CLI-first) =========
selftest: init
	@echo ">>> Selftest (fast)"
	$(CLI) selftest --fast

selftest-deep: init
	@echo ">>> Selftest (deep)"
	$(CLI) selftest --deep

calibrate: init
	@echo ">>> Calibrate"
	$(CLI) calibrate --outdir "$(OUT_DIR)" $(EXTRA_ARGS)

calibrate-temp: init
	@echo ">>> Temperature scaling"
	$(CLI) calibrate-temp --outdir "$(OUT_DIR)" $(EXTRA_ARGS)

corel-train: init
	@echo ">>> COREL conformal training"
	$(CLI) corel-train --outdir "$(OUT_DIR)" $(EXTRA_ARGS)

train: init
	@echo ">>> Train"
	@echo "    DEVICE    : $(DEVICE)"
	@echo "    EPOCHS    : $(EPOCHS)"
	@echo "    OVERRIDES : $(OVERRIDES)"
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) --outdir "$(OUT_DIR)" $(EXTRA_ARGS)

predict: init
	@echo ">>> Predict"
	$(CLI) predict --outdir "$(PRED_DIR)" $(EXTRA_ARGS)

diagnose: init
	@echo ">>> Diagnostics — smoothness + dashboard"
	$(CLI) diagnose smoothness --outdir "$(DIAG_DIR)" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$(DIAG_DIR)" $(EXTRA_ARGS) || true

ablate: init
	@echo ">>> Ablations"
	$(CLI) ablate --outdir "$(OUT_DIR)" $(EXTRA_ARGS)

submit: init
	@echo ">>> Submission packaging"
	$(CLI) submit --zip-out "$(SUBMIT_ZIP)" $(EXTRA_ARGS)
	@echo ">>> Submission bundle at: $(SUBMIT_ZIP)"

analyze-log: init
	@echo ">>> Analyze CLI debug log"
	$(CLI) analyze-log --md "$(OUT_DIR)/log_table.md" --csv "$(OUT_DIR)/log_table.csv" $(EXTRA_ARGS)
	@echo ">>> Reports: $(OUT_DIR)/log_table.md, $(OUT_DIR)/log_table.csv"

check-cli-map:
	@echo ">>> Check CLI command-to-file mapping"
	$(CLI) check-cli-map $(EXTRA_ARGS)

# ========= Data via DVC =========
dvc-pull:
	@echo ">>> DVC pull"
	dvc pull || true

dvc-push:
	@echo ">>> DVC push"
	dvc push || true

# ========= Benchmarks (local parity with CI) =========
# Benchmarks preserve the original user flow, with light polish for parity & summaries.
bench-selftest:
	@echo ">>> Running pipeline selftest (fast)..."
	$(CLI) selftest --fast

benchmark: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-cpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=cpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-gpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=gpu EPOCHS=$(EPOCHS) OVERRIDES='$(OVERRIDES)' EXTRA_ARGS='$(EXTRA_ARGS)'

benchmark-run:
	@set -euo pipefail
	OUTDIR="benchmarks/$(TS)_$(DEVICE)"
	mkdir -p "$$OUTDIR"
	@echo ">>> Benchmark start"
	@echo "    DEVICE     : $(DEVICE)"
	@echo "    EPOCHS     : $(EPOCHS)"
	@echo "    OUTDIR     : $$OUTDIR"
	@echo "    OVERRIDES  : $(OVERRIDES)"
	@echo "    EXTRA_ARGS : $(EXTRA_ARGS)"

	# Training (minimal epochs with Hydra overrides)
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) --outdir "$$OUTDIR" $(EXTRA_ARGS)

	# Diagnostics (smoothness + dashboard) to mirror CI artifacts
	$(CLI) diagnose smoothness --outdir "$$OUTDIR" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$$OUTDIR" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$$OUTDIR" $(EXTRA_ARGS) || true

	# Summarize environment + timing
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

	@echo ">>> Benchmark complete"
	@echo "    Summary: $$OUTDIR/summary.txt"

benchmark-report:
	@echo ">>> Aggregating benchmark summaries"
	mkdir -p aggregated
	{ \
	  echo "# SpectraMind V50 Benchmark Report"; \
	  echo ""; \
	  for f in $$(find benchmarks -type f -name summary.txt | sort); do \
	    echo "## $$f"; echo ""; cat "$$f"; echo ""; \
	  done; \
	} > aggregated/report.md
	@echo ">>> Aggregated: aggregated/report.md"

benchmark-clean:
	@echo ">>> Removing local benchmark artifacts (benchmarks/ and aggregated/)"
	rm -rf benchmarks aggregated

# ========= Cleanup =========
clean:
	@echo ">>> Cleaning outputs"
	rm -rf "$(OUT_DIR)" "$(DIAG_DIR)" "$(PRED_DIR)" "$(SUBMIT_DIR)"

realclean: clean
	@echo ">>> Cleaning caches"
	rm -rf .pytest_cache .ruff_cache .mypy_cache