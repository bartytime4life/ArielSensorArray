# Makefile — Benchmarks wired to GitHub Actions workflow (local parity)
# SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
#
# Usage:
#   make benchmark                 # run CPU benchmark locally
#   make benchmark DEVICE=gpu      # run GPU benchmark locally
#   make benchmark-cpu             # force CPU
#   make benchmark-gpu             # force GPU
#   make benchmark-report          # aggregate summaries under aggregated/report.md
#   make benchmark-clean           # remove local benchmark artifacts
#
# Overrideables:
#   EPOCHS=1 OUTDIR=benchmarks/<auto> OVERRIDES="+benchmark=true"
#   CLI="poetry run spectramind"

.PHONY: help benchmark benchmark-cpu benchmark-gpu benchmark-run \
        benchmark-report benchmark-clean bench-selftest

# -------- Defaults (override via CLI: make VAR=value) -------------------------
PYTHON        ?= python3
POETRY        ?= poetry
CLI           ?= $(POETRY) run spectramind
DEVICE        ?= cpu                  # cpu|gpu
EPOCHS        ?= 1
TS            := $(shell date +%Y%m%d_%H%M%S)
OUTDIR        ?= benchmarks/$(TS)_$(DEVICE)
OVERRIDES     ?= +benchmark=true
EXTRA_ARGS    ?=
SHELL         := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS   := -euo pipefail -c

help:
	@echo "SpectraMind V50 — Benchmark targets"
	@echo "  make benchmark               # run CPU benchmark (default)"
	@echo "  make benchmark DEVICE=gpu    # run GPU benchmark"
	@echo "  make benchmark-cpu           # force CPU benchmark"
	@echo "  make benchmark-gpu           # force GPU benchmark"
	@echo "  make benchmark-report        # aggregate summaries into aggregated/report.md"
	@echo "  make benchmark-clean         # remove local benchmark artifacts"
	@echo ""
	@echo "Override EPOCHS, OUTDIR, OVERRIDES, CLI"
	@echo "Example:"
	@echo "  make benchmark DEVICE=gpu EPOCHS=2 OVERRIDES='+benchmark=true +data.split=toy'"

# -------- Sanity check (parity with CI selftest) ------------------------------
bench-selftest:
	@echo ">>> Running pipeline selftest (fast)..."
	$(CLI) test --fast

# -------- Unified benchmark entrypoint (parity with CI steps) -----------------
benchmark: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=$(DEVICE) EPOCHS=$(EPOCHS) OUTDIR=$(OUTDIR) OVERRIDES="$(OVERRIDES)" EXTRA_ARGS="$(EXTRA_ARGS)"

benchmark-cpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=cpu EPOCHS=$(EPOCHS) OUTDIR=$(OUTDIR) OVERRIDES="$(OVERRIDES)" EXTRA_ARGS="$(EXTRA_ARGS)"

benchmark-gpu: bench-selftest
	@$(MAKE) --no-print-directory benchmark-run DEVICE=gpu EPOCHS=$(EPOCHS) OUTDIR=$(OUTDIR) OVERRIDES="$(OVERRIDES)" EXTRA_ARGS="$(EXTRA_ARGS)"

benchmark-run:
	@echo ">>> Benchmark start"
	@echo "    DEVICE     : $(DEVICE)"
	@echo "    EPOCHS     : $(EPOCHS)"
	@echo "    OUTDIR     : $(OUTDIR)"
	@echo "    OVERRIDES  : $(OVERRIDES)"
	@echo "    EXTRA_ARGS : $(EXTRA_ARGS)"
	mkdir -p "$(OUTDIR)"

	# Training (minimal epochs with Hydra overrides)
	$(CLI) train +training.epochs=$(EPOCHS) $(OVERRIDES) --device $(DEVICE) --outdir "$(OUTDIR)" $(EXTRA_ARGS)

	# Diagnostics (smoothness + dashboard) to mirror CI artifacts
	$(CLI) diagnose smoothness --outdir "$(OUTDIR)" $(EXTRA_ARGS)
	$(CLI) diagnose dashboard --no-umap --no-tsne --outdir "$(OUTDIR)" $(EXTRA_ARGS) || \
	$(CLI) diagnose dashboard --outdir "$(OUTDIR)" $(EXTRA_ARGS) || true

	# Summarize environment + timing
	{ \
	  echo "Benchmark summary"; \
	  date; \
	  echo "python  : $$($(PYTHON) --version 2>&1)"; \
	  echo "poetry  : $$($(POETRY) --version 2>&1 || true)"; \
	  echo "device  : $(DEVICE)"; \
	  echo "epochs  : $(EPOCHS)"; \
	  echo "overrides: $(OVERRIDES)"; \
	  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true; \
	  echo ""; \
	  echo "Artifacts in $(OUTDIR):"; \
	  ls -lh "$(OUTDIR)" || true; \
	} > "$(OUTDIR)/summary.txt"

	@echo ">>> Benchmark complete"
	@echo "    Summary: $(OUTDIR)/summary.txt"

# -------- Aggregate reports (local analogue to CI aggregation) ----------------
benchmark-report:
	@echo ">>> Aggregating benchmark summaries"
	mkdir -p aggregated
	echo "# SpectraMind V50 Benchmark Report" > aggregated/report.md
	echo "" >> aggregated/report.md
	for f in $$(find benchmarks -type f -name summary.txt | sort); do \
	  echo "## $$f" >> aggregated/report.md; \
	  echo "" >> aggregated/report.md; \
	  cat "$$f" >> aggregated/report.md; \
	  echo "" >> aggregated/report.md; \
	done
	@echo ">>> Aggregated report at aggregated/report.md"

# -------- Cleanup -------------------------------------------------------------
benchmark-clean:
	@echo ">>> Removing local benchmark artifacts (benchmarks/ and aggregated/)"
	rm -rf benchmarks aggregated