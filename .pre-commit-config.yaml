SHELL := /usr/bin/env bash
.ONESHELL:
.SHELLFLAGS := -eo pipefail -c

======================================================================

SpectraMind V50 — Master Makefile

–––––––––––––––––––––––––––––––––––

Provides convenience targets for training, inference, calibration,

diagnostics, submission packaging, CI smoke tests, and version bumping.

======================================================================

=== Defaults ===

.DEFAULT_GOAL := help

=== Variables ===

PYTHON          := poetry run python
PIP             := poetry run pip
POETRY          := poetry
CLI             := poetry run spectramind
OUT_DIR         := outputs
LOGS_DIR        := logs
DIAG_DIR        := $(OUT_DIR)/diagnostics
SUBMISSION      := $(OUT_DIR)/submission.csv
REPORT_HTML     := $(DIAG_DIR)/report.html
CONFIG          := configs/config_v50.yaml
TIMESTAMP       := $(shell date +”%Y%m%d_%H%M%S”)
RUN_HASH_FILE   := run_hash_summary_v50.json

Colors for pretty output

BOLD:=\033[1m
DIM :=\033[2m
RED :=\033[31m
GRN :=\033[32m
YEL :=\033[33m
BLU :=\033[34m
RST :=\033[0m

Ensure required dirs exist before most targets

REQUIRED_DIRS := $(OUT_DIR) $(LOGS_DIR) $(DIAG_DIR)

=== Phony targets ===

.PHONY: help init env lock fmt lint test precommit install-hooks 
all e2e train predict calibrate diagnose submit selftest 
open-report analyze-log ci clean deep-clean clean-submission 
bump-patch bump-minor bump-major docker-build docker-run dvc-pull

–––––––––––––––––––––––––––––––––––

Help

–––––––––––––––––––––––––––––––––––

help:
@echo “”
@echo “$(BOLD)SpectraMind V50 — Make targets$(RST)”
@echo “  $(BOLD)make e2e$(RST)            : Selftest → Train → Predict → Calibrate → Dashboard”
@echo “  $(BOLD)make train/predict/…$(RST) : Run individual pipeline stages”
@echo “  $(BOLD)make fmt lint test$(RST)   : Code quality (ruff/black/isort/mypy/pytest)”
@echo “  $(BOLD)make submit$(RST)          : Validate and bundle submission”
@echo “  $(BOLD)make ci$(RST)              : CI smoke (selftest + dry-runs)”
@echo “  $(BOLD)make init$(RST)            : Poetry install + pre-commit hooks”
@echo “  $(BOLD)make open-report$(RST)     : Open latest diagnostics HTML”
@echo “  $(BOLD)make clean/deep-clean$(RST): Remove artifacts and caches”
@echo “  $(BOLD)make bump-*(patch|minor|major)$(RST): Version bumps via scripts/bump_version.sh”
@echo “”

–––––––––––––––––––––––––––––––––––

Project setup

–––––––––––––––––––––––––––––––––––

init: $(REQUIRED_DIRS)
@echo “$(BLU)[INIT]$(RST) Installing dependencies via Poetry”
$(POETRY) install –no-interaction
@$(MAKE) install-hooks

env:
@echo “$(BLU)[ENV]$(RST) Exporting minimal env diagnostics”
@python3 -c ‘import sys,platform;print(“python”,sys.version);print(“platform”,platform.platform())’
@$(POETRY) –version || true
@nvidia-smi || true

lock:
@echo “$(BLU)[LOCK]$(RST) Refreshing poetry.lock”
$(POETRY) lock –no-update
$(POETRY) install –no-interaction

install-hooks:
@echo “$(BLU)[GIT]$(RST) Installing pre-commit hooks”
$(POETRY) run pre-commit install -t pre-commit -t pre-push || true

–––––––––––––––––––––––––––––––––––

Code quality

–––––––––––––––––––––––––––––––––––

fmt:
@echo “$(BLU)[FMT]$(RST) isort → black → ruff”
$(POETRY) run isort src tests
$(POETRY) run black src tests
$(POETRY) run ruff check –fix src tests

lint:
@echo “$(BLU)[LINT]$(RST) ruff + mypy”
$(POETRY) run ruff check src tests
$(POETRY) run mypy src

test:
@echo “$(BLU)[TEST]$(RST) Running pytest”
$(POETRY) run pytest

precommit:
@echo “$(BLU)[PRE-COMMIT]$(RST) Running all hooks on repository”
$(POETRY) run pre-commit run –all-files

–––––––––––––––––––––––––––––––––––

End-to-End pipeline (selftest → train → predict → validate → calibrate → report)

–––––––––––––––––––––––––––––––––––

e2e: $(REQUIRED_DIRS)
@echo “$(GRN)[E2E]$(RST) Starting full pipeline”
$(CLI) selftest –fast
$(CLI) train –config $(CONFIG)
$(CLI) predict –out-csv $(SUBMISSION) –config $(CONFIG)
@echo “$(BLU)[VALIDATE]$(RST) Validating submission schema”
@if [ -f scripts/validate_submission.py ]; then 
$(PYTHON) scripts/validate_submission.py $(SUBMISSION); 
else 
echo “$(YEL)[WARN]$(RST) scripts/validate_submission.py not found — skipping”; 
fi
$(CLI) calibrate –config $(CONFIG)
@mkdir -p $(DIAG_DIR)
$(CLI) diagnose dashboard –html-out $(REPORT_HTML) –config $(CONFIG)
@echo “$(GRN)[E2E]$(RST) Complete. Artifacts in $(OUT_DIR)/”

–––––––––––––––––––––––––––––––––––

Individual pipeline stages

–––––––––––––––––––––––––––––––––––

train: $(REQUIRED_DIRS)
$(CLI) train –config $(CONFIG)

predict: $(REQUIRED_DIRS)
$(CLI) predict –out-csv $(SUBMISSION) –config $(CONFIG)

calibrate: $(REQUIRED_DIRS)
$(CLI) calibrate –config $(CONFIG)

diagnose: $(REQUIRED_DIRS)
@mkdir -p $(DIAG_DIR)
$(CLI) diagnose dashboard –html-out $(REPORT_HTML) –config $(CONFIG)

submit: $(REQUIRED_DIRS)
$(CLI) submit –bundle –validate –config $(CONFIG)

selftest: $(REQUIRED_DIRS)
$(CLI) selftest –deep

open-report:
@if [ -f “$(REPORT_HTML)” ]; then 
echo “$(BLU)[OPEN]$(RST) Opening $(REPORT_HTML)”; 
(xdg-open “$(REPORT_HTML)” || open “$(REPORT_HTML)” || true) >/dev/null 2>&1; 
else 
echo “$(RED)[ERROR]$(RST) $(REPORT_HTML) not found. Run ‘make diagnose’ first.”; 
exit 1; 
fi

analyze-log:
@if [ -f “v50_debug_log.md” ]; then 
$(CLI) analyze-log –md-out $(OUT_DIR)/log_table.md –csv-out $(OUT_DIR)/log_table.csv; 
echo “$(GRN)[ANALYZE]$(RST) Wrote $(OUT_DIR)/log_table.{md,csv}”; 
else 
echo “$(YEL)[WARN]$(RST) v50_debug_log.md not found — skipping”; 
fi

–––––––––––––––––––––––––––––––––––

CI entrypoint (lightweight pipeline)

–––––––––––––––––––––––––––––––––––

ci: $(REQUIRED_DIRS)
@echo “$(BLU)[CI]$(RST) Smoke test”
$(CLI) selftest –fast
$(CLI) train –dry-run –config $(CONFIG)
$(CLI) predict –dry-run –config $(CONFIG)

–––––––––––––––––––––––––––––––––––

Data / DVC helpers

–––––––––––––––––––––––––––––––––––

dvc-pull:
@if command -v dvc >/dev/null 2>&1; then 
echo “$(BLU)[DVC]$(RST) Pulling tracked data”; 
dvc pull; 
else 
echo “$(YEL)[WARN]$(RST) dvc not installed — skipping dvc pull”; 
fi

–––––––––––––––––––––––––––––––––––

Docker helpers (optional)

–––––––––––––––––––––––––––––––––––

docker-build:
@echo “$(BLU)[DOCKER]$(RST) Building image spectramindv50:dev”
docker build -t spectramindv50:dev .

docker-run:
@echo “$(BLU)[DOCKER]$(RST) Running dev shell (GPU if available)”
docker run –gpus all -it –rm -v $$PWD:/workspace spectramindv50:dev bash

–––––––––––––––––––––––––––––––––––

Repo hygiene

–––––––––––––––––––––––––––––––––––

clean:
@echo “$(BLU)[CLEAN]$(RST) Removing common artifacts”
rm -rf $(OUT_DIR)/.csv $(OUT_DIR)/.zip $(OUT_DIR)/.json 
$(OUT_DIR)/.html $(LOGS_DIR)/*.log .pytest_cache .mypy_cache

clean-submission:
@echo “$(BLU)[CLEAN]$(RST) Removing submission artifacts”
rm -f $(OUT_DIR)/submission*.csv $(OUT_DIR)/submission*.zip

deep-clean: clean
@echo “$(BLU)[DEEP-CLEAN]$(RST) Removing outputs/, logs/, hydra, multirun, .dvc/cache”
rm -rf $(OUT_DIR) $(LOGS_DIR) hydra multirun .dvc/cache

–––––––––––––––––––––––––––––––––––

Version bump (semver)

–––––––––––––––––––––––––––––––––––

bump-patch:
@./scripts/bump_version.sh patch

bump-minor:
@./scripts/bump_version.sh minor

bump-major:
@./scripts/bump_version.sh major

–––––––––––––––––––––––––––––––––––

Internal: ensure required dirs

–––––––––––––––––––––––––––––––––––

$(REQUIRED_DIRS):
@mkdir -p $(REQUIRED_DIRS)
@echo “$(DIM)[MKDIR] Ensured: $(REQUIRED_DIRS)$(RST)”