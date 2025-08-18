# ======================================================================
# SpectraMind V50 — Master Makefile
# ----------------------------------------------------------------------
# Provides convenience targets for training, inference, calibration,
# diagnostics, submission packaging, and version bumping.
# ======================================================================

# === Variables ===
PYTHON      := poetry run python
CLI         := poetry run spectramind
OUT_DIR     := outputs
SUBMISSION  := $(OUT_DIR)/submission.csv
REPORT_HTML := $(OUT_DIR)/diagnostics/report.html

# === Phony targets ===
.PHONY: all e2e train predict calibrate diagnose submit \
        selftest clean deep-clean ci bump-patch bump-minor bump-major

# ----------------------------------------------------------------------
# End-to-End pipeline (selftest → train → predict → calibrate → report)
# ----------------------------------------------------------------------
e2e:
	@echo ">>> [E2E] Starting full pipeline"
	$(CLI) selftest --fast
	$(CLI) train
	$(CLI) predict --out-csv $(SUBMISSION)
	$(PYTHON) scripts/validate_submission.py $(SUBMISSION)
	$(CLI) calibrate
	$(CLI) diagnose dashboard --html-out $(REPORT_HTML)
	@echo ">>> [E2E] Complete. Artifacts in $(OUT_DIR)/"

# ----------------------------------------------------------------------
# Individual pipeline stages
# ----------------------------------------------------------------------
train:
	$(CLI) train

predict:
	$(CLI) predict --out-csv $(SUBMISSION)

calibrate:
	$(CLI) calibrate

diagnose:
	$(CLI) diagnose dashboard --html-out $(REPORT_HTML)

submit:
	$(CLI) submit --bundle --validate

selftest:
	$(CLI) selftest --deep

# ----------------------------------------------------------------------
# Repo hygiene
# ----------------------------------------------------------------------
clean:
	rm -rf $(OUT_DIR)/*.csv $(OUT_DIR)/*.zip $(OUT_DIR)/*.json \
	       $(OUT_DIR)/*.html logs/*.log .pytest_cache .mypy_cache

deep-clean: clean
	rm -rf $(OUT_DIR) logs hydra multirun .dvc/cache

# ----------------------------------------------------------------------
# CI entrypoint (lightweight pipeline)
# ----------------------------------------------------------------------
ci:
	@echo ">>> [CI] Smoke test"
	$(CLI) selftest --fast
	$(CLI) train --dry-run
	$(CLI) predict --dry-run

# ----------------------------------------------------------------------
# Version bump (semver)
# ----------------------------------------------------------------------
bump-patch:
	@./scripts/bump_version.sh patch

bump-minor:
	@./scripts/bump_version.sh minor

bump-major:
	@./scripts/bump_version.sh major