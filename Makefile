# === Convenience targets ===
.PHONY: e2e bump-patch bump-minor bump-major

e2e:
	@echo ">>> E2E: train → predict → validate → calibrate → report"
	@poetry run asa train
	@poetry run asa predict
	@poetry run python scripts/validate_submission.py outputs/submission.csv
	@poetry run python -m asa.calib.temperature
	@poetry run python -m asa.diagnostics.report
	@echo "Artifacts in outputs/: submission.csv, submission_calibrated.csv, report.html"

bump-patch:
	@./scripts/bump_version.sh patch

bump-minor:
	@./scripts/bump_version.sh minor

bump-major:
	@./scripts/bump_version.sh major
