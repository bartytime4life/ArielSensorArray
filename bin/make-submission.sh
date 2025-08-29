#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — make-submission.sh
# ------------------------------------------------------------------------------
# Purpose:
#   Orchestrates the pipeline to create a Kaggle-ready submission bundle.
#   Wraps around the `spectramind` CLI: predict → validate → package.
#
# Usage:
#   ./bin/make-submission.sh [options]
#
# Options:
#   --dry-run        Run without producing final bundle (default: true).
#   --open           Open output directory after completion.
#   --tag <string>   Add a version tag to the submission bundle.
#
# Logging:
#   Every call is logged into logs/v50_debug_log.md with timestamp + config hash.
# ==============================================================================

set -euo pipefail

# Defaults
DRY_RUN=true
OPEN_AFTER=false
TAG=""

# Parse args
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true; shift ;;
    --open) OPEN_AFTER=true; shift ;;
    --tag) TAG="$2"; shift 2 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

# Ensure logs dir exists
mkdir -p logs outputs submissions

echo "[make-submission] Starting pipeline at $(date -Iseconds)" | tee -a logs/v50_debug_log.md

# 1. Self-test
spectramind test --fast || { echo "❌ Selftest failed"; exit 1; }

# 2. Run prediction
spectramind predict --config configs/config_v50.yaml --out outputs/predictions.csv

# 3. Validate predictions
spectramind validate --input outputs/predictions.csv

# 4. Package submission
CMD="spectramind bundle --pred outputs/predictions.csv --out submissions/"
if [ -n "$TAG" ]; then
  CMD="$CMD --tag $TAG"
fi

if [ "$DRY_RUN" = true ]; then
  echo "[make-submission] Dry run: $CMD"
else
  echo "[make-submission] Executing: $CMD"
  eval "$CMD"
fi

# 5. Open output dir (optional)
if [ "$OPEN_AFTER" = true ]; then
  open submissions/ || xdg-open submissions/ || true
fi

echo "[make-submission] Completed at $(date -Iseconds)" | tee -a logs/v50_debug_log.md
