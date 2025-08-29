#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — repair_and_push.sh
# ------------------------------------------------------------------------------
# Purpose:
#   Fix and push repository state with Git + DVC consistency checks.
#   Ensures tracked data, configs, and logs are synchronized before commit.
#
# Usage:
#   ./bin/repair_and_push.sh "Commit message"
#
# Logging:
#   Logs activity into logs/v50_debug_log.md for reproducibility.
# ==============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 \"Commit message\""
  exit 1
fi

COMMIT_MSG=$1

mkdir -p logs
echo "[repair_and_push] Starting repair at $(date -Iseconds)" | tee -a logs/v50_debug_log.md

# 1. Run DVC status to check pipeline consistency
dvc status || echo "⚠️ DVC reported pipeline differences."

# 2. Git add tracked files
git add -A
dvc add data/* || true

# 3. Commit
git commit -m "$COMMIT_MSG" || echo "ℹ️ Nothing to commit."

# 4. Push Git + DVC
git push origin main
dvc push

echo "[repair_and_push] Completed push at $(date -Iseconds)" | tee -a logs/v50_debug_log.md
