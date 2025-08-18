#!/usr/bin/env bash
set -euo pipefail
branch="$(git rev-parse --abbrev-ref HEAD)"
msg="$(git log -1 --pretty=%B 2>/dev/null || echo "chore(update): sync changes")"
git add -A
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi
git commit -m "$msg" || true
git push origin "$branch"
