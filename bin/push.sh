#!/usr/bin/env bash
# bin/push.sh — SpectraMind V50 guarded push helper
# -------------------------------------------------
# Safe, reproducible, and chatty wrapper around the common
# "commit → validate → dvc sync → push" flow.
#
# Defaults:
#   - Runs preflight checks (venv/poetry), selftest, lint, tests
#   - Syncs DVC if repo has DVC
#   - Pushes current branch to origin
#
# Flags:
#   -m "msg"   Commit message (optional; if provided, will git add -A && git commit)
#   -b BRANCH  Target branch to push (default: current)
#   -t TAG     Create and push an annotated tag (e.g. v0.1.0)
#   -n         Dry-run (show what would run; no side effects)
#   -F         Force push (use with care)
#   -S         Skip tests (unit/integration)
#   -L         Skip lint & style (pre-commit)
#   -D         Skip DVC push
#   -C         Skip CLI selftest (spectramind selftest)
#   -K         Skip Kaggle CLI verification (if you enable it below)
#   -q         Quiet mode (less chatter)
#   -h         Help
#
# Examples:
#   bin/push.sh -m "feat(train): add COREL coverage report"
#   bin/push.sh -m "chore: bump deps" -t v0.2.0
#   bin/push.sh -S -L -D    # minimal, just push
#
set -euo pipefail

# ---------- styling ----------
if command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"
  GRN="$(tput setaf 2)"; YLW="$(tput setaf 3)"; CYN="$(tput setaf 6)"; RST="$(tput sgr0)"
else
  BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; CYN=""; RST=""
fi

log() { printf "%b\n" "${1-}"; }
info() { log "${CYN}›${RST} ${1-}"; }
ok()   { log "${GRN}✓${RST} ${1-}"; }
warn() { log "${YLW}!${RST} ${1-}"; }
err()  { log "${RED}✗${RST} ${1-}"; }

usage() {
  sed -n '1,120p' "$0" | sed -n '1,60p' | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# ---------- args ----------
COMMIT_MSG=""
TARGET_BRANCH=""
TAG_NAME=""
DRYRUN=0
FORCE=0
SKIP_TESTS=0
SKIP_LINT=0
SKIP_DVC=0
SKIP_SELFTEST=0
SKIP_KAGGLE=1   # default: skip; set to 0 to enable in code block below
QUIET=0

while getopts ":m:b:t:nFSLDCKqh" opt; do
  case "$opt" in
    m) COMMIT_MSG="$OPTARG" ;;
    b) TARGET_BRANCH="$OPTARG" ;;
    t) TAG_NAME="$OPTARG" ;;
    n) DRYRUN=1 ;;
    F) FORCE=1 ;;
    S) SKIP_TESTS=1 ;;
    L) SKIP_LINT=1 ;;
    D) SKIP_DVC=1 ;;
    C) SKIP_SELFTEST=1 ;;
    K) SKIP_KAGGLE=1 ;;  # keep compatibility; -K enforces skip
    q) QUIET=1 ;;
    h) usage 0 ;;
    \?) err "Unknown option: -$OPTARG"; usage 1 ;;
    :) err "Option -$OPTARG requires an argument"; usage 1 ;;
  esac
done

run() {
  if [ "$DRYRUN" -eq 1 ]; then
    printf "%s\n" "${DIM}[dry-run]${RST} $*"
  else
    if [ "$QUIET" -eq 1 ]; then
      "$@" >/dev/null
    else
      "$@"
    fi
  fi
}

assert_cmd() {
  command -v "$1" >/dev/null 2>&1 || { err "Missing required command: $1"; exit 127; }
}

# ---------- preflight ----------
assert_cmd git
GIT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
[ -n "$GIT_ROOT" ] || { err "Not inside a git repository."; exit 1; }
cd "$GIT_ROOT"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
BRANCH="${TARGET_BRANCH:-$CURRENT_BRANCH}"

REMOTE="$(git remote 2>/dev/null | head -n1 || true)"
[ -n "$REMOTE" ] || { err "No git remote configured."; exit 1; }

if [ "$FORCE" -eq 1 ]; then
  warn "Force push enabled (-F). Use with care."
fi

# Ensure we have a clean index if no commit message supplied.
if [ -z "$COMMIT_MSG" ]; then
  if [ -n "$(git status --porcelain)" ]; then
    warn "Working tree has changes but no commit message provided."
    warn "Either commit manually or provide -m \"message\" to auto-commit."
    exit 1
  fi
fi

# Poetry is optional; support both poetry & pip flows.
HAS_POETRY=0
if command -v poetry >/dev/null 2>&1; then HAS_POETRY=1; fi

# DVC presence?
HAS_DVC=0
if command -v dvc >/dev/null 2>&1 && [ -d ".dvc" ]; then HAS_DVC=1; fi

# Pre-commit presence?
HAS_PRECOMMIT=0
if command -v pre-commit >/dev/null 2>&1 && [ -f ".pre-commit-config.yaml" ]; then HAS_PRECOMMIT=1; fi

# SpectraMind CLI presence?
HAS_SM=0
if [ -f "spectramind.py" ] || command -v spectramind >/dev/null 2>&1; then HAS_SM=1; fi

# ---------- commit (optional) ----------
if [ -n "$COMMIT_MSG" ]; then
  info "Staging & committing changes"
  run git add -A
  if git diff --cached --quiet; then
    warn "No staged changes to commit; skipping commit step."
  else
    run git commit -m "$COMMIT_MSG"
    ok "Committed: ${COMMIT_MSG}"
  fi
fi

# ---------- environment / dependencies ----------
if [ "$HAS_POETRY" -eq 1 ]; then
  info "Ensuring Poetry env & dependencies"
  run poetry install --no-root
else
  warn "Poetry not found. Assuming environment is already prepared."
fi

# ---------- quality gates ----------
if [ "$SKIP_SELFTEST" -ne 1 ] && [ "$HAS_SM" -eq 1 ]; then
  info "Running SpectraMind selftest"
  if command -v spectramind >/dev/null 2>&1; then
    run spectramind selftest
  else
    run poetry run python spectramind.py selftest
  fi
  ok "Selftest passed"
else
  warn "Skipping CLI selftest (-C) or spectramind CLI not detected."
fi

if [ "$SKIP_LINT" -ne 1 ] && [ "$HAS_PRECOMMIT" -eq 1 ]; then
  info "Running pre-commit hooks (lint/format/security)"
  run pre-commit run --all-files
  ok "Lint & format checks passed"
else
  warn "Skipping lint/format (-L) or pre-commit not configured."
fi

if [ "$SKIP_TESTS" -ne 1 ]; then
  info "Running tests"
  if [ "$HAS_POETRY" -eq 1 ]; then
    run poetry run pytest -q
  else
    run pytest -q
  fi
  ok "Tests passed"
else
  warn "Skipping tests (-S)"
fi

# ---------- optional Kaggle CLI verify (disabled by default) ----------
if [ "$SKIP_KAGGLE" -ne 1 ]; then
  if command -v kaggle >/dev/null 2>&1; then
    info "Verifying Kaggle CLI login"
    if run kaggle competitions list >/dev/null 2>&1; then
      ok "Kaggle CLI OK"
    else
      warn "Kaggle CLI not authenticated (skip or login)."
    fi
  else
    warn "Kaggle CLI not installed."
  fi
fi

# ---------- helpful project housekeeping ----------
if [ "$HAS_SM" -eq 1 ]; then
  info "Generating short CLI activity summary"
  if command -v spectramind >/dev/null 2>&1; then
    run spectramind analyze-log-short || true
  else
    run poetry run python spectramind.py analyze-log-short || true
  fi
fi

# ---------- DVC sync ----------
if [ "$HAS_DVC" -eq 1 ] && [ "$SKIP_DVC" -ne 1 ]; then
  info "DVC status"
  run dvc status || true
  info "Pushing DVC-tracked artifacts (if any)"
  run dvc push
  ok "DVC push complete"
else
  if [ "$HAS_DVC" -eq 1 ]; then
    warn "Skipping DVC push (-D)"
  else
    warn "No DVC detected"
  fi
fi

# ---------- push ----------
PUSH_ARGS=("$REMOTE" "$BRANCH")
if [ "$FORCE" -eq 1 ]; then PUSH_ARGS+=("--force"); fi

info "Pushing branch: ${BOLD}${BRANCH}${RST} → ${BOLD}${REMOTE}${RST}"
run git push "${PUSH_ARGS[@]}"
ok "Branch pushed"

if [ -n "$TAG_NAME" ]; then
  info "Tagging: ${BOLD}${TAG_NAME}${RST}"
  if git rev-parse -q --verify "refs/tags/$TAG_NAME" >/dev/null; then
    warn "Tag ${TAG_NAME} already exists locally; pushing existing tag"
  else
    run git tag -a "$TAG_NAME" -m "$TAG_NAME"
  fi
  run git push "$REMOTE" "$TAG_NAME"
  ok "Tag pushed: ${TAG_NAME}"
fi

# ---------- summary ----------
GIT_SHA_SHORT="$(git rev-parse --short HEAD)"
if [ "$QUIET" -ne 1 ]; then
  echo
  ok "Done."
  echo "  Repo : $GIT_ROOT"
  echo "  Branch: $BRANCH"
  echo "  Commit: $GIT_SHA_SHORT"
  [ -n "$TAG_NAME" ] && echo "  Tag   : $TAG_NAME"
  [ "$DRYRUN" -eq 1 ] && echo "  Note  : dry-run (no changes applied)"
fi