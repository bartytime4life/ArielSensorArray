#!/usr/bin/env bash
# ==============================================================================
# bin/sync-lock.sh — Keep Poetry lock & requirements files perfectly in sync
# ------------------------------------------------------------------------------
# Purpose
#   Ensure the single source of truth (Poetry lock) matches the pip requirements
#   artifacts in the repo. Detect drift, show diffs, and optionally regenerate.
#
# What it does
#   • (Optional) Rebuilds Poetry lock (no-update or with update)
#   • Exports fresh requirements via bin/export-reqs.sh to a temp dir
#   • Compares against committed artifacts (requirements*.txt)
#   • Verifies VERSION == pyproject.toml version
#   • Reports a concise drift summary; can auto-write & optionally freeze
#
# Usage
#   bin/sync-lock.sh [options]
#
# Common examples
#   # CI-style check: fail if any requirements drift from the lock
#   bin/sync-lock.sh --check
#
#   # Rebuild lock (no update) and export all requirement variants
#   bin/sync-lock.sh --lock --write --all
#
#   # Update lock to latest compatible versions, then export only main+dev
#   bin/sync-lock.sh --update --write --main --dev
#
# Options
#   --lock               Run `poetry lock --no-update`
#   --update             Run `poetry lock` (allow resolver updates)
#   --write              Write/replace repo requirements files with fresh export
#   --check              Only check for drift (non-zero exit if mismatched)
#   --freeze             Also write requirements.freeze.txt from current env
#   --all                Target all exported flavors (main, dev, min, kaggle)
#   --main               Include requirements.txt
#   --dev                Include requirements-dev.txt
#   --min                Include requirements-min.txt
#   --kaggle             Include requirements-kaggle.txt
#   --groups <csv>       Extra Poetry groups to include during export (e.g. viz,hf)
#   --hashes             Include hashes in Poetry export (default: no hashes)
#   --no-poetry          Do not use Poetry (will only run VERSION/pyproject checks)
#   --outdir <dir>       Where requirements live (default: repo root)
#   --quiet              Reduce verbosity
#   --dry-run            Show what would happen, do not modify files
#   -h|--help            Show help
#
# Exit codes
#   0 OK / in-sync (or write completed)
#   1 Drift detected (with --check) or fatal error
#   2 Bad usage
# ==============================================================================

set -euo pipefail

# ---------- pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[SYNC]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[SYNC]%s %s\n' "${YLW}" "${RST}" "$*" >&2; }
fail() { printf '%s[SYNC]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- defaults ----------
DO_LOCK=0
DO_UPDATE=0
DO_WRITE=0
DO_CHECK=0
DO_FREEZE=0
ALL=0
INC_MAIN=0
INC_DEV=0
INC_MIN=0
INC_KAGGLE=0
GROUPS=""
WITH_HASHES=0
USE_POETRY=1
OUTDIR=""
DRY=0
QUIET=0

usage() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --lock) DO_LOCK=1 ;;
    --update) DO_UPDATE=1 ;;
    --write) DO_WRITE=1 ;;
    --check) DO_CHECK=1 ;;
    --freeze) DO_FREEZE=1 ;;
    --all) ALL=1 ;;
    --main) INC_MAIN=1 ;;
    --dev) INC_DEV=1 ;;
    --min) INC_MIN=1 ;;
    --kaggle) INC_KAGGLE=1 ;;
    --groups) GROUPS="${2:?}"; shift ;;
    --hashes) WITH_HASHES=1 ;;
    --no-poetry) USE_POETRY=0 ;;
    --outdir) OUTDIR="${2:?}"; shift ;;
    --dry-run) DRY=1 ;;
    --quiet) QUIET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

# ---------- repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; exit 1; }
fi

REPO_ROOT="$PWD"
[[ -z "$OUTDIR" ]] && OUTDIR="$REPO_ROOT"

VERSION_FILE="$REPO_ROOT/VERSION"
PYPROJECT_FILE="$REPO_ROOT/pyproject.toml"
EXPORT_SCRIPT="$REPO_ROOT/bin/export-reqs.sh"

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }

read_version_file() { [[ -f "$VERSION_FILE" ]] && sed -n '1s/[[:space:]]//gp' "$VERSION_FILE" || true; }

read_pyproj_version() { sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PYPROJECT_FILE" | head -n1 || true; }

diff_one() {
  local left="$1" right="$2" label="$3"
  if ! diff -u "$left" "$right" >/dev/null 2>&1; then
    echo "---- drift: $label ----"
    diff -u "$left" "$right" || true
    echo "------------------------"
    return 1
  fi
  return 0
}

export_temp() {
  local tmpdir="$1"
  local args=()
  [[ -n "$GROUPS" ]] && args+=(--groups "$GROUPS")
  [[ $WITH_HASHES -eq 1 ]] && args+=(--hashes)

  # Decide flavors
  local want_main=$INC_MAIN
  local want_dev=$INC_DEV
  local want_min=$INC_MIN
  local want_kaggle=$INC_KAGGLE
  if [[ $ALL -eq 1 ]]; then want_main=1; want_dev=1; want_min=1; want_kaggle=1; fi
  # If none specified, default to main
  if [[ $want_main -eq 0 && $want_dev -eq 0 && $want_min -eq 0 && $want_kaggle -eq 0 ]]; then
    want_main=1
  fi

  # poetry-export via our exporter (preferred)
  if [[ $USE_POETRY -eq 1 && -x "$EXPORT_SCRIPT" ]]; then
    [[ $want_main  -eq 1 ]] && "$EXPORT_SCRIPT" --outdir "$tmpdir" --main  "${args[@]}" >/dev/null
    [[ $want_dev   -eq 1 ]] && "$EXPORT_SCRIPT" --outdir "$tmpdir" --dev   "${args[@]}" >/dev/null
    [[ $want_min   -eq 1 ]] && "$EXPORT_SCRIPT" --outdir "$tmpdir" --min   "${args[@]}" >/dev/null
    [[ $want_kaggle -eq 1 ]] && "$EXPORT_SCRIPT" --outdir "$tmpdir" --kaggle "${args[@]}" >/dev/null
    [[ $DO_FREEZE -eq 1   ]] && "$EXPORT_SCRIPT" --outdir "$tmpdir" --freeze >/dev/null || true
    return 0
  fi

  # poetry export fallback
  if [[ $USE_POETRY -eq 1 && $(have poetry) ]]; then
    local base=(-f requirements.txt --without-hashes)
    [[ $WITH_HASHES -eq 1 ]] && base=(-f requirements.txt)
    [[ -n "$GROUPS" ]] && base+=("--with" "$GROUPS")
    [[ $want_main -eq 1  ]] && poetry export "${base[@]}" -o "$tmpdir/requirements.txt" >/dev/null
    [[ $want_dev  -eq 1  ]] && poetry export "${base[@]}" --with dev -o "$tmpdir/requirements-dev.txt" >/dev/null
    # Derive min/kaggle from main export
    if [[ $want_min -eq 1 || $want_kaggle -eq 1 ]]; then
      if [[ ! -f "$tmpdir/requirements.txt" ]]; then
        poetry export "${base[@]}" -o "$tmpdir/requirements.txt" >/dev/null
      fi
      if [[ $want_min -eq 1 ]]; then
        awk 'BEGIN{IGNORECASE=1}/^(mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets|black|ruff|flake8|mypy|pytest|coverage|pre-commit|mlflow|wandb|dvc|ray|spark)(\b|[=<>])/ {next} {print}' \
          "$tmpdir/requirements.txt" > "$tmpdir/requirements-min.txt"
      fi
      if [[ $want_kaggle -eq 1 ]]; then
        awk 'BEGIN{IGNORECASE=1}
          /^(torch|torchvision|torchaudio|torch-geometric|dvc|mlflow|wandb|astropy|ray|jax|flax|mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets)(\b|[=<>])/ {next}
          {print}' "$tmpdir/requirements.txt" > "$tmpdir/requirements-kaggle.txt"
      fi
    fi
    [[ $DO_FREEZE -eq 1 ]] && python -m pip freeze > "$tmpdir/requirements.freeze.txt" || true
    return 0
  fi

  warn "Poetry/export script not available; cannot generate fresh requirements."
  return 1
}

# ---------- preflight ----------
[[ -f "$PYPROJECT_FILE" ]] || warn "pyproject.toml not found at repo root (some checks will be skipped)."

if [[ $USE_POETRY -eq 1 && ! $(have poetry) ]]; then
  warn "Poetry not found; disabling Poetry-dependent steps."
  USE_POETRY=0
fi

# ---------- lock / update (optional) ----------
if [[ $USE_POETRY -eq 1 ]]; then
  if [[ $DO_LOCK -eq 1 && $DO_UPDATE -eq 1 ]]; then
    fail "Choose either --lock or --update (not both)."
    exit 2
  fi
  if [[ $DO_LOCK -eq 1 ]]; then
    say "Running: poetry lock --no-update"
    [[ $DRY -eq 1 ]] || poetry lock --no-update
  fi
  if [[ $DO_UPDATE -eq 1 ]]; then
    say "Running: poetry lock   # (resolver update)"
    [[ $DRY -eq 1 ]] || poetry lock
  fi
fi

# ---------- VERSION vs pyproject version ----------
if [[ -f "$VERSION_FILE" && -f "$PYPROJECT_FILE" ]]; then
  v_file="$(read_version_file)"
  v_py="$(read_pyproj_version)"
  if [[ -n "$v_file" && -n "$v_py" ]]; then
    if [[ "$v_file" != "$v_py" ]]; then
      warn "VERSION ($v_file) differs from pyproject.toml version ($v_py)."
      [[ $DO_CHECK -eq 1 ]] && { fail "Version mismatch"; exit 1; }
    else
      say "VERSION matches pyproject: $v_file"
    fi
  fi
fi

# ---------- export fresh requirements to temp ----------
TMPDIR="$(mktemp -d -t sync_lock_XXXX)"
trap 'rm -rf "$TMPDIR"' EXIT

if ! export_temp "$TMPDIR"; then
  if [[ $DO_CHECK -eq 1 || $DO_WRITE -eq 1 ]]; then
    fail "Failed to generate fresh requirements."
    exit 1
  else
    warn "Skipping export; nothing to compare."
  fi
fi

# Decide list of artifacts to compare/write
ARTS=()
if [[ $ALL -eq 1 || $INC_MAIN -eq 1 || ( $INC_MAIN -eq 0 && $INC_DEV -eq 0 && $INC_MIN -eq 0 && $INC_KAGGLE -eq 0 ) ]]; then
  ARTS+=("requirements.txt")
fi
[[ $ALL -eq 1 || $INC_DEV -eq 1     ]] && ARTS+=("requirements-dev.txt")
[[ $ALL -eq 1 || $INC_MIN -eq 1     ]] && ARTS+=("requirements-min.txt")
[[ $ALL -eq 1 || $INC_KAGGLE -eq 1  ]] && ARTS+=("requirements-kaggle.txt")
[[ $DO_FREEZE -eq 1                 ]] && ARTS+=("requirements.freeze.txt")

# ---------- compare & possibly write ----------
DRIFT=0
for f in "${ARTS[@]}"; do
  src="$TMPDIR/$f"
  dst="$OUTDIR/$f"
  # Skip if the source doesn't exist (exporter omitted it)
  [[ -f "$src" ]] || { warn "Fresh export missing $f (skipping)."; continue; }

  if [[ ! -f "$dst" ]]; then
    warn "Missing $dst in repo."
    if [[ $DO_WRITE -eq 1 ]]; then
      say "Creating $dst"
      [[ $DRY -eq 1 ]] || cp -f "$src" "$dst"
    else
      DRIFT=1
      continue
    fi
  else
    if ! diff -u "$dst" "$src" >/dev/null 2>&1; then
      say "Drift detected in $f"
      [[ $QUIET -eq 1 ]] || diff -u "$dst" "$src" || true
      if [[ $DO_WRITE -eq 1 ]]; then
        say "Updating $dst"
        [[ $DRY -eq 1 ]] || cp -f "$src" "$dst"
      else
        DRIFT=1
      fi
    else
      say "$f is in sync."
    fi
  fi
done

# ---------- result ----------
if [[ $DO_CHECK -eq 1 && $DRIFT -ne 0 ]]; then
  fail "Requirements are NOT in sync with the lock."
  exit 1
fi

if [[ $DO_WRITE -eq 1 ]]; then
  say "${GRN}Wrote updated requirements artifacts.${RST}"
  [[ $DRY -eq 1 ]] && say "(dry-run: no files actually modified)"
elif [[ $DO_CHECK -eq 1 ]]; then
  say "${GRN}All checked artifacts are in sync.${RST}"
else
  say "Sync summary completed. Use --check for CI or --write to update files."
fi

exit 0