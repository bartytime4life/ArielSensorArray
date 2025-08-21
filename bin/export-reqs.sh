#!/usr/bin/env bash
# ==============================================================================
# bin/export-reqs.sh — Export reproducible requirements files (Poetry → pip)
# ------------------------------------------------------------------------------
# What it does
#   • Exports Poetry lock to pip-compatible requirement files
#   • Generates variants:
#       - requirements.txt           (runtime, no hashes)
#       - requirements-dev.txt       (runtime + dev groups)
#       - requirements-min.txt       (minimal portable subset)
#       - requirements-kaggle.txt    (Kaggle-safe: removes torch* & heavy MLOps)
#       - requirements.freeze.txt    (pip freeze of the CURRENT env, optional)
#   • Lets you pick extra groups, toggle hashes, change output paths, and dry-run
#
# Usage
#   bin/export-reqs.sh [options]
#
# Common examples
#   # Full set (runtime, dev, min, kaggle):
#   bin/export-reqs.sh --all
#
#   # Only main + selected groups (viz,hf), dev too:
#   bin/export-reqs.sh --groups viz,hf --dev
#
#   # Also produce a pinned snapshot from the current env:
#   bin/export-reqs.sh --freeze
#
# Options
#   --outdir <dir>       Directory for outputs (default: .)
#   --main               Export requirements.txt                 (default: on when --all)
#   --dev                Export requirements-dev.txt             (default: off)
#   --min                Export requirements-min.txt             (default: off)
#   --kaggle             Export requirements-kaggle.txt          (default: off)
#   --freeze             Export requirements.freeze.txt          (default: off)
#   --groups <csv>       Poetry groups to include with --with (e.g. viz,hf,lightning)
#   --hashes             Include hashes in Poetry export          (default: no hashes)
#   --no-poetry          Do not use Poetry; minimal fallbacks only (freeze/min from env)
#   -n, --dry-run        Show what would happen; no files written
#   -q, --quiet          Less verbose
#   -h, --help           Show help
#
# Notes
#   • Requires Poetry for lock → pip export. If missing, --freeze still works.
#   • Kaggle variant removes torch/torchvision/torchaudio (preinstalled) and heavy MLOps.
#   • Minimal variant removes heavy/GUI/dev stacks, keeping a portable core.
#   • Order is preserved from Poetry export; filters keep relative order when possible.
# ==============================================================================

set -euo pipefail

# ---------- pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[REQS]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[REQS]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[REQS]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- defaults ----------
OUTDIR="."
DO_MAIN=0
DO_DEV=0
DO_MIN=0
DO_KAGGLE=0
DO_FREEZE=0
ALL=0
USE_POETRY=1
WITH_HASHES=0
GROUPS=""
DRY=0
QUIET=0

# ---------- args ----------
usage() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//' | sed 's/^bin\/export-reqs\.sh/export-reqs.sh/'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir) OUTDIR="${2:?}"; shift ;;
    --main) DO_MAIN=1 ;;
    --dev) DO_DEV=1 ;;
    --min) DO_MIN=1 ;;
    --kaggle) DO_KAGGLE=1 ;;
    --freeze) DO_FREEZE=1 ;;
    --all) ALL=1 ;;
    --groups) GROUPS="${2:?}"; shift ;;
    --hashes) WITH_HASHES=1 ;;
    --no-poetry) USE_POETRY=0 ;;
    -n|--dry-run) DRY=1 ;;
    -q|--quiet) QUIET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done

# implicit defaults
if [[ $ALL -eq 1 ]]; then
  DO_MAIN=1; DO_DEV=1; DO_MIN=1; DO_KAGGLE=1
fi
# if nothing chosen, default to main
if [[ $DO_MAIN -eq 0 && $DO_DEV -eq 0 && $DO_MIN -eq 0 && $DO_KAGGLE -eq 0 && $DO_FREEZE -eq 0 ]]; then
  DO_MAIN=1
fi

mkdir -p "$OUTDIR"

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
out() { printf '%s\n' "$*" > "$1"; }
copy_or_write() {
  local path="$1"; shift
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] would write $path"
    return 0
  fi
  printf '%s\n' "$@" > "$path"
}

poetry_export() {
  local outfile="$1"; shift
  local args=(-f requirements.txt -o "$outfile")
  [[ -n "$GROUPS" ]] && args+=(--with "$GROUPS")
  [[ $WITH_HASHES -eq 0 ]] && args+=(--without-hashes)
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] poetry export ${args[*]}"
  else
    poetry export "${args[@]}"
  fi
}

poetry_export_dev() {
  local outfile="$1"
  local args=(-f requirements.txt -o "$outfile" --with dev)
  [[ -n "$GROUPS" ]] && args+=(--with "$GROUPS")
  [[ $WITH_HASHES -eq 0 ]] && args+=(--without-hashes)
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] poetry export ${args[*]}"
  else
    poetry export "${args[@]}"
  fi
}

pip_freeze_to() {
  local outfile="$1"
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] pip freeze > $outfile"
  else
    python - <<'PY' >"$outfile"
import sys, pkgutil, subprocess
# Use plain 'pip freeze' to capture current env
subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True)
PY
  fi
}

# Filter helpers using awk/grep while preserving order
filter_kaggle() {
  # stdin→stdout; remove heavy deps for Kaggle
  # - torch/torchvision/torchaudio (preinstalled)
  # - dvc*, mlflow, wandb, torch-geometric, astropy, jax, ray
  # - big GUI/docs extras that aren’t needed
  awk '
    BEGIN{
      IGNORECASE=1
    }
    /^[[:space:]]*#/ {next}
    /^[[:space:]]*$/ {next}
    tolower($0) ~ /^torch([-=].*|$)/ {next}
    tolower($0) ~ /^torchvision([-=].*|$)/ {next}
    tolower($0) ~ /^torchaudio([-=].*|$)/ {next}
    tolower($0) ~ /^torch-geometric([-=].*|$)/ {next}
    tolower($0) ~ /^dvc([-=].*|$)/ {next}
    tolower($0) ~ /^(mlflow|wandb)([-=].*|$)/ {next}
    tolower($0) ~ /^(astropy|ray|jax|flax)([-=].*|$)/ {next}
    tolower($0) ~ /^(mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets)([-=].*|$)/ {next}
    {print}
  '
}

filter_min() {
  # stdin→stdout; remove heavy/GUI/dev stacks; keep a portable core.
  awk '
    BEGIN{ IGNORECASE=1 }
    /^[[:space:]]*#/ {next}
    /^[[:space:]]*$/ {next}
    # strip dev/ci/doc/gui stacks
    tolower($0) ~ /^(mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets|black|ruff|flake8|mypy|pytest|pytest-.*|coverage|pre-commit)([-=].*|$)/ {next}
    # strip heavy ops
    tolower($0) ~ /^(mlflow|wandb|dvc|ray|spark)([-=].*|$)/ {next}
    # keep most of scientific python; leave torch in (portable); user can remove later if needed
    {print}
  '
}

# ---------- sanity ----------
if [[ $USE_POETRY -eq 1 && ! $(have poetry) ]]; then
  warn "Poetry not found. Fallbacks: --freeze and filter-based --min/--kaggle only."
  USE_POETRY=0
fi
if ! have python; then
  fail "python not found in PATH"
  exit 127
fi

# ---------- export: main ----------
if [[ $DO_MAIN -eq 1 ]]; then
  OUT="$OUTDIR/requirements.txt"
  if [[ $USE_POETRY -eq 1 ]]; then
    say "Exporting main → $OUT"
    poetry_export "$OUT"
  else
    warn "Poetry unavailable; using pip freeze (not lock-based) → $OUT"
    pip_freeze_to "$OUT"
  fi
fi

# ---------- export: dev ----------
if [[ $DO_DEV -eq 1 ]]; then
  OUT="$OUTDIR/requirements-dev.txt"
  if [[ $USE_POETRY -eq 1 ]]; then
    say "Exporting dev → $OUT"
    poetry_export_dev "$OUT"
  else
    warn "Poetry unavailable; generating dev from current env → $OUT"
    pip_freeze_to "$OUT"
  fi
fi

# ---------- export: kaggle ----------
if [[ $DO_KAGGLE -eq 1 ]]; then
  SRC="$OUTDIR/requirements.txt"
  OUT="$OUTDIR/requirements-kaggle.txt"
  if [[ ! -f "$SRC" ]]; then
    warn "Missing $SRC; exporting main first (temporary stream)…"
    TMP="$(mktemp)"
    if [[ $USE_POETRY -eq 1 ]]; then
      poetry_export "$TMP"
    else
      pip_freeze_to "$TMP"
    fi
    say "Writing Kaggle variant → $OUT"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] would filter Kaggle rules"
    else
      { echo "# requirements-kaggle.txt (auto-generated)"; echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"; } > "$OUT"
      filter_kaggle < "$TMP" >> "$OUT"
      rm -f "$TMP"
    fi
  else
    say "Deriving Kaggle variant from $SRC → $OUT"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] would filter Kaggle rules"
    else
      { echo "# requirements-kaggle.txt (auto-generated)"; echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"; } > "$OUT"
      filter_kaggle < "$SRC" >> "$OUT"
    fi
  fi
fi

# ---------- export: min ----------
if [[ $DO_MIN -eq 1 ]]; then
  SRC="$OUTDIR/requirements.txt"
  OUT="$OUTDIR/requirements-min.txt"
  if [[ ! -f "$SRC" ]]; then
    warn "Missing $SRC; exporting main first (temporary stream)…"
    TMP="$(mktemp)"
    if [[ $USE_POETRY -eq 1 ]]; then
      poetry_export "$TMP"
    else
      pip_freeze_to "$TMP"
    fi
    say "Writing minimal variant → $OUT"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] would filter minimal rules"
    else
      { echo "# requirements-min.txt (auto-generated)"; echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"; } > "$OUT"
      filter_min < "$TMP" >> "$OUT"
      rm -f "$TMP"
    fi
  else
    say "Deriving minimal variant from $SRC → $OUT"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] would filter minimal rules"
    else
      { echo "# requirements-min.txt (auto-generated)"; echo "# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"; } > "$OUT"
      filter_min < "$SRC" >> "$OUT"
    fi
  fi
fi

# ---------- export: freeze (current env) ----------
if [[ $DO_FREEZE -eq 1 ]]; then
  OUT="$OUTDIR/requirements.freeze.txt"
  say "Freezing current environment → $OUT"
  pip_freeze_to "$OUT"
fi

# ---------- epilog ----------
say "${GRN}Done exporting requirements.${RST}"
if [[ $DO_KAGGLE -eq 1 ]]; then
  say "  • Kaggle variant removes torch*, dvc/mlflow/wandb, GUI/dev/docs stacks."
fi
if [[ $DO_MIN -eq 1 ]]; then
  say "  • Minimal variant removes heavy/GUI/dev stacks for portability."
fi
exit 0