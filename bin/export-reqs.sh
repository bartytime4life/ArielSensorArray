#!/usr/bin/env bash

==============================================================================

bin/export-reqs.sh — Export reproducible requirements files (Poetry → pip)

——————————————————————————

What it does

• Exports Poetry lock to pip-compatible requirement files

• Generates variants:

- requirements.txt           (runtime, no hashes by default)

- requirements-dev.txt       (runtime + dev groups)

- requirements-min.txt       (minimal portable subset)

- requirements-kaggle.txt    (Kaggle-safe: removes torch* & heavy MLOps)

- requirements.freeze.txt    (pip freeze of the CURRENT env, optional)

• Lets you pick extra groups, toggle hashes, change output paths, and dry-run

• Optional JSON summary for CI

• CI/Kaggle-safe, idempotent, order-preserving filters



Usage

bin/export-reqs.sh [options]



Common examples

# Full set (runtime, dev, min, kaggle):

bin/export-reqs.sh –all



# Only main + selected groups (viz,hf), dev too:

bin/export-reqs.sh –groups viz,hf –dev



# Also produce a pinned snapshot from the current env:

bin/export-reqs.sh –freeze



Options

–outdir        Directory for outputs (default: .)

–main               Export requirements.txt                 (default: on when –all; else on if none picked)

–dev                Export requirements-dev.txt             (default: off)

–min                Export requirements-min.txt             (default: off)

–kaggle             Export requirements-kaggle.txt          (default: off)

–freeze             Export requirements.freeze.txt          (default: off)

–groups        Poetry groups to include (e.g., viz,hf,lightning)

–hashes             Include hashes in Poetry export          (default: no hashes)

–no-poetry          Do not use Poetry; minimal fallbacks only (freeze/min/kaggle from env)

–json               Emit JSON summary to stdout

-n, –dry-run        Show what would happen; no files written

-q, –quiet          Less verbose

-h, –help           Show help



Notes

• Requires Poetry for lock → pip export. If missing, –freeze still works.

• Kaggle variant removes torch/torchvision/torchaudio (preinstalled) and heavy MLOps.

• Minimal variant removes heavy/GUI/dev stacks, keeping a portable core.

• Filters keep relative order where possible.

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– pretty –––––

is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; CYN=”$(tput setaf 6)”; YLW=”$(tput setaf 3)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; CYN=””; YLW=””; RST=””
fi
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[REQS]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[REQS]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[REQS]%s %s\n’ “${RED}” “${RST}” “$*” >&2; }

––––– defaults –––––

OUTDIR=”.”
DO_MAIN=0
DO_DEV=0
DO_MIN=0
DO_KAGGLE=0
DO_FREEZE=0
ALL=0
USE_POETRY=1
WITH_HASHES=0
GROUPS=””
DRY=0
QUIET=0
JSON=0

––––– args –––––

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’ | sed ‘s/^bin/export-reqs.sh/export-reqs.sh/’; }

while [[ $# -gt 0 ]]; do
case “$1” in
–outdir) OUTDIR=”${2:?}”; shift ;;
–main) DO_MAIN=1 ;;
–dev) DO_DEV=1 ;;
–min) DO_MIN=1 ;;
–kaggle) DO_KAGGLE=1 ;;
–freeze) DO_FREEZE=1 ;;
–all) ALL=1 ;;
–groups) GROUPS=”${2:?}”; shift ;;
–hashes) WITH_HASHES=1 ;;
–no-poetry) USE_POETRY=0 ;;
–json) JSON=1 ;;
-n|–dry-run) DRY=1 ;;
-q|–quiet) QUIET=1 ;;
-h|–help) usage; exit 0 ;;
*) fail “Unknown arg: $1”; usage; exit 2 ;;
esac
shift
done

––––– repo root (best effort) –––––

if command -v git >/dev/null 2>&1 && git rev-parse –show-toplevel >/dev/null 2>&1; then
cd “$(git rev-parse –show-toplevel)”
fi

mkdir -p “$OUTDIR”

––––– helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }

timestamp() { date -u +”%Y-%m-%dT%H:%M:%SZ”; }

header_to() {
local path=”$1”; shift
{
printf “# %s (auto-generated)\n” “$(basename “$path”)”
printf “# Generated: %s\n” “$(timestamp)”
[[ $# -gt 0 ]] && printf “# %s\n” “$*”
} > “$path”
}

poetry_present() {
[[ $USE_POETRY -eq 1 ]] && have poetry && [[ -f “pyproject.toml” ]]
}

poetry_export() {
local outfile=”$1”; shift
local args=(-f requirements.txt -o “$outfile”)
[[ -n “$GROUPS” ]] && args+=(–with “$GROUPS”)
[[ $WITH_HASHES -eq 0 ]] && args+=(–without-hashes)
if [[ $DRY -eq 1 ]]; then
say “[dry-run] poetry export ${args[*]}”
else
poetry export “${args[@]}”
fi
}

poetry_export_dev() {
local outfile=”$1”
local args=(-f requirements.txt -o “$outfile” –with dev)
[[ -n “$GROUPS” ]] && args+=(–with “$GROUPS”)
[[ $WITH_HASHES -eq 0 ]] && args+=(–without-hashes)
if [[ $DRY -eq 1 ]]; then
say “[dry-run] poetry export ${args[*]}”
else
poetry export “${args[@]}”
fi
}

pip_freeze_to() {
local outfile=”$1”
if [[ $DRY -eq 1 ]]; then
say “[dry-run] pip freeze > $outfile”
else
python - <<‘PY’ >”$outfile”
import sys, subprocess
subprocess.run([sys.executable, “-m”, “pip”, “freeze”], check=True)
PY
fi
}

––––– filters (stdin→stdout) –––––

filter_kaggle() {

Remove heavy deps for Kaggle kernels (torch preinstalled; skip MLOps + big GUI stacks)

awk ’
BEGIN{ IGNORECASE=1 }
/^[[:space:]]#/ {next}
/^[[:space:]]$/ {next}
# Preinstalled / heavy CUDA stacks
tolower($0) ~ /^torch([-=].|$)/          {next}
tolower($0) ~ /^torchvision([-=].|$)/    {next}
tolower($0) ~ /^torchaudio([-=].|$)/     {next}
tolower($0) ~ /^torch-geometric([-=].|$)/{next}
# MLOps / remote artifacts (often blocked)
tolower($0) ~ /^(dvc|mlflow|wandb)([-=].|$)/ {next}
# Very heavy scientific or alt backends
tolower($0) ~ /^(astropy|ray|jax|flax|tensorflow|pyspark)([-=].|$)/ {next}
# Interactive/GUI/doc stacks
tolower($0) ~ /^(mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets|matplotlib-inline)([-=].*|$)/ {next}
{print}
’
}

filter_min() {

Minimal portable: trim dev, GUI, docs, and big orchestration

awk ’
BEGIN{ IGNORECASE=1 }
/^[[:space:]]#/ {next}
/^[[:space:]]$/ {next}
# Dev & QA
tolower($0) ~ /^(black|ruff|flake8|mypy|pytest(-.)?|coverage|pre-commit)([-=].|$)/ {next}
# Docs / notebooks
tolower($0) ~ /^(mkdocs|sphinx|jupyter|notebook|ipykernel|ipywidgets)([-=].|$)/ {next}
# Orchestration / MLOps
tolower($0) ~ /^(mlflow|wandb|dvc|ray|pyspark)([-=].|$)/ {next}
{print}
’
}

––––– implicit selection –––––

if [[ $ALL -eq 1 ]]; then
DO_MAIN=1; DO_DEV=1; DO_MIN=1; DO_KAGGLE=1
fi
if [[ $DO_MAIN -eq 0 && $DO_DEV -eq 0 && $DO_MIN -eq 0 && $DO_KAGGLE -eq 0 && $DO_FREEZE -eq 0 ]]; then
DO_MAIN=1
fi

––––– sanity –––––

if ! have python; then
fail “python not found in PATH”
exit 127
fi
if [[ $USE_POETRY -eq 1 && ! $(have poetry) ]]; then
warn “Poetry not found. Falling back to –freeze and filter-based variants from current env.”
USE_POETRY=0
fi
if [[ $USE_POETRY -eq 1 && ! -f “poetry.lock” ]]; then
warn “poetry.lock not found; export will use current pyproject state.”
fi

––––– MAIN –––––

MAIN_PATH=”$OUTDIR/requirements.txt”
DEV_PATH=”$OUTDIR/requirements-dev.txt”
MIN_PATH=”$OUTDIR/requirements-min.txt”
KAG_PATH=”$OUTDIR/requirements-kaggle.txt”
FRZ_PATH=”$OUTDIR/requirements.freeze.txt”

main

if [[ $DO_MAIN -eq 1 ]]; then
say “Exporting main → $MAIN_PATH”
if poetry_present; then
if [[ $DRY -eq 1 ]]; then
say “[dry-run] will write $MAIN_PATH from poetry export”
else
poetry_export “$MAIN_PATH”
fi
else
warn “Poetry unavailable; generating main from current environment (pip freeze) → $MAIN_PATH”
pip_freeze_to “$MAIN_PATH”
fi
fi

dev

if [[ $DO_DEV -eq 1 ]]; then
say “Exporting dev → $DEV_PATH”
if poetry_present; then
if [[ $DRY -eq 1 ]]; then
say “[dry-run] will write $DEV_PATH from poetry export (with dev)”
else
poetry_export_dev “$DEV_PATH”
fi
else
warn “Poetry unavailable; generating dev from current environment (pip freeze) → $DEV_PATH”
pip_freeze_to “$DEV_PATH”
fi
fi

kaggle

if [[ $DO_KAGGLE -eq 1 ]]; then
say “Producing Kaggle variant → $KAG_PATH”

choose base stream (prefer main)

if [[ -f “$MAIN_PATH” ]]; then
SRC=”$MAIN_PATH”
else
SRC=”$(mktemp)”; trap ‘rm -f “$SRC” 2>/dev/null || true’ EXIT
if poetry_present; then
[[ $DRY -eq 1 ]] && say “[dry-run] poetry export > (tmp)” || poetry_export “$SRC”
else
[[ $DRY -eq 1 ]] && say “[dry-run] pip freeze > (tmp)” || pip_freeze_to “$SRC”
fi
fi
if [[ $DRY -eq 1 ]]; then
say “[dry-run] filter_kaggle < $SRC > $KAG_PATH”
else
header_to “$KAG_PATH” “Kaggle-safe: removes torch*, dvc/mlflow/wandb, GUI/dev stacks”
filter_kaggle < “$SRC” >> “$KAG_PATH”
fi
fi

min

if [[ $DO_MIN -eq 1 ]]; then
say “Producing minimal variant → $MIN_PATH”
if [[ -f “$MAIN_PATH” ]]; then
SRC=”$MAIN_PATH”
else
SRC=”$(mktemp)”; trap ‘rm -f “$SRC” 2>/dev/null || true’ EXIT
if poetry_present; then
[[ $DRY -eq 1 ]] && say “[dry-run] poetry export > (tmp)” || poetry_export “$SRC”
else
[[ $DRY -eq 1 ]] && say “[dry-run] pip freeze > (tmp)” || pip_freeze_to “$SRC”
fi
fi
if [[ $DRY -eq 1 ]]; then
say “[dry-run] filter_min < $SRC > $MIN_PATH”
else
header_to “$MIN_PATH” “Minimal portable subset (no dev/docs/GUI/MLOps stacks)”
filter_min < “$SRC” >> “$MIN_PATH”
fi
fi

freeze

if [[ $DO_FREEZE -eq 1 ]]; then
say “Freezing current environment → $FRZ_PATH”
if [[ $DRY -eq 1 ]]; then
say “[dry-run] pip freeze > $FRZ_PATH”
else
header_to “$FRZ_PATH” “Pinned snapshot from current environment (not from lock)”
pip_freeze_to “$FRZ_PATH”
fi
fi

––––– JSON summary –––––

if [[ $JSON -eq 1 ]]; then
printf ‘{’
printf ’“ok”: true, ’
printf ’“outdir”: “%s”, ’ “$OUTDIR”
printf ’“selection”: {“main”: %s, “dev”: %s, “min”: %s, “kaggle”: %s, “freeze”: %s}, ’ 
“$([[ $DO_MAIN -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_DEV -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_MIN -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_KAGGLE -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_FREEZE -eq 1 ]] && echo true || echo false)”
printf ‘“groups”: “%s”, “hashes”: %s, “used_poetry”: %s’ 
“$GROUPS” “$([[ $WITH_HASHES -eq 1 ]] && echo true || echo false)” “$([[ $(poetry_present && echo 1 || echo 0) -eq 1 ]] && echo true || echo false)”
printf ‘}\n’
fi

––––– footer –––––

say “${GRN}Done exporting requirements.${RST}”
if [[ $DO_KAGGLE -eq 1 ]]; then
say “  • Kaggle variant removes torch*, dvc/mlflow/wandb, and GUI/dev/docs stacks.”
fi
if [[ $DO_MIN -eq 1 ]]; then
say “  • Minimal variant removes dev/GUI/docs/MLOps for portability.”
fi
exit 0