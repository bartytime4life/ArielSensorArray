#!/usr/bin/env bash

==============================================================================

bin/sync-lock.sh — Keep Poetry lock & requirements artifacts perfectly in sync

——————————————————————————

Purpose

Ensure the single source of truth (Poetry lock) matches the pip requirements

artifacts committed in the repo. Detect drift, print diffs, optionally

regenerate artifacts, and (optionally) emit a machine-readable JSON summary.



What it does

• (Optional) Rebuilds Poetry lock (no-update or with resolver updates)

• Exports fresh requirements via bin/export-reqs.sh (preferred) or fallback

• Compares vs committed artifacts (requirements*.txt), with normalization

• Verifies VERSION == pyproject.toml version

• Reports concise drift summary; can auto-write & optionally freeze

• (Optional) Emits JSON result for CI



Usage

bin/sync-lock.sh [options]



Common examples

# CI-style check: fail if any requirements drift from the lock

bin/sync-lock.sh –check



# Rebuild lock (no update) and export all requirement variants, then write

bin/sync-lock.sh –lock –write –all



# Update lock to latest compatible versions, then export only main+dev

bin/sync-lock.sh –update –write –main –dev



# Export with hashes, include extra Poetry groups, and print JSON for CI

bin/sync-lock.sh –all –hashes –groups viz,hf –check –json



Options

–lock               Run poetry lock --no-update

–update             Run poetry lock (allow resolver updates)

–write              Write/replace repo requirements files with fresh export

–check              Only check for drift (non-zero exit if mismatched)

–freeze             Also write requirements.freeze.txt from current env

–all                Target all exported flavors (main, dev, min, kaggle)

–main               Include requirements.txt

–dev                Include requirements-dev.txt

–min                Include requirements-min.txt

–kaggle             Include requirements-kaggle.txt

–groups        Extra Poetry groups to include during export (e.g. viz,hf)

–hashes             Include hashes in Poetry export (default: without hashes)

–no-poetry          Do not use Poetry (only VERSION/pyproject checks)

–pip-tools          Allow fallback to pip-compile for main/dev (if present)

–outdir        Where requirements live (default: repo root)

–json               Emit machine-readable JSON summary to stdout

–quiet              Reduce verbosity

–dry-run            Show actions; do not modify files

-h|–help            Show help



Exit codes

0 OK / in-sync (or write completed)

1 Drift detected (with –check) or fatal error

2 Bad usage

==============================================================================

set -euo pipefail
IFS=$’\n\t’

––––– pretty –––––

if [[ -t 1 ]]; then
BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; YLW=$’\033[33m’; CYN=$’\033[36m’; RST=$’\033[0m’
else
BOLD=’’; DIM=’’; RED=’’; GRN=’’; YLW=’’; CYN=’’; RST=’’
fi
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[SYNC]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[SYNC]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
err()  { printf ‘%s[SYNC]%s %s\n’ “${RED}” “${RST}” “$” >&2; }
fail() { err “$”; }

––––– defaults –––––

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
GROUPS=””
WITH_HASHES=0
USE_POETRY=1
ALLOW_PIP_TOOLS=0
OUTDIR=””
DRY=0
QUIET=0
EMIT_JSON=0

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; }

––––– args –––––

while [[ $# -gt 0 ]]; do
case “$1” in
–lock) DO_LOCK=1 ;;
–update) DO_UPDATE=1 ;;
–write) DO_WRITE=1 ;;
–check) DO_CHECK=1 ;;
–freeze) DO_FREEZE=1 ;;
–all) ALL=1 ;;
–main) INC_MAIN=1 ;;
–dev) INC_DEV=1 ;;
–min) INC_MIN=1 ;;
–kaggle) INC_KAGGLE=1 ;;
–groups) GROUPS=”${2:?}”; shift ;;
–hashes) WITH_HASHES=1 ;;
–no-poetry) USE_POETRY=0 ;;
–pip-tools) ALLOW_PIP_TOOLS=1 ;;
–outdir) OUTDIR=”${2:?}”; shift ;;
–json) EMIT_JSON=1 ;;
–dry-run) DRY=1 ;;
–quiet) QUIET=1 ;;
-h|–help) usage; exit 0 ;;
*) fail “Unknown arg: $1”; usage; exit 2 ;;
esac
shift
done

––––– repo root –––––

if command -v git >/dev/null 2>&1 && git rev-parse –show-toplevel >/dev/null 2>&1; then
cd “$(git rev-parse –show-toplevel)”
else
SCRIPT_DIR=”$(cd – “$(dirname – “${BASH_SOURCE[0]}”)” && pwd)”
cd “$SCRIPT_DIR/..” || { fail “Cannot locate repo root”; exit 1; }
fi

REPO_ROOT=”$PWD”
[[ -z “$OUTDIR” ]] && OUTDIR=”$REPO_ROOT”
mkdir -p “$OUTDIR”

VERSION_FILE=”$REPO_ROOT/VERSION”
PYPROJECT_FILE=”$REPO_ROOT/pyproject.toml”
POETRY_LOCK_FILE=”$REPO_ROOT/poetry.lock”
EXPORT_SCRIPT=”$REPO_ROOT/bin/export-reqs.sh”

––––– helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }

sha256_of() {
local f=”$1”
if command -v sha256sum >/dev/null 2>&1; then sha256sum – “$f” 2>/dev/null | awk ‘{print $1}’; 
elif command -v shasum >/dev/null 2>&1; then shasum -a 256 – “$f” 2>/dev/null | awk ‘{print $1}’; 
else echo “”; fi
}

read_version_file() {
[[ -f “$VERSION_FILE” ]] && sed -n ‘1s/[[:space:]]//gp’ “$VERSION_FILE” || true
}

read_pyproj_version() {

Handles common pyproject layouts

sed -n ‘s/^[[:space:]]version[[:space:]]=[[:space:]]”.”.*/\1/p’ “$PYPROJECT_FILE” | head -n1 || true
}

make_tmpdir() {
local t
if t=$(mktemp -d -t sync_lock_XXXX 2>/dev/null); then
printf ‘%s\n’ “$t”
else
t=”${TMPDIR:-/tmp}/sync_lock_$$.$RANDOM”
mkdir -p “$t”
printf ‘%s\n’ “$t”
fi
}

json_escape() {

robust JSON string quoting via Python if available

if have python3 || have python; then
local py=python3; have python3 || py=python
“$py” - <<‘PY’ “$1” 2>/dev/null || printf ‘%s’ “$1”
import json, sys
print(json.dumps(sys.argv[1]))
PY
else
# minimal fallback (unsafe for exotic chars, but best-effort)
printf ‘%s’ “$1” | sed -e ‘s/\/\\/g’ -e ‘s/”/\”/g’
fi
}

emit_json() {
[[ $EMIT_JSON -eq 1 ]] || return 0
local ok=”$1” drift=”$2” wrote=”$3” details=”$4”
printf ‘{ “ok”: %s, “drift”: %s, “wrote”: %s, “details”: %s }\n’ 
“$([[ “$ok” == “true” ]] && echo true || echo false)” 
“$([[ “$drift” == “true” ]] && echo true || echo false)” 
“$([[ “$wrote” == “true” ]] && echo true || echo false)” 
“${details:-null}”
}

normalize_requirements() {

normalize file to canonical form for diff: strip comments, trim spaces, collapse blanks, sort

usage: normalize_requirements  

local in=”$1” out=”$2”
awk ’
BEGIN{IGNORECASE=0}
/^[[:space:]]#/ {next}
/^[[:space:]]$/ {next}
{gsub(/\r$/,””)}
{gsub(/[[:space:]]+$/,””)}
{print}
’ “$in” | LC_ALL=C sort -u > “$out”
}

––––– preflight –––––

if [[ $USE_POETRY -eq 1 && ! $(have poetry) ]]; then
warn “Poetry not found; disabling Poetry-dependent steps.”
USE_POETRY=0
fi

if [[ $DO_LOCK -eq 1 && $DO_UPDATE -eq 1 ]]; then
fail “Choose either –lock or –update (not both).”
exit 2
fi

if [[ ! -f “$PYPROJECT_FILE” ]]; then
warn “pyproject.toml not found at repo root; lock/export steps limited.”
USE_POETRY=0
fi

––––– VERSION vs pyproject version –––––

VERSION_MISMATCH=0
if [[ -f “$VERSION_FILE” && -f “$PYPROJECT_FILE” ]]; then
v_file=”$(read_version_file)”
v_py=”$(read_pyproj_version)”
if [[ -n “$v_file” && -n “$v_py” ]]; then
if [[ “$v_file” != “$v_py” ]]; then
VERSION_MISMATCH=1
warn “VERSION ($v_file) differs from pyproject.toml version ($v_py).”
if [[ $DO_CHECK -eq 1 ]]; then
fail “Version mismatch”
emit_json false true false null
exit 1
fi
else
say “VERSION matches pyproject: $v_file”
fi
fi
fi

––––– lock / update (optional) –––––

if [[ $USE_POETRY -eq 1 ]]; then
if [[ $DO_LOCK -eq 1 ]]; then
say “Running: poetry lock –no-update”
[[ $DRY -eq 1 ]] || poetry lock –no-update
fi
if [[ $DO_UPDATE -eq 1 ]]; then
say “Running: poetry lock   # (resolver update)”
[[ $DRY -eq 1 ]] || poetry lock
fi
fi

––––– export fresh requirements to temp –––––

TMPDIR=”$(make_tmpdir)”
trap ‘rm -rf “$TMPDIR”’ EXIT

export_temp() {
local tmpdir=”$1”
local args=()
[[ -n “$GROUPS” ]] && args+=(–groups “$GROUPS”)
[[ $WITH_HASHES -eq 1 ]] && args+=(–hashes)

Decide flavors

local want_main=$INC_MAIN
local want_dev=$INC_DEV
local want_min=$INC_MIN
local want_kaggle=$INC_KAGGLE
if [[ $ALL -eq 1 ]]; then want_main=1; want_dev=1; want_min=1; want_kaggle=1; fi

If none specified, default to main

if [[ $want_main -eq 0 && $want_dev -eq 0 && $want_min -eq 0 && $want_kaggle -eq 0 ]]; then
want_main=1
fi

Preferred path: our explicit exporter script

if [[ -x “$EXPORT_SCRIPT” ]]; then
say “Using exporter: $EXPORT_SCRIPT”
[[ $want_main   -eq 1 ]] && “$EXPORT_SCRIPT” –outdir “$tmpdir” –main   “${args[@]}” >/dev/null
[[ $want_dev    -eq 1 ]] && “$EXPORT_SCRIPT” –outdir “$tmpdir” –dev    “${args[@]}” >/dev/null
[[ $want_min    -eq 1 ]] && “$EXPORT_SCRIPT” –outdir “$tmpdir” –min    “${args[@]}” >/dev/null
[[ $want_kaggle -eq 1 ]] && “$EXPORT_SCRIPT” –outdir “$tmpdir” –kaggle “${args[@]}” >/dev/null
[[ $DO_FREEZE   -eq 1 ]] && “$EXPORT_SCRIPT” –outdir “$tmpdir” –freeze >/dev/null || true
return 0
fi

Poetry export fallback

if [[ $USE_POETRY -eq 1 && $(have poetry) ]]; then
say “Using poetry export fallback”
local base=(-f requirements.txt –without-hashes)
[[ $WITH_HASHES -eq 1 ]] && base=(-f requirements.txt)
[[ -n “$GROUPS” ]] && base+=(”–with” “$GROUPS”)
[[ $want_main -eq 1  ]] && poetry export “${base[@]}” -o “$tmpdir/requirements.txt” >/dev/null
[[ $want_dev  -eq 1  ]] && poetry export “${base[@]}” –with dev -o “$tmpdir/requirements-dev.txt” >/dev/null

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

pip-tools fallback (limited to main/dev) if allowed

if [[ $ALLOW_PIP_TOOLS -eq 1 && $(have pip-compile) ]]; then
say “Using pip-compile fallback (pip-tools)”
[[ $want_main -eq 1 ]] && pip-compile –no-header –no-annotate -q -o “$tmpdir/requirements.txt” pyproject.toml || true
[[ $want_dev  -eq 1 ]] && pip-compile –extra dev –no-header –no-annotate -q -o “$tmpdir/requirements-dev.txt” pyproject.toml || true
[[ $DO_FREEZE -eq 1 ]] && python -m pip freeze > “$tmpdir/requirements.freeze.txt” || true
# No min/kaggle derivation here (keep scope clear)
return 0
fi

warn “No exporter available (bin/export-reqs.sh or poetry export or pip-compile).”
return 1
}

if ! export_temp “$TMPDIR”; then
if [[ $DO_CHECK -eq 1 || $DO_WRITE -eq 1 ]]; then
fail “Failed to generate fresh requirements.”
emit_json false true false null
exit 1
else
warn “Skipping export; nothing to compare.”
fi
fi

––––– decide artifacts –––––

ARTS=()
if [[ $ALL -eq 1 || $INC_MAIN -eq 1 || ( $INC_MAIN -eq 0 && $INC_DEV -eq 0 && $INC_MIN -eq 0 && $INC_KAGGLE -eq 0 ) ]]; then
ARTS+=(“requirements.txt”)
fi
[[ $ALL -eq 1 || $INC_DEV -eq 1     ]] && ARTS+=(“requirements-dev.txt”)
[[ $ALL -eq 1 || $INC_MIN -eq 1     ]] && ARTS+=(“requirements-min.txt”)
[[ $ALL -eq 1 || $INC_KAGGLE -eq 1  ]] && ARTS+=(“requirements-kaggle.txt”)
[[ $DO_FREEZE -eq 1                 ]] && ARTS+=(“requirements.freeze.txt”)

––––– compare (normalized) & possibly write –––––

DRIFT=0
WROTE=0
DETAILS=”{”
FIRST=1

for f in “${ARTS[@]}”; do
src=”$TMPDIR/$f”
dst=”$OUTDIR/$f”

skip if source doesn’t exist (exporter omitted it)

if [[ ! -f “$src” ]]; then
warn “Fresh export missing $f (skipping).”
continue
fi

prepare normalized copies for drift check

ns=”$TMPDIR/.norm_source_${f}”
nd=”$TMPDIR/.norm_dest_${f}”
normalize_requirements “$src” “$ns”
if [[ -f “$dst” ]]; then
normalize_requirements “$dst” “$nd”
else
# create empty file to compare against “missing”
: > “$nd”
fi

JSON per-file state

file_state=”"$f": {”
file_state+=”"exists": $( [[ -f “$dst” ]] && echo true || echo false )”

if [[ ! -f “$dst” ]]; then
warn “Missing $dst in repo.”
if [[ $DO_WRITE -eq 1 ]]; then
say “Creating $dst”
[[ $DRY -eq 1 ]] || cp -f “$src” “$dst”
WROTE=1
file_state+=”, "action": "created", "sha256": "$(sha256_of “$src”)"”
else
DRIFT=1
file_state+=”, "action": "drift_missing"”
fi
else
if ! diff -u “$nd” “$ns” >/dev/null 2>&1; then
say “Drift detected in $f”
if [[ $QUIET -ne 1 ]]; then
echo “–– normalized diff: $f ––”
diff -u “$nd” “$ns” || true
echo “——————————”
fi
if [[ $DO_WRITE -eq 1 ]]; then
say “Updating $dst”
[[ $DRY -eq 1 ]] || cp -f “$src” “$dst”
WROTE=1
file_state+=”, "action": "updated", "sha256": "$(sha256_of “$src”)"”
else
DRIFT=1
file_state+=”, "action": "drift", "sha256": "$(sha256_of “$dst”)"”
fi
else
say “$f is in sync.”
file_state+=”, "action": "ok", "sha256": "$(sha256_of “$dst”)"”
fi
fi

file_state+=”}”
if [[ $FIRST -eq 1 ]]; then
DETAILS+=”$file_state”
FIRST=0
else
DETAILS+=”, $file_state”
fi
done
DETAILS+=”}”

––––– result –––––

if [[ $DO_CHECK -eq 1 && $DRIFT -ne 0 ]]; then
fail “Requirements are NOT in sync with the lock.”
emit_json false true false “$DETAILS”
exit 1
fi

if [[ $DO_WRITE -eq 1 ]]; then
say “${GRN}Wrote updated requirements artifacts.${RST}”
[[ $DRY -eq 1 ]] && say “(dry-run: no files actually modified)”
emit_json true “$( [[ $DRIFT -ne 0 ]] && echo true || echo false )” true “$DETAILS”
elif [[ $DO_CHECK -eq 1 ]]; then
say “${GRN}All checked artifacts are in sync.${RST}”
emit_json true false false “$DETAILS”
else
say “Sync summary completed. Use –check for CI or –write to update files.”
emit_json true “$( [[ $DRIFT -ne 0 ]] && echo true || echo false )” false “$DETAILS”
fi

––––– extra context (best-effort) –––––

if [[ -f “$POETRY_LOCK_FILE” && $QUIET -ne 1 ]]; then
say “Lock fingerprint: $(sha256_of “$POETRY_LOCK_FILE” | cut -c1-12)”
fi

exit 0