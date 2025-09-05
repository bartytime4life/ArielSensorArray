#!/usr/bin/env bash

==============================================================================

SpectraMind V50 — CLI shim (ultimate, upgraded)

Runs the unified Typer app with smart fallbacks across Poetry / venv / module.



Usage:

bin/spectramind [shim-flags] – [app-args]

bin/spectramind [app-args]               # shim-flags optional



Examples:

bin/spectramind –version

bin/spectramind selftest –deep

bin/spectramind train +training.epochs=1 –device gpu



Shim flags (optional; must precede “–” or normal args):

–no-poetry         Do not try Poetry even if installed

–venv PATH         Activate a specific venv (default: .venv if present)

–no-venv           Do not auto-activate .venv

–conda NAME|PATH   Activate a conda env by name or path (if conda present)

–no-conda          Do not attempt conda activation

–prefer-script     Prefer running local script if present (repo dev)

–prefer-module     Prefer python -m spectramind (default on Kaggle)

–dry-run           Print resolved command without executing

–quiet             Reduce shim verbosity

–log-line          Append a one-line invocation record to logs/v50_debug_log.md

–print-python      Print the resolved python interpreter path and exit

-h|–help           Show this help and exit



Resolution order (non-Kaggle, defaults):

1) Poetry:        poetry run spectramind

2) Module:        python -m spectramind      (injects src/ to PYTHONPATH if needed)

3) Repo script:   python spectramind.py      (or src package main)



On Kaggle (auto-detected): skip Poetry, prefer Module → Script fallback.



Environment hardening:

• PYTHONUNBUFFERED=1, HYDRA_FULL_ERROR=1, MPLBACKEND=Agg

• TOKENIZERS_PARALLELISM=false (quiets HF spam)

• CUBLAS_WORKSPACE_CONFIG=:16:8 for determinism (non-fatal if ignored)

• Creates outputs/, logs/, outputs/{diagnostics,predictions,submission}/



Exit codes:

0 success, 1 failure to locate entrypoint / runtime error, 2 usage error

==============================================================================

set -Eeuo pipefail

–––– colors / logging (TTY-aware) ––––

if [[ -t 1 ]]; then
CYN=$’\033[36m’; YLW=$’\033[33m’; RED=$’\033[31m’; GRN=$’\033[32m’; DIM=$’\033[2m’; RST=$’\033[0m’
else
CYN=’’; YLW=’’; RED=’’; GRN=’’; DIM=’’; RST=’’
fi
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf “%s[SM]%s %s\n” “$CYN” “$RST” “$”; }
warn() { printf “%s[SM]%s %s\n” “$YLW” “$RST” “$” >&2; }
fail() { printf “%s[SM]%s %s\n” “$RED” “$RST” “$*” >&2; exit 1; }

–––– defaults (shim) ––––

NO_POETRY=0
VENV_PATH=””
NO_VENV=0
CONDA_ARG=””
NO_CONDA=0
PREFER_SCRIPT=0
PREFER_MODULE=0
DRY=0
QUIET=0
LOG_LINE=0
PRINT_PY=0

print_help() {
sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’
}

–––– parse shim flags (up to –) ––––

APP_ARGS=()
while [[ $# -gt 0 ]]; do
case “${1:-}” in
–) shift; APP_ARGS=(”$@”); break ;;
–no-poetry)       NO_POETRY=1 ;;
–venv)            VENV_PATH=”${2:-}”; shift ;;
–no-venv)         NO_VENV=1 ;;
–conda)           CONDA_ARG=”${2:-}”; shift ;;
–no-conda)        NO_CONDA=1 ;;
–prefer-script)   PREFER_SCRIPT=1 ;;
–prefer-module)   PREFER_MODULE=1 ;;
–dry-run)         DRY=1 ;;
–quiet)           QUIET=1 ;;
–log-line)        LOG_LINE=1 ;;
–print-python)    PRINT_PY=1 ;;
–version)         APP_ARGS=(–version); shift; break ;; # convenience passthrough
-h|–help)         print_help; exit 0 ;;
–*)               APP_ARGS=(”$@”); break ;;            # forward remainder to app
*)                 APP_ARGS=(”$@”); break ;;            # first non-flag = app args
esac
shift
done

–––– locate repo root (best-effort) ––––

if command -v git >/dev/null 2>&1 && git_root=”$(git rev-parse –show-toplevel 2>/dev/null)”; then
ROOT=”$git_root”
else
SCRIPT_DIR=”$(cd – “$(dirname – “${BASH_SOURCE[0]}”)” && pwd)”
ROOT=”$(cd “$SCRIPT_DIR/..” && pwd)”
fi
cd “$ROOT”

–––– basic dirs to keep tools happy ––––

mkdir -p outputs logs outputs/diagnostics outputs/predictions outputs/submission

–––– env hardening for reproducibility ––––

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLBACKEND=”${MPLBACKEND:-Agg}”
export TOKENIZERS_PARALLELISM=”${TOKENIZERS_PARALLELISM:-false}”
export CUBLAS_WORKSPACE_CONFIG=”${CUBLAS_WORKSPACE_CONFIG:-:16:8}”

–––– detect Kaggle ––––

is_kaggle=0
if [[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]]; then
is_kaggle=1
fi

–––– helpers ––––

have() { command -v “$1” >/dev/null 2>&1; }

py_cmd() {

Choose the most specific, existing python

if [[ -n “${VIRTUAL_ENV:-}” && -x “${VIRTUAL_ENV}/bin/python” ]]; then
printf ‘%s’ “${VIRTUAL_ENV}/bin/python”; return 0
fi
if have python3; then printf ‘%s’ “python3”; return 0; fi
if have python;  then printf ‘%s’ “python”;  return 0; fi
printf ‘%s’ “python3”
}

activate_conda() {
[[ $NO_CONDA -eq 1 ]] && return 0
[[ -n “$CONDA_ARG” ]] || return 0
if have conda; then
# shellcheck disable=SC1091
local conda_sh
conda_sh=”$(conda info –base 2>/dev/null)/etc/profile.d/conda.sh”
if [[ -f “$conda_sh” ]]; then
# shellcheck disable=SC1090
source “$conda_sh”
if [[ -d “$CONDA_ARG” ]]; then
conda activate “$CONDA_ARG” || warn “Conda activate by path failed: $CONDA_ARG”
else
conda activate “$CONDA_ARG” || warn “Conda activate by name failed: $CONDA_ARG”
fi
else
warn “Conda profile script not found; skipping conda activation.”
fi
fi
}

activate_venv() {
[[ $NO_VENV -eq 1 ]] && return 0
local candidate=””
if [[ -n “$VENV_PATH” && -d “$VENV_PATH” ]]; then
candidate=”$VENV_PATH”
elif [[ -d “$ROOT/.venv” ]]; then
candidate=”$ROOT/.venv”
fi
if [[ -n “$candidate” && -f “$candidate/bin/activate” ]]; then
# shellcheck disable=SC1090
source “$candidate/bin/activate” || warn “Failed to activate venv: $candidate”
fi
}

can_import_spectramind() {
local py; py=”$(py_cmd)”
“$py” - <<‘PY’ >/dev/null 2>&1
try:
import spectramind  # noqa: F401
except Exception:
raise SystemExit(1)
else:
raise SystemExit(0)
PY
}

inject_src_path() {
if [[ -d “$ROOT/src” ]]; then
export PYTHONPATH=”${PYTHONPATH:+$PYTHONPATH:}$ROOT/src”
fi
}

resolve_python_path() {
if [[ $PRINT_PY -eq 1 ]]; then
command -v “$(py_cmd)” || true
exit 0
fi
}

append_log_line() {
[[ $LOG_LINE -ne 1 ]] && return 0
local log=“logs/v50_debug_log.md”
mkdir -p “$(dirname “$log”)”
local ts sha cfg ver
ts=”$(date -u +%Y-%m-%dT%H:%M:%SZ)”
if command -v git >/dev/null 2>&1; then
sha=”$(git rev-parse –short HEAD 2>/dev/null || echo nogit)”
else
sha=“nogit”
fi

Best-effort: CLI version and config hash

ver=“unknown”; cfg=”-”
{
if have spectramind; then
ver=”$(spectramind –version 2>/dev/null | head -n1 || echo unknown)”
if spectramind –help 2>/dev/null | grep -qiE – “–print-config-hash|hash-config”; then
cfg=”$(spectramind –print-config-hash 2>/dev/null || spectramind hash-config 2>/dev/null || echo -)”
fi
fi
} || true
printf ‘%s cmd=%s git=%s cfg_hash=%s cli_ver=”%s” argv=”%s”\n’ 
“$ts” “spectramind-shim” “$sha” “$cfg” “$ver” “$*” >> “$log” || true
}

–––– runners (exec on success) ––––

run_with_poetry() {
command -v poetry >/dev/null 2>&1 || return 1
[[ $NO_POETRY -eq 1 ]] && return 1
say “Running via Poetry: spectramind ${APP_ARGS[]}”
if [[ $DRY -eq 1 ]]; then
printf “%s\n” “poetry run spectramind ${APP_ARGS[]}”
return 0
fi
exec poetry run spectramind “${APP_ARGS[@]}”
}

run_with_module() {
local py; py=”$(py_cmd)”
say “Running via module: $py -m spectramind ${APP_ARGS[]}”
if [[ $DRY -eq 1 ]]; then
printf “%s\n” “$py -m spectramind ${APP_ARGS[]}”
return 0
fi
exec “$py” -m spectramind “${APP_ARGS[@]}”
}

run_with_script() {
local py; py=”$(py_cmd)”
if [[ -f “$ROOT/spectramind.py” ]]; then
say “Running local script: $py spectramind.py ${APP_ARGS[]}”
if [[ $DRY -eq 1 ]]; then
printf “%s\n” “$py $ROOT/spectramind.py ${APP_ARGS[]}”
return 0
fi
exec “$py” “$ROOT/spectramind.py” “${APP_ARGS[@]}”
fi
if [[ -f “$ROOT/src/spectramind/main.py” || -f “$ROOT/src/spectramind/init.py” ]]; then
inject_src_path
say “Running src package: $py -m spectramind ${APP_ARGS[]}”
if [[ $DRY -eq 1 ]]; then
printf “%s\n” “$py -m spectramind ${APP_ARGS[]}”
return 0
fi
exec “$py” -m spectramind “${APP_ARGS[@]}”
fi
return 1
}

–––– activate environments (order: conda → venv) ––––

activate_conda
activate_venv
resolve_python_path

–––– fast path on Kaggle ––––

if [[ “$is_kaggle” -eq 1 ]]; then
inject_src_path
append_log_line “${APP_ARGS[*]}”
if [[ $PREFER_SCRIPT -eq 1 ]]; then
run_with_script || run_with_module || fail “Could not locate SpectraMind entrypoint (Kaggle).”
else
run_with_module || run_with_script || fail “Could not locate SpectraMind entrypoint (Kaggle).”
fi
fi

–––– non-Kaggle dispatch ––––

Optional preference switches

if [[ $PREFER_SCRIPT -eq 1 ]]; then
append_log_line “${APP_ARGS[*]}”
run_with_script || run_with_poetry || { inject_src_path; run_with_module; } || fail “Could not locate SpectraMind entrypoint.”
fi

if [[ $PREFER_MODULE -eq 1 ]]; then
inject_src_path
append_log_line “${APP_ARGS[*]}”
run_with_module || run_with_poetry || run_with_script || fail “Could not locate SpectraMind entrypoint.”
fi

Default: Poetry → Module (with src inject if needed) → Script

if ! can_import_spectramind >/dev/null 2>&1; then
inject_src_path
fi

append_log_line “${APP_ARGS[*]}”
run_with_poetry || run_with_module || run_with_script || fail “Could not locate SpectraMind entrypoint.

Tried (in order):
	1.	poetry run spectramind              $( [[ $NO_POETRY -eq 1 ]] && echo ‘(skipped by –no-poetry)’ )
	2.	python -m spectramind               $( [[ -n “${PYTHONPATH:-}” ]] && echo ‘(with PYTHONPATH)’ )
	3.	python spectramind.py               (repo dev script)

Hints:
• If using Poetry: ‘poetry install –no-root’ then re-run.
• Or install editable: ‘python -m pip install -e .’
• Ensure one of: ‘spectramind.py’ or ‘src/spectramind/init.py’ exists.
“