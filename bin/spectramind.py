#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — CLI shim (upgraded)
# Runs the unified Typer app with smart fallbacks across Poetry / venv / module.
#
# Usage:
#   bin/spectramind [shim-flags] -- [app-args]
#   bin/spectramind [app-args]               # shim-flags optional
#
# Examples:
#   bin/spectramind --version
#   bin/spectramind selftest --deep
#   bin/spectramind train +training.epochs=1 --device gpu
#
# Shim flags (optional; must precede “--” or normal args):
#   --no-poetry       Do not try Poetry even if installed
#   --venv PATH       Activate a specific venv (default: .venv if present)
#   --no-venv         Do not auto-activate .venv
#   --prefer-script   Prefer running local script if present (repo dev)
#   --prefer-module   Prefer python -m spectramind (default on Kaggle)
#   --dry-run         Print resolved command without executing
#   --quiet           Reduce shim verbosity
#   -h|--help         Show this help and exit
#
# Resolution order (non-Kaggle, defaults):
#   1) Poetry:        poetry run spectramind
#   2) Module:        python -m spectramind
#   3) Repo script:   python spectramind.py  (or src path injection)
#
# On Kaggle (auto-detected): skip Poetry, prefer Module → Script fallback.
#
# Environment hardening:
#   • PYTHONUNBUFFERED=1, HYDRA_FULL_ERROR=1, MPLBACKEND=Agg
#   • TOKENIZERS_PARALLELISM=false (quiets HF spam)
#   • Creates outputs/, logs/, outputs/{diagnostics,predictions,submission}/
#
# Exit codes:
#   0 success, 1 failure to locate entrypoint / runtime error
# ==============================================================================

set -euo pipefail

# -------- colors / logging --------
CYN=$'\033[36m'; YLW=$'\033[33m'; RED=$'\033[31m'; GRN=$'\033[32m'; DIM=$'\033[2m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf "%s[SM]%s %s\n" "$CYN" "$RST" "$*"; }
warn() { printf "%s[SM]%s %s\n" "$YLW" "$RST" "$*" >&2; }
fail() { printf "%s[SM]%s %s\n" "$RED" "$RST" "$*" >&2; exit 1; }

# -------- defaults (shim) --------
NO_POETRY=0
VENV_PATH=""
NO_VENV=0
PREFER_SCRIPT=0
PREFER_MODULE=0
DRY=0
QUIET=0

print_help() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }

# -------- parse shim flags (up to --) --------
SHIM_ARGS=()
APP_ARGS=()
if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do
    case "${1:-}" in
      --) shift; APP_ARGS=("$@"); break ;;
      --no-poetry)    NO_POETRY=1 ;;
      --venv)         VENV_PATH="${2:-}"; shift ;;
      --no-venv)      NO_VENV=1 ;;
      --prefer-script) PREFER_SCRIPT=1 ;;
      --prefer-module) PREFER_MODULE=1 ;;
      --dry-run)      DRY=1 ;;
      --quiet)        QUIET=1 ;;
      -h|--help)      print_help; exit 0 ;;
      --version)      # Pass-through convenience
        APP_ARGS=(--version); shift; break ;;
      --*)            warn "Unknown shim flag: $1 (forwarding to app)"; APP_ARGS=("$@"); break ;;
      *)              APP_ARGS=("$@"); break ;;
    esac
    shift
  done
fi

# -------- locate repo root (best-effort) --------
if git_root="$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT="$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

# -------- basic dirs to keep tools happy --------
mkdir -p outputs logs outputs/diagnostics outputs/predictions outputs/submission

# -------- env hardening for reproducibility --------
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export MPLBACKEND=${MPLBACKEND:-Agg}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# -------- detect Kaggle --------
is_kaggle=0
if [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  is_kaggle=1
fi

# -------- python locator / helpers --------
py_cmd() {
  local py="python3"
  command -v "$py" >/dev/null 2>&1 || py="python"
  printf "%s" "$py"
}

can_import_spectramind() {
  local py; py="$(py_cmd)"
  "$py" - <<'PY' >/dev/null 2>&1
try:
    import spectramind  # noqa: F401
except Exception:
    raise SystemExit(1)
else:
    raise SystemExit(0)
PY
}

# Add repo src to PYTHONPATH if module not installed but src/ exists
inject_src_path() {
  if [[ -d "$ROOT/src" ]]; then
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/src"
  fi
}

# -------- runners --------
run_with_poetry() {
  command -v poetry >/dev/null 2>&1 || return 1
  [[ $NO_POETRY -eq 1 ]] && return 1
  say "Running via Poetry: spectramind ${APP_ARGS[*]}"
  [[ $DRY -eq 1 ]] && { printf "%s\n" "poetry run spectramind ${APP_ARGS[*]}"; return 0; }
  exec poetry run spectramind "${APP_ARGS[@]}"
}

run_with_module() {
  local py; py="$(py_cmd)"
  say "Running via module: $py -m spectramind ${APP_ARGS[*]}"
  [[ $DRY -eq 1 ]] && { printf "%s\n" "$py -m spectramind ${APP_ARGS[*]}"; return 0; }
  exec "$py" -m spectramind "${APP_ARGS[@]}"
}

run_with_script() {
  local py; py="$(py_cmd)"
  if [[ -f "$ROOT/spectramind.py" ]]; then
    say "Running local script: $py spectramind.py ${APP_ARGS[*]}"
    [[ $DRY -eq 1 ]] && { printf "%s\n" "$py $ROOT/spectramind.py ${APP_ARGS[*]}"; return 0; }
    exec "$py" "$ROOT/spectramind.py" "${APP_ARGS[@]}"
  fi
  # Try common dev entry (src package main)
  if [[ -f "$ROOT/src/spectramind/__main__.py" ]]; then
    inject_src_path
    say "Running src package: $py -m spectramind ${APP_ARGS[*]}"
    [[ $DRY -eq 1 ]] && { printf "%s\n" "$py -m spectramind ${APP_ARGS[*]}"; return 0; }
    exec "$py" -m spectramind "${APP_ARGS[@]}"
  fi
  return 1
}

# -------- venv activation (non-Kaggle) --------
if [[ $is_kaggle -eq 0 && $NO_VENV -eq 0 ]]; then
  if [[ -n "$VENV_PATH" && -d "$VENV_PATH" ]]; then
    # shellcheck disable=SC1091
    [[ -f "$VENV_PATH/bin/activate" ]] && source "$VENV_PATH/bin/activate"
  elif [[ -d "$ROOT/.venv" ]]; then
    # shellcheck disable=SC1091
    [[ -f "$ROOT/.venv/bin/activate" ]] && source "$ROOT/.venv/bin/activate"
  fi
fi

# -------- fast path on Kaggle --------
if [[ "$is_kaggle" -eq 1 ]]; then
  # Kaggle often has checkout under /kaggle/working and not installed as package
  inject_src_path
  # Prefer module unless user asked for script
  if [[ $PREFER_SCRIPT -eq 1 ]]; then
    run_with_script || run_with_module || fail "Could not locate SpectraMind entrypoint."
  else
    run_with_module || run_with_script || fail "Could not locate SpectraMind entrypoint."
  fi
fi

# -------- non-Kaggle dispatch --------
# Optional preference switches
if [[ $PREFER_SCRIPT -eq 1 ]]; then
  run_with_script || run_with_poetry || run_with_module || fail "Could not locate SpectraMind entrypoint."
fi
if [[ $PREFER_MODULE -eq 1 ]]; then
  inject_src_path
  run_with_module || run_with_poetry || run_with_script || fail "Could not locate SpectraMind entrypoint."
fi

# Default: Poetry → Module → Script
# If spectramind module is not importable, try to inject src path before module run
if ! can_import_spectramind >/dev/null 2>&1; then
  inject_src_path
fi

run_with_poetry || run_with_module || run_with_script || fail "Could not locate SpectraMind entrypoint.

Tried (in order):
  1) poetry run spectramind              $( [[ $NO_POETRY -eq 1 ]] && echo '(skipped by --no-poetry)' )
  2) python -m spectramind               $( [[ -n "${PYTHONPATH:-}" ]] && echo '(with PYTHONPATH) ')
  3) python spectramind.py               (repo dev script)

Hints:
  • If using Poetry: 'poetry install --no-root' then re-run.
  • Or install editable: 'python -m pip install -e .'
  • Ensure one of: 'spectramind.py', 'src/spectramind/__init__.py' exists.
"