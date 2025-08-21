#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — CLI shim
# Runs the unified Typer app via Poetry if available, with fallbacks.
# Usage:
#   bin/spectramind [subcommand] [--flags...]
# Examples:
#   bin/spectramind --version
#   bin/spectramind selftest --deep
#   bin/spectramind train +training.epochs=1 --device gpu
# ==============================================================================

set -euo pipefail

# -------- locate repo root (best-effort) --------
if git_root="$(git rev-parse --show-toplevel 2>/dev/null)"; then
  ROOT="$git_root"
else
  # fallback to the directory containing this script, then its parent
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

# -------- basic dirs to keep tools happy --------
mkdir -p outputs logs outputs/diagnostics outputs/predictions outputs/submission

# -------- helpers --------
say()   { printf "\033[36m[SM]\033[0m %s\n" "$*"; }
warn()  { printf "\033[33m[SM]\033[0m %s\n" "$*" >&2; }
fail()  { printf "\033[31m[SM]\033[0m %s\n" "$*" >&2; exit 1; }

# Detect Kaggle env (no Poetry there by default)
is_kaggle=0
if [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]]; then
  is_kaggle=1
fi

# -------- choose runner in priority order --------
run_with_poetry() {
  if command -v poetry >/dev/null 2>&1; then
    say "Running via Poetry: spectramind $*"
    exec poetry run spectramind "$@"
  fi
  return 1
}

run_with_module() {
  local py=python3
  command -v "$py" >/dev/null 2>&1 || py=python
  say "Running via module: $py -m spectramind $*"
  exec "$py" -m spectramind "$@"
}

run_with_script() {
  local py=python3
  command -v "$py" >/dev/null 2>&1 || py=python
  if [[ -f "$ROOT/spectramind.py" ]]; then
    say "Running local script: $py spectramind.py $*"
    exec "$py" "$ROOT/spectramind.py" "$@"
  fi
  return 1
}

# -------- main dispatch --------
# Fast path for Kaggle (skip Poetry detection noise)
if [[ "$is_kaggle" -eq 1 ]]; then
  run_with_module "$@" || run_with_script "$@" || fail "Could not locate SpectraMind entrypoint."
fi

# Local/dev path: prefer Poetry if available
run_with_poetry "$@" || run_with_module "$@" || run_with_script "$@" || fail "Could not locate SpectraMind entrypoint.

Tried:
  1) poetry run spectramind
  2) python -m spectramind
  3) python spectramind.py

Hint:
  • If you use Poetry: 'poetry install --no-root'
  • Or run: 'python -m pip install -e .' if you expose a console_script.
  • Ensure spectramind.py or src/spectramind/__init__.py exists."