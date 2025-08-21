#!/usr/bin/env bash
# ==============================================================================
# bin/ci-smoke.sh — SpectraMind V50 CI/Kaggle-friendly smoke test
# ------------------------------------------------------------------------------
# Purpose
#   Run a fast, hermetic verification of the repo in CI (or locally) without
#   heavy compute. Validates env, installs deps, runs CLI selftest, produces
#   a light diagnostics dashboard, and (optionally) performs a tiny train/predict
#   smoke on CPU/GPU. Safe defaults prefer speed and portability.
#
# Usage
#   bin/ci-smoke.sh [options]
#
# Common examples
#   # Fast path: env + install + selftest + light dashboard + log summary
#   bin/ci-smoke.sh --fast
#
#   # Deeper path: also do tiny train + predict on GPU if available
#   bin/ci-smoke.sh --deep --device gpu
#
# Options
#   --fast                Minimal checks (default if none specified)
#   --deep                Include tiny training & predict smoke, extra probes
#   --device <cpu|gpu>    Device hint for tiny training (default: cpu)
#   --outdir <dir>        Outputs directory (default: outputs/diagnostics)
#   --timeout <sec>       Per-step timeout if `timeout` cmd exists (default: 600)
#   --no-install          Skip dependency installation
#   --no-selftest         Skip `spectramind selftest`
#   --no-diagnose         Skip dashboard generation
#   --no-train            Skip tiny training/predict step (even in --deep)
#   --no-log-summary      Skip `spectramind analyze-log-short`
#   --no-poetry           Do not use Poetry; run via system python/module
#   --dry-run             Print actions but perform no side-effects
#   --quiet               Reduce verbosity
#   -h|--help             Show help
#
# Exit codes
#   0 OK, 1 failure, 2 bad usage
#
# Notes
#   • CI-safe and Kaggle-safe. Avoids opening files; chooses light diagnostics.
#   • Does not download external data; relies on repo scripts to generate stubs.
# ==============================================================================

set -euo pipefail

# ---------- Pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[SMOKE]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[SMOKE]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[SMOKE]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- Defaults ----------
FAST=0
DEEP=0
DEVICE="cpu"
OUTDIR="outputs/diagnostics"
STEP_TIMEOUT=600
NO_INSTALL=0
NO_SELFTEST=0
NO_DIAGNOSE=0
NO_TRAIN=0
NO_LOG_SUMMARY=0
USE_POETRY=1
DRY=0
QUIET=0

# ---------- Args ----------
usage() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fast) FAST=1 ;;
    --deep) DEEP=1 ;;
    --device) DEVICE="${2:?}"; shift ;;
    --outdir) OUTDIR="${2:?}"; shift ;;
    --timeout) STEP_TIMEOUT="${2:?}"; shift ;;
    --no-install) NO_INSTALL=1 ;;
    --no-selftest) NO_SELFTEST=1 ;;
    --no-diagnose) NO_DIAGNOSE=1 ;;
    --no-train) NO_TRAIN=1 ;;
    --no-log-summary) NO_LOG_SUMMARY=1 ;;
    --no-poetry) USE_POETRY=0 ;;
    --dry-run) DRY=1 ;;
    --quiet) QUIET=1 ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown arg: $1"; usage; exit 2 ;;
  esac
  shift
done
# Default to fast if neither selected
if [[ $FAST -eq 0 && $DEEP -eq 0 ]]; then FAST=1; fi
# Deep implies not-fast
if [[ $DEEP -eq 1 ]]; then FAST=0; fi

# ---------- Repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; exit 1; }
fi

mkdir -p "$OUTDIR" logs outputs

# ---------- Env detection ----------
IS_CI=0; [[ -n "${GITHUB_ACTIONS:-}" || -n "${CI:-}" ]] && IS_CI=1
IS_KAGGLE=0; [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1

say "CI: ${BOLD}${IS_CI}${RST}  Kaggle: ${BOLD}${IS_KAGGLE}${RST}  Device: ${BOLD}${DEVICE}${RST}  Out: ${BOLD}${OUTDIR}${RST}"

# ---------- Timeout wrapper ----------
HAVE_TIMEOUT=0
if command -v timeout >/dev/null 2>&1; then HAVE_TIMEOUT=1; fi
with_timeout() {
  if [[ $HAVE_TIMEOUT -eq 1 && $STEP_TIMEOUT -gt 0 ]]; then
    timeout "${STEP_TIMEOUT}s" "$@"
  else
    "$@"
  fi
}

# ---------- Runner ----------
run() {
  local desc="$1"; shift
  [[ $DRY -eq 1 ]] && { say "[dry-run] $desc :: $*"; return 0; }
  [[ $QUIET -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "${desc}" "${RST}"
  if ! with_timeout "$@"; then
    fail "${desc} — command failed."
    return 1
  fi
}

FAILED=0

# ---------- Resolve CLI ----------
CLI_BIN=""
if [[ $USE_POETRY -eq 1 && -x "$(command -v poetry)" ]]; then
  CLI_BIN="poetry run spectramind"
elif command -v spectramind >/dev/null 2>&1; then
  CLI_BIN="spectramind"
else
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  CLI_BIN="$PY -m spectramind"
fi
say "CLI launcher: ${BOLD}${CLI_BIN}${RST}"

# ---------- Step: install deps ----------
if [[ $NO_INSTALL -eq 1 ]]; then
  warn "Skipping dependency installation (--no-install)."
else
  if [[ $USE_POETRY -eq 1 && -x "$(command -v poetry)" ]]; then
    run "Poetry install (no-root)" poetry install --no-root || FAILED=1
  else
    warn "Poetry not available; assuming environment already satisfied."
  fi
fi

# ---------- Step: basic env checks ----------
if command -v python3 >/dev/null 2>&1; then ok_py=1; else ok_py=0; fi
if [[ $ok_py -eq 0 ]]; then fail "python3 not found"; exit 1; fi
if command -v nvidia-smi >/dev/null 2>&1; then
  if [[ $DEEP -eq 1 ]]; then run "nvidia-smi" nvidia-smi || true; fi
else
  warn "CUDA not detected (this is OK for CPU smoke)."
fi

# ---------- Step: spectramind --version ----------
run "spectramind --version" ${CLI_BIN} --version || FAILED=1

# ---------- Step: selftest ----------
if [[ $NO_SELFTEST -eq 1 ]]; then
  warn "Skipping CLI selftest (--no-selftest)."
else
  if [[ $DEEP -eq 1 ]]; then
    run "spectramind selftest --deep" ${CLI_BIN} selftest --deep || FAILED=1
  else
    run "spectramind selftest" ${CLI_BIN} selftest || FAILED=1
  fi
fi

# ---------- Step: light diagnostics ----------
if [[ $NO_DIAGNOSE -eq 1 ]]; then
  warn "Skipping diagnostics (--no-diagnose)."
else
  # Light path always disables UMAP/t-SNE
  run "diagnose smoothness" ${CLI_BIN} diagnose smoothness --outdir "$OUTDIR" || FAILED=1
  run "diagnose dashboard (light)" ${CLI_BIN} diagnose dashboard --no-umap --no-tsne --outdir "$OUTDIR" || FAILED=1
fi

# ---------- Step: tiny train / predict ----------
if [[ $DEEP -eq 1 && $NO_TRAIN -eq 0 ]]; then
  if [[ $IS_KAGGLE -eq 1 ]]; then
    warn "Skipping tiny training in Kaggle environment (to conserve session quota)."
  else
    run "tiny training run" ${CLI_BIN} train +training.epochs=1 --device "$DEVICE" --outdir "outputs/checkpoints/_smoke_${DEVICE}" || FAILED=1
    # produce a small CSV to ensure predict path exercises I/O
    run "predict (stub CSV)" ${CLI_BIN} predict --out-csv "outputs/predictions/_smoke_submission.csv" || FAILED=1
  fi
fi

# ---------- Step: analyze log short ----------
if [[ $NO_LOG_SUMMARY -eq 1 ]]; then
  warn "Skipping log summary (--no-log-summary)."
else
  run "analyze-log-short" ${CLI_BIN} analyze-log-short || FAILED=1
fi

# ---------- Optional: DVC status (non-fatal) ----------
if command -v dvc >/dev/null 2>&1 && [[ -d ".dvc" ]]; then
  run "dvc status (non-fatal)" dvc status || true
fi

# ---------- Result ----------
if [[ $FAILED -eq 0 ]]; then
  printf "%s✔%s CI smoke passed.\n" "${GRN}" "${RST}"
  # Print the latest diagnostics HTML path (if present)
  latest="$(ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true)"
  [[ -n "$latest" ]] && say "Latest diagnostics: ${BOLD}${latest}${RST}"
  exit 0
else
  printf "%s✘%s CI smoke encountered issues.\n" "${RED}" "${RST}"
  exit 1
fi