#!/usr/bin/env bash
# ==============================================================================
# bin/ci-smoke.sh — SpectraMind V50 CI/Kaggle-friendly smoke test (upgraded)
# ------------------------------------------------------------------------------
# Purpose
#   Run a fast, hermetic verification of the repo in CI (or locally) without
#   heavy compute. Validates env, (optionally) installs deps, runs CLI selftest,
#   produces a light diagnostics dashboard, and (optionally) performs a tiny
#   train/predict smoke on CPU/GPU. Safe defaults prefer speed and portability.
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
#   Modes
#     --fast                Minimal checks (default if none specified)
#     --deep                Include tiny training & predict smoke, extra probes
#   Runtime
#     --device <cpu|gpu>    Device hint for tiny training (default: cpu)
#     --outdir <dir>        Outputs directory (default: outputs/diagnostics)
#     --timeout <sec>       Per-step timeout if `timeout` exists (default: 600)
#   Gates / toggles
#     --no-install          Skip dependency installation
#     --no-selftest         Skip `spectramind selftest`
#     --no-diagnose         Skip dashboard generation
#     --no-train            Skip tiny training/predict step (even in --deep)
#     --no-log-summary      Skip `spectramind analyze-log-short`
#     --no-poetry           Do not use Poetry; run via system python/module
#     --dvc-pull            Run `bin/dvc-pull.sh --status-only` then pull
#     --sync-lock           Enforce requirements sync (bin/sync-lock.sh --check)
#     --fix-lock            If sync-lock fails, attempt fix (--lock --write)
#   Output
#     --log <path>          Log file (default: logs/ci/ci-smoke-<ts>.log)
#     --json                Emit JSON result summary to stdout
#     --dry-run             Print actions but perform no side-effects
#     --quiet               Reduce verbosity
#     -h|--help             Show help
#
# Exit codes
#   0 OK, 1 failure, 2 bad usage
#
# Notes
#   • CI-safe and Kaggle-safe. Avoids opening files; chooses light diagnostics.
#   • Does not download external data; relies on repo scripts to generate stubs.
#   • Idempotent: repeated runs should not mutate model/data state.
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
STEP_TIMEOUT="${CI_SMOKE_TIMEOUT:-600}"
NO_INSTALL=0
NO_SELFTEST=0
NO_DIAGNOSE=0
NO_TRAIN=0
NO_LOG_SUMMARY=0
USE_POETRY=1
DO_DVC_PULL=0
ENFORCE_SYNC=0
FIX_LOCK=0
DRY=0
QUIET=0
JSON_OUT=0

TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH_DEFAULT="logs/ci/ci-smoke-${TS}.log"
LOG_PATH=""

# ---------- Args ----------
usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

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
    --dvc-pull) DO_DVC_PULL=1 ;;
    --sync-lock) ENFORCE_SYNC=1 ;;
    --fix-lock) FIX_LOCK=1 ;;
    --log) LOG_PATH="${2:?}"; shift ;;
    --json) JSON_OUT=1 ;;
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

mkdir -p "$OUTDIR" logs outputs logs/ci

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

# ---------- Logging ----------
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="$LOG_PATH_DEFAULT"
fi
# tee logs unless quiet
if [[ "$QUIET" -eq 0 ]]; then
  # shellcheck disable=SC2094
  exec > >(tee -a "${LOG_PATH}") 2>&1
else
  exec >> "${LOG_PATH}" 2>&1
fi
say "Log file: ${BOLD}${LOG_PATH}${RST}"

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

# ---------- Optional: preflight DVC pull ----------
if [[ $DO_DVC_PULL -eq 1 && -d ".dvc" && -x "bin/dvc-pull.sh" ]]; then
  say "DVC preflight: status-only against remote…"
  run "bin/dvc-pull.sh --status-only --json" bash -lc "bin/dvc-pull.sh --status-only --json" || true
  say "DVC preflight: pulling workspace objects…"
  run "bin/dvc-pull.sh --json" bash -lc "bin/dvc-pull.sh --json" || FAILED=1
fi

# ---------- Optional: sync-lock enforcement ----------
if [[ $ENFORCE_SYNC -eq 1 && -x "bin/sync-lock.sh" ]]; then
  say "Enforcing requirements sync via bin/sync-lock.sh --check"
  if run "sync-lock check" bash -lc "bin/sync-lock.sh --check --all --json >/dev/null"; then
    say "Requirements are in sync."
  else
    if [[ $FIX_LOCK -eq 1 ]]; then
      warn "Sync-lock drift detected; attempting fix with --lock --write --all"
      run "sync-lock write" bash -lc "bin/sync-lock.sh --lock --write --all" || FAILED=1
    else
      fail "Requirements drift (use --fix-lock to regenerate)."
      FAILED=1
    fi
  fi
fi

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
if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  fail "Python not found"; exit 1
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  [[ $DEEP -eq 1 ]] && run "nvidia-smi" nvidia-smi || true
else
  warn "CUDA not detected (CPU mode)."
fi

# ---------- Step: version alignment (non-fatal gate) ----------
if [[ -f "bin/version_tools.py" ]]; then
  run "version validate" python bin/version_tools.py --validate || FAILED=1
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
    warn "Skipping tiny training in Kaggle environment (conserve quota)."
  else
    run "tiny training run" ${CLI_BIN} train +training.epochs=1 --device "$DEVICE" --outdir "outputs/checkpoints/_smoke_${DEVICE}" || FAILED=1
    # produce a small CSV to ensure predict path exercises I/O
    mkdir -p outputs/predictions
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
LATEST_HTML="$(ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true)"
if [[ $FAILED -eq 0 ]]; then
  printf "%s✔%s CI smoke passed.\n" "${GRN}" "${RST}"
  [[ -n "$LATEST_HTML" ]] && say "Latest diagnostics: ${BOLD}${LATEST_HTML}${RST}"
  OK=true; EC=0
else
  printf "%s✘%s CI smoke encountered issues.\n" "${RED}" "${RST}"
  OK=false; EC=1
fi

# ---------- JSON summary ----------
if [[ $JSON_OUT -eq 1 ]]; then
  START_ISO="${TS:0:4}-${TS:4:2}-${TS:6:2}T${TS:9:2}:${TS:11:2}:${TS:13:2}Z"
  END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{'
  printf '"ok": %s, ' "$([[ $OK == true ]] && echo true || echo false)"
  printf '"mode": "%s", ' "$([[ $DEEP -eq 1 ]] && echo deep || echo fast)"
  printf '"device": "%s", ' "$DEVICE"
  printf '"outdir": "%s", ' "$OUTDIR"
  printf '"latest_html": "%s", ' "${LATEST_HTML:-}"
  printf '"log": "%s", ' "$LOG_PATH"
  printf '"ci": %s, "kaggle": %s, ' "$([[ $IS_CI -eq 1 ]] && echo true || echo false)" "$([[ $IS_KAGGLE -eq 1 ]] && echo true || echo false)"
  printf '"start": "%s", "end": "%s"' "$START_ISO" "$END_ISO"
  printf '}\n'
fi

exit "$EC"
