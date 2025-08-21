#!/usr/bin/env bash
# ==============================================================================
# bin/selftest.sh — SpectraMind V50 quick/deep health check
# ------------------------------------------------------------------------------
# Goals:
#   • Verify local environment (Python/Poetry/CLI/CUDA/DVC) and repo layout
#   • Exercise the CLI in a safe, CI/Kaggle‑friendly way (no heavy runs by default)
#   • Emit a concise pass/fail summary and proper exit code for automation
#
# Usage:
#   bin/selftest.sh [--deep] [--quick] [--skip-heavy] [--no-poetry]
#                   [--log <path>] [--device <cpu|gpu>] [--quiet]
#
# Flags:
#   --deep         : run additional checks (CUDA/DVC, extended CLI tests)
#   --quick        : minimal checks only (fast path; overrides --deep parts)
#   --skip-heavy   : skip even the light diagnostics/dashboard step
#   --no-poetry    : do not use Poetry; try `spectramind` or `python -m spectramind`
#   --device <d>   : cpu|gpu (hint for the training/test step; default: cpu)
#   --log <path>   : path to write a copy of stdout/stderr (default: logs/selftest.log)
#   --quiet        : less verbose output
#
# Exit codes:
#   0 OK, 1 failure, 2 bad usage
#
# Notes:
#   • Safe on Kaggle: auto‑detects env and skips non‑portable steps.
#   • No data mutation; produces only logs and lightweight artifacts.
# ==============================================================================

set -euo pipefail

# ---------- Pretty printing ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'

say()   { printf '%s[SELFTEST]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn()  { printf '%s[SELFTEST]%s %s\n' "${YLW}" "${RST}" "$*" >&2; }
fail()  { printf '%s[SELFTEST]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

hr()    { printf '%s\n' "────────────────────────────────────────────────────────────────────"; }

# ---------- Defaults / args ----------
DEEP=0
QUICK=0
SKIP_HEAVY=0
USE_POETRY=1
DEVICE="cpu"
QUIET=0
LOG_PATH="logs/selftest.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deep)        DEEP=1 ;;
    --quick)       QUICK=1 ;;
    --skip-heavy)  SKIP_HEAVY=1 ;;
    --no-poetry)   USE_POETRY=0 ;;
    --device)      DEVICE="${2:-cpu}"; shift ;;
    --log)         LOG_PATH="${2:-logs/selftest.log}"; shift ;;
    --quiet)       QUIET=1 ;;
    -h|--help)
      sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      fail "Unknown argument: $1"
      exit 2
      ;;
  esac
  shift
done

# QUICK overrides DEEP heavy bits
if [[ "$QUICK" -eq 1 ]]; then
  DEEP=0
  SKIP_HEAVY=1
fi

START_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# ---------- Locate repo root ----------
# Prefer git; fallback to script dir parent
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  ROOT="$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT"

# ---------- Prepare dirs & logging ----------
mkdir -p logs outputs outputs/diagnostics
# tee logs unless quiet
if [[ "$QUIET" -eq 0 ]]; then
  # shellcheck disable=SC2094
  exec > >(tee -a "${LOG_PATH}") 2>&1
else
  exec >> "${LOG_PATH}" 2>&1
fi

say "SpectraMind V50 selftest started at ${BOLD}${START_TS}${RST}"
say "Repository root: ${BOLD}${ROOT}${RST}"
hr

# ---------- Utility: runner with echo ----------
run() {
  local desc="$1"; shift
  if [[ "$QUIET" -eq 0 ]]; then
    printf "%s→ %s%s%s\n" "${DIM}" "${desc}" "${RST}" ""
    printf "%s$ %s%s\n" "${DIM}" "$*" "${RST}"
  fi
  if ! "$@"; then
    fail "${desc} — command failed: $*"
    return 1
  fi
}

okmark()  { printf "%s✔%s %s\n" "${GRN}" "${RST}" "$*"; }
badmark() { printf "%s✘%s %s\n" "${RED}" "${RST}" "$*"; }

# ---------- Detect Kaggle / CI context ----------
IS_KAGGLE=0
if [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]]; then
  IS_KAGGLE=1
  warn "Kaggle environment detected: enabling safe/skipped steps as needed."
fi

IS_CI=0
if [[ -n "${GITHUB_ACTIONS:-}" || -n "${CI:-}" ]]; then
  IS_CI=1
  say "CI environment detected."
fi

# ---------- Resolve CLI launcher ----------
CLI_BIN=""
if [[ "$USE_POETRY" -eq 1 ]] && command -v poetry >/dev/null 2>&1; then
  CLI_BIN="poetry run spectramind"
elif command -v spectramind >/dev/null 2>&1; then
  CLI_BIN="spectramind"
else
  # last resort: module run
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  CLI_BIN="$PY -m spectramind"
fi
say "CLI launcher: ${BOLD}${CLI_BIN}${RST}"

# ---------- Checks matrix ----------
FAILED=0

check() {
  local name="$1"; shift
  if "$@"; then
    okmark "$name"
  else
    badmark "$name"
    FAILED=1
  fi
}

# Basic presence
say "Basic environment checks…"
check "Python present" bash -lc 'command -v python3 >/dev/null || command -v python >/dev/null'
check "Pip present"    bash -lc 'command -v pip3 >/dev/null || command -v pip >/dev/null'

if [[ "$USE_POETRY" -eq 1 ]]; then
  check "Poetry present" bash -lc 'command -v poetry >/dev/null'
fi

# Repo structure
say "Repository structure checks…"
check "configs/ exists" test -d "configs"
check "src/ exists"     test -d "src"
check "logs/ exists"    test -d "logs"
check "outputs/ exists" test -d "outputs"

# Optional tools (informational; not fatal)
say "Optional tool probes…"
if command -v nvidia-smi >/dev/null 2>&1; then
  okmark "CUDA device(s) visible"
  if [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]]; then
    run "nvidia-smi snapshot" nvidia-smi || true
  fi
else
  warn "CUDA not detected (this is OK on CPU or in many CI contexts)"
fi

if command -v dvc >/dev/null 2>&1; then
  okmark "DVC present"
  if [[ "$DEEP" -eq 1 ]]; then
    run "DVC status (non-fatal)" dvc status || true
  fi
else
  warn "DVC not found (OK if you don't use remote data in this run)"
fi

hr

# ---------- Version banner ----------
say "CLI version & run-hash…"
if ${CLI_BIN} --version; then
  okmark "spectramind --version"
else
  badmark "spectramind --version"
  FAILED=1
fi

# ---------- Core selftest ----------
say "Running CLI selftest…"
SELFTEST_ARGS=()
[[ "$DEEP" -eq 1 ]] && SELFTEST_ARGS+=(--deep)
if ${CLI_BIN} selftest "${SELFTEST_ARGS[@]}"; then
  okmark "CLI selftest"
else
  badmark "CLI selftest"
  FAILED=1
fi

# ---------- Analyze log (short) ----------
say "Analyzing CLI log (short)…"
if ${CLI_BIN} analyze-log-short; then
  okmark "analyze-log-short"
else
  badmark "analyze-log-short"
  FAILED=1
fi

# ---------- Light diagnostics (skippable) ----------
if [[ "$SKIP_HEAVY" -eq 1 ]]; then
  warn "Skipping diagnostics/dashboard per --skip-heavy / --quick."
else
  say "Building light diagnostics dashboard…"
  # Always use the no-UMAP/TSNE fast path (pure HTML assembly from stubs)
  if ${CLI_BIN} diagnose dashboard --no-umap --no-tsne --outdir "outputs/diagnostics"; then
    okmark "diagnose dashboard (light)"
    # show the most recent report path if available
    latest_html=$(ls -t outputs/diagnostics/*.html 2>/dev/null | head -n1 || true)
    if [[ -n "$latest_html" ]]; then
      say "Latest diagnostics HTML: ${BOLD}${latest_html}${RST}"
    fi
  else
    badmark "diagnose dashboard (light)"
    FAILED=1
  fi
fi

# ---------- Optional: device smoke (training one epoch) ----------
if [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]]; then
  say "Device smoke test (tiny training run)…"
  if ${CLI_BIN} train +training.epochs=1 --device "${DEVICE}" --outdir "outputs/checkpoints/_selftest_${DEVICE}" ; then
    okmark "tiny training run (${DEVICE})"
  else
    badmark "tiny training run (${DEVICE})"
    FAILED=1
  fi
else
  warn "Skipping tiny training run (not requested with --deep or running in Kaggle)."
fi

# ---------- Final summary ----------
hr
END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
say "Selftest finished at ${BOLD}${END_TS}${RST}"
hr

if [[ "$FAILED" -eq 0 ]]; then
  printf "%s%sALL CHECKS PASSED%s\n" "${GRN}" "${BOLD}" "${RST}"
  printf "Log: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
  exit 0
else
  printf "%s%sSOME CHECKS FAILED%s\n" "${RED}" "${BOLD}" "${RST}"
  printf "See log for details: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
  exit 1
fi