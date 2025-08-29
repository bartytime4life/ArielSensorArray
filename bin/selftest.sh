#!/usr/bin/env bash
# ==============================================================================
# bin/selftest.sh — SpectraMind V50 quick/deep health check (upgraded)
# ------------------------------------------------------------------------------
# Goals:
#   • Verify local environment (Python/Poetry/CLI/CUDA/DVC) and repo layout
#   • Exercise the CLI safely (CI/Kaggle-friendly; no heavy runs by default)
#   • Emit a concise pass/fail summary and correct exit code for automation
#   • (Optional) JSON result for CI dashboards
#
# Usage:
#   bin/selftest.sh [--deep] [--quick] [--skip-heavy] [--no-poetry]
#                   [--device <cpu|gpu>] [--log <path>] [--quiet]
#                   [--no-cli] [--no-dvc] [--no-cuda]
#                   [--timeout <sec>] [--json]
#
# Flags:
#   --deep         : run additional checks (CUDA/DVC, tiny train smoke)
#   --quick        : minimal checks only (overrides heavy/deep parts)
#   --skip-heavy   : skip diagnostics/dashboard step even if available
#   --no-poetry    : do not use Poetry; try `spectramind` or `python -m spectramind`
#   --no-cli       : skip CLI invocations (env-only checks)
#   --no-dvc       : skip DVC checks even if DVC is installed
#   --no-cuda      : skip CUDA probes even if nvidia-smi exists
#   --device <d>   : cpu|gpu (hint for the tiny train smoke; default: cpu)
#   --log <path>   : write a copy of stdout/stderr (default: logs/selftest.log)
#   --timeout <s>  : timeout for sub-steps (default: 180)
#   --json         : emit a JSON summary to stdout (in addition to text)
#   --quiet        : less verbose output
#
# Exit codes:
#   0 OK, 1 failure, 2 bad usage
#
# Notes:
#   • Safe on Kaggle: auto-detects env and skips non-portable steps.
#   • No data mutation; produces only logs and lightweight artifacts.
# ==============================================================================

set -euo pipefail

# ---------- Pretty printing ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'
YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'

say()   { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[SELFTEST]%s %s\n' "${CYN}" "${RST}" "$*"; }
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
NO_CLI=0
NO_DVC=0
NO_CUDA=0
JSON_OUT=0
STEP_TIMEOUT="${SELFTEST_TIMEOUT:-180}" # override via env if desired

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deep)        DEEP=1 ;;
    --quick)       QUICK=1 ;;
    --skip-heavy)  SKIP_HEAVY=1 ;;
    --no-poetry)   USE_POETRY=0 ;;
    --device)      DEVICE="${2:-cpu}"; shift ;;
    --log)         LOG_PATH="${2:-logs/selftest.log}"; shift ;;
    --quiet)       QUIET=1 ;;
    --no-cli)      NO_CLI=1 ;;
    --no-dvc)      NO_DVC=1 ;;
    --no-cuda)     NO_CUDA=1 ;;
    --timeout)     STEP_TIMEOUT="${2:-180}"; shift ;;
    --json)        JSON_OUT=1 ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
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

# ---------- Utility: timeout wrapper + runner with echo ----------
have()  { command -v "$1" >/dev/null 2>&1; }

run_raw() {
  # run_raw <desc> <cmd...>
  local desc="$1"; shift
  local tcmd=("$@")
  local to=""; local use_to=0
  if have timeout; then
    use_to=1
    to=("timeout" "--preserve-status" "--signal=TERM" "${STEP_TIMEOUT}")
  fi
  if [[ "$QUIET" -eq 0 ]]; then
    printf "%s→ %s%s\n" "${DIM}" "${desc}" "${RST}"
    printf "%s$ %s%s\n" "${DIM}" "${tcmd[*]}" "${RST}"
  fi
  if [[ $use_to -eq 1 ]]; then
    "${to[@]}" "${tcmd[@]}"
  else
    "${tcmd[@]}"
  fi
}

run() {
  # run <desc> <cmd...> -> returns 0/1 but does not exit
  if run_raw "$@"; then
    return 0
  else
    fail "$1 — command failed"
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
if [[ "$NO_CLI" -eq 0 ]]; then
  if [[ "$USE_POETRY" -eq 1 ]] && have poetry; then
    CLI_BIN="poetry run spectramind"
  elif have spectramind; then
    CLI_BIN="spectramind"
  else
    if have python3; then PY=python3; else PY=python; fi
    CLI_BIN="$PY -m spectramind"
  fi
  say "CLI launcher: ${BOLD}${CLI_BIN}${RST}"
else
  warn "Skipping CLI invocations per --no-cli"
fi

# ---------- JSON accumulator ----------
JSON_DETAILS="{"
JSON_FIRST=1
json_put() {
  # json_put "key" "value (already encoded or raw string)"
  local k="$1"; shift
  local v="$1"; shift || true
  local pair
  # crude escaper using python if available
  if have python3 || have python; then
    local py=python3; have python3 || py=python
    k=$("$py" - <<PY 2>/dev/null
import json,sys; print(json.dumps(sys.argv[1]))
PY "$k")
    v=$("$py" - <<PY 2>/dev/null
import json,sys; print(json.dumps(sys.argv[1]))
PY "$v")
  else
    k="\"$k\""; v="\"$v\""
  fi
  pair="$k: $v"
  if [[ $JSON_FIRST -eq 1 ]]; then
    JSON_DETAILS+="$pair"; JSON_FIRST=0
  else
    JSON_DETAILS+=", $pair"
  fi
}

# ---------- Checks matrix ----------
FAILED=0

check() {
  local name="$1"; shift
  if "$@"; then
    okmark "$name"; return 0
  else
    badmark "$name"; FAILED=1; return 1
  fi
}

# Basic presence
say "Basic environment checks…"
check "Python present" bash -lc 'command -v python3 >/dev/null || command -v python >/dev/null'
PY_VER="$(bash -lc 'python3 -V 2>/dev/null || python -V 2>/dev/null || true')"
json_put "python_version" "${PY_VER:-unknown}"

check "Pip present"    bash -lc 'command -v pip3 >/dev/null || command -v pip >/dev/null'
PIP_VER="$(bash -lc 'pip3 -V 2>/dev/null || pip -V 2>/dev/null || true')"
json_put "pip_version" "${PIP_VER:-unknown}"

if [[ "$USE_POETRY" -eq 1 ]]; then
  if check "Poetry present" bash -lc 'command -v poetry >/dev/null'; then
    POETRY_VER="$(bash -lc 'poetry --version 2>/dev/null || true')"
    json_put "poetry_version" "${POETRY_VER:-unknown}"
  else
    json_put "poetry_version" "absent"
  fi
else
  json_put "poetry_version" "skipped"
fi

# Repo structure
say "Repository structure checks…"
check "configs/ exists" test -d "configs"
check "src/ exists"     test -d "src"
check "logs/ exists"    test -d "logs"
check "outputs/ exists" test -d "outputs"

# Optional tools (informational; not fatal)
say "Optional tool probes…"
if [[ "$NO_CUDA" -eq 0 ]] && have nvidia-smi; then
  okmark "CUDA device(s) visible"
  json_put "cuda_available" "true"
  if [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]]; then
    run "nvidia-smi snapshot" nvidia-smi || true
  fi
else
  warn "CUDA not detected or skipped (OK on CPU/CI/Kaggle)."
  json_put "cuda_available" "false"
fi

if [[ "$NO_DVC" -eq 0 ]] && have dvc; then
  okmark "DVC present"
  json_put "dvc_present" "true"
  if [[ "$DEEP" -eq 1 ]]; then
    run "DVC status (non-fatal)" dvc status || true
  fi
else
  warn "DVC not found or skipped (OK if no remote data needed)."
  json_put "dvc_present" "false"
fi

# VERSION vs pyproject.toml
if [[ -f VERSION && -f pyproject.toml ]]; then
  V_FILE="$(sed -n '1s/[[:space:]]//gp' VERSION || true)"
  V_PY="$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' pyproject.toml | head -n1 || true)"
  if [[ -n "$V_FILE" && -n "$V_PY" && "$V_FILE" == "$V_PY" ]]; then
    okmark "VERSION matches pyproject: $V_FILE"
    json_put "version_match" "true"
    json_put "version" "$V_FILE"
  else
    warn "VERSION and pyproject.toml differ (VERSION='$V_FILE' pyproject='$V_PY')."
    json_put "version_match" "false"
    json_put "version" "${V_FILE:-unknown}"
    [[ "$QUICK" -eq 0 ]] && FAILED=1
  fi
fi

hr

# ---------- CLI checks ----------
if [[ "$NO_CLI" -eq 0 ]]; then
  say "CLI version & run-hash…"
  if run "spectramind --version" bash -lc "${CLI_BIN} --version"; then
    okmark "spectramind --version"
  else
    badmark "spectramind --version"
    FAILED=1
  fi

  # CLI selftest (fast)
  say "Running CLI selftest…"
  SELFTEST_ARGS=()
  [[ "$DEEP" -eq 1 ]] && SELFTEST_ARGS+=(--deep)
  if run "CLI selftest" bash -lc "${CLI_BIN} selftest ${SELFTEST_ARGS[*]}"; then
    okmark "CLI selftest"
  else
    badmark "CLI selftest"
    FAILED=1
  fi

  # Analyze log (short)
  say "Analyzing CLI log (short)…"
  if run "analyze-log-short" bash -lc "${CLI_BIN} analyze-log-short"; then
    okmark "analyze-log-short"
  else
    badmark "analyze-log-short"
    FAILED=1
  fi

  # Light diagnostics (skippable)
  if [[ "$SKIP_HEAVY" -eq 1 ]]; then
    warn "Skipping diagnostics/dashboard per --skip-heavy / --quick."
  else
    say "Building light diagnostics dashboard…"
    if run "diagnose dashboard (light)" bash -lc "${CLI_BIN} diagnose dashboard --no-umap --no-tsne --outdir outputs/diagnostics"; then
      okmark "diagnose dashboard (light)"
      latest_html=$(ls -t outputs/diagnostics/*.html 2>/dev/null | head -n1 || true)
      if [[ -n "$latest_html" ]]; then
        say "Latest diagnostics HTML: ${BOLD}${latest_html}${RST}"
      fi
    else
      badmark "diagnose dashboard (light)"
      FAILED=1
    fi
  fi

  # Optional tiny training smoke
  if [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]]; then
    say "Device smoke test (tiny training run)…"
    if run "tiny train (${DEVICE})" bash -lc "${CLI_BIN} train +training.epochs=1 --device ${DEVICE} --outdir outputs/checkpoints/_selftest_${DEVICE}"; then
      okmark "tiny training run (${DEVICE})"
    else
      badmark "tiny training run (${DEVICE})"
      FAILED=1
    fi
  else
    warn "Skipping tiny training run (not requested with --deep, or running in Kaggle)."
  fi
else
  warn "CLI checks skipped per --no-cli"
fi

# ---------- Final summary ----------
hr
END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
say "Selftest finished at ${BOLD}${END_TS}${RST}"
hr

if [[ "$FAILED" -eq 0 ]]; then
  printf "%s%sALL CHECKS PASSED%s\n" "${GRN}" "${BOLD}" "${RST}"
  printf "Log: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
  STATUS_OK=true
else
  printf "%s%sSOME CHECKS FAILED%s\n" "${RED}" "${BOLD}" "${RST}"
  printf "See log for details: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
  STATUS_OK=false
fi

# ---------- JSON summary ----------
if [[ "$JSON_OUT" -eq 1 ]]; then
  # finalize details map
  JSON_DETAILS+="}"
  if have python3 || have python; then
    py=python3; have python3 || py=python
    "$py" - <<PY 2>/dev/null || true
import json, sys, os
details = ${JSON_DETAILS}
summary = {
  "ok": ${STATUS_OK:-false},
  "kaggle": ${IS_KAGGLE:-0} == 1,
  "ci": ${IS_CI:-0} == 1,
  "quick": ${QUICK:-0} == 1,
  "deep": ${DEEP:-0} == 1,
  "skip_heavy": ${SKIP_HEAVY:-0} == 1,
  "start": ${START_TS!r},
  "end": ${END_TS!r},
  "log": ${LOG_PATH!r},
  "details": details
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
  else
    # Minimal JSON fallback (no pretty, just status)
    printf '{"ok": %s, "quick": %s, "deep": %s, "log": "%s"}\n' \
      "$( [[ "$STATUS_OK" == "true" ]] && echo true || echo false )" \
      "$( [[ "$QUICK" -eq 1 ]] && echo true || echo false )" \
      "$( [[ "$DEEP" -eq 1 ]] && echo true || echo false )" \
      "$LOG_PATH"
  fi
fi

[[ "$FAILED" -eq 0 ]] && exit 0 || exit 1