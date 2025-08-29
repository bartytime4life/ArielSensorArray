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

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[SELFTEST]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[SELFTEST]%s %s\n' "${YLW}" "${RST}" "$*" >&2; }
err()  { printf '%s[SELFTEST]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

hr()   { printf '%s\n' "────────────────────────────────────────────────────────────────────"; }

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
      err "Unknown argument: $1"
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
if command -v git >/dev/null 2>&1 && ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then
  cd "$ROOT"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  cd "$ROOT"
fi

# ---------- Prepare dirs & logging ----------
mkdir -p logs outputs outputs/diagnostics
if [[ "$QUIET" -eq 0 ]]; then
  exec > >(tee -a "${LOG_PATH}") 2>&1
else
  exec >> "${LOG_PATH}" 2>&1
fi

say "SpectraMind V50 selftest started at ${BOLD}${START_TS}${RST}"
say "Repository root: ${BOLD}${ROOT}${RST}"
hr

# ---------- Utils ----------
have()  { command -v "$1" >/dev/null 2>&1; }
with_timeout() {
  if have timeout && [[ "${STEP_TIMEOUT:-0}" -gt 0 ]]; then
    timeout --preserve-status --signal=TERM "${STEP_TIMEOUT}" "$@"
  else
    "$@"
  fi
}
run_raw() {
  local desc="$1"; shift
  [[ "$QUIET" -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "$desc" "${RST}"
  with_timeout "$@"
}
run() {
  local desc="$1"; shift
  if run_raw "$desc" "$@"; then return 0; else err "$desc — command failed"; return 1; fi
}
okmark()  { printf "%s✔%s %s\n" "${GRN}" "${RST}" "$*"; }
badmark() { printf "%s✘%s %s\n" "${RED}" "${RST}" "$*"; }

# ---------- Detect Kaggle / CI ----------
IS_KAGGLE=0
[[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1 && warn "Kaggle detected: enabling safe/skipped steps."
IS_CI=0
[[ -n "${GITHUB_ACTIONS:-}" || -n "${CI:-}" ]] && IS_CI=1 && say "CI environment detected."

# ---------- Resolve CLI ----------
CLI_BIN=""
if [[ "$NO_CLI" -eq 0 ]]; then
  if [[ "$USE_POETRY" -eq 1 ]] && have poetry; then
    CLI_BIN="poetry run spectramind"
  elif have spectramind; then
    CLI_BIN="spectramind"
  elif have python3; then
    CLI_BIN="python3 -m spectramind"
  elif have python; then
    CLI_BIN="python -m spectramind"
  else
    CLI_BIN=""
  fi
  say "CLI launcher: ${BOLD}${CLI_BIN:-(not found)}${RST}"
fi

# ---------- JSON accumulator ----------
JSON="{"
JFIRST=1
json_put() {
  local k="$1"; shift
  local v="$1"; shift || true
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
  if [[ $JFIRST -eq 1 ]]; then JSON+="$k:$v"; JFIRST=0; else JSON+=", $k:$v"; fi
}

# ---------- Checks ----------
FAILED=0
check() { local name="$1"; shift; if "$@"; then okmark "$name"; return 0; else badmark "$name"; FAILED=1; return 1; fi; }

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

say "Repository structure checks…"
check "configs/ exists" test -d "configs"
check "src/ exists"     test -d "src"
check "logs/ exists"    test -d "logs"
check "outputs/ exists" test -d "outputs"

say "Optional tool probes…"
if [[ "$NO_CUDA" -eq 0 ]] && have nvidia-smi; then
  okmark "CUDA device(s) visible"
  json_put "cuda_available" "true"
  [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]] && run "nvidia-smi snapshot" nvidia-smi || true
else
  warn "CUDA not detected or skipped"
  json_put "cuda_available" "false"
fi

if [[ "$NO_DVC" -eq 0 ]] && have dvc; then
  okmark "DVC present"
  json_put "dvc_present" "true"
  [[ "$DEEP" -eq 1 ]] && run "DVC status (non-fatal)" dvc status || true
else
  warn "DVC not found or skipped"
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
    warn "VERSION and pyproject.toml differ (VERSION='$V_FILE' pyproject='$V_PY')"
    json_put "version_match" "false"
    json_put "version" "${V_FILE:-unknown}"
    [[ "$QUICK" -eq 0 ]] && FAILED=1
  fi
fi

hr

# ---------- CLI checks ----------
if [[ "$NO_CLI" -eq 0 && -n "$CLI_BIN" ]]; then
  say "CLI version & config hash…"
  run "spectramind --version" bash -lc "${CLI_BIN} --version" || FAILED=1

  # Try to get a config hash (best-effort)
  CFG_HASH="-"
  if bash -lc "${CLI_BIN} --help" >/dev/null 2>&1; then
    if bash -lc "${CLI_BIN} --help | grep -qi 'print-config-hash'"; then
      CFG_HASH="$(bash -lc "${CLI_BIN} --print-config-hash" 2>/dev/null || echo -)"
    elif bash -lc "${CLI_BIN} --help | grep -qi 'hash-config'"; then
      CFG_HASH="$(bash -lc "${CLI_BIN} hash-config" 2>/dev/null || echo -)"
    fi
  fi
  json_put "cfg_hash" "${CFG_HASH:-"-"}"

  # CLI selftest (fast) — different from this script; skip if it maps here
  if bash -lc "${CLI_BIN} --help | grep -qi '^ *selftest'"; then
    say "Running CLI selftest (fast)…"
    run "CLI selftest" bash -lc "${CLI_BIN} selftest --quick" || FAILED=1
  fi

  # Light analyze-log
  if bash -lc "${CLI_BIN} --help | grep -qi 'analyze-log'"; then
    say "Analyzing CLI log (short)…"
    run "analyze-log" bash -lc "${CLI_BIN} analyze-log --no-refresh --tail 20 --md outputs/_selftest_log.md --csv outputs/_selftest_log.csv" || FAILED=1
  fi

  # Light diagnostics (skippable)
  if [[ "$SKIP_HEAVY" -eq 1 ]]; then
    warn "Skipping diagnostics/dashboard per --skip-heavy / --quick."
  else
    if bash -lc "${CLI_BIN} --help | grep -qi 'diagnose'"; then
      say "Building light diagnostics dashboard…"
      run "diagnose dashboard" bash -lc "${CLI_BIN} diagnose dashboard --no-umap --no-tsne --outdir outputs/diagnostics" || FAILED=1
    fi
  fi

  # Optional tiny train smoke (deep only, not on Kaggle)
  if [[ "$DEEP" -eq 1 && "$IS_KAGGLE" -eq 0 ]]; then
    say "Device smoke test (tiny training)…"
    run "tiny train (${DEVICE})" bash -lc "${CLI_BIN} train +training.epochs=1 --device ${DEVICE} --outdir outputs/checkpoints/_selftest_${DEVICE}" || FAILED=1
  else
    warn "Skipping tiny training run (not requested with --deep, or running in Kaggle)."
  fi
else
  [[ "$NO_CLI" -eq 1 ]] && warn "CLI checks skipped per --no-cli" || { err "spectramind CLI not available"; FAILED=1; }
fi

# ---------- Final summary ----------
hr
END_TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
say "Selftest finished at ${BOLD}${END_TS}${RST}"
hr

STATUS_OK=true
if [[ "$FAILED" -ne 0 ]]; then
  STATUS_OK=false
  printf "%s%sSOME CHECKS FAILED%s\n" "${RED}" "${BOLD}" "${RST}"
  printf "See log for details: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
else
  printf "%s%sALL CHECKS PASSED%s\n" "${GRN}" "${BOLD}" "${RST}"
  printf "Log: %s%s%s\n" "${BOLD}" "${LOG_PATH}" "${RST}"
fi

# ---------- JSON summary ----------
if [[ "$JSON_OUT" -eq 1 ]]; then
  JSON+=", \"ok\": $( $STATUS_OK && echo true || echo false )"
  JSON+=", \"kaggle\": $( [[ $IS_KAGGLE -eq 1 ]] && echo true || echo false )"
  JSON+=", \"ci\": $( [[ $IS_CI -eq 1 ]] && echo true || echo false )"
  JSON+=", \"quick\": $( [[ $QUICK -eq 1 ]] && echo true || echo false )"
  JSON+=", \"deep\": $( [[ $DEEP -eq 1 ]] && echo true || echo false )"
  JSON+=", \"skip_heavy\": $( [[ $SKIP_HEAVY -eq 1 ]] && echo true || echo false )"
  JSON+=", \"start\": \"${START_TS}\""
  JSON+=", \"end\": \"${END_TS}\""
  JSON+=", \"log\": \"${LOG_PATH}\""
  JSON+="}"
  printf '%s\n' "$JSON"
fi

$STATUS_OK && exit 0 || exit 1