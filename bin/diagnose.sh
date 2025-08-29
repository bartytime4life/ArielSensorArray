#!/usr/bin/env bash
# ==============================================================================
# bin/diagnose.sh — Generate SpectraMind V50 diagnostics (smoothness + dashboard)
# ------------------------------------------------------------------------------
# What it does
#   • Runs light or full diagnostics via the SpectraMind CLI
#   • Produces smoothness HTML and a versioned dashboard report in outputs/diagnostics
#   • Lets you toggle UMAP / t-SNE, open the report, control verbosity, and emit JSON
#   • Safe for CI/Kaggle (light mode default; avoids browser open in those envs)
#
# Usage
#   bin/diagnose.sh [options]
#
# Common examples
#   # Fast, CI/Kaggle-safe dashboard (no UMAP/t-SNE), plus smoothness:
#   bin/diagnose.sh --light
#
#   # Full dashboard (UMAP + t-SNE) and open the resulting HTML:
#   bin/diagnose.sh --full --open
#
# Options
#   --outdir <dir>       Output directory (default: outputs/diagnostics)
#   --light              Light mode (skip UMAP/t-SNE; fastest & CI/Kaggle-safe)
#   --full               Full mode (attempt UMAP + t-SNE; may be slower)
#   --no-umap            Skip UMAP even in --full
#   --no-tsne            Skip t-SNE even in --full
#   --smoothness-only    Only generate the smoothness HTML (skip dashboard)
#   --dashboard-only     Only generate dashboard (skip smoothness)
#   --open               Open the newest dashboard HTML (non-Kaggle only)
#   --no-poetry          Do not use Poetry; call spectramind directly / via python -m
#   --timeout <sec>      Timeout per CLI step (default: 300)
#   --log <path>         Write a combined stdout/stderr log (default: logs/diagnostics/diag-<ts>.log)
#   --json               Emit a JSON result summary to stdout (in addition to text)
#   --quiet              Less verbose output
#   -h|--help            Show help and exit
#
# Exit codes
#   0 OK, 1 failure, 2 bad usage
#
# Notes
#   • Auto-detects Kaggle/CI and avoids opening files there.
#   • Works even if Poetry isn’t installed (module fallback).
# ==============================================================================

set -euo pipefail

# ---------- Pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[DIAG]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[DIAG]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[DIAG]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- Defaults ----------
OUTDIR="outputs/diagnostics"
LIGHT=0
FULL=0
NO_UMAP=0
NO_TSNE=0
SMOOTHNESS_ONLY=0
DASHBOARD_ONLY=0
OPEN_AFTER=0
USE_POETRY=1
TIMEOUT="${DIAG_TIMEOUT:-300}"
LOG_DIR_DEFAULT="logs/diagnostics"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_PATH=""
QUIET=0
JSON_OUT=0

# ---------- Args ----------
usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)          OUTDIR="${2:?}"; shift ;;
    --light)           LIGHT=1 ;;
    --full)            FULL=1 ;;
    --no-umap)         NO_UMAP=1 ;;
    --no-tsne)         NO_TSNE=1 ;;
    --smoothness-only) SMOOTHNESS_ONLY=1 ;;
    --dashboard-only)  DASHBOARD_ONLY=1 ;;
    --open)            OPEN_AFTER=1 ;;
    --no-poetry)       USE_POETRY=0 ;;
    --timeout)         TIMEOUT="${2:?}"; shift ;;
    --log)             LOG_PATH="${2:?}"; shift ;;
    --json)            JSON_OUT=1 ;;
    --quiet)           QUIET=1 ;;
    -h|--help)         usage; exit 0 ;;
    *)                 fail "Unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# Normalize modes: default is "light" if neither specified
if [[ "$LIGHT" -eq 0 && "$FULL" -eq 0 ]]; then
  LIGHT=1
fi

# If smoothness-only & dashboard-only are both set → error
if [[ "$SMOOTHNESS_ONLY" -eq 1 && "$DASHBOARD_ONLY" -eq 1 ]]; then
  fail "Choose either --smoothness-only or --dashboard-only (not both)."
  exit 2
fi

# ---------- Move to repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; exit 1; }
fi

mkdir -p "$OUTDIR" "$LOG_DIR_DEFAULT"

# ---------- Detect env ----------
IS_KAGGLE=0
[[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1
IS_CI=0
[[ -n "${GITHUB_ACTIONS:-}" || -n "${CI:-}" ]] && IS_CI=1

# ---------- Pick CLI ----------
CLI_BIN=""
if [[ "$USE_POETRY" -eq 1 ]] && command -v poetry >/dev/null 2>&1; then
  CLI_BIN="poetry run spectramind"
elif command -v spectramind >/dev/null 2>&1; then
  CLI_BIN="spectramind"
else
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  CLI_BIN="$PY -m spectramind"
fi

say "Diagnostics outdir: ${BOLD}${OUTDIR}${RST}"
say "CLI launcher:       ${BOLD}${CLI_BIN}${RST}"

# ---------- Logging ----------
if [[ -z "$LOG_PATH" ]]; then
  LOG_PATH="${LOG_DIR_DEFAULT}/diag-${TS}.log"
fi

if [[ "$QUIET" -eq 0 ]]; then
  # shellcheck disable=SC2094
  exec > >(tee -a "${LOG_PATH}") 2>&1
else
  exec >> "${LOG_PATH}" 2>&1
fi

say "Log file: ${BOLD}${LOG_PATH}${RST}"

# ---------- Helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }

run_step() {
  # run_step <name> <cmd...>
  local name="$1"; shift
  local -a cmd=("$@")
  [[ "$QUIET" -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "${name}" "${RST}"
  if have timeout; then
    if ! timeout --preserve-status --signal=TERM "$TIMEOUT" "${cmd[@]}"; then
      fail "${name} — command failed or timed out"
      return 1
    fi
  else
    if ! "${cmd[@]}"; then
      fail "${name} — command failed"
      return 1
    fi
  fi
  return 0
}

latest_html() { ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true; }

# ---------- Build argument sets ----------
DASH_ARGS=(diagnose dashboard --outdir "$OUTDIR")
if [[ "$LIGHT" -eq 1 ]]; then
  DASH_ARGS+=(--no-umap --no-tsne)
else
  [[ "$NO_UMAP" -eq 1 ]] && DASH_ARGS+=(--no-umap)
  [[ "$NO_TSNE" -eq 1 ]] && DASH_ARGS+=(--no-tsne)
fi

FAILED=0
START_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ---------- Smoothness ----------
SMOOTH_PATH=""
if [[ "$DASHBOARD_ONLY" -eq 0 ]]; then
  say "Generating smoothness HTML…"
  if run_step "spectramind diagnose smoothness" bash -lc "${CLI_BIN} diagnose smoothness --outdir '$OUTDIR'"; then
    SMOOTH_PATH="$(latest_html || true)"
    say "Smoothness step completed."
  else
    warn "Smoothness step failed."
    FAILED=1
  fi
fi

# ---------- Dashboard ----------
REPORT=""
if [[ "$SMOOTHNESS_ONLY" -eq 0 ]]; then
  say "Building diagnostics dashboard…"
  if run_step "spectramind ${DASH_ARGS[*]}" bash -lc "${CLI_BIN} ${DASH_ARGS[*]}"; then
    REPORT="$(latest_html)"
    if [[ -n "$REPORT" ]]; then
      say "Dashboard report: ${BOLD}${REPORT}${RST}"
    else
      warn "Dashboard completed but no HTML found in ${OUTDIR}."
      FAILED=1
    fi
  else
    warn "Dashboard step failed."
    FAILED=1
  fi
fi

# ---------- Optionally open report ----------
if [[ "$OPEN_AFTER" -eq 1 ]]; then
  if [[ "$IS_KAGGLE" -eq 1 || "$IS_CI" -eq 1 ]]; then
    warn "Skipping --open in CI/Kaggle."
  else
    REPORT_TO_OPEN="$(latest_html)"
    if [[ -n "$REPORT_TO_OPEN" ]]; then
      if command -v xdg-open >/dev/null 2>&1; then
        say "Opening ${REPORT_TO_OPEN}…"
        xdg-open "$REPORT_TO_OPEN" >/dev/null 2>&1 || warn "Failed to open with xdg-open"
      elif command -v open >/dev/null 2>&1; then
        say "Opening ${REPORT_TO_OPEN}…"
        open "$REPORT_TO_OPEN" >/dev/null 2>&1 || warn "Failed to open with open"
      else
        warn "No opener (xdg-open/open) found; skipping viewer."
      fi
    else
      warn "No dashboard HTML to open."
    fi
  fi
fi

# ---------- Summary / Exit ----------
END_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
if [[ "$FAILED" -eq 0 ]]; then
  printf "%s✔%s Diagnostics completed.\n" "${GRN}" "${RST}"
  OK=true
  EC=0
else
  printf "%s✘%s Diagnostics encountered issues.\n" "${RED}" "${RST}"
  OK=false
  EC=1
fi

# ---------- JSON summary (optional) ----------
if [[ "$JSON_OUT" -eq 1 ]]; then
  # minimal JSON (no external deps)
  printf '{'
  printf '"ok": %s, ' "$([[ $OK == true ]] && echo true || echo false)"
  printf '"outdir": "%s", ' "$OUTDIR"
  printf '"light": %s, ' "$([[ $LIGHT -eq 1 ]] && echo true || echo false)"
  printf '"full": %s, ' "$([[ $FULL -eq 1 ]] && echo true || echo false)"
  printf '"no_umap": %s, ' "$([[ $NO_UMAP -eq 1 ]] && echo true || echo false)"
  printf '"no_tsne": %s, ' "$([[ $NO_TSNE -eq 1 ]] && echo true || echo false)"
  printf '"smoothness_only": %s, ' "$([[ $SMOOTHNESS_ONLY -eq 1 ]] && echo true || echo false)"
  printf '"dashboard_only": %s, ' "$([[ $DASHBOARD_ONLY -eq 1 ]] && echo true || echo false)"
  printf '"report": "%s", ' "${REPORT:-}"
  printf '"smoothness_html": "%s", ' "${SMOOTH_PATH:-}"
  printf '"log": "%s", ' "$LOG_PATH"
  printf '"start": "%s", "end": "%s"' "$START_ISO" "$END_ISO"
  printf '}\n'
fi

exit "$EC"
