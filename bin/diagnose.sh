#!/usr/bin/env bash
# ==============================================================================
# bin/diagnose.sh — Generate SpectraMind V50 diagnostics (smoothness + dashboard)
# ------------------------------------------------------------------------------
# What it does
#   • Runs light or full diagnostics via the SpectraMind CLI
#   • Produces smoothness HTML and a versioned dashboard report in outputs/diagnostics
#   • Lets you toggle UMAP / t‑SNE, open the report, and control verbosity
#
# Usage
#   bin/diagnose.sh [options]
#
# Common examples
#   # Fast, CI/Kaggle‑safe dashboard (no UMAP/t‑SNE), plus smoothness:
#   bin/diagnose.sh --light
#
#   # Full dashboard (UMAP + t‑SNE) and open the resulting HTML:
#   bin/diagnose.sh --full --open
#
# Options
#   --outdir <dir>     Output directory (default: outputs/diagnostics)
#   --light            Light mode (skip UMAP/t‑SNE; fastest & CI/Kaggle‑safe)
#   --full             Full mode (attempt UMAP + t‑SNE; may be slower)
#   --no-umap          Skip UMAP even in --full
#   --no-tsne          Skip t‑SNE even in --full
#   --smoothness-only  Only generate the smoothness HTML (skip dashboard)
#   --dashboard-only   Only generate dashboard (skip smoothness)
#   --open             Open the newest dashboard HTML (non‑Kaggle only)
#   --no-poetry        Do not use Poetry; call spectramind directly / via python -m
#   --quiet            Less verbose output
#   -h|--help          Show help and exit
#
# Exit codes
#   0 OK, 1 failure, 2 bad usage
#
# Notes
#   • Auto‑detects Kaggle/CI and avoids opening files there.
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
QUIET=0

# ---------- Args ----------
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
    --quiet)           QUIET=1 ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) fail "Unknown arg: $1"; exit 2 ;;
  esac
  shift
done

# Normalize modes: default is "light" if neither specified
if [[ "$LIGHT" -eq 0 && "$FULL" -eq 0 ]]; then
  LIGHT=1
fi

# If smoothness-only, don’t run dashboard; if dashboard-only, don’t run smoothness
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

mkdir -p "$OUTDIR"

# ---------- Detect env ----------
IS_KAGGLE=0
[[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1

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

# ---------- Helpers ----------
run() {
  local desc="$1"; shift
  [[ "$QUIET" -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "${desc}" "${RST}"
  if ! "$@"; then
    fail "${desc} — command failed: $*"
    return 1
  fi
}

latest_html() {
  ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true
}

FAILED=0

# ---------- Smoothness ----------
if [[ "$DASHBOARD_ONLY" -eq 0 ]]; then
  say "Generating smoothness HTML…"
  if run "spectramind diagnose smoothness" ${CLI_BIN} diagnose smoothness --outdir "$OUTDIR"; then
    say "Smoothness output ready."
  else
    warn "Smoothness step failed."
    FAILED=1
  fi
fi

# ---------- Dashboard ----------
if [[ "$SMOOTHNESS_ONLY" -eq 0 ]]; then
  # Build args based on mode
  DASH_ARGS=(diagnose dashboard --outdir "$OUTDIR")
  if [[ "$LIGHT" -eq 1 ]]; then
    # Always skip heavy steps in light mode
    DASH_ARGS+=(--no-umap --no-tsne)
  else
    # Full mode; honor overrides
    [[ "$NO_UMAP" -eq 1 ]] && DASH_ARGS+=(--no-umap)
    [[ "$NO_TSNE" -eq 1 ]] && DASH_ARGS+=(--no-tsne)
  fi

  say "Building diagnostics dashboard…"
  if run "spectramind ${DASH_ARGS[*]}" ${CLI_BIN} "${DASH_ARGS[@]}"; then
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
  if [[ "$IS_KAGGLE" -eq 1 ]]; then
    warn "Skipping --open on Kaggle."
  else
    REPORT="$(latest_html)"
    if [[ -n "$REPORT" ]]; then
      if command -v xdg-open >/dev/null 2>&1; then
        say "Opening ${REPORT}…"
        xdg-open "$REPORT" >/dev/null 2>&1 || warn "Failed to open with xdg-open"
      elif command -v open >/dev/null 2>&1; then
        say "Opening ${REPORT}…"
        open "$REPORT" >/dev/null 2>&1 || warn "Failed to open with open"
      else
        warn "No opener (xdg-open/open) found; skipping viewer."
      fi
    else
      warn "No dashboard HTML to open."
    fi
  fi
fi

# ---------- Summary ----------
if [[ "$FAILED" -eq 0 ]]; then
  printf "%s✔%s Diagnostics completed.\n" "${GRN}" "${RST}"
  exit 0
else
  printf "%s✘%s Diagnostics encountered issues.\n" "${RED}" "${RST}"
  exit 1
fi