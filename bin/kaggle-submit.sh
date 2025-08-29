#!/usr/bin/env bash
# ==============================================================================
# bin/kaggle-submit.sh — Safe, ergonomic Kaggle submission helper (upgraded)
# ------------------------------------------------------------------------------
# Features
#   • DRY-RUN BY DEFAULT (use --yes to actually submit)
#   • Auto-detects submission file (multiple common locations)
#   • Validates Kaggle CLI & auth, competition slug, file existence/size
#   • Optional gzip support and auto-compress if --gzip and file endswith .csv
#   • Nice logs, optional retries, optional polling of recent submissions
#   • Optional JSON summary for CI dashboards
#   • Works locally and inside Kaggle kernels (best-effort)
#
# Usage
#   bin/kaggle-submit.sh [--comp SLUG] [--file PATH] [--message "msg"] [--yes]
#                        [--retries N] [--sleep SEC]
#                        [--open|--no-open] [--gzip] [--json] [--quiet]
#
# Examples
#   bin/kaggle-submit.sh --yes
#   bin/kaggle-submit.sh --comp neurips-2025-ariel \
#                        --file outputs/predictions/submission.csv \
#                        --message "V50 run #42" --yes
#
# Notes
#   • Requires: kaggle CLI + valid auth (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR)
#   • Default competition: neurips-2025-ariel (override with --comp or $KAGGLE_COMP)
#   • DRY-RUN prints what would happen and recent submissions tail (no network submit)
# ==============================================================================

set -euo pipefail

# ----- Colors ---------------------------------------------------------------
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; CYN=$'\033[36m'; RST=$'\033[0m'
else
  BOLD=''; DIM=''; RED=''; GRN=''; YLW=''; CYN=''; RST=''
fi

log()   { printf "%b\n" "${*}"; }
info()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; log "${CYN}::${RST} ${*}"; }
ok()    { [[ "${QUIET:-0}" -eq 1 ]] && return 0; log "${GRN}✓${RST} ${*}"; }
warn()  { log "${YLW}⚠${RST} ${*}"; }
err()   { log "${RED}✗${RST} ${*}"; }
die()   { err "$*"; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }

# ----- Defaults -------------------------------------------------------------
COMPETITION="${KAGGLE_COMP:-neurips-2025-ariel}"
SUBMIT_FILE=""
MESSAGE="${KAGGLE_MSG:-SpectraMind V50 auto-submit}"
YES=0
RETRIES="${KAGGLE_RETRIES:-0}"
SLEEP="${KAGGLE_SLEEP:-15}"
OPEN=1
GZIP=0
QUIET=0
JSON=0

# ----- Helpers --------------------------------------------------------------
usage() {
  cat <<EOF
${BOLD}kaggle-submit.sh${RST} — submit a CSV to a Kaggle competition (safe by default)

${BOLD}Options${RST}
  --comp SLUG        Kaggle competition slug (default: ${COMPETITION})
  --file PATH        Path to submission CSV (auto-detected if omitted)
  --message TEXT     Submission message (default: "${MESSAGE}")
  --yes              Actually submit (DRY-RUN by default)
  --retries N        Retries if Kaggle CLI transiently fails (default: ${RETRIES})
  --sleep SEC        Sleep between retries (default: ${SLEEP})
  --gzip             Gzip the submission (auto .csv.gz) before upload
  --open | --no-open Open the competition submission page after submit (default: open)
  --json             Emit a JSON summary (useful for CI)
  --quiet            Minimal logs
  -h, --help         Show help

${BOLD}Auto-detect CSV (first match wins)${RST}
  • predictions/submission.csv
  • outputs/predictions/submission.csv
  • outputs/submission.csv
  • outputs/submission/submission.csv
  • submission.csv

Examples:
  bin/kaggle-submit.sh --yes
  bin/kaggle-submit.sh --comp neurips-2025-ariel --file outputs/predictions/submission.csv --message "V50 run #42" --yes
EOF
}

detect_submit_file() {
  local candidates=(
    "predictions/submission.csv"
    "outputs/predictions/submission.csv"
    "outputs/submission.csv"
    "outputs/submission/submission.csv"
    "submission.csv"
  )
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then SUBMIT_FILE="$f"; return 0; fi
  done
  return 1
}

kaggle_authenticated() {
  if [[ -n "${KAGGLE_CONFIG_DIR:-}" ]]; then
    [[ -f "${KAGGLE_CONFIG_DIR%/}/kaggle.json" ]]
  else
    [[ -f "$HOME/.kaggle/kaggle.json" ]]
  fi
}

print_recent_submissions() {
  [[ $QUIET -eq 1 ]] && return 0
  info "Recent submissions (top 6):"
  if ! kaggle competitions submissions -c "$COMPETITION" -v 2>/dev/null | head -n 7; then
    warn "Could not list recent submissions (permission or CLI issue)."
  fi
}

open_competition_page() {
  local url="https://www.kaggle.com/competitions/${COMPETITION}/submissions"
  [[ $OPEN -eq 1 ]] || return 0
  if have xdg-open; then xdg-open "$url" >/dev/null 2>&1 || true
  elif have open; then open "$url" >/dev/null 2>&1 || true
  fi
}

submit_once() {
  local -r comp="$1" file="$2" msg="$3"
  if [[ $QUIET -eq 1 ]]; then
    kaggle competitions submit -c "$comp" -f "$file" -m "$msg" >/dev/null
  else
    kaggle competitions submit -c "$comp" -f "$file" -m "$msg"
  fi
}

json_emit() {
  [[ $JSON -eq 1 ]] || return 0
  # naive JSON emission, values already shell-safe
  printf '{'
  printf '"ok": %s, '   "${1:-false}"
  printf '"competition": "%s", ' "${2}"
  printf '"file": "%s", '        "${3}"
  printf '"message": "%s", '     "${4}"
  printf '"attempts": %s, '      "${5:-0}"
  printf '"dry_run": %s'         "$([[ $YES -eq 1 ]] && echo false || echo true)"
  printf '}\n'
}

# ----- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --comp)     COMPETITION="${2:?}"; shift 2 ;;
    --file)     SUBMIT_FILE="${2:?}"; shift 2 ;;
    --message)  MESSAGE="${2:?}"; shift 2 ;;
    --yes)      YES=1; shift ;;
    --retries)  RETRIES="${2:-0}"; shift 2 ;;
    --sleep)    SLEEP="${2:-15}"; shift 2 ;;
    --open)     OPEN=1; shift ;;
    --no-open)  OPEN=0; shift ;;
    --gzip)     GZIP=1; shift ;;
    --json)     JSON=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          err "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# ----- Kaggle kernel detection (best-effort) ---------------------------------
IS_KAGGLE=0
if [[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]]; then
  IS_KAGGLE=1
  info "Kaggle environment detected."
fi

# ----- Validations ----------------------------------------------------------
[[ -n "$COMPETITION" ]] || die "Competition slug is empty (use --comp)."
have kaggle || die "kaggle CLI not found. Install via 'pip install kaggle' and auth."
kaggle_authenticated || die "Kaggle auth not found (~/.kaggle/kaggle.json or \$KAGGLE_CONFIG_DIR). See Kaggle > Account > Create API Token."

if [[ -z "$SUBMIT_FILE" ]]; then
  if ! detect_submit_file; then
    die "No submission file found. Provide --file PATH or ensure submission.csv exists in a common location."
  fi
fi
[[ -f "$SUBMIT_FILE" ]] || die "File not found: $SUBMIT_FILE"
[[ -s "$SUBMIT_FILE" ]] || die "Submission file is empty: $SUBMIT_FILE"

# Gzip if requested
ORIG_FILE="$SUBMIT_FILE"
if [[ $GZIP -eq 1 ]]; then
  if [[ "$SUBMIT_FILE" =~ \.csv$ ]]; then
    GZ="${SUBMIT_FILE}.gz"
    info "Gzipping submission → ${BOLD}${GZ}${RST}"
    gzip -c "$SUBMIT_FILE" > "$GZ"
    SUBMIT_FILE="$GZ"
  else
    warn "Skipping --gzip: file does not end with .csv (${SUBMIT_FILE})."
  fi
fi

# Try to enrich default message if untouched
if [[ "$MESSAGE" == "SpectraMind V50 auto-submit" ]]; then
  # Prefer VERSION if available; else short git SHA
  if [[ -f "VERSION" ]]; then
    V=$(sed -n '1s/[[:space:]]//gp' VERSION || true)
    [[ -n "$V" ]] && MESSAGE="SpectraMind V50 ${V}"
  elif have git; then
    SHA=$(git rev-parse --short HEAD 2>/dev/null || true)
    [[ -n "$SHA" ]] && MESSAGE="SpectraMind V50 ${SHA}"
  fi
fi

# Show context
if [[ $QUIET -ne 1 ]]; then
  info "Competition : ${BOLD}${COMPETITION}${RST}"
  info "CSV         : ${BOLD}${SUBMIT_FILE}${RST}"
  info "Message     : ${BOLD}${MESSAGE}${RST}"
  info "Mode        : ${BOLD}$([[ $YES -eq 1 ]] && echo "SUBMIT" || echo "DRY-RUN")${RST}"
  # Best-effort: recent submissions tail
  print_recent_submissions
fi

# ----- Dry-run guard --------------------------------------------------------
if [[ $YES -ne 1 ]]; then
  warn "DRY-RUN only (no submission). Use ${BOLD}--yes${RST} to submit."
  json_emit false "$COMPETITION" "$SUBMIT_FILE" "$MESSAGE" 0
  exit 0
fi

# ----- Submit with optional retries ----------------------------------------
attempt=0
while :; do
  attempt=$((attempt+1))
  info "Submitting (attempt ${attempt})…"
  if submit_once "$COMPETITION" "$SUBMIT_FILE" "$MESSAGE"; then
    ok "Submitted to ${COMPETITION}."
    open_competition_page
    print_recent_submissions
    json_emit true "$COMPETITION" "$SUBMIT_FILE" "$MESSAGE" "$attempt"
    exit 0
  fi
  if (( attempt > RETRIES )); then
    json_emit false "$COMPETITION" "$SUBMIT_FILE" "$MESSAGE" "$attempt"
    die "Submission failed after ${RETRIES} retries."
  fi
  warn "Submission failed. Retrying in ${SLEEP}s…"
  sleep "${SLEEP}"
done