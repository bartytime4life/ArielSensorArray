#!/usr/bin/env bash
# kaggle-submit.sh — Safe, ergonomic Kaggle submission helper for SpectraMind V50
# ------------------------------------------------------------------------------
# Features
#   • Dry-run by default (use --yes to actually submit)
#   • Auto-detects submission file (predictions/submission.csv or outputs/submission.csv)
#   • Validates Kaggle CLI & auth, competition slug, file existence
#   • Nice logs, optional retries/poll, and quick recent-submissions tail
#
# Usage
#   bin/kaggle-submit.sh [--comp SLUG] [--file PATH] [--message "msg"] [--yes]
#                        [--retries N] [--sleep SEC] [--open|--no-open] [--quiet]
#
# Examples
#   bin/kaggle-submit.sh --yes
#   bin/kaggle-submit.sh --comp neurips-2025-ariel --file outputs/predictions/submission.csv --message "V50 run #42" --yes
#
# Notes
#   • Requires: kaggle CLI + valid auth (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR)
#   • Default competition: neurips-2025-ariel
# ------------------------------------------------------------------------------

set -euo pipefail

# ----- Colors ---------------------------------------------------------------
if [[ -t 1 ]]; then
  BOLD='\033[1m'; DIM='\033[2m'; RED='\033[31m'; GRN='\033[32m'; YLW='\033[33m'; CYN='\033[36m'; RST='\033[0m'
else
  BOLD=''; DIM=''; RED=''; GRN=''; YLW=''; CYN=''; RST=''
fi

log()   { printf "%b\n" "${*}"; }
info()  { log "${CYN}::${RST} ${*}"; }
ok()    { log "${GRN}✓${RST} ${*}"; }
warn()  { log "${YLW}⚠${RST} ${*}"; }
err()   { log "${RED}✗${RST} ${*}"; }
die()   { err "$*"; exit 1; }

# ----- Defaults -------------------------------------------------------------
COMPETITION=${COMPETITION:-neurips-2025-ariel}
SUBMIT_FILE=""
MESSAGE="SpectraMind V50 auto-submit"
YES=0
RETRIES=0
SLEEP=15
OPEN=1
QUIET=0

# ----- Helpers --------------------------------------------------------------
usage() {
  cat <<EOF
${BOLD}kaggle-submit.sh${RST} — submit a CSV to a Kaggle competition (safe by default)

${BOLD}Options${RST}
  --comp SLUG        Kaggle competition slug (default: ${COMPETITION})
  --file PATH        Path to submission CSV (auto-detected if omitted)
  --message TEXT     Submission message (default: "${MESSAGE}")
  --yes              Actually submit (dry-run by default)
  --retries N        Retries if Kaggle CLI transiently fails (default: ${RETRIES})
  --sleep SEC        Sleep between retries (default: ${SLEEP})
  --open | --no-open Open the competition page after submit (default: open)
  --quiet            Minimal logs
  -h, --help         Show help

${BOLD}Auto-detect CSV${RST}
  • predictions/submission.csv
  • outputs/predictions/submission.csv
  • outputs/submission.csv

Examples:
  bin/kaggle-submit.sh --yes
  bin/kaggle-submit.sh --comp neurips-2025-ariel --file outputs/predictions/submission.csv --message "V50 run #42" --yes
EOF
}

is_cmd() { command -v "$1" >/dev/null 2>&1; }

detect_submit_file() {
  local candidates=(
    "predictions/submission.csv"
    "outputs/predictions/submission.csv"
    "outputs/submission.csv"
  )
  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then SUBMIT_FILE="$f"; return 0; fi
  done
  return 1
}

print_recent_submissions() {
  if [[ $QUIET -eq 1 ]]; then return 0; fi
  info "Recent submissions (top 6):"
  if ! kaggle competitions submissions -c "$COMPETITION" -v 2>/dev/null | head -n 7; then
    warn "Could not list recent submissions (permission or CLI issue)."
  fi
}

open_competition_page() {
  local url="https://www.kaggle.com/competitions/${COMPETITION}/submissions"
  if [[ $OPEN -eq 1 ]]; then
    if is_cmd xdg-open; then xdg-open "$url" >/dev/null 2>&1 || true
    elif is_cmd open; then open "$url" >/dev/null 2>&1 || true
    fi
  fi
}

kaggle_authenticated() {
  # Kaggle CLI prints a helpful message if unauthenticated; we check for config presence.
  if [[ -n "${KAGGLE_CONFIG_DIR:-}" ]]; then
    [[ -f "${KAGGLE_CONFIG_DIR%/}/kaggle.json" ]]
  else
    [[ -f "$HOME/.kaggle/kaggle.json" ]]
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

# ----- Parse args -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --comp)     COMPETITION="$2"; shift 2 ;;
    --file)     SUBMIT_FILE="$2"; shift 2 ;;
    --message)  MESSAGE="$2"; shift 2 ;;
    --yes)      YES=1; shift ;;
    --retries)  RETRIES="${2:-0}"; shift 2 ;;
    --sleep)    SLEEP="${2:-15}"; shift 2 ;;
    --open)     OPEN=1; shift ;;
    --no-open)  OPEN=0; shift ;;
    --quiet)    QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *)          err "Unknown option: $1"; usage; exit 2 ;;
  esac
done

# ----- Validations ----------------------------------------------------------
[[ -n "$COMPETITION" ]] || die "Competition slug is empty."
is_cmd kaggle || die "kaggle CLI not found. Install via 'pip install kaggle' and auth."
kaggle_authenticated || die "Kaggle auth not found (~/.kaggle/kaggle.json or \$KAGGLE_CONFIG_DIR). See Kaggle > Account > Create API Token."

if [[ -z "$SUBMIT_FILE" ]]; then
  if ! detect_submit_file; then
    die "No submission file found. Provide --file PATH or ensure predictions/submission.csv exists."
  fi
fi
[[ -f "$SUBMIT_FILE" ]] || die "File not found: $SUBMIT_FILE"
[[ -s "$SUBMIT_FILE" ]] || die "Submission file is empty: $SUBMIT_FILE"

# Show context
if [[ $QUIET -ne 1 ]]; then
  info "Competition : ${BOLD}${COMPETITION}${RST}"
  info "CSV         : ${BOLD}${SUBMIT_FILE}${RST}"
  info "Message     : ${BOLD}${MESSAGE}${RST}"
  info "Mode        : ${BOLD}$([[ $YES -eq 1 ]] && echo "SUBMIT" || echo "DRY-RUN")${RST}"
  print_recent_submissions
fi

# ----- Dry-run guard --------------------------------------------------------
if [[ $YES -ne 1 ]]; then
  warn "Dry-run only (no submission). Use ${BOLD}--yes${RST} to submit."
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
    exit 0
  fi
  if (( attempt > RETRIES )); then
    die "Submission failed after ${RETRIES} retries."
  fi
  warn "Submission failed. Retrying in ${SLEEP}s…"
  sleep "${SLEEP}"
done