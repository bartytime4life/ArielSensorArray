#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — make-submission.sh (Upgraded, mission-grade)
# ------------------------------------------------------------------------------
# Purpose:
#   Orchestrate a *safe, reproducible* submission flow:
#     1) selftest  →  2) predict  →  3) validate  →  4) bundle  →  (5) optional Kaggle submit
#
# Philosophy:
#   • CLI-first (wraps the `spectramind` Typer CLI)
#   • Hydra-safe (no hard-coded params; pass via --config/--overrides/--extra)
#   • Reproducibility (write a run manifest; log config hash, git, ts)
#   • Safe-by-default (dry-run skips final bundle & submit unless disabled)
#
# Usage:
#   ./bin/make-submission.sh [flags]
#
# Common flags:
#   --dry-run | --no-dry-run       Keep / skip the final bundle + submit (default: --dry-run)
#   --open                         Open the submissions/ dir when done
#   --tag <string>                 Version tag for the bundle filename and manifest
#   --config <path>                Hydra config (e.g., configs/config_v50.yaml)
#   --overrides "<hydra args>"     Quoted Hydra overrides (e.g., '+training.epochs=1 model=v50')
#   --extra "<cli args>"           Extra args passed to spectramind subcommands
#   --pred-out <path.csv>          Predictions CSV path (default: outputs/predictions.csv)
#   --bundle-out <dir-or-zip>      Bundle output (dir or .zip; default: submissions/)
#   --kaggle-submit                Submit to Kaggle (requires kaggle CLI & --kaggle-comp)
#   --kaggle-comp <slug>           Kaggle competition slug (e.g., neurips-2025-ariel)
#   --kaggle-msg "<msg>"           Kaggle submission message
#   -h | --help                    Show this help
#
# Environment overrides:
#   SPECTRAMIND_CLI    (default: 'spectramind')
#   LOG_FILE           (default: logs/v50_debug_log.md)
#
# Logging:
#   Appends to $LOG_FILE (timestamp + git sha + config hash + command lines).
# ==============================================================================

set -Eeuo pipefail

# ---------- Defaults ----------
DRY_RUN=true
OPEN_AFTER=false
TAG=""
CONFIG="configs/config_v50.yaml"
OVERRIDES=""
EXTRA=""
PRED_CSV="outputs/predictions.csv"
BUNDLE_OUT="submissions/"
KAGGLE_DO_SUBMIT=false
KAGGLE_COMP=""
KAGGLE_MSG=""
CLI="${SPECTRAMIND_CLI:-spectramind}"
LOG_FILE="${LOG_FILE:-logs/v50_debug_log.md}"

# ---------- Colors ----------
BOLD="\033[1m"; DIM="\033[2m"; RED="\033[31m"; GRN="\033[32m"; YLW="\033[33m"; CYN="\033[36m"; RST="\033[0m"

# ---------- Helpers ----------
usage() {
  sed -n '1,/^# =/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

log()   { printf "%b\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
info()  { printf "${CYN}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
ok()    { printf "${GRN}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
warn()  { printf "${YLW}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
err()   { printf "${RED}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }

ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha(){ git rev-parse --short HEAD 2>/dev/null || echo "nogit"; }

open_path() {
  local path="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$path" || true
  elif command -v open >/devnull 2>&1 || command -v open >/dev/null 2>&1; then open "$path" || true
  fi
}

# ---------- getopt parsing (robust) ----------
if command -v getopt >/dev/null 2>&1; then
  PARSED=$(getopt -o h --long help,dry-run,no-dry-run,open,tag:,config:,overrides:,extra:,pred-out:,bundle-out:,kaggle-submit,kaggle-comp:,kaggle-msg: -- "$@") || { usage; }
  eval set -- "$PARSED"
  while true; do
    case "$1" in
      -h|--help) usage ;;
      --dry-run) DRY_RUN=true; shift ;;
      --no-dry-run) DRY_RUN=false; shift ;;
      --open) OPEN_AFTER=true; shift ;;
      --tag) TAG="$2"; shift 2 ;;
      --config) CONFIG="$2"; shift 2 ;;
      --overrides) OVERRIDES="$2"; shift 2 ;;
      --extra) EXTRA="$2"; shift 2 ;;
      --pred-out) PRED_CSV="$2"; shift 2 ;;
      --bundle-out) BUNDLE_OUT="$2"; shift 2 ;;
      --kaggle-submit) KAGGLE_DO_SUBMIT=true; shift ;;
      --kaggle-comp) KAGGLE_COMP="$2"; shift 2 ;;
      --kaggle-msg) KAGGLE_MSG="$2"; shift 2 ;;
      --) shift; break ;;
      *) err "Unknown option: $1"; exit 2 ;;
    esac
  done
else
  # Fallback: minimal parser
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage ;;
      --dry-run) DRY_RUN=true ;;
      --no-dry-run) DRY_RUN=false ;;
      --open) OPEN_AFTER=true ;;
      --tag) TAG="$2"; shift ;;
      --config) CONFIG="$2"; shift ;;
      --overrides) OVERRIDES="$2"; shift ;;
      --extra) EXTRA="$2"; shift ;;
      --pred-out) PRED_CSV="$2"; shift ;;
      --bundle-out) BUNDLE_OUT="$2"; shift ;;
      --kaggle-submit) KAGGLE_DO_SUBMIT=true ;;
      --kaggle-comp) KAGGLE_COMP="$2"; shift ;;
      --kaggle-msg) KAGGLE_MSG="$2"; shift ;;
      *) err "Unknown option: $1"; exit 2 ;;
    esac
    shift
  done
fi

# ---------- Setup & context ----------
mkdir -p "$(dirname "$LOG_FILE")" outputs "$(dirname "$PRED_CSV")" "$BUNDLE_OUT"

RUN_TS="$(ts)"
GIT_SHA="$(gitsha)"
RUN_ID="${RUN_TS}-${GIT_SHA}"

# Trap for graceful failure logging
trap 'err "[make-submission] ❌ Failed at $(ts) (RUN_ID=${RUN_ID})"; exit 1' ERR

log   "[make-submission] ========================================================"
log   "[make-submission] Start  : $(ts)"
log   "[make-submission] RUN_ID : ${RUN_ID}"
log   "[make-submission] CLI    : ${CLI}"
log   "[make-submission] DRYRUN : ${DRY_RUN}"
[ -n "$TAG" ] && log "[make-submission] TAG    : ${TAG}"

# Show config / overrides succinctly
log "[make-submission] CONFIG : ${CONFIG}"
[ -n "$OVERRIDES" ] && log "[make-submission] OVERR  : ${OVERRIDES}"
[ -n "$EXTRA" ]     && log "[make-submission] EXTRA  : ${EXTRA}"

# Compute a config hash (best-effort)
CFG_HASH="$($CLI hash-config 2>/dev/null || echo "")"
[ -n "$CFG_HASH" ] && log "[make-submission] CFGHASH: ${CFG_HASH}"

# ---------- Guards ----------
command -v "$CLI"      >/dev/null 2>&1 || { err "Missing CLI: ${CLI}"; exit 1; }
[ -f "$CONFIG" ]       || { warn "Config not found: $CONFIG (continuing; may be overridden by CLI default)"; }

# ---------- 1) Selftest ----------
info  "▶ Selftest (fast)"
$CLI test --fast || { err "Selftest failed"; exit 1; }
ok    "Selftest OK"

# ---------- 2) Predict ----------
info  "▶ Predict → ${PRED_CSV}"
PRED_DIR="$(dirname "$PRED_CSV")"; mkdir -p "$PRED_DIR"
set -x
$CLI predict --config "$CONFIG" --out-csv "$PRED_CSV" $EXTRA ${OVERRIDES:+$OVERRIDES}
set +x
[ -s "$PRED_CSV" ] || { err "Prediction CSV not produced: $PRED_CSV"; exit 1; }
ok    "Predictions ready: $PRED_CSV"

# ---------- 3) Validate ----------
info  "▶ Validate predictions"
set -x
$CLI validate --input "$PRED_CSV" $EXTRA
set +x
ok    "Validation OK"

# ---------- 4) Bundle (skippable in dry-run) ----------
BUNDLE_CMD="$CLI bundle --pred \"$PRED_CSV\" --out \"$BUNDLE_OUT\""
[ -n "$TAG" ] && BUNDLE_CMD="$BUNDLE_CMD --tag \"$TAG\""

if "$DRY_RUN"; then
  warn  "Dry-run enabled — skipping bundle. Would run:"
  printf "%s\n" "  $BUNDLE_CMD" | tee -a "$LOG_FILE" >/dev/null
else
  info  "▶ Bundle → $BUNDLE_OUT"
  set -x
  eval "$BUNDLE_CMD"
  set +x
  ok    "Bundle complete"
fi

# ---------- 5) Optional Kaggle submission ----------
if "$KAGGLE_DO_SUBMIT"; then
  if "$DRY_RUN"; then
    warn "Dry-run: Kaggle submission suppressed."
  else
    command -v kaggle >/dev/null 2>&1 || { err "kaggle CLI not found"; exit 1; }
    [ -n "$KAGGLE_COMP" ] || { err "--kaggle-submit requires --kaggle-comp <slug>"; exit 1; }
    SUB_FILE="$PRED_CSV"
    info "▶ Kaggle submit → -c \"$KAGGLE_COMP\" -f \"$SUB_FILE\""
    set -x
    kaggle competitions submit -c "$KAGGLE_COMP" -f "$SUB_FILE" -m "${KAGGLE_MSG:-SpectraMind V50 submit ($RUN_ID)}"
    set +x
    ok "Kaggle submit command issued"
  fi
fi

# ---------- Run manifest ----------
MANIFEST_DIR="outputs/manifests"; mkdir -p "$MANIFEST_DIR"
MANIFEST_PATH="$MANIFEST_DIR/run_manifest_${RUN_ID}.json"
{
  printf '{\n'
  printf '  "run_id": "%s",\n'        "$RUN_ID"
  printf '  "ts_utc": "%s",\n'        "$RUN_TS"
  printf '  "git_sha": "%s",\n'       "$GIT_SHA"
  printf '  "cfg_hash": "%s",\n'      "$CFG_HASH"
  printf '  "config": "%s",\n'        "$CONFIG"
  printf '  "overrides": %s,\n'       "$(printf '%s' "${OVERRIDES:-}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  printf '  "extra_args": %s,\n'      "$(printf '%s' "${EXTRA:-}"     | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  printf '  "pred_csv": "%s",\n'      "$PRED_CSV"
  printf '  "bundle_out": "%s",\n'    "$BUNDLE_OUT"
  printf '  "tag": "%s",\n'           "$TAG"
  printf '  "dry_run": %s,\n'         "$( $DRY_RUN && echo true || echo false )"
  printf '  "kaggle": { "submit": %s, "comp": "%s", "message": %s }\n' \
         "$( $KAGGLE_DO_SUBMIT && echo true || echo false )" \
         "$KAGGLE_COMP" \
         "$(printf '%s' "${KAGGLE_MSG:-}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  printf '}\n'
} > "$MANIFEST_PATH"
ok "Manifest: $MANIFEST_PATH"

# ---------- Optional: open output dir ----------
if "$OPEN_AFTER"; then
  open_path "$BUNDLE_OUT"
fi

ok   "[make-submission] Completed at $(ts)  (RUN_ID=${RUN_ID})"
log  "[make-submission] ========================================================"
