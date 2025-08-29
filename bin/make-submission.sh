#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — make-submission.sh (Mission-grade, upgraded)
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
#   --open                         Open the submissions/ dir or bundle when done
#   --tag <string>                 Version tag for bundle & manifest
#   --config <path>                Hydra config (e.g., configs/config_v50.yaml)
#   --overrides "<hydra args>"     Quoted Hydra overrides (e.g., '+training.epochs=1 model=v50')
#   --extra "<cli args>"           Extra args passed to spectramind subcommands
#   --pred-out <path.csv>          Predictions CSV path (default: outputs/predictions.csv)
#   --bundle-out <dir-or-zip>      Bundle output (dir or .zip; default: submissions/bundle.zip)
#   --kaggle-submit                Submit to Kaggle (requires kaggle CLI & --kaggle-comp)
#   --kaggle-comp <slug>           Kaggle competition slug (e.g., neurips-2025-ariel)
#   --kaggle-msg "<msg>"           Kaggle submission message
#   --manifest                     Also write a compact run manifest summary to stdout
#   -h | --help                    Show this help
#
# Environment overrides:
#   SPECTRAMIND_CLI    (default: 'spectramind')
#   LOG_FILE           (default: logs/v50_debug_log.md)
#
# Logging:
#   Appends a single structured line to $LOG_FILE (timestamp + git sha + config hash + paths).
#
# Exit codes:
#   0  success
#   1  generic failure
#   2  usage / invalid arguments
#   3  self-test failed
#   4  validation failed
#   5  bundling failed
#   6  kaggle submit failed
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Defaults ----------
DRY_RUN=true
OPEN_AFTER=false
TAG=""
CONFIG="configs/config_v50.yaml"
OVERRIDES=""
EXTRA=""
PRED_CSV="outputs/predictions.csv"
BUNDLE_OUT="submissions/bundle.zip"
KAGGLE_DO_SUBMIT=false
KAGGLE_COMP=""
KAGGLE_MSG=""
EMIT_MANIFEST_STDOUT=false

CLI="${SPECTRAMIND_CLI:-spectramind}"
LOG_FILE="${LOG_FILE:-logs/v50_debug_log.md}"

# ---------- Colors ----------
BOLD="\033[1m"; DIM="\033[2m"; RED="\033[31m"; GRN="\033[32m"; YLW="\033[33m"; CYN="\033[36m"; RST="\033[0m"

# ---------- Helpers ----------
usage() {
  sed -n '1,/^# ==============================================================================/{p}' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

iso_ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
git_sha() { git rev-parse --short HEAD 2>/dev/null || echo "nogit"; }

open_path() {
  local path="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$path" >/dev/null 2>&1 || true
  elif command -v open     >/dev/null 2>&1; then open "$path"     >/dev/null 2>&1 || true
  fi
}

cfg_hash() {
  # Best-effort: prefer a direct CLI flag, fallback to a subcommand, else "-"
  if "$CLI" --help 2>/dev/null | grep -qi "print-config-hash"; then
    "$CLI" --print-config-hash 2>/dev/null || echo "-"
  elif "$CLI" --help 2>/dev/null | grep -qi "hash-config"; then
    "$CLI" hash-config 2>/dev/null || echo "-"
  else
    echo "-"
  fi
}

log_line() {
  mkdir -p "$(dirname "$LOG_FILE")"
  printf '%s cmd=%s git=%s cfg_hash=%s tag=%s pred=%s bundle=%s notes="%s"\n' \
    "$(iso_ts)" "make-submission" "$(git_sha)" "$1" "${TAG:-"-"}" "$2" "$3" "competition=${KAGGLE_COMP:-"-"}" \
    >> "$LOG_FILE"
}

die()   { printf "${RED}ERROR:${RST} %s\n" "$*" >&2; exit 1; }
fail()  { printf "${RED}%s${RST}\n" "$*" >&2; }
info()  { printf "${CYN}%s${RST}\n" "$*"; }
ok()    { printf "${GRN}%s${RST}\n" "$*"; }
warn()  { printf "${YLW}%s${RST}\n" "$*"; }

# ---------- getopt parsing (robust) ----------
if command -v getopt >/dev/null 2>&1; then
  PARSED=$(getopt -o h --long help,dry-run,no-dry-run,open,tag:,config:,overrides:,extra:,pred-out:,bundle-out:,kaggle-submit,kaggle-comp:,kaggle-msg:,manifest -- "$@") || { usage; }
  eval set -- "$PARSED"
  while true; do
    case "$1" in
      -h|--help) usage ;;
      --dry-run) DRY_RUN=true; shift ;;
      --no-dry-run) DRY_RUN=false; shift ;;
      --open) OPEN_AFTER=true; shift ;;
      --tag) TAG="${2:-}"; shift 2 ;;
      --config) CONFIG="${2:-}"; shift 2 ;;
      --overrides) OVERRIDES="${2:-}"; shift 2 ;;
      --extra) EXTRA="${2:-}"; shift 2 ;;
      --pred-out) PRED_CSV="${2:-}"; shift 2 ;;
      --bundle-out) BUNDLE_OUT="${2:-}"; shift 2 ;;
      --kaggle-submit) KAGGLE_DO_SUBMIT=true; shift ;;
      --kaggle-comp) KAGGLE_COMP="${2:-}"; shift 2 ;;
      --kaggle-msg) KAGGLE_MSG="${2:-}"; shift 2 ;;
      --manifest) EMIT_MANIFEST_STDOUT=true; shift ;;
      --) shift; break ;;
      *) die "Unknown option: $1" ;;
    esac
  done
else
  # Minimal fallback parser
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage ;;
      --dry-run) DRY_RUN=true ;;
      --no-dry-run) DRY_RUN=false ;;
      --open) OPEN_AFTER=true ;;
      --tag) TAG="${2:-}"; shift ;;
      --config) CONFIG="${2:-}"; shift ;;
      --overrides) OVERRIDES="${2:-}"; shift ;;
      --extra) EXTRA="${2:-}"; shift ;;
      --pred-out) PRED_CSV="${2:-}"; shift ;;
      --bundle-out) BUNDLE_OUT="${2:-}"; shift ;;
      --kaggle-submit) KAGGLE_DO_SUBMIT=true ;;
      --kaggle-comp) KAGGLE_COMP="${2:-}"; shift ;;
      --kaggle-msg) KAGGLE_MSG="${2:-}"; shift ;;
      --manifest) EMIT_MANIFEST_STDOUT=true ;;
      *) die "Unknown option: $1" ;;
    esac
    shift
  done
fi

# ---------- Setup & context ----------
mkdir -p "outputs" "$(dirname "$PRED_CSV")" "$(dirname "$BUNDLE_OUT")" "submissions" "$(dirname "$LOG_FILE")"

RUN_TS="$(iso_ts)"
GIT_SHA="$(git_sha)"
RUN_ID="${RUN_TS}-${GIT_SHA}"

# Trap for graceful failure logging
trap 'fail "[make-submission] ❌ Failed at $(iso_ts) (RUN_ID=${RUN_ID})"; exit 1' ERR

printf '%s\n' "[make-submission] ========================================================"
printf '%s\n' "[make-submission] Start  : $(iso_ts)"
printf '%s\n' "[make-submission] RUN_ID : ${RUN_ID}"
printf '%s\n' "[make-submission] CLI    : ${CLI}"
printf '%s\n' "[make-submission] DRYRUN : ${DRY_RUN}"
[ -n "$TAG" ]     && printf '%s\n' "[make-submission] TAG    : ${TAG}"
printf '%s\n'     "[make-submission] CONFIG : ${CONFIG}"
[ -n "$OVERRIDES" ] && printf '%s\n' "[make-submission] OVERR  : ${OVERRIDES}"
[ -n "$EXTRA" ]     && printf '%s\n' "[make-submission] EXTRA  : ${EXTRA}"

# Compute a config hash (best-effort)
CFG_HASH="$(cfg_hash)"
[ -n "$CFG_HASH" ] && printf '%s\n' "[make-submission] CFGHASH: ${CFG_HASH}"

# ---------- Guards ----------
command -v "$CLI" >/dev/null 2>&1 || die "Missing CLI: ${CLI}"
# CONFIG may be generated at runtime; warn if not found
if [[ ! -f "$CONFIG" ]]; then
  warn "Config not found: $CONFIG (continuing; CLI may resolve defaults)"
fi

# ---------- 1) Selftest ----------
info  "▶ Selftest (fast)"
if ! "$CLI" test --fast; then
  fail  "Selftest failed"
  log_line "$CFG_HASH" "-"
  exit 3
fi
ok    "Selftest OK"

# ---------- 2) Predict ----------
info  "▶ Predict → ${PRED_CSV}"
PRED_DIR="$(dirname "$PRED_CSV")"; mkdir -p "$PRED_DIR"
set -x
"$CLI" predict --config "$CONFIG" --out-csv "$PRED_CSV" ${EXTRA:+$EXTRA} ${OVERRIDES:+$OVERRIDES}
set +x
[[ -s "$PRED_CSV" ]] || { fail "Prediction CSV not produced: $PRED_CSV"; log_line "$CFG_HASH" "-"; exit 1; }
ok    "Predictions ready: $PRED_CSV"

# ---------- 3) Validate ----------
info  "▶ Validate predictions"
set -x
"$CLI" validate --input "$PRED_CSV" ${EXTRA:+$EXTRA}
VRC=$?
set +x
if [[ $VRC -ne 0 ]]; then
  fail "Validation failed"
  log_line "$CFG_HASH" "$PRED_CSV"
  exit 4
fi
ok    "Validation OK"

# ---------- 4) Bundle (skippable in dry-run) ----------
BUNDLE_CMD=( "$CLI" bundle --pred "$PRED_CSV" --out "$BUNDLE_OUT" )
[[ -n "$TAG" ]] && BUNDLE_CMD+=( --tag "$TAG" )
[[ -n "$EXTRA" ]] && BUNDLE_CMD+=( $EXTRA )

if "$DRY_RUN"; then
  warn  "Dry-run enabled — skipping bundle. Would run:"
  printf '  %q' "${BUNDLE_CMD[@]}"; printf '\n'
  BUNDLE_SAFE="-"
else
  info  "▶ Bundle → $BUNDLE_OUT"
  set -x
  "${BUNDLE_CMD[@]}"
  BRC=$?
  set +x
  if [[ $BRC -ne 0 ]]; then
    fail "Bundling failed"
    log_line "$CFG_HASH" "$PRED_CSV" "-"
    exit 5
  fi
  # If bundle is a file, verify non-empty
  if [[ "$BUNDLE_OUT" != */ && -f "$BUNDLE_OUT" ]]; then
    [[ -s "$BUNDLE_OUT" ]] || { fail "Bundle appears empty: $BUNDLE_OUT"; log_line "$CFG_HASH" "$PRED_CSV" "$BUNDLE_OUT"; exit 5; }
  fi
  BUNDLE_SAFE="$BUNDLE_OUT"
  ok    "Bundle complete"
fi

# ---------- 5) Optional Kaggle submission ----------
if "$KAGGLE_DO_SUBMIT"; then
  if "$DRY_RUN"; then
    warn "Dry-run: Kaggle submission suppressed."
  else
    command -v kaggle >/dev/null 2>&1 || { fail "kaggle CLI not found"; log_line "$CFG_HASH" "$PRED_CSV" "$BUNDLE_SAFE"; exit 6; }
    [[ -n "$KAGGLE_COMP" ]] || { fail "--kaggle-submit requires --kaggle-comp <slug>"; log_line "$CFG_HASH" "$PRED_CSV" "$BUNDLE_SAFE"; exit 2; }
    SUB_FILE="$PRED_CSV"
    info "▶ Kaggle submit → -c \"$KAGGLE_COMP\" -f \"$SUB_FILE\""
    set -x
    kaggle competitions submit -c "$KAGGLE_COMP" -f "$SUB_FILE" -m "${KAGGLE_MSG:-SpectraMind V50 submit ($RUN_ID)}"
    KRC=$?
    set +x
    if [[ $KRC -ne 0 ]]; then
      fail "Kaggle submit failed"
      log_line "$CFG_HASH" "$PRED_CSV" "$BUNDLE_SAFE"
      exit 6
    fi
    ok "Kaggle submit command issued"
  fi
fi

# ---------- Structured log line ----------
log_line "$CFG_HASH" "$PRED_CSV" "${BUNDLE_SAFE:-"-"}"

# ---------- Full JSON manifest on disk ----------
MANIFEST_DIR="outputs/manifests"; mkdir -p "$MANIFEST_DIR"
MANIFEST_PATH="$MANIFEST_DIR/run_manifest_${RUN_ID}.json"
# JSON encode strings via Python (portable & safe)
json_escape() { python3 - <<'PY' "$1"; import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else "")) ; PY
}
CFG_JSON=$(json_escape "$CONFIG")
OVR_JSON=$(json_escape "$OVERRIDES")
EXT_JSON=$(json_escape "$EXTRA")
TAG_JSON=$(json_escape "$TAG")
KMSG_JSON=$(json_escape "$KAGGLE_MSG")

cat > "$MANIFEST_PATH" <<JSON
{
  "run_id": "$(echo "$RUN_ID")",
  "ts_utc": "$(echo "$RUN_TS")",
  "git_sha": "$(echo "$GIT_SHA")",
  "cfg_hash": "$(echo "$CFG_HASH")",
  "config": $CFG_JSON,
  "overrides": $OVR_JSON,
  "extra_args": $EXT_JSON,
  "pred_csv": "$(echo "$PRED_CSV")",
  "bundle_out": "$(echo "$BUNDLE_OUT")",
  "tag": $TAG_JSON,
  "dry_run": $( $DRY_RUN && echo true || echo false ),
  "kaggle": {
    "submit": $( $KAGGLE_DO_SUBMIT && echo true || echo false ),
    "comp": "$(echo "$KAGGLE_COMP")",
    "message": $KMSG_JSON
  }
}
JSON
ok "Manifest: $MANIFEST_PATH"

# ---------- Optional concise manifest to stdout ----------
if "$EMIT_MANIFEST_STDOUT"; then
  printf '%s\n' "$MANIFEST_PATH"
fi

# ---------- Optional: open dir/bundle ----------
if "$OPEN_AFTER"; then
  if "$DRY_RUN"; then
    open_path "$(dirname "$BUNDLE_OUT")"
  else
    if [[ -f "$BUNDLE_OUT" ]]; then open_path "$BUNDLE_OUT"; else open_path "$(dirname "$BUNDLE_OUT")"; fi
  fi
fi

ok   "[make-submission] Completed at $(iso_ts)  (RUN_ID=${RUN_ID})"
printf '%s\n' "[make-submission] ========================================================"
exit 0
