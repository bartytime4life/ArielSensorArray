#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — diagnostics.sh
# ------------------------------------------------------------------------------
# Purpose:
#   Run rich diagnostics (smoothness, dashboard, optional symbolic overlays)
#   for a given run/output directory, with manifest + log append.
#
# Usage:
#   ./bin/diagnostics.sh [options]
#
# Common options:
#   --outdir <dir>          Target directory for diagnostics (default: outputs/diagnostics/<ts>)
#   --source <path>         Optional source dir or file (e.g., predictions.csv) to diagnose
#   --overrides "<hydra>"   Quoted Hydra overrides
#   --extra "<cli>"         Extra args to pass to spectramind diagnose
#   --no-umap               Skip UMAP in dashboard
#   --no-tsne               Skip t-SNE in dashboard
#   --symbolic              Include symbolic overlays/violation tables if available
#   --open                  Open latest HTML after run
#   --manifest              Write JSON manifest into outdir
#   -h|--help               Show help
#
# Notes:
#   - Appends a one-line entry to logs/v50_debug_log.md
#   - Idempotent; safe to re-run
# ==============================================================================

set -Eeuo pipefail

OUTDIR=""
SOURCE=""
OVERRIDES=""
EXTRA=""
NO_UMAP=false
NO_TSNE=false
WITH_SYMBOLIC=false
OPEN_AFTER=false
WRITE_MANIFEST=false
CLI="${SPECTRAMIND_CLI:-spectramind}"
LOG_FILE="${LOG_FILE:-logs/v50_debug_log.md}"

usage() { sed -n '1,/^# =/p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
ts()     { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha() { git rev-parse --short HEAD 2>/dev/null || echo "nogit"; }
log()    { mkdir -p "$(dirname "$LOG_FILE")"; printf "%b\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
info()   { printf "\033[36m%b\033[0m\n" "$*"; }
ok()     { printf "\033[32m%b\033[0m\n" "$*"; }
warn()   { printf "\033[33m%b\033[0m\n" "$*"; }
err()    { printf "\033[31m%b\033[0m\n" "$*"; }

open_path() {
  local p="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$p" || true
  elif command -v open >/dev/null 2>&1; then open "$p" || true
  fi
}

if command -v getopt >/dev/null 2>&1; then
  PARSED=$(getopt -o h --long help,outdir:,source:,overrides:,extra:,no-umap,no-tsne,symbolic,open,manifest -- "$@") || usage
  eval set -- "$PARSED"
  while true; do
    case "$1" in
      -h|--help) usage ;;
      --outdir) OUTDIR="$2"; shift 2;;
      --source) SOURCE="$2"; shift 2;;
      --overrides) OVERRIDES="$2"; shift 2;;
      --extra) EXTRA="$2"; shift 2;;
      --no-umap) NO_UMAP=true; shift ;;
      --no-tsne) NO_TSNE=true; shift ;;
      --symbolic) WITH_SYMBOLIC=true; shift ;;
      --open) OPEN_AFTER=true; shift ;;
      --manifest) WRITE_MANIFEST=true; shift ;;
      --) shift; break ;;
      *) err "Unknown option: $1"; exit 2;;
    esac
  done
else
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage ;;
      --outdir) OUTDIR="$2"; shift ;;
      --source) SOURCE="$2"; shift ;;
      --overrides) OVERRIDES="$2"; shift ;;
      --extra) EXTRA="$2"; shift ;;
      --no-umap) NO_UMAP=true ;;
      --no-tsne) NO_TSNE=true ;;
      --symbolic) WITH_SYMBOLIC=true ;;
      --open) OPEN_AFTER=true ;;
      --manifest) WRITE_MANIFEST=true ;;
      *) err "Unknown option: $1"; exit 2;;
    esac; shift
  done
fi

RUN_TS="$(ts)"
GIT_SHA="$(gitsha)"
RUN_ID="${RUN_TS}-${GIT_SHA}"
[ -n "$OUTDIR" ] || OUTDIR="outputs/diagnostics/${RUN_TS}"
mkdir -p "$OUTDIR" logs

trap 'err "[diagnostics] ❌ Failed at $(ts) (RUN_ID=${RUN_ID})"; exit 1' ERR

log "[diagnostics] ======================================================"
log "[diagnostics] Start  : ${RUN_TS}"
log "[diagnostics] RUN_ID : ${RUN_ID}"
[ -n "$SOURCE" ]   && log "[diagnostics] Source  : ${SOURCE}"
[ -n "$OVERRIDES" ] && log "[diagnostics] Overrides: ${OVERRIDES}"
[ -n "$EXTRA" ]     && log "[diagnostics] Extra    : ${EXTRA}"

info "▶ Self-test (fast)"
$CLI test --fast || { err "Self-test failed"; exit 3; }

# Build dashboard flags
DB_FLAGS=( --outdir "$OUTDIR" )
$NO_UMAP && DB_FLAGS+=( --no-umap )
$NO_TSNE && DB_FLAGS+=( --no-tsne )
[ -n "$EXTRA" ] && DB_FLAGS+=( $EXTRA )

# Smoothness
info "▶ Diagnostics: smoothness"
SMOOTH_CMD=( $CLI diagnose smoothness --outdir "$OUTDIR" )
[ -n "$EXTRA" ] && SMOOTH_CMD+=( $EXTRA )
[ -n "$OVERRIDES" ] && SMOOTH_CMD+=( $OVERRIDES )
"${SMOOTH_CMD[@]}" || warn "smoothness returned non-zero"

# Symbolic overlays (optional)
if $WITH_SYMBOLIC; then
  info "▶ Diagnostics: symbolic overlays"
  SYM_CMD=( $CLI diagnose symbolic-rank --outdir "$OUTDIR" )
  [ -n "$EXTRA" ] && SYM_CMD+=( $EXTRA )
  [ -n "$OVERRIDES" ] && SYM_CMD+=( $OVERRIDES )
  "${SYM_CMD[@]}" || warn "symbolic-rank returned non-zero"
fi

# Dashboard
info "▶ Diagnostics: dashboard"
DB_CMD=( $CLI diagnose dashboard "${DB_FLAGS[@]}" )
[ -n "$OVERRIDES" ] && DB_CMD+=( $OVERRIDES )
"${DB_CMD[@]}" || warn "dashboard returned non-zero"

# Summary
SUMMARY="${OUTDIR}/diagnostics_summary.txt"
{
  echo "Diagnostics summary"
  echo "time_utc : $(ts)"
  echo "run_id   : ${RUN_ID}"
  echo "git_sha  : ${GIT_SHA}"
  echo "outdir   : ${OUTDIR}"
  echo "source   : ${SOURCE:-n/a}"
  echo "symbolic : ${WITH_SYMBOLIC}"
  echo "flags    : no_umap=${NO_UMAP} no_tsne=${NO_TSNE}"
} > "$SUMMARY"
ok "Summary → $SUMMARY"

# Manifest (optional)
if $WRITE_MANIFEST; then
  MANIFEST="${OUTDIR}/diagnostics_manifest_${RUN_ID}.json"
  python - <<PY > "$MANIFEST"
import json,os,time
print(json.dumps({
  "run_id": os.environ.get("RUN_ID","${RUN_ID}"),
  "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "git_sha": "${GIT_SHA}",
  "outdir": "${OUTDIR}",
  "source": "${SOURCE}",
  "symbolic": ${WITH_SYMBOLIC}
}, indent=2))
PY
  ok "Manifest → $MANIFEST"
fi

# Open (optional)
if $OPEN_AFTER; then
  latest_html="$(ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true)"
  if [ -n "$latest_html" ]; then
    info "Opening ${latest_html}"
    open_path "$latest_html"
  else
    warn "No HTML report found in ${OUTDIR}"
  fi
fi

log "[diagnostics] Completed at $(ts)  (RUN_ID=${RUN_ID})"
log "[diagnostics] ======================================================"
ok "[diagnostics] ✅ Done"
