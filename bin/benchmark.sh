#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — benchmark.sh
# ------------------------------------------------------------------------------
# Purpose:
#   Run a standardized benchmark pass with logging and reproducibility controls.
#   Trains for a small, configurable number of epochs, runs diagnostics, and
#   writes a summary file + optional manifest.
#
# Usage:
#   ./bin/benchmark.sh [options]
#
# Common options:
#   --profile {cpu|gpu}     Device profile (default: gpu)
#   --epochs <N>            Training epochs (default: 1)
#   --seed <N>              Random seed (default: 42)
#   --overrides "<hydra>"   Quoted Hydra overrides (e.g. '+training.lr=3e-4')
#   --extra "<cli>"         Extra args to pass through to CLI commands
#   --outdir <dir>          Output dir (default: benchmarks/<ts>_<profile>)
#   --tag <str>             Tag label for logs/summary (default: "")
#   --dry-run               Plan steps, do not execute training/diagnostics
#   --open-report           Open latest HTML in outdir after run
#   --manifest              Write JSON manifest in outdir
#   -h|--help               Show help
#
# Notes:
#   - Appends a one-line log to logs/v50_debug_log.md.
#   - Emits <outdir>/summary.txt and (if --manifest) manifest JSON.
# ==============================================================================

set -Eeuo pipefail

# ---------- defaults ----------
PROFILE="gpu"
EPOCHS=1
SEED=42
OVERRIDES=""
EXTRA=""
OUTDIR=""
TAG=""
DRY_RUN=false
OPEN_AFTER=false
WRITE_MANIFEST=false
CLI="${SPECTRAMIND_CLI:-spectramind}"
LOG_FILE="${LOG_FILE:-logs/v50_debug_log.md}"

# ---------- helpers ----------
usage() { sed -n '1,/^# =/p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

ts()      { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha()  { git rev-parse --short HEAD 2>/dev/null || echo "nogit"; }
log()     { mkdir -p "$(dirname "$LOG_FILE")"; printf "%b\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
info()    { printf "\033[36m%b\033[0m\n" "$*"; }
ok()      { printf "\033[32m%b\033[0m\n" "$*"; }
warn()    { printf "\033[33m%b\033[0m\n" "$*"; }
err()     { printf "\033[31m%b\033[0m\n" "$*"; }

open_path() {
  local p="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$p" || true
  elif command -v open >/dev/null 2>&1; then open "$p" || true
  fi
}

retry() {
  # retry <attempts> <sleep_s> -- cmd...
  local n=1 attempts="$1" sleep_s="$2"; shift 2
  until "$@"; do
    if (( n >= attempts )); then return 1; fi
    warn "Retry $n/$attempts failed; sleeping ${sleep_s}s ..."
    sleep "$sleep_s"; ((n++))
  done
}

# ---------- argparse ----------
if command -v getopt >/dev/null 2>&1; then
  PARSED=$(getopt -o h --long help,profile:,epochs:,seed:,overrides:,extra:,outdir:,tag:,dry-run,open-report,manifest -- "$@") || usage
  eval set -- "$PARSED"
  while true; do
    case "$1" in
      -h|--help) usage ;;
      --profile) PROFILE="$2"; shift 2;;
      --epochs) EPOCHS="$2"; shift 2;;
      --seed) SEED="$2"; shift 2;;
      --overrides) OVERRIDES="$2"; shift 2;;
      --extra) EXTRA="$2"; shift 2;;
      --outdir) OUTDIR="$2"; shift 2;;
      --tag) TAG="$2"; shift 2;;
      --dry-run) DRY_RUN=true; shift ;;
      --open-report) OPEN_AFTER=true; shift ;;
      --manifest) WRITE_MANIFEST=true; shift ;;
      --) shift; break ;;
      *) err "Unknown option: $1"; exit 2;;
    esac
  done
else
  # minimal fallback
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage ;;
      --profile) PROFILE="$2"; shift ;;
      --epochs) EPOCHS="$2"; shift ;;
      --seed) SEED="$2"; shift ;;
      --overrides) OVERRIDES="$2"; shift ;;
      --extra) EXTRA="$2"; shift ;;
      --outdir) OUTDIR="$2"; shift ;;
      --tag) TAG="$2"; shift ;;
      --dry-run) DRY_RUN=true ;;
      --open-report) OPEN_AFTER=true ;;
      --manifest) WRITE_MANIFEST=true ;;
      *) err "Unknown option: $1"; exit 2;;
    esac; shift
  done
fi

RUN_TS="$(ts)"
GIT_SHA="$(gitsha)"
RUN_ID="${RUN_TS}-${GIT_SHA}"
[ -n "$OUTDIR" ] || OUTDIR="benchmarks/${RUN_TS}_${PROFILE}"
mkdir -p "$OUTDIR" logs

trap 'err "[benchmark] ❌ Failed at $(ts) (RUN_ID=${RUN_ID})"; exit 1' ERR

log "[benchmark] ========================================================"
log "[benchmark] Start  : ${RUN_TS}"
log "[benchmark] RUN_ID : ${RUN_ID}"
log "[benchmark] Profile: ${PROFILE}  Epochs:${EPOCHS}  Seed:${SEED}  Tag:${TAG}"
[ -n "$OVERRIDES" ] && log "[benchmark] Overrides: ${OVERRIDES}"
[ -n "$EXTRA" ] && log "[benchmark] Extra    : ${EXTRA}"

# ---------- self-test ----------
info "▶ Self-test (fast)"
if ! $DRY_RUN; then
  $CLI test --fast
else
  warn "[dry-run] Skipping self-test execution"
fi

# ---------- train ----------
info "▶ Train (${EPOCHS} epochs, seed=${SEED}, device=${PROFILE})"
TRAIN_CMD=( $CLI train +training.epochs="$EPOCHS" +training.seed="$SEED" --device "$PROFILE" --outdir "$OUTDIR" )
[ -n "$OVERRIDES" ] && TRAIN_CMD+=( $OVERRIDES )
[ -n "$EXTRA" ] && TRAIN_CMD+=( $EXTRA )

if $DRY_RUN; then
  warn "[dry-run] ${TRAIN_CMD[*]}"
else
  retry 2 3 -- "${TRAIN_CMD[@]}"
fi

# ---------- diagnostics ----------
info "▶ Diagnostics (smoothness + dashboard)"
DIAG1_CMD=( $CLI diagnose smoothness --outdir "$OUTDIR" )
DIAG2_CMD=( $CLI diagnose dashboard --outdir "$OUTDIR" )
[ -n "$EXTRA" ] && DIAG1_CMD+=( $EXTRA )
[ -n "$EXTRA" ] && DIAG2_CMD+=( $EXTRA )

if $DRY_RUN; then
  warn "[dry-run] ${DIAG1_CMD[*]}"
  warn "[dry-run] ${DIAG2_CMD[*]}"
else
  "${DIAG1_CMD[@]}" || warn "smoothness diagnostics returned non-zero"
  "${DIAG2_CMD[@]}" || warn "dashboard diagnostics returned non-zero"
fi

# ---------- summary ----------
SUMMARY="${OUTDIR}/summary.txt"
{
  echo "Benchmark summary"
  echo "time_utc : $(ts)"
  echo "run_id   : ${RUN_ID}"
  echo "git_sha  : ${GIT_SHA}"
  echo "profile  : ${PROFILE}"
  echo "epochs   : ${EPOCHS}"
  echo "seed     : ${SEED}"
  echo "tag      : ${TAG}"
  echo "outdir   : ${OUTDIR}"
} > "$SUMMARY"
ok "Summary → $SUMMARY"

# ---------- manifest (optional) ----------
if $WRITE_MANIFEST; then
  MANIFEST="${OUTDIR}/benchmark_manifest_${RUN_ID}.json"
  python - <<PY > "$MANIFEST"
import json,os,time
print(json.dumps({
  "run_id": os.environ.get("RUN_ID","${RUN_ID}"),
  "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "git_sha": "${GIT_SHA}",
  "profile": "${PROFILE}",
  "epochs": ${EPOCHS},
  "seed": ${SEED},
  "tag": "${TAG}",
  "outdir": "${OUTDIR}"
}, indent=2))
PY
  ok "Manifest → $MANIFEST"
fi

# ---------- open (optional) ----------
if $OPEN_AFTER; then
  latest_html="$(ls -t "${OUTDIR}"/*.html 2>/dev/null | head -n1 || true)"
  if [ -n "$latest_html" ]; then
    info "Opening ${latest_html}"
    open_path "$latest_html"
  else
    warn "No HTML report found in ${OUTDIR}"
  fi
fi

log "[benchmark] Completed at $(ts)  (RUN_ID=${RUN_ID})"
log "[benchmark] ========================================================"
ok "[benchmark] ✅ Done"
