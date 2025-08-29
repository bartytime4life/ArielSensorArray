#!/usr/bin/env bash
# ==============================================================================
# 🛰️ SpectraMind V50 — repair_and_push.sh (Upgraded, mission-grade)
# ------------------------------------------------------------------------------
# Purpose:
#   Safely repair, commit, and push repository state with Git + DVC consistency.
#   - Verifies working tree and branch
#   - (Optional) runs fast selftests and pre-commit hooks
#   - Adds + commits code & data (DVC)
#   - Pushes Git and DVC with retries
#   - Writes a run manifest and appends log lines to logs/v50_debug_log.md
#
# Usage:
#   ./bin/repair_and_push.sh --msg "Commit message" [flags]
#
# Flags:
#   --msg "<text>"           Commit message (required unless --allow-empty)
#   --allow-empty            Allow an empty commit if nothing changed
#   --allow-non-main         Allow pushing from non-main branch
#   --run-tests              Run `spectramind test --fast` before commit
#   --run-pre-commit         Run `pre-commit run --all-files` if available
#   --no-dvc                 Skip DVC add/status/push
#   --no-push                Skip Git/DVC push (local-only repair)
#   --tag "<vX.Y.Z>"         Create an annotated tag on the commit (and push tag)
#   --manifest               Write a JSON manifest under outputs/manifests/
#   -h | --help              Show help
#
# Env overrides:
#   SPECTRAMIND_CLI   (default: 'spectramind')
#   LOG_FILE          (default: logs/v50_debug_log.md)
# ==============================================================================

set -Eeuo pipefail

# ---------- colors ----------
BOLD="\033[1m"; DIM="\033[2m"; RED="\033[31m"; GRN="\033[32m"; YLW="\033[33m"; CYN="\033[36m"; RST="\033[0m"

# ---------- defaults ----------
CLI="${SPECTRAMIND_CLI:-spectramind}"
LOG_FILE="${LOG_FILE:-logs/v50_debug_log.md}"

COMMIT_MSG=""
ALLOW_EMPTY=false
ALLOW_NON_MAIN=false
RUN_TESTS=false
RUN_PRE_COMMIT=false
DO_DVC=true
DO_PUSH=true
TAG_NAME=""
WRITE_MANIFEST=false

# ---------- helpers ----------
usage() {
  sed -n '1,/^# =/p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}
ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha(){ git rev-parse --short HEAD 2>/dev/null || echo "nogit"; }
branch(){ git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "nobranch"; }

log()   { printf "%b\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
info()  { printf "${CYN}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
ok()    { printf "${GRN}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
warn()  { printf "${YLW}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }
err()   { printf "${RED}%b${RST}\n" "$*" | tee -a "$LOG_FILE" >/dev/null; }

retry() {
  # retry <attempts> <sleep> -- <cmd...>
  local attempts="$1" sleep_s="$2"; shift 2
  local n=1
  until "$@"; do
    if (( n >= attempts )); then return 1; fi
    warn "Retry $n/$attempts failed, sleeping ${sleep_s}s …"
    sleep "${sleep_s}"
    ((n++))
  done
}

open_path() {
  local p="$1"
  if command -v xdg-open >/dev/null 2>&1; then xdg-open "$p" || true
  elif command -v open >/dev/null 2>&1; then open "$p" || true
  fi
}

# ---------- getopt parsing ----------
if command -v getopt >/dev/null 2>&1; then
  PARSED=$(getopt -o h --long help,msg:,allow-empty,allow-non-main,run-tests,run-pre-commit,no-dvc,no-push,tag:,manifest -- "$@") || usage
  eval set -- "$PARSED"
  while true; do
    case "$1" in
      -h|--help) usage ;;
      --msg) COMMIT_MSG="$2"; shift 2 ;;
      --allow-empty) ALLOW_EMPTY=true; shift ;;
      --allow-non-main) ALLOW_NON_MAIN=true; shift ;;
      --run-tests) RUN_TESTS=true; shift ;;
      --run-pre-commit) RUN_PRE_COMMIT=true; shift ;;
      --no-dvc) DO_DVC=false; shift ;;
      --no-push) DO_PUSH=false; shift ;;
      --tag) TAG_NAME="$2"; shift 2 ;;
      --manifest) WRITE_MANIFEST=true; shift ;;
      --) shift; break ;;
      *) err "Unknown option: $1"; exit 2 ;;
    esac
  done
else
  # minimal fallback parser
  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help) usage ;;
      --msg) COMMIT_MSG="$2"; shift ;;
      --allow-empty) ALLOW_EMPTY=true ;;
      --allow-non-main) ALLOW_NON_MAIN=true ;;
      --run-tests) RUN_TESTS=true ;;
      --run-pre-commit) RUN_PRE_COMMIT=true ;;
      --no-dvc) DO_DVC=false ;;
      --no-push) DO_PUSH=false ;;
      --tag) TAG_NAME="$2"; shift ;;
      --manifest) WRITE_MANIFEST=true ;;
      *) err "Unknown option: $1"; exit 2 ;;
    esac
    shift
  done
fi

mkdir -p "$(dirname "$LOG_FILE")"

RUN_TS="$(ts)"
GIT_SHA="$(gitsha)"
BRANCH="$(branch)"
RUN_ID="${RUN_TS}-${GIT_SHA}"

trap 'err "[repair_and_push] ❌ Failed at $(ts) (RUN_ID=${RUN_ID})"; exit 1' ERR

log   "[repair_and_push] ========================================================"
log   "[repair_and_push] Start  : $(ts)"
log   "[repair_and_push] RUN_ID : ${RUN_ID}"
log   "[repair_and_push] BRANCH : ${BRANCH}"
[ -n "$TAG_NAME" ] && log "[repair_and_push] TAG    : ${TAG_NAME}"

# ---------- guards ----------
command -v git >/dev/null 2>&1 || { err "git not found"; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { err "Not a git repo"; exit 1; }

if [ -z "$COMMIT_MSG" ] && [ "$ALLOW_EMPTY" != true ]; then
  err "Commit message required (use --msg \"...\"), or pass --allow-empty"
  exit 2
fi

if [ "$ALLOW_NON_MAIN" != true ] && [ "$BRANCH" != "main" ] && [ "$BRANCH" != "master" ]; then
  err "Refusing to push from non-main branch (${BRANCH}). Use --allow-non-main to override."
  exit 2
fi

# ---------- optional tests ----------
if [ "$RUN_TESTS" = true ]; then
  if command -v "$CLI" >/dev/null 2>&1; then
    info "▶ Running fast selftest"
    $CLI test --fast
    ok "Selftest OK"
  else
    warn "SpectraMind CLI not found; skipping selftest."
  fi
fi

# ---------- pre-commit hooks (if requested and available) ----------
if [ "$RUN_PRE_COMMIT" = true ]; then
  if command -v pre-commit >/dev/null 2>&1; then
    info "▶ Running pre-commit hooks"
    pre-commit run --all-files || warn "pre-commit reported issues"
  else
    warn "pre-commit not installed; skipping."
  fi
fi

# ---------- DVC check ----------
if [ "$DO_DVC" = true ]; then
  if command -v dvc >/dev/null 2>&1; then
    info "▶ DVC status"
    dvc status || warn "DVC status reported differences."
    # Add newly created data files if any conventionally tracked under data/
    if [ -d "data" ]; then
      info "▶ DVC add (best-effort) data/*"
      dvc add data/* >/dev/null 2>&1 || true
    fi
  else
    warn "DVC not installed; --no-dvc equivalent assumed."
    DO_DVC=false
  fi
fi

# ---------- stage changes ----------
info "▶ Git add -A"
git add -A

# quick summary for the log
CHANGES="$(git status --porcelain)"
if [ -n "$CHANGES" ]; then
  echo "$CHANGES" | sed 's/^/  /' | tee -a "$LOG_FILE" >/dev/null
else
  warn "Working tree has no changes."
fi

# ---------- commit ----------
if [ -n "$CHANGES" ] || [ "$ALLOW_EMPTY" = true ]; then
  info "▶ Git commit"
  if [ -n "$COMMIT_MSG" ]; then
    git commit -m "$COMMIT_MSG" || { warn "Nothing to commit"; true; }
  else
    git commit --allow-empty -m "chore: empty repair commit ($(ts))"
  fi
else
  warn "Skipping commit (no changes)."
fi

# ---------- create tag (optional) ----------
if [ -n "$TAG_NAME" ]; then
  info "▶ Tag ${TAG_NAME}"
  git tag -a "$TAG_NAME" -m "$COMMIT_MSG" || warn "Tag exists or failed; continuing."
fi

# ---------- push (optional) ----------
if [ "$DO_PUSH" = true ]; then
  info "▶ Git push to origin ${BRANCH}"
  retry 3 2 git push -u origin "${BRANCH}" || { err "git push failed"; exit 1; }

  if [ -n "$TAG_NAME" ]; then
    info "▶ Pushing tag ${TAG_NAME}"
    retry 3 2 git push origin "$TAG_NAME" || warn "Tag push failed"
  fi

  if [ "$DO_DVC" = true ]; then
    info "▶ DVC push"
    retry 3 5 dvc push || warn "DVC push encountered errors"
  fi

  ok "Push completed"
else
  warn "Push disabled (--no-push). Local commit only."
fi

# ---------- manifest ----------
if [ "$WRITE_MANIFEST" = true ]; then
  MANIFEST_DIR="outputs/manifests"; mkdir -p "$MANIFEST_DIR"
  MANIFEST_PATH="$MANIFEST_DIR/repair_manifest_${RUN_ID}.json"
  CFG_HASH=""
  if command -v "$CLI" >/dev/null 2>&1; then
    CFG_HASH="$($CLI hash-config 2>/dev/null || echo "")"
  fi
  {
    printf '{\n'
    printf '  "run_id": "%s",\n'        "$RUN_ID"
    printf '  "ts_utc": "%s",\n'        "$RUN_TS"
    printf '  "git_sha": "%s",\n'       "$GIT_SHA"
    printf '  "branch": "%s",\n'        "$BRANCH"
    printf '  "cfg_hash": "%s",\n'      "$CFG_HASH"
    printf '  "do_dvc": %s,\n'          "$( $DO_DVC && echo true || echo false )"
    printf '  "do_push": %s,\n'         "$( $DO_PUSH && echo true || echo false )"
    printf '  "tag": "%s",\n'           "$TAG_NAME"
    printf '  "commit_msg": %s\n'       "$(printf '%s' "${COMMIT_MSG}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
    printf '}\n'
  } > "$MANIFEST_PATH"
  ok "Manifest: $MANIFEST_PATH"
fi

ok  "[repair_and_push] Completed at $(ts) (RUN_ID=${RUN_ID})"
log "[repair_and_push] ========================================================"
