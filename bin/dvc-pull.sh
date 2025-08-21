#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — DVC Pull Helper
# Syncs data/artifacts from configured DVC remotes with retries, logging, and
# guardrails. Safe for local dev, CI, and Kaggle/dockerized environments.
#
# Usage:
#   bin/dvc-pull.sh [options] [--] [TARGET ...]
#
# Options:
#   -r, --remote NAME      Pull from a specific DVC remote (default: auto)
#   -j, --jobs N           Parallel jobs for DVC (default: auto)
#   -R, --rev REV          Pull a specific Git/DVC rev (branch/tag/sha)
#   -a, --all              Pull all outputs (dvc pull -a) ignoring targets
#   -f, --force            Force pull (ignore run-cache skip heuristics)
#   -n, --dry-run          Show what would be pulled; do not modify the workspace
#   -q, --quiet            Quieter output (pass through to dvc)
#   --no-run-cache         Disable using run-cache when pulling
#   --max-retries N        Retry attempts on transient errors (default: 3)
#   --backoff SEC          Initial backoff between retries (default: 4)
#   -h, --help             Show this help
#
# Examples:
#   bin/dvc-pull.sh
#   bin/dvc-pull.sh -r origin-aws -j 8
#   bin/dvc-pull.sh -a --max-retries 5 --backoff 6
#   bin/dvc-pull.sh -- data/raw/ data/processed/
#
# Notes:
#   • Writes logs under logs/ops/dvc-pull-YYYYmmdd_HHMMSS.log
#   • Exits non-zero on failure; safe in CI pipelines
#   • Auto-detects repo root and .dvc presence; prints actionable errors
# ==============================================================================

set -Eeuo pipefail

# ---------- tiny stdlib ----------
bold()   { printf "\033[1m%s\033[0m" "$*"; }
dim()    { printf "\033[2m%s\033[0m" "$*"; }
green()  { printf "\033[32m%s\033[0m" "$*"; }
yellow() { printf "\033[33m%s\033[0m" "$*"; }
red()    { printf "\033[31m%s\033[0m" "$*"; }

die()    { printf "%s %s\n" "$(red "✖")" "$*" >&2; exit 1; }
info()   { printf "%s %s\n" "$(green "✔")" "$*"; }
warn()   { printf "%s %s\n" "$(yellow "∙")" "$*"; }
note()   { printf "%s %s\n" "$(dim "·")" "$*"; }

usage() {
  sed -n '2,60p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------- defaults ----------
REMOTE=""
JOBS=""
REV=""
ALL=false
FORCE=false
DRY_RUN=false
QUIET=false
NO_RUN_CACHE=false
MAX_RETRIES=3
BACKOFF=4
TARGETS=()

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--remote)        REMOTE="${2:-}"; shift 2 ;;
    -j|--jobs)          JOBS="${2:-}"; shift 2 ;;
    -R|--rev)           REV="${2:-}"; shift 2 ;;
    -a|--all)           ALL=true; shift ;;
    -f|--force)         FORCE=true; shift ;;
    -n|--dry-run)       DRY_RUN=true; shift ;;
    -q|--quiet)         QUIET=true; shift ;;
    --no-run-cache)     NO_RUN_CACHE=true; shift ;;
    --max-retries)      MAX_RETRIES="${2:-}"; shift 2 ;;
    --backoff)          BACKOFF="${2:-}"; shift 2 ;;
    -h|--help)          usage ;;
    --)                 shift; TARGETS+=("$@"); break ;;
    -*)
      die "Unknown option: $1 (use -h for help)"
      ;;
    *)
      TARGETS+=("$1"); shift ;;
  esac
done

# ---------- repo root & logging ----------
# Find repo root (directory containing .git or pyproject/README fallback)
find_repo_root() {
  local d="$PWD"
  while [[ "$d" != "/" ]]; do
    [[ -d "$d/.git" || -f "$d/pyproject.toml" || -f "$d/README.md" ]] && { echo "$d"; return; }
    d="$(dirname "$d")"
  done
  echo "$PWD"
}

ROOT="$(find_repo_root)"
cd "$ROOT"

LOG_DIR="$ROOT/logs/ops"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/dvc-pull-$TS.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

note "Log: $LOG_FILE"
note "Repo: $ROOT"

# ---------- preflight checks ----------
command -v dvc >/dev/null 2>&1 || die "dvc not found on PATH. Install DVC (pip install dvc[<remote>])"
command -v git >/dev/null 2>&1 || warn "git not found on PATH (continuing, but revision checks will be limited)"

[[ -d "$ROOT/.dvc" ]] || die "No .dvc directory found at repo root ($ROOT). Initialize with 'dvc init' or run from project root."

# Safety: inform dirty git tree (non-fatal)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Git working tree is dirty. Consider committing or stashing before pulling large artifacts."
  fi
fi

# If user passed a rev, attempt checkout (non-destructive unless detached)
if [[ -n "$REV" ]]; then
  note "Checking out revision: $(bold "$REV")"
  git fetch --all --tags --prune >/dev/null 2>&1 || warn "git fetch failed or remote missing (continuing)"
  git checkout --quiet "$REV" || die "Failed to checkout rev '$REV'"
fi

# Infer remote if not provided (prefer first configured remote w/ url)
if [[ -z "$REMOTE" ]]; then
  set +e
  REMOTE="$(dvc remote list | awk '{print $1}' | head -n1)"
  set -e
  [[ -n "$REMOTE" ]] && note "Auto-selected DVC remote: $(bold "$REMOTE")" || warn "No DVC remote configured. Will rely on default."
fi

# ---------- show status summary (non-fatal) ----------
note "DVC status (cache vs workspace)…"
if ! dvc status -c ${REMOTE:+-r "$REMOTE"} || ! dvc status; then
  warn "dvc status encountered issues (continuing)."
fi

# ---------- build dvc pull command ----------
DVC_ARGS=()
$ALL        && DVC_ARGS+=("-a")
[[ -n "$JOBS" ]]  && DVC_ARGS+=("-j" "$JOBS")
[[ -n "$REMOTE" ]]&& DVC_ARGS+=("-r" "$REMOTE")
$QUIET      && DVC_ARGS+=("-q")
$NO_RUN_CACHE && DVC_ARGS+=("--no-run-cache")
$FORCE      && DVC_ARGS+=("--force")
if $DRY_RUN; then
  DVC_SUBCMD=(dvc list "${REMOTE:-.}" ${REV:+--rev "$REV"} .)
  note "Dry-run: showing listing instead of pulling…"
  printf "%s " "${DVC_SUBCMD[@]}"; echo
  "${DVC_SUBCMD[@]}" || warn "Dry-run listing failed (remote may not support listing)."
  info "Dry-run complete."
  exit 0
fi

# ---------- retry wrapper ----------
retry() {
  local tries="$1"; shift
  local backoff="$1"; shift
  local attempt=1
  until "$@"; do
    exit_code=$?
    if (( attempt >= tries )); then
      return "$exit_code"
    fi
    warn "Command failed (attempt $attempt/$tries). Retrying in ${backoff}s…"
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    attempt=$(( attempt + 1 ))
  done
}

# ---------- pull ----------
note "Starting dvc pull…"
printf "%s " "dvc pull ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo

if ! retry "$MAX_RETRIES" "$BACKOFF" dvc pull "${DVC_ARGS[@]}" "${TARGETS[@]}" ; then
  die "dvc pull failed after $MAX_RETRIES attempt(s). See log: $LOG_FILE"
fi

# ---------- post-checks ----------
note "Verifying workspace integrity (dvc doctor)…"
if ! dvc doctor >/dev/null 2>&1; then
  warn "dvc doctor reported issues (see above for details)."
fi

# Summarize cache usage (non-fatal)
note "Cache summary:"
if ! dvc gc --workspace --dry-run >/dev/null 2>&1; then
  warn "Unable to compute cache gc preview (continuing)."
fi

info "DVC pull completed successfully."
note "Tip: consider committing updated .dvc.lock / .dvc files if they changed."

# ---------- minimal JSON audit line (for external log consumers) ----------
AUDIT_LINE=$(jq -nc \
  --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg action "dvc-pull" \
  --arg remote "${REMOTE:-default}" \
  --arg rev "${REV:-current}" \
  --arg jobs "${JOBS:-auto}" \
  --arg all "$ALL" \
  --arg force "$FORCE" \
  --arg no_run_cache "$NO_RUN_CACHE" \
  --arg retries "$MAX_RETRIES" \
  --arg backoff_init "$BACKOFF" \
  --arg targets "${TARGETS[*]:-}" \
  '{ts:$ts, action:$action, params:{remote:$remote, rev:$rev, jobs:$jobs, all:$all, force:$force, no_run_cache:$no_run_cache, retries:$retries, backoff_initial:$backoff_init, targets:$targets}}' \
  2>/dev/null || echo "{}")
echo "$AUDIT_LINE" >> "$LOG_DIR/dvc-pull.audit.jsonl" || true