#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — DVC Push Helper
# Syncs local DVC-tracked cache/artifacts to remote storage with guardrails,
# retries, logging, and optional scope controls. Safe for local dev & CI.
#
# Usage:
#   bin/dvc-push.sh [options] [--] [TARGET ...]
#
# Options:
#   -r, --remote NAME       Push to a specific DVC remote (default: auto)
#   -j, --jobs N            Parallel jobs for DVC (default: auto)
#   -R, --rev REV           Git/DVC revision to use (default: current HEAD)
#   -a, --all               Push all outputs referenced by current workspace
#   --all-commits           Push cache for all commits (dvc push -a)
#   --all-branches          Push cache for all branches (dvc push -A)
#   --all-tags              Push cache for all tags (dvc push -T)
#   -f, --force             Force push (ignore optimizations)
#   -n, --dry-run           Show what would be pushed; do not modify remote
#   -q, --quiet             Quieter output (pass through to dvc)
#   --max-retries N         Retry attempts on transient errors (default: 3)
#   --backoff SEC           Initial backoff between retries (default: 4)
#   -h, --help              Show this help
#
# Examples:
#   bin/dvc-push.sh
#   bin/dvc-push.sh -r s3-prod -j 8
#   bin/dvc-push.sh --all-commits --all-tags
#   bin/dvc-push.sh -- data/raw/ data/processed/
#
# Notes:
#   • Writes logs under logs/ops/dvc-push-YYYYmmdd_HHMMSS.log
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
  sed -n '2,70p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

# ---------- defaults ----------
REMOTE=""
JOBS=""
REV=""
ALL=false               # alias for pushing current workspace targets
ALL_COMMITS=false       # dvc push -a
ALL_BRANCHES=false      # dvc push -A
ALL_TAGS=false          # dvc push -T
FORCE=false
DRY_RUN=false
QUIET=false
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
    --all-commits)      ALL_COMMITS=true; shift ;;
    --all-branches)     ALL_BRANCHES=true; shift ;;
    --all-tags)         ALL_TAGS=true; shift ;;
    -f|--force)         FORCE=true; shift ;;
    -n|--dry-run)       DRY_RUN=true; shift ;;
    -q|--quiet)         QUIET=true; shift ;;
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
LOG_FILE="$LOG_DIR/dvc-push-$TS.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

note "Log: $LOG_FILE"
note "Repo: $ROOT"

# ---------- preflight checks ----------
command -v dvc >/dev/null 2>&1 || die "dvc not found on PATH. Install DVC (pip install dvc[<remote>])"
command -v git >/dev/null 2>&1 || warn "git not found on PATH (continuing, but revision checks will be limited)"

[[ -d "$ROOT/.dvc" ]] || die "No .dvc directory found at repo root ($ROOT). Initialize with 'dvc init' or run from project root."

# Inform dirty git tree (non-fatal)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Git working tree is dirty. Consider committing versions of updated .dvc/.lock files."
  fi
fi

# If user passed a rev, attempt checkout (non-destructive unless detached)
if [[ -n "$REV" ]]; then
  note "Checking out revision: $(bold "$REV")"
  git fetch --all --tags --prune >/dev/null 2>&1 || warn "git fetch failed or remote missing (continuing)"
  git checkout --quiet "$REV" || die "Failed to checkout rev '$REV'"
fi

# Infer remote if not provided (prefer first configured remote)
if [[ -z "$REMOTE" ]]; then
  set +e
  REMOTE="$(dvc remote list | awk '{print $1}' | head -n1)"
  set -e
  [[ -n "$REMOTE" ]] && note "Auto-selected DVC remote: $(bold "$REMOTE")" || warn "No DVC remote configured. Will rely on default."
fi

# ---------- show status summary (non-fatal) ----------
note "DVC status (cache vs remote)…"
if ! dvc status -r "${REMOTE:-default}" || ! dvc status; then
  warn "dvc status encountered issues (continuing)."
fi

# ---------- build dvc push command ----------
DVC_ARGS=()
$ALL_COMMITS  && DVC_ARGS+=("-a")
$ALL_BRANCHES && DVC_ARGS+=("-A")
$ALL_TAGS     && DVC_ARGS+=("-T")
[[ -n "$JOBS"   ]] && DVC_ARGS+=("-j" "$JOBS")
[[ -n "$REMOTE" ]] && DVC_ARGS+=("-r" "$REMOTE")
$QUIET         && DVC_ARGS+=("-q")
$FORCE         && DVC_ARGS+=("--force")

# Sanity: if no explicit scope flags and no targets, default to current workspace
if ! $ALL_COMMITS && ! $ALL_BRANCHES && ! $ALL_TAGS && [[ ${#TARGETS[@]} -eq 0 ]]; then
  if $ALL; then
    note "Using current workspace scope (equivalent to dvc push for visible outputs)."
  else
    note "No scope flags provided; pushing current workspace outputs."
  fi
fi

# Dry-run: summarize what would happen
if $DRY_RUN; then
  note "Dry-run: computing candidates (this does not contact remote)."
  # dvc status -c shows missing cache in remote for current workspace
  if ! dvc status -c ${REMOTE:+-r "$REMOTE"}; then
    warn "Unable to compute status against remote (it may be empty)."
  fi
  printf "%s " "dvc push ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo
  info "Dry-run complete (no data pushed)."
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

# ---------- push ----------
note "Starting dvc push…"
printf "%s " "dvc push ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo

if ! retry "$MAX_RETRIES" "$BACKOFF" dvc push "${DVC_ARGS[@]}" "${TARGETS[@]}"; then
  die "dvc push failed after $MAX_RETRIES attempt(s). See log: $LOG_FILE"
fi

# ---------- post-checks ----------
note "Verifying remote state (dvc status -c)…"
if ! dvc status -c ${REMOTE:+-r "$REMOTE"}; then
  warn "Remote status check reported differences (some objects may still be missing)."
fi

# Summarize cache usage (non-fatal preview)
note "Cache GC preview (workspace only, dry-run)…"
if ! dvc gc --workspace --dry-run >/dev/null 2>&1; then
  warn "Unable to compute cache gc preview (continuing)."
fi

info "DVC push completed successfully."
note "Tip: commit updated .dvc.lock / .dvc files if they changed."

# ---------- minimal JSON audit line (for external log consumers) ----------
AUDIT_LINE=$(jq -nc \
  --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --arg action "dvc-push" \
  --arg remote "${REMOTE:-default}" \
  --arg rev "${REV:-current}" \
  --arg jobs "${JOBS:-auto}" \
  --arg all_commits "$ALL_COMMITS" \
  --arg all_branches "$ALL_BRANCHES" \
  --arg all_tags "$ALL_TAGS" \
  --arg force "$FORCE" \
  --arg retries "$MAX_RETRIES" \
  --arg backoff_init "$BACKOFF" \
  --arg targets "${TARGETS[*]:-}" \
  '{ts:$ts, action:$action, params:{remote:$remote, rev:$rev, jobs:$jobs, all_commits:$all_commits, all_branches:$all_branches, all_tags:$all_tags, force:$force, retries:$retries, backoff_initial:$backoff_init, targets:$targets}}' \
  2>/dev/null || echo "{}")
echo "$AUDIT_LINE" >> "$LOG_DIR/dvc-push.audit.jsonl" || true