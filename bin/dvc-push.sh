#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — DVC Push Helper (upgraded)
# ------------------------------------------------------------------------------
# Syncs local DVC-tracked cache/artifacts to remote storage with guardrails,
# retries, logging, JSON summary (optional), and flexible scope controls.
# Safe for local dev & CI (non-interactive, explicit failures).
#
# Usage:
#   bin/dvc-push.sh [options] [--] [TARGET ...]
#
# Options:
#   -r, --remote NAME       Push to a specific DVC remote (default: auto)
#   -j, --jobs N            Parallel jobs for DVC (default: auto)
#   -R, --rev REV           Git/DVC revision to use (default: current HEAD)
#   -a, --all               Push current workspace outputs (alias for no-scope flags)
#       --all-commits       Push cache for all commits       (dvc push -a)
#       --all-branches      Push cache for all branches      (dvc push -A)
#       --all-tags          Push cache for all tags          (dvc push -T)
#   -f, --force             Force push (ignore optimizations)
#   -n, --dry-run           Show what would be pushed; do not modify remote
#   -q, --quiet             Quieter output (pass through to dvc)
#       --max-retries N     Retry attempts on transient errors (default: 3)
#       --backoff SEC       Initial backoff between retries  (default: 4)
#       --timeout SEC       Timeout per DVC/Git step (default: 900)
#       --json              Emit JSON summary to stdout (in addition to logs)
#       --status-only       Only show status vs remote; do not push
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
#   • If --rev is given, we temporarily checkout and restore the original ref
# ==============================================================================

set -Eeuo pipefail

# ---------- tiny stdlib / colors ----------
is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"
  GRN="$(tput setaf 2)"; YLW="$(tput setaf 3)"; CYN="$(tput setaf 6)"; RST="$(tput sgr0)"
else
  BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; CYN=""; RST=""
fi

die()   { printf "%s %s\n" "${RED}✖${RST}" "$*" >&2; exit 1; }
info()  { printf "%s %s\n" "${GRN}✔${RST}" "$*"; }
warn()  { printf "%s %s\n" "${YLW}∙${RST}" "$*"; }
note()  { printf "%s %s\n" "${DIM}·${RST}" "$*"; }

usage() {
  sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

have() { command -v "$1" >/dev/null 2>&1; }

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
TIMEOUT="${DVC_PUSH_TIMEOUT:-900}"
JSON=false
STATUS_ONLY=false
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
    --timeout)          TIMEOUT="${2:-}"; shift 2 ;;
    --json)             JSON=true; shift ;;
    --status-only)      STATUS_ONLY=true; shift ;;
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

note "Log : $LOG_FILE"
note "Repo: $ROOT"

# ---------- preflight checks ----------
have dvc  || die "dvc not found on PATH. Install DVC (pip install dvc[<remote>])"
have git  || warn "git not found on PATH (continuing, but revision checks will be limited)"
[[ -d "$ROOT/.dvc" ]] || die "No .dvc directory found at repo root ($ROOT). Initialize with 'dvc init' or run from project root."

# Inform dirty git tree (non-fatal)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Git working tree is dirty. Consider committing versions of updated .dvc/.lock files."
  fi
fi

# If user passed a rev, attempt checkout (restore later)
ORIG_REF=""
RESTORE_REF=false
if [[ -n "$REV" ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  ORIG_REF="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD 2>/dev/null || true)"
  note "Checking out revision: ${BOLD}${REV}${RST}"
  git fetch --all --tags --prune >/dev/null 2>&1 || warn "git fetch failed or remote missing (continuing)"
  git checkout --quiet "$REV" || die "Failed to checkout rev '$REV'"
  RESTORE_REF=true
fi

restore_ref() {
  if $RESTORE_REF && [[ -n "$ORIG_REF" ]]; then
    note "Restoring to original ref: ${BOLD}${ORIG_REF}${RST}"
    git checkout --quiet "$ORIG_REF" || warn "Failed to restore original ref ($ORIG_REF)."
  fi
}
trap restore_ref EXIT

# Infer remote if not provided (prefer first configured remote)
if [[ -z "$REMOTE" ]]; then
  set +e
  REMOTE="$(dvc remote list 2>/dev/null | awk '{print $1}' | head -n1)"
  set -e
  [[ -n "$REMOTE" ]] && note "Auto-selected DVC remote: ${BOLD}${REMOTE}${RST}" || warn "No DVC remote configured. Will rely on default."
fi

# ---------- status summary ----------
status_cmd=(dvc status)
[[ -n "$REMOTE" ]] && status_cmd+=(-r "$REMOTE")
$QUIET && status_cmd+=(-q)

note "DVC status (cache vs remote)…"
if ! "${status_cmd[@]}" || ! dvc status; then
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

# ---------- dry-run / status-only ----------
if $DRY_RUN || $STATUS_ONLY; then
  note "$([[ $STATUS_ONLY == true ]] && echo "Status-only" || echo "Dry-run"): computing candidates (no remote writes)."
  # dvc status -c shows missing cache in remote for current workspace
  if ! dvc status -c ${REMOTE:+-r "$REMOTE"}; then
    warn "Unable to compute status against remote (it may be empty)."
  fi
  printf "%s " "dvc push ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo
  if $JSON; then
    printf '{ "ok": true, "mode": "%s", "remote": "%s", "log": "%s" }\n' \
      "$([[ $STATUS_ONLY == true ]] && echo status || echo dry-run)" "${REMOTE:-default}" "$LOG_FILE"
  fi
  exit 0
fi

# ---------- retry + timeout wrapper ----------
retry() {
  local tries="$1"; shift
  local backoff="$1"; shift
  local attempt=1
  until "$@"; do
    local exit_code=$?
    if (( attempt >= tries )); then
      return "$exit_code"
    fi
    warn "Command failed (attempt $attempt/$tries). Retrying in ${backoff}s…"
    sleep "$backoff"
    backoff=$(( backoff * 2 ))
    attempt=$(( attempt + 1 ))
  done
}

run_to() {
  # run_to <timeout_sec> <cmd...>
  local t="$1"; shift
  if have timeout; then
    timeout --preserve-status --signal=TERM "$t" "$@"
  else
    "$@"
  fi
}

# ---------- push ----------
note "Starting dvc push…"
printf "%s " "dvc push ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo

if ! retry "$MAX_RETRIES" "$BACKOFF" run_to "$TIMEOUT" dvc push "${DVC_ARGS[@]}" "${TARGETS[@]}"; then
  [[ $JSON == true ]] && printf '{ "ok": false, "error": "dvc push failed", "remote": "%s", "log": "%s" }\n' "${REMOTE:-default}" "$LOG_FILE"
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

# ---------- minimal JSON summary ----------
if $JSON; then
  # Try jq for richer details; fallback to minimal JSON
  if have jq; then
    jq -nc \
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
      --arg timeout "$TIMEOUT" \
      --arg log "$LOG_FILE" \
      --arg targets "${TARGETS[*]:-}" \
      '{ok:true, ts:$ts, action:$action,
        params:{remote:$remote, rev:$rev, jobs:$jobs, timeout:$timeout,
                all_commits:$all_commits, all_branches:$all_branches, all_tags:$all_tags,
                force:$force, retries:$retries, backoff_initial:$backoff_init, targets:$targets},
        log:$log}' || true
  else
    printf '{ "ok": true, "remote": "%s", "rev": "%s", "log": "%s" }\n' "${REMOTE:-default}" "${REV:-current}" "$LOG_FILE"
  fi
fi

# ---------- audit line (append to file) ----------
{
  printf '{'
  printf '"ts":"%s",'   "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '"action":"dvc-push",'
  printf '"remote":"%s",' "${REMOTE:-default}"
  printf '"rev":"%s",'    "${REV:-current}"
  printf '"jobs":"%s",'   "${JOBS:-auto}"
  printf '"targets":"%s"' "${TARGETS[*]:-}"
  printf '}\n'
} >> "$LOG_DIR/dvc-push.audit.jsonl" || true

exit 0