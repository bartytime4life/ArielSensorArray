#!/usr/bin/env bash
# ==============================================================================
# SpectraMind V50 — DVC Pull Helper (upgraded)
# ------------------------------------------------------------------------------
# Syncs data/artifacts from configured DVC remotes with retries, logging, JSON,
# and guardrails. Safe for local dev, CI, and Kaggle/dockerized environments.
#
# Usage:
#   bin/dvc-pull.sh [options] [--] [TARGET ...]
#
# Options:
#   -r, --remote NAME      Pull from a specific DVC remote (default: auto)
#   -j, --jobs N           Parallel jobs for DVC (default: auto)
#   -R, --rev REV          Git/DVC revision to use (branch/tag/sha)
#   -a, --all              Pull all outputs (dvc pull -a) ignoring targets
#   -f, --force            Force pull (ignore run-cache skip heuristics)
#   -n, --dry-run          Show what would be pulled; do not modify workspace
#   -q, --quiet            Quieter output (pass through to dvc)
#       --no-run-cache     Disable using run-cache when pulling
#       --max-retries N    Retry attempts on transient errors (default: 3)
#       --backoff SEC      Initial backoff between retries (default: 4)
#       --timeout SEC      Timeout per DVC/Git step (default: 900)
#       --status-only      Only show status/listing; do not pull
#       --json             Emit JSON summary to stdout
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
have()  { command -v "$1" >/dev/null 2>&1; }

usage() {
  sed -n '2,120p' "$0" | sed 's/^# \{0,1\}//'
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
TIMEOUT="${DVC_PULL_TIMEOUT:-900}"
STATUS_ONLY=false
JSON=false
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
    --timeout)          TIMEOUT="${2:-}"; shift 2 ;;
    --status-only)      STATUS_ONLY=true; shift ;;
    --json)             JSON=true; shift ;;
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
LOG_FILE="$LOG_DIR/dvc-pull-$TS.log"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

note "Log : $LOG_FILE"
note "Repo: $ROOT"

# ---------- preflight checks ----------
have dvc  || die "dvc not found on PATH. Install DVC (pip install dvc[<remote>])"
if ! have git; then
  warn "git not found on PATH (continuing, but revision checks will be limited)"
fi
[[ -d "$ROOT/.dvc" ]] || die "No .dvc directory found at repo root ($ROOT). Initialize with 'dvc init' or run from project root."

# Safety: inform dirty git tree (non-fatal)
if have git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Git working tree is dirty. Consider committing or stashing before pulling large artifacts."
  fi
fi

# If user passed a rev, attempt checkout (restore later)
ORIG_REF=""
RESTORE_REF=false
if [[ -n "$REV" ]] && have git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
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
note "DVC status (cache vs workspace)…"
if ! dvc status -c ${REMOTE:+-r "$REMOTE"} || ! dvc status; then
  warn "dvc status encountered issues (continuing)."
fi

# ---------- build dvc pull command ----------
DVC_ARGS=()
$ALL           && DVC_ARGS+=("-a")
[[ -n "$JOBS"   ]] && DVC_ARGS+=("-j" "$JOBS")
[[ -n "$REMOTE" ]] && DVC_ARGS+=("-r" "$REMOTE")
$QUIET         && DVC_ARGS+=("-q")
$NO_RUN_CACHE  && DVC_ARGS+=("--no-run-cache")
$FORCE         && DVC_ARGS+=("--force")

# ---------- dry-run / status-only ----------
if $DRY_RUN || $STATUS_ONLY; then
  if $STATUS_ONLY; then
    note "Status-only: showing remote listing or cache differences."
  else
    note "Dry-run: showing listing instead of pulling…"
  fi
  # Prefer status -c vs remote; fallback to remote list
  if ! dvc status -c ${REMOTE:+-r "$REMOTE"}; then
    if [[ -n "$REMOTE" ]]; then
      printf "%s " "dvc list $REMOTE ${REV:+--rev "$REV"} ."; echo
      dvc list "$REMOTE" ${REV:+--rev "$REV"} . || warn "Remote may not support listing."
    else
      warn "No remote selected; cannot list."
    fi
  fi
  if $JSON; then
    printf '{ "ok": true, "mode": "%s", "remote": "%s", "log": "%s" }\n' \
      "$([[ $STATUS_ONLY == true ]] && echo status || echo dry-run)" "${REMOTE:-default}" "$LOG_FILE"
  fi
  exit 0
fi

# ---------- retry + timeout ----------
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
  local t="$1"; shift
  if have timeout; then
    timeout --preserve-status --signal=TERM "$t" "$@"
  else
    "$@"
  fi
}

# ---------- pull ----------
note "Starting dvc pull…"
printf "%s " "dvc pull ${DVC_ARGS[*]} ${TARGETS[*]:-}"; echo

if ! retry "$MAX_RETRIES" "$BACKOFF" run_to "$TIMEOUT" dvc pull "${DVC_ARGS[@]}" "${TARGETS[@]}"; then
  [[ $JSON == true ]] && printf '{ "ok": false, "error": "dvc pull failed", "remote": "%s", "log": "%s" }\n' "${REMOTE:-default}" "$LOG_FILE"
  die "dvc pull failed after $MAX_RETRIES attempt(s). See log: $LOG_FILE"
fi

# ---------- post-checks ----------
note "Verifying workspace integrity (dvc doctor)…"
if ! dvc doctor >/dev/null 2>&1; then
  warn "dvc doctor reported issues (see above for details)."
fi

# Summarize cache usage (non-fatal)
note "Cache GC preview (workspace only, dry-run)…"
if ! dvc gc --workspace --dry-run >/dev/null 2>&1; then
  warn "Unable to compute cache gc preview (continuing)."
fi

info "DVC pull completed successfully."
note "Tip: consider committing updated .dvc.lock / .dvc files if they changed."

# ---------- minimal JSON summary ----------
if $JSON; then
  if have jq; then
    jq -nc \
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
      --arg timeout "$TIMEOUT" \
      --arg log "$LOG_FILE" \
      --arg targets "${TARGETS[*]:-}" \
      '{ok:true, ts:$ts, action:$action,
        params:{remote:$remote, rev:$rev, jobs:$jobs, all:$all, force:$force, no_run_cache:$no_run_cache,
                retries:$retries, backoff_initial:$backoff_init, timeout:$timeout, targets:$targets},
        log:$log}' || true
  else
    printf '{ "ok": true, "remote": "%s", "rev": "%s", "log": "%s" }\n' "${REMOTE:-default}" "${REV:-current}" "$LOG_FILE"
  fi
fi

# ---------- audit line (append to file) ----------
{
  printf '{'
  printf '"ts":"%s",'   "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '"action":"dvc-pull",'
  printf '"remote":"%s",' "${REMOTE:-default}"
  printf '"rev":"%s",'    "${REV:-current}"
  printf '"jobs":"%s",'   "${JOBS:-auto}"
  printf '"targets":"%s"' "${TARGETS[*]:-}"
  printf '}\n'
} >> "$LOG_DIR/dvc-pull.audit.jsonl" || true

exit 0