#!/usr/bin/env bash

==============================================================================

🛰️ SpectraMind V50 — repair_and_push.sh (Upgraded • mission-grade • ultimate)

——————————————————————————

Purpose:

Safely repair, commit, and push repository state with Git + DVC consistency.

- Verifies repo, remote, branch, and optional “main/master” guard

- Optional quick selftests and pre-commit hooks

- Adds & commits code; DVC add/status/push for data artifacts (opt-out)

- Pushes Git and DVC with retries + timeouts

- Writes a JSON run manifest + appends log lines to logs/v50_debug_log.md



TL;DR:

./bin/repair_and_push.sh –msg “Fix data lineage + regen docs” –manifest



Flags:

–msg “”           Commit message (required unless –allow-empty)

–allow-empty            Allow an empty commit if nothing changed

–allow-non-main         Allow pushing from non-main branch (default: guarded)

–remote           Git remote name (default: first configured remote)

–branch           Branch to push (default: current branch)

–run-tests              Run spectramind test --fast before commit (if available)

–run-pre-commit         Run pre-commit run --all-files (if configured)

–poetry-install         Run poetry install --no-root (if poetry present)

–security-scan          Run pip-audit if installed (fail on findings)

–no-dvc                 Skip DVC status/add/push

–no-push                Skip Git/DVC push (local-only repair)

–tag “<vX.Y.Z>”         Create tag on the commit and push it

–gpg-sign               Sign the tag (-s) instead of annotate (-a)

–manifest               Write JSON manifest under outputs/manifests/

–open-manifest          Open manifest or its folder when done (best-effort)

–json                   Also echo compact JSON summary to stdout

–dry-run                Explain what would happen; no side-effects

–timeout           Timeout per step (default: 240)

–retries             Retries for network steps (git push, dvc push). Default: 3

–sleep             Sleep between retries (default: 5)

–no-fetch               Do not fetch/compare remote before pushing

–help                   Show this help



Env Overrides:

SPECTRAMIND_CLI           (default: ‘spectramind’)

LOG_FILE                  (default: logs/v50_debug_log.md)

REPAIR_ALLOW_MAIN=1       (override guard; equivalent to –allow-non-main when on main)

REPAIR_TIMEOUT=      (equivalent to –timeout)

REPAIR_RETRIES=        (equivalent to –retries)

REPAIR_SLEEP=        (equivalent to –sleep)

==============================================================================

set -Eeuo pipefail
IFS=$’\n\t’

––––– Colors –––––

if [[ -t 1 ]]; then
BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; YLW=$’\033[33m’; CYN=$’\033[36m’; RST=$’\033[0m’
else
BOLD=’’; DIM=’’; RED=’’; GRN=’’; YLW=’’; CYN=’’; RST=’’
fi

––––– Defaults –––––

CLI=”${SPECTRAMIND_CLI:-spectramind}”
LOG_FILE=”${LOG_FILE:-logs/v50_debug_log.md}”

COMMIT_MSG=””
ALLOW_EMPTY=false
ALLOW_NON_MAIN=false
REMOTE_OVERRIDE=””
BRANCH_OVERRIDE=””

RUN_TESTS=false
RUN_PRE_COMMIT=false
POETRY_INSTALL=false
SECURITY_SCAN=false

DO_DVC=true
DO_PUSH=true
TAG_NAME=””
GPG_SIGN=false
WRITE_MANIFEST=false
OPEN_MANIFEST=false
EMIT_JSON=false
DRY_RUN=false

STEP_TIMEOUT=”${REPAIR_TIMEOUT:-240}”
RETRIES=”${REPAIR_RETRIES:-3}”
SLEEP_BETWEEN=”${REPAIR_SLEEP:-5}”
NO_FETCH=false

––––– Helpers –––––

usage() {
sed -n ‘1,/^# ==============================================================================/{p}’ “$0” | sed ‘s/^# {0,1}//’
exit 0
}

ts()        { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha()    { git rev-parse –short HEAD 2>/dev/null || echo “nogit”; }
hostname_s(){ hostname 2>/dev/null || uname -n 2>/dev/null || echo “unknown-host”; }

open_path() {
local p=”$1”
if command -v xdg-open >/dev/null 2>&1; then xdg-open “$p” >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then open “$p” >/dev/null 2>&1 || true
fi
}

have() { command -v “$1” >/dev/null 2>&1; }

Logging (tee only metadata lines; command outputs are printed directly)

mkdir -p “$(dirname “$LOG_FILE”)”
log_line() { printf “%b\n” “$” | tee -a “$LOG_FILE” >/dev/null; }
info()     { printf “%b\n” “${CYN}${}${RST}”; }
ok()       { printf “%b\n” “${GRN}${}${RST}”; }
warn()     { printf “%b\n” “${YLW}${}${RST}”; }
err()      { printf “%b\n” “${RED}${*}${RST}” >&2; }

die() { err “$*”; exit 1; }

run() {

run <cmd…> respecting DRY_RUN

if $DRY_RUN; then
printf “%s\n” “${DIM}[dry-run]${RST} $*”
return 0
fi
“$@”
}

run_q() {

run quietly when not in dry-run

if $DRY_RUN; then
printf “%s\n” “${DIM}[dry-run]${RST} $*”
return 0
fi
“$@” >/dev/null 2>&1
}

run_to() {

run_to  <cmd…> — honors timeout + DRY_RUN

local label=”$1”; shift
local start end rc
info “▶ ${label}”
if $DRY_RUN; then
printf “%s\n” “${DIM}[dry-run]${RST} (timeout=${STEP_TIMEOUT}s) $*”
return 0
fi
set +e
if have timeout; then
timeout –preserve-status –signal=TERM “$STEP_TIMEOUT” “$@”; rc=$?
else
“$@”; rc=$?
fi
set -e
start=”$(date +%s)”; end=”$(date +%s)”
if [[ “${rc:-0}” -ne 0 ]]; then
err “Step failed: ${label} (rc=${rc})”
return “$rc”
fi
ok “Done: ${label}”
return 0
}

retry() {

retry   – <cmd…>

local attempts=”$1” sleep_s=”$2”; shift 2
local n=1
local rc=0
while true; do
if $DRY_RUN; then
printf “%s\n” “${DIM}[dry-run]${RST} $*”
return 0
fi
“$@”; rc=$? || rc=$?
[[ $rc -eq 0 ]] && return 0
if (( n >= attempts )); then
return “$rc”
fi
warn “Attempt ${n}/${attempts} failed (rc=${rc}); sleeping ${sleep_s}s…”
sleep “${sleep_s}”
n=$((n+1))
done
}

json_escape() { python3 - <<‘PY’ “$1”; import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else “”)); PY
}

––––– Arg parsing –––––

if have getopt; then
PARSED=$(getopt -o h –long help,msg:,allow-empty,allow-non-main,remote:,branch:,run-tests,run-pre-commit,poetry-install,security-scan,no-dvc,no-push,tag:,gpg-sign,manifest,open-manifest,json,dry-run,timeout:,retries:,sleep:,no-fetch – “$@”) || usage
eval set – “$PARSED”
while true; do
case “$1” in
-h|–help) usage ;;
–msg) COMMIT_MSG=”${2:-}”; shift 2 ;;
–allow-empty) ALLOW_EMPTY=true; shift ;;
–allow-non-main) ALLOW_NON_MAIN=true; shift ;;
–remote) REMOTE_OVERRIDE=”${2:-}”; shift 2 ;;
–branch) BRANCH_OVERRIDE=”${2:-}”; shift 2 ;;
–run-tests) RUN_TESTS=true; shift ;;
–run-pre-commit) RUN_PRE_COMMIT=true; shift ;;
–poetry-install) POETRY_INSTALL=true; shift ;;
–security-scan) SECURITY_SCAN=true; shift ;;
–no-dvc) DO_DVC=false; shift ;;
–no-push) DO_PUSH=false; shift ;;
–tag) TAG_NAME=”${2:-}”; shift 2 ;;
–gpg-sign) GPG_SIGN=true; shift ;;
–manifest) WRITE_MANIFEST=true; shift ;;
–open-manifest) OPEN_MANIFEST=true; shift ;;
–json) EMIT_JSON=true; shift ;;
–dry-run) DRY_RUN=true; shift ;;
–timeout) STEP_TIMEOUT=”${2:-240}”; shift 2 ;;
–retries) RETRIES=”${2:-3}”; shift 2 ;;
–sleep) SLEEP_BETWEEN=”${2:-5}”; shift 2 ;;
–no-fetch) NO_FETCH=true; shift ;;
–) shift; break ;;
*) die “Unknown option: $1” ;;
esac
done
else

Minimal parser

while [[ $# -gt 0 ]]; do
case “$1” in
-h|–help) usage ;;
–msg) COMMIT_MSG=”${2:-}”; shift ;;
–allow-empty) ALLOW_EMPTY=true ;;
–allow-non-main) ALLOW_NON_MAIN=true ;;
–remote) REMOTE_OVERRIDE=”${2:-}”; shift ;;
–branch) BRANCH_OVERRIDE=”${2:-}”; shift ;;
–run-tests) RUN_TESTS=true ;;
–run-pre-commit) RUN_PRE_COMMIT=true ;;
–poetry-install) POETRY_INSTALL=true ;;
–security-scan) SECURITY_SCAN=true ;;
–no-dvc) DO_DVC=false ;;
–no-push) DO_PUSH=false ;;
–tag) TAG_NAME=”${2:-}”; shift ;;
–gpg-sign) GPG_SIGN=true ;;
–manifest) WRITE_MANIFEST=true ;;
–open-manifest) OPEN_MANIFEST=true ;;
–json) EMIT_JSON=true ;;
–dry-run) DRY_RUN=true ;;
–timeout) STEP_TIMEOUT=”${2:-240}”; shift ;;
–retries) RETRIES=”${2:-3}”; shift ;;
–sleep) SLEEP_BETWEEN=”${2:-5}”; shift ;;
–no-fetch) NO_FETCH=true ;;
*) die “Unknown option: $1” ;;
esac
shift
done
fi

––––– Start banner –––––

RUN_TS=”$(ts)”
RUN_HOST=”$(hostname_s)”
GIT_SHA=”$(gitsha)”
trap ‘err “[repair_and_push] ❌ Failed at $(ts) (RUN_ID=${RUN_ID})”; exit 1’ ERR

––––– Preflight checks –––––

have git || die “git not found in PATH”
git rev-parse –is-inside-work-tree >/dev/null 2>&1 || die “Not a git repository”

GIT_ROOT=”$(git rev-parse –show-toplevel)”
cd “$GIT_ROOT”

CURRENT_BRANCH=”$(git rev-parse –abbrev-ref HEAD)”
BRANCH=”${BRANCH_OVERRIDE:-$CURRENT_BRANCH}”

if [[ -n “$REMOTE_OVERRIDE” ]]; then
REMOTE=”$REMOTE_OVERRIDE”
else
REMOTE=”$(git remote 2>/dev/null | head -n1 || true)”
fi
[[ -n “$REMOTE” ]] || die “No git remote configured (add one or pass –remote )”

RUN_ID=”${RUN_TS}-${GIT_SHA}”
log_line “[repair_and_push] ========================================================”
log_line “[repair_and_push] start=${RUN_TS} host=${RUN_HOST} repo=${GIT_ROOT}”
log_line “[repair_and_push] run_id=${RUN_ID} branch=${BRANCH} remote=${REMOTE} tag=${TAG_NAME:-”-”} dry_run=${DRY_RUN}”

Guard: refuse to push non-main unless allowed? (inverse of typical)

Your original flag means “allow pushing from non-main”. Default is disallow on non-main.

if ! $ALLOW_NON_MAIN; then
if [[ “$BRANCH” != “main” && “$BRANCH” != “master” && “${REPAIR_ALLOW_MAIN:-0}” -ne 1 ]]; then
die “Refusing to push from non-main branch (’$BRANCH’). Use –allow-non-main to override.”
fi
fi

Ensure message or allow-empty

if [[ -z “$COMMIT_MSG” && “$ALLOW_EMPTY” != true ]]; then
die “Commit message required (–msg "…") or pass –allow-empty”
fi

Poetry/pre-commit/DVC presence

HAS_POETRY=0; have poetry && HAS_POETRY=1
HAS_PRECOMMIT=0; have pre-commit && [[ -f “.pre-commit-config.yaml” ]] && HAS_PRECOMMIT=1
HAS_DVC=0; have dvc && [[ -d “.dvc” ]] && HAS_DVC=1
HAS_CLI=0; { [[ -f “spectramind.py” ]] || have “$CLI”; } && HAS_CLI=1

––––– Optional: fetch + divergence info –––––

if ! $NO_FETCH; then
info “Fetching ${REMOTE}/${BRANCH} (state check)”
run git fetch “$REMOTE” “$BRANCH” || true
if git rev-parse –verify “refs/remotes/$REMOTE/$BRANCH” >/dev/null 2>&1; then
LOCAL=”$(git rev-parse “$BRANCH”)”
REMOTE_SHA=”$(git rev-parse “refs/remotes/$REMOTE/$BRANCH”)”
BASE=”$(git merge-base “$BRANCH” “refs/remotes/$REMOTE/$BRANCH”)”
if [[ “$LOCAL” = “$REMOTE_SHA” ]]; then
info “Branch up-to-date with ${REMOTE}/${BRANCH}”
elif [[ “$LOCAL” = “$BASE” ]]; then
warn “Local branch is behind ${REMOTE}/${BRANCH} (consider pull/rebase).”
elif [[ “$REMOTE_SHA” = “$BASE” ]]; then
info “Local branch is ahead of remote (push will fast-forward).”
else
warn “Local/remote have diverged (consider rebase/merge).”
fi
fi
else
warn “Skipping remote fetch (–no-fetch)”
fi

––––– Optional: environment prep –––––

if $POETRY_INSTALL && [[ $HAS_POETRY -eq 1 ]]; then
run_to “Poetry install” poetry install –no-root
fi

if $SECURITY_SCAN; then
if have pip-audit; then
run_to “pip-audit security scan” pip-audit
else
warn “pip-audit not installed; skipping security scan.”
fi
fi

––––– Optional: tests / pre-commit –––––

if $RUN_TESTS; then
if [[ $HAS_CLI -eq 1 ]]; then
if have “$CLI”; then
run_to “spectramind test –fast” “$CLI” test –fast
else
# fallback via poetry if CLI module only
if [[ $HAS_POETRY -eq 1 ]]; then
run_to “poetry run spectramind test –fast” poetry run “$CLI” test –fast
else
warn “SpectraMind CLI not invocable (no entrypoint). Skipping tests.”
fi
fi
else
warn “SpectraMind CLI not detected; skipping tests.”
fi
fi

if $RUN_PRE_COMMIT; then
if [[ $HAS_PRECOMMIT -eq 1 ]]; then
run_to “pre-commit run –all-files” pre-commit run –all-files
else
warn “pre-commit not configured; skipping.”
fi
fi

––––– DVC pre-commit (status + add) –––––

if $DO_DVC; then
if [[ $HAS_DVC -eq 1 ]]; then
info “DVC status (informational)”
run dvc status || true
# Best-effort auto-add new content under conventional data/ or dvc-tracked paths
if [[ -d “data” ]]; then
info “DVC add (data/) — best-effort”
run dvc add data/ >/dev/null 2>&1 || true
fi
else
warn “DVC not detected; continuing with –no-dvc behavior.”
DO_DVC=false
fi
fi

––––– Stage & commit –––––

info “Staging: git add -A”
run git add -A

CHANGES=”$(git status –porcelain || true)”
if [[ -n “$CHANGES” ]]; then
printf “%s\n” “$CHANGES” | sed ‘s/^/  /’
else
warn “Working tree has no changes.”
fi

if [[ -n “$CHANGES” || “$ALLOW_EMPTY” == true ]]; then
info “Committing changes”
if [[ -n “$COMMIT_MSG” ]]; then
run git commit -m “$COMMIT_MSG” || warn “Nothing to commit (possible race); continuing.”
else
run git commit –allow-empty -m “chore: empty repair commit ($(ts))”
fi
else
warn “Skipping commit (no staged changes).”
fi

––––– Tag (optional) –––––

if [[ -n “$TAG_NAME” ]]; then
info “Tagging ${TAG_NAME}”
if git rev-parse -q –verify “refs/tags/$TAG_NAME” >/dev/null; then
warn “Tag ${TAG_NAME} already exists locally; will push existing tag.”
else
if $GPG_SIGN; then
run git tag -s “$TAG_NAME” -m “$COMMIT_MSG”
else
run git tag -a “$TAG_NAME” -m “$COMMIT_MSG”
fi
fi
fi

––––– Push Git + DVC (optional) –––––

if $DO_PUSH; then
info “Pushing branch: ${BOLD}${BRANCH}${RST} → ${BOLD}${REMOTE}${RST}”
retry “$RETRIES” “$SLEEP_BETWEEN” – run git push -u “$REMOTE” “$BRANCH”

if [[ -n “$TAG_NAME” ]]; then
info “Pushing tag: ${BOLD}${TAG_NAME}${RST}”
retry “$RETRIES” “$SLEEP_BETWEEN” – run git push “$REMOTE” “$TAG_NAME” || warn “Tag push failed (continuing)”
fi

if $DO_DVC; then
info “Pushing DVC artifacts”
retry “$RETRIES” “$SLEEP_BETWEEN” – run dvc push || warn “DVC push encountered errors”
fi

ok “Push sequence completed”
else
warn “Push disabled (–no-push). Local commit only.”
fi

––––– Compute config hash (best-effort) –––––

CFG_HASH=”-”
if [[ $HAS_CLI -eq 1 ]]; then
if “$CLI” –help 2>/dev/null | grep -qiE – “–print-config-hash|hash-config”; then
if “$CLI” –print-config-hash >/dev/null 2>&1; then
CFG_HASH=”$(”$CLI” –print-config-hash 2>/dev/null || echo “-”)”
else
CFG_HASH=”$(”$CLI” hash-config 2>/dev/null || echo “-”)”
fi
fi
fi

––––– Structured log line –––––

printf -v LOG_SUMMARY ‘%s cmd=%s git=%s branch=%s remote=%s cfg_hash=%s tag=%s note=”%s”’ 
“$(ts)” “repair_and_push” “$(gitsha)” “$BRANCH” “$REMOTE” “$CFG_HASH” “${TAG_NAME:-”-”}” “dvc=$($DO_DVC && echo on || echo off) push=$($DO_PUSH && echo on || echo off)”
log_line “$LOG_SUMMARY”

––––– Manifest –––––

MANIFEST_PATH=””
if $WRITE_MANIFEST; then
MANIFEST_DIR=“outputs/manifests”
mkdir -p “$MANIFEST_DIR”
MANIFEST_PATH=”$MANIFEST_DIR/repair_manifest_${RUN_ID}.json”

MSG_JSON=$(json_escape “$COMMIT_MSG”)
TAG_JSON=$(json_escape “$TAG_NAME”)
REMOTE_JSON=$(json_escape “$REMOTE”)
BRANCH_JSON=$(json_escape “$BRANCH”)

{
printf ‘{\n’
printf ’  “run”: {\n’
printf ’    “id”: “%s”,\n’       “$RUN_ID”
printf ’    “timestamp_utc”: “%s”,\n’ “$(ts)”
printf ’    “host”: “%s”,\n’     “$RUN_HOST”
printf ’    “git_sha”: “%s”,\n’  “$(gitsha)”
printf ’    “cfg_hash”: “%s”\n’  “$CFG_HASH”
printf ’  },\n’
printf ’  “repo”: {\n’
printf ’    “root”: “%s”,\n’     “$GIT_ROOT”
printf ’    “remote”: %s,\n’     “$REMOTE_JSON”
printf ’    “branch”: %s\n’      “$BRANCH_JSON”
printf ’  },\n’
printf ’  “actions”: {\n’
printf ’    “dvc”: %s,\n’        “$($DO_DVC && echo true || echo false)”
printf ’    “push”: %s,\n’       “$($DO_PUSH && echo true || echo false)”
printf ’    “tag”: %s,\n’        “$([[ -n “$TAG_NAME” ]] && echo true || echo false)”
printf ’    “gpg_sign”: %s\n’    “$($GPG_SIGN && echo true || echo false)”
printf ’  },\n’
printf ’  “inputs”: {\n’
printf ’    “message”: %s,\n’    “$MSG_JSON”
printf ’    “allow_empty”: %s,\n’ “$($ALLOW_EMPTY && echo true || echo false)”
printf ’    “allow_non_main”: %s,\n’ “$($ALLOW_NON_MAIN && echo true || echo false)”
printf ’    “poetry_install”: %s,\n’ “$($POETRY_INSTALL && echo true || echo false)”
printf ’    “run_tests”: %s,\n’  “$($RUN_TESTS && echo true || echo false)”
printf ’    “run_pre_commit”: %s,\n’ “$($RUN_PRE_COMMIT && echo true || echo false)”
printf ’    “security_scan”: %s\n’ “$($SECURITY_SCAN && echo true || echo false)”
printf ’  },\n’
printf ’  “tag”: %s\n’          “$TAG_JSON”
printf ‘}\n’
} > “$MANIFEST_PATH”

ok “Manifest: $MANIFEST_PATH”
if $OPEN_MANIFEST; then
if [[ -f “$MANIFEST_PATH” ]]; then open_path “$MANIFEST_PATH”; else open_path “$MANIFEST_DIR”; fi
fi
fi

––––– JSON summary (stdout) –––––

if $EMIT_JSON; then
SUMMARY=$(
printf ‘{’
printf ’“ok”: true, ’
printf ’“run_id”:”%s”, ’ “$RUN_ID”
printf ’“branch”:”%s”, ’ “$BRANCH”
printf ’“remote”:”%s”, ’ “$REMOTE”
printf ’“git_sha”:”%s”, ’ “$(gitsha)”
printf ’“cfg_hash”:”%s”, ’ “$CFG_HASH”
printf ’“tag”:”%s”, ’ “$TAG_NAME”
printf ’“dvc”:%s, ’ “$($DO_DVC && echo true || echo false)”
printf ’“pushed”:%s, ’ “$($DO_PUSH && echo true || echo false)”
printf ‘“manifest”:”%s”’ “${MANIFEST_PATH}”
printf ‘}\n’
)
printf ‘%s\n’ “$SUMMARY”
fi

ok  “[repair_and_push] Completed at $(ts) (RUN_ID=${RUN_ID})”
log_line “[repair_and_push] ========================================================”
exit 0