#!/usr/bin/env bash

==============================================================================

bin/push.sh — SpectraMind V50 guarded push helper (ultimate, upgraded)

——————————————————————————

Safe, reproducible, and chatty wrapper around the common

“commit → validate → dvc sync → push” flow with mission-grade guardrails.



Defaults:

- Preflight checks: git present, inside repo, remote configured

- Optional Poetry env install, pre-commit lint, unit tests

- Optional SpectraMind CLI selftest (fast)

- Optional sync-lock drift enforcement via bin/sync-lock.sh

- VERSION ↔ pyproject.toml version consistency check

- DVC push (if repo uses DVC)

- Pushes current branch to origin (or chosen remote/branch)

- JSON summary (optional) for CI dashboards



Backward-compatible short flags:

-m “msg”   Commit message (optional; if given, will git add -A && git commit)

-b BRANCH  Target branch to push (default: current)

-t TAG     Create and push a tag (annotated by default; see –gpg-sign)

-n         Dry-run (show what would run; no side effects)

-F         Force push (use with care)

-S         Skip tests (unit/integration)

-L         Skip lint & style (pre-commit)

-D         Skip DVC push

-C         Skip CLI selftest (spectramind selftest)

-K         Skip Kaggle CLI verification (kept for compatibility; default skip)

-q         Quiet mode (less chatter)

-h         Help



New long flags:

–json                 Emit JSON summary to stdout (in addition to text)

–gpg-sign             Sign tag (-s) instead of annotate (-a) when -t is used

–sync-lock            Enforce requirements sync using bin/sync-lock.sh –check

–fix-lock             Regenerate requirements (bin/sync-lock.sh –lock –write)

–timeout         Timeout per step (default: 240)

–no-sync-lock         Disable sync-lock checks (overrides env)

–remote         Git remote to push to (default: first remote)

–allow-main           Allow pushing to main/master without prompt guard

–no-fetch             Do not fetch before push (skip up-to-date check)

–no-ff                Push with –no-ff merge style if a merge is performed (advisory)

–security-scan        Run pip-audit (if available) before pushing



Environment toggles:

PUSH_ENFORCE_SYNC=1     (fail if sync-lock drift)

PUSH_SIGN_TAGS=1        (equivalent to –gpg-sign)

PUSH_TIMEOUT=      (equivalent to –timeout)

PUSH_REMOTE=      (default remote name)

PUSH_ALLOW_MAIN=1       (equivalent to –allow-main)

PUSH_NO_FETCH=1         (equivalent to –no-fetch)

PUSH_SECURITY_SCAN=1    (equivalent to –security-scan)



Examples:

bin/push.sh -m “feat(train): add COREL coverage report”

bin/push.sh -m “chore: bump deps” -t v0.2.0

bin/push.sh -S -L -D    # minimal, just push

bin/push.sh –sync-lock # enforce lock/requirements in CI

==============================================================================

set -euo pipefail
IFS=$’\n\t’

––––– styling –––––

if command -v tput >/dev/null 2>&1 && [[ -t 1 ]]; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; YLW=”$(tput setaf 3)”; CYN=”$(tput setaf 6)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; YLW=””; CYN=””; RST=””
fi

log()  { printf “%b\n” “${1-}”; }
info() { [[ “${QUIET:-0}” -eq 1 ]] && return 0; log “${CYN}›${RST} ${1-}”; }
ok()   { [[ “${QUIET:-0}” -eq 1 ]] && return 0; log “${GRN}✓${RST} ${1-}”; }
warn() { log “${YLW}!${RST} ${1-}”; }
err()  { log “${RED}✗${RST} ${1-}”; }

usage() {
sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’
exit “${1:-0}”
}

have() { command -v “$1” >/dev/null 2>&1; }

––––– args (short) –––––

COMMIT_MSG=””
TARGET_BRANCH=””
TAG_NAME=””
DRYRUN=0
FORCE=0
SKIP_TESTS=0
SKIP_LINT=0
SKIP_DVC=0
SKIP_SELFTEST=0
SKIP_KAGGLE=1   # default skip
QUIET=0

––––– args (long/defaults) –––––

JSON_SUM=0
GPG_SIGN=”${PUSH_SIGN_TAGS:-0}”
ENFORCE_SYNC=”${PUSH_ENFORCE_SYNC:-0}”
FIX_LOCK=0
NO_SYNC_LOCK=0
STEP_TIMEOUT=”${PUSH_TIMEOUT:-240}”
REMOTE_OVERRIDE=”${PUSH_REMOTE:-}”
ALLOW_MAIN=”${PUSH_ALLOW_MAIN:-0}”
NO_FETCH=”${PUSH_NO_FETCH:-0}”
NO_FF=0
SECURITY_SCAN=”${PUSH_SECURITY_SCAN:-0}”

Parse short options

while getopts “:m:b:t:nFSLDCKqh” opt; do
case “$opt” in
m) COMMIT_MSG=”$OPTARG” ;;
b) TARGET_BRANCH=”$OPTARG” ;;
t) TAG_NAME=”$OPTARG” ;;
n) DRYRUN=1 ;;
F) FORCE=1 ;;
S) SKIP_TESTS=1 ;;
L) SKIP_LINT=1 ;;
D) SKIP_DVC=1 ;;
C) SKIP_SELFTEST=1 ;;
K) SKIP_KAGGLE=1 ;;  # kept for compatibility; -K ensures skip
q) QUIET=1 ;;
h) usage 0 ;;
?) err “Unknown option: -$OPTARG”; usage 1 ;;
:) err “Option -$OPTARG requires an argument”; usage 1 ;;
esac
done
shift $((OPTIND -1))

Parse long options

while [[ $# -gt 0 ]]; do
case “$1” in
–json) JSON_SUM=1 ;;
–gpg-sign) GPG_SIGN=1 ;;
–sync-lock) ENFORCE_SYNC=1 ;;
–fix-lock) FIX_LOCK=1 ;;
–no-sync-lock) NO_SYNC_LOCK=1 ;;
–timeout) STEP_TIMEOUT=”${2:?}”; shift ;;
–remote) REMOTE_OVERRIDE=”${2:?}”; shift ;;
–allow-main) ALLOW_MAIN=1 ;;
–no-fetch) NO_FETCH=1 ;;
–no-ff) NO_FF=1 ;;
–security-scan) SECURITY_SCAN=1 ;;
–help) usage 0 ;;
*) err “Unknown flag: $1”; usage 1 ;;
esac
shift
done

If explicit disable requested, override env

if [[ $NO_SYNC_LOCK -eq 1 ]]; then ENFORCE_SYNC=0; FIX_LOCK=0; fi

run() {

run <cmd…>  — obey DRYRUN and QUIET

if [[ $DRYRUN -eq 1 ]]; then
printf “%s\n” “${DIM}[dry-run]${RST} $*”
else
if [[ $QUIET -eq 1 ]]; then
“$@” >/dev/null
else
“$@”
fi
fi
}

run_to() {

run_to  <cmd…> — with timeout & DRYRUN handling

local desc=”$1”; shift
local tcmd=(”$@”)
if [[ $DRYRUN -eq 1 ]]; then
printf “%s\n” “${DIM}[dry-run]${RST} ($STEP_TIMEOUT)s ${tcmd[*]}”
return 0
fi
if have timeout; then
if [[ $QUIET -eq 1 ]]; then
timeout –preserve-status –signal=TERM “$STEP_TIMEOUT” “${tcmd[@]}” >/dev/null
else
timeout –preserve-status –signal=TERM “$STEP_TIMEOUT” “${tcmd[@]}”
fi
else
run “${tcmd[@]}”
fi
}

assert_cmd() {
have “$1” || { err “Missing required command: $1”; exit 127; }
}

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

––––– preflight –––––

assert_cmd git
GIT_ROOT=”$(git rev-parse –show-toplevel 2>/dev/null || true)”
[[ -n “$GIT_ROOT” ]] || { err “Not inside a git repository.”; exit 1; }
cd “$GIT_ROOT”

CURRENT_BRANCH=”$(git rev-parse –abbrev-ref HEAD)”
BRANCH=”${TARGET_BRANCH:-$CURRENT_BRANCH}”

Remote selection

if [[ -n “$REMOTE_OVERRIDE” ]]; then
REMOTE=”$REMOTE_OVERRIDE”
else
REMOTE=”$(git remote 2>/dev/null | head -n1 || true)”
fi
[[ -n “$REMOTE” ]] || { err “No git remote configured.”; exit 1; }

if [[ $FORCE -eq 1 ]]; then
warn “Force push enabled (-F). Use with care.”
fi

Reject pushing to main/master unless allowed

if [[ “$BRANCH” =~ ^(main|master)$ && $ALLOW_MAIN -ne 1 ]]; then
warn “Pushing to ‘$BRANCH’ is guarded. Use –allow-main or set PUSH_ALLOW_MAIN=1 to proceed.”
exit 1
fi

Ensure we have a clean index if no commit message supplied.

if [[ -z “$COMMIT_MSG” ]]; then
if [[ -n “$(git status –porcelain)” ]]; then
warn “Working tree has changes but no commit message provided.”
warn “Either commit manually or provide -m "message" to auto-commit.”
exit 1
fi
fi

Poetry optional

HAS_POETRY=0; have poetry && HAS_POETRY=1

DVC presence?

HAS_DVC=0; have dvc && [[ -d “.dvc” ]] && HAS_DVC=1

pre-commit?

HAS_PRECOMMIT=0; have pre-commit && [[ -f “.pre-commit-config.yaml” ]] && HAS_PRECOMMIT=1

SpectraMind CLI?

HAS_SM=0; { [[ -f “spectramind.py” ]] || have spectramind; } && HAS_SM=1

––––– JSON init –––––

JSON_FIRST=0
JSON_BODY=”{”
json_field() { local k=”$1”; local v=”$2”; JSON_BODY+=$(printf ‘%s”%s”:”%s”’ “${JSON_FIRST:+,}” “$k” “$v”); JSON_FIRST=1; }
START_TS=”$(now_utc)”
json_field “repo” “$GIT_ROOT”
json_field “branch” “$BRANCH”
json_field “remote” “$REMOTE”
json_field “start” “$START_TS”

––––– commit (optional) –––––

if [[ -n “$COMMIT_MSG” ]]; then
info “Staging & committing changes”
run git add -A
if git diff –cached –quiet; then
warn “No staged changes to commit; skipping commit step.”
json_field “commit” “skipped”
else
run git commit -m “$COMMIT_MSG”
ok “Committed: ${COMMIT_MSG}”
json_field “commit” “ok”
fi
fi

––––– environment / dependencies –––––

if [[ $HAS_POETRY -eq 1 ]]; then
info “Ensuring Poetry env & dependencies”
run_to “poetry install” poetry install –no-root
json_field “poetry_install” “ok”
else
warn “Poetry not found. Assuming environment is already prepared.”
json_field “poetry_install” “absent”
fi

––––– VERSION ↔ pyproject sync –––––

if [[ -f VERSION && -f pyproject.toml ]]; then
V_FILE=”$(sed -n ‘1s/[[:space:]]//gp’ VERSION || true)”
V_PY=”$(sed -n ‘s/^[[:space:]]version[[:space:]]=[[:space:]]”.”.*/\1/p’ pyproject.toml | head -n1 || true)”
if [[ -n “$V_FILE” && -n “$V_PY” && “$V_FILE” != “$V_PY” ]]; then
err “VERSION ($V_FILE) differs from pyproject.toml ($V_PY).”
json_field “version_match” “false”
exit 1
else
ok “VERSION matches pyproject: ${V_FILE:-unknown}”
json_field “version_match” “true”
fi
fi

––––– optional sync-lock enforcement –––––

SYNC_SCRIPT=“bin/sync-lock.sh”
if [[ $ENFORCE_SYNC -eq 1 && -x “$SYNC_SCRIPT” && $NO_SYNC_LOCK -eq 0 ]]; then
info “Enforcing requirements sync via ${SYNC_SCRIPT} –check”
if run_to “sync-lock check” “$SYNC_SCRIPT” –check –all –json >/tmp/sync-lock.json 2>/dev/null; then
ok “Requirements in sync”
json_field “sync_lock” “ok”
else
if [[ $FIX_LOCK -eq 1 ]]; then
warn “Drift detected; attempting fix via ${SYNC_SCRIPT} –lock –write –all”
run_to “sync-lock write” “$SYNC_SCRIPT” –lock –write –all || { err “sync-lock fix failed”; exit 1; }
ok “Requirements updated from lock”
json_field “sync_lock” “fixed”
else
err “Requirements are NOT in sync with lock (use –fix-lock to regenerate).”
json_field “sync_lock” “drift”
exit 1
fi
fi
else
if [[ -x “$SYNC_SCRIPT” ]]; then
warn “Sync-lock enforcement disabled (use –sync-lock to enable).”
else
warn “Sync-lock script not found (bin/sync-lock.sh).”
fi
fi

––––– Security scan (optional) –––––

if [[ $SECURITY_SCAN -eq 1 ]]; then
if have pip-audit; then
info “Running pip-audit (security scan)”
run_to “pip-audit” pip-audit || { err “pip-audit reported issues (fail or ignore as needed)”; exit 1; }
ok “pip-audit passed”
json_field “security_scan” “ok”
else
warn “pip-audit not installed; skipping security scan.”
json_field “security_scan” “absent”
fi
fi

––––– quality gates –––––

if [[ $SKIP_SELFTEST -ne 1 && $HAS_SM -eq 1 ]]; then
info “Running SpectraMind selftest”
if have spectramind; then
run_to “spectramind selftest” spectramind selftest –quick
else
run_to “poetry run spectramind selftest” poetry run spectramind selftest –quick
fi
ok “Selftest passed”
json_field “selftest” “ok”
else
warn “Skipping CLI selftest (-C) or spectramind CLI not detected.”
json_field “selftest” “skipped”
fi

if [[ $SKIP_LINT -ne 1 && $HAS_PRECOMMIT -eq 1 ]]; then
info “Running pre-commit hooks (lint/format/security)”
run_to “pre-commit” pre-commit run –all-files
ok “Lint & format checks passed”
json_field “lint” “ok”
else
warn “Skipping lint/format (-L) or pre-commit not configured.”
json_field “lint” “skipped”
fi

if [[ $SKIP_TESTS -ne 1 ]]; then
info “Running tests”
if [[ $HAS_POETRY -eq 1 ]]; then
run_to “pytest” poetry run pytest -q
else
run_to “pytest” pytest -q
fi
ok “Tests passed”
json_field “tests” “ok”
else
warn “Skipping tests (-S)”
json_field “tests” “skipped”
fi

––––– optional Kaggle CLI verify (disabled by default) –––––

if [[ $SKIP_KAGGLE -ne 1 ]]; then
if have kaggle; then
info “Verifying Kaggle CLI login”
if run_to “kaggle check” kaggle competitions list >/dev/null 2>&1; then
ok “Kaggle CLI OK”
json_field “kaggle_cli” “ok”
else
warn “Kaggle CLI not authenticated (skip or login).”
json_field “kaggle_cli” “unauth”
fi
else
warn “Kaggle CLI not installed.”
json_field “kaggle_cli” “absent”
fi
else
json_field “kaggle_cli” “skipped”
fi

––––– helpful project housekeeping –––––

if [[ $HAS_SM -eq 1 ]]; then
info “Generating short CLI activity summary”
if have spectramind; then
run spectramind analyze-log-short || true
else
run poetry run spectramind analyze-log-short || true
fi
fi

––––– Large untracked file warning (avoid accidental pushes) –––––

if git ls-files –others –exclude-standard | grep -q .; then

Warn if any untracked file > 50MB

BIG=”$(git ls-files –others –exclude-standard -z | xargs -0 -I{} sh -c ‘b=$(wc -c < “{}” 2>/dev/null || echo 0); if [ “$b” -gt 52428800 ]; then echo “{} ($b bytes)”; fi’ || true)”
if [[ -n “$BIG” ]]; then
warn “Untracked large files detected (won’t be pushed):”
printf ‘%s\n’ “$BIG”
fi
fi

––––– fetch & up-to-date guard –––––

if [[ $NO_FETCH -ne 1 ]]; then
info “Fetching ‘$REMOTE’ to verify branch state”
run git fetch “$REMOTE” “$BRANCH” || true
if git rev-parse –verify “refs/remotes/$REMOTE/$BRANCH” >/dev/null 2>&1; then
LOCAL=”$(git rev-parse “$BRANCH”)”
REMOTE_SHA=”$(git rev-parse “refs/remotes/$REMOTE/$BRANCH”)”
BASE=”$(git merge-base “$BRANCH” “refs/remotes/$REMOTE/$BRANCH”)”
if [[ “$LOCAL” = “$REMOTE_SHA” ]]; then
info “Branch is up-to-date with $REMOTE/$BRANCH”
elif [[ “$LOCAL” = “$BASE” ]]; then
warn “Local branch is behind $REMOTE/$BRANCH (consider pulling/rebasing).”
elif [[ “$REMOTE_SHA” = “$BASE” ]]; then
info “Local branch is ahead of remote; pushing will fast-forward remote.”
else
warn “Local and remote have diverged. Consider merge/rebase before pushing.”
fi
fi
else
warn “Skipping fetch/up-to-date check (–no-fetch).”
fi

––––– DVC sync –––––

if [[ $HAS_DVC -eq 1 && $SKIP_DVC -ne 1 ]]; then
info “DVC status”
run dvc status || true
info “Pushing DVC-tracked artifacts (if any)”
run dvc push
ok “DVC push complete”
json_field “dvc_push” “ok”
else
if [[ $HAS_DVC -eq 1 ]]; then
warn “Skipping DVC push (-D)”
json_field “dvc_push” “skipped”
else
warn “No DVC detected”
json_field “dvc_push” “absent”
fi
fi

––––– push –––––

PUSH_ARGS=(”$REMOTE” “$BRANCH”)
[[ $FORCE -eq 1 ]] && PUSH_ARGS+=(”–force”)

info “Pushing branch: ${BOLD}${BRANCH}${RST} → ${BOLD}${REMOTE}${RST}”
run git push “${PUSH_ARGS[@]}”
ok “Branch pushed”
json_field “branch_push” “ok”

Tag handling

if [[ -n “$TAG_NAME” ]]; then
info “Tagging: ${BOLD}${TAG_NAME}${RST}”
if git rev-parse -q –verify “refs/tags/$TAG_NAME” >/dev/null; then
warn “Tag ${TAG_NAME} already exists locally; pushing existing tag”
else
if [[ $GPG_SIGN -eq 1 ]]; then
run git tag -s “$TAG_NAME” -m “$TAG_NAME”
else
run git tag -a “$TAG_NAME” -m “$TAG_NAME”
fi
fi
run git push “$REMOTE” “$TAG_NAME”
ok “Tag pushed: ${TAG_NAME}”
json_field “tag_push” “$TAG_NAME”
else
json_field “tag_push” “none”
fi

––––– summary –––––

GIT_SHA_SHORT=”$(git rev-parse –short HEAD)”
END_TS=”$(now_utc)”
if [[ $QUIET -ne 1 ]]; then
echo
ok “Done.”
echo “  Repo   : $GIT_ROOT”
echo “  Remote : $REMOTE”
echo “  Branch : $BRANCH”
echo “  Commit : $GIT_SHA_SHORT”
[[ -n “$TAG_NAME” ]] && echo “  Tag    : $TAG_NAME”
[[ $DRYRUN -eq 1 ]] && echo “  Note   : dry-run (no changes applied)”
fi

JSON summary

if [[ $JSON_SUM -eq 1 ]]; then
JSON_BODY+=’, “end”: “’”$END_TS”’”’
JSON_BODY+=’}’
printf ‘%s\n’ “$JSON_BODY”
fi

exit 0