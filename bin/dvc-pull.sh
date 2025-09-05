#!/usr/bin/env bash

==============================================================================

SpectraMind V50 — bin/dvc-pull.sh (ultimate, upgraded)

——————————————————————————

Syncs data/artifacts from configured DVC remotes with retries, logging, JSON,

and guardrails. Safe for local dev, CI, Kaggle, and dockerized environments.



Features (new/expanded)

• Auto repo-root detection; graceful messages if .dvc missing

• Status-only and dry-run modes (no workspace mutation)

• Robust retry with exponential backoff + per-step timeout

• Remote autodetect (first configured), or pick one via –remote

• Optional Git/DVC revision checkout and restore

• Parallel jobs (-j), run-cache toggle, force mode

• CI/Kaggle-safe tuning; structured audit lines (logs/ops/*.jsonl) + v50_debug_log.md

• JSON summary with params + log path; quiet mode; explicit targets



Usage:

bin/dvc-pull.sh [options] [–] [TARGET …]



Options:

-r, –remote NAME      Pull from a specific DVC remote (default: auto)

-j, –jobs N           Parallel jobs for DVC (default: auto)

-R, –rev REV          Git/DVC revision to use (branch/tag/sha)

-a, –all              Pull all outputs (dvc pull -a) ignoring targets

-f, –force            Force pull (ignore run-cache skip heuristics)

-n, –dry-run          Show what would be pulled; do not modify workspace

-q, –quiet            Quieter output (pass through to dvc)

–no-run-cache     Disable using run-cache when pulling

–max-retries N    Retry attempts on transient errors (default: 3)

–backoff SEC      Initial backoff between retries (default: 4)

–timeout SEC      Timeout per DVC/Git step (default: 900)

–status-only      Only show status/listing; do not pull

–json             Emit JSON summary to stdout

–json-path PATH   Also write JSON summary to file

-h, –help             Show this help



Examples:

bin/dvc-pull.sh

bin/dvc-pull.sh -r origin-aws -j 8

bin/dvc-pull.sh -a –max-retries 5 –backoff 6

bin/dvc-pull.sh – data/raw/ data/processed/



Notes:

• Writes logs under logs/ops/dvc-pull-YYYYmmdd_HHMMSS.log

• Exits non-zero on failure; safe in CI pipelines

• Appends a structured line to logs/v50_debug_log.md (cmd=dvc-pull)

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– tiny stdlib –––––

is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; YLW=”$(tput setaf 3)”; CYN=”$(tput setaf 6)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; YLW=””; CYN=””; RST=””
fi

die()   { printf “%s %s\n” “${RED}✖${RST}” “$” >&2; exit 1; }
info()  { [[ “${QUIET:-false}” == “true” ]] && return 0; printf “%s %s\n” “${GRN}✔${RST}” “$”; }
warn()  { printf “%s %s\n” “${YLW}∙${RST}” “$” >&2; }
note()  { [[ “${QUIET:-false}” == “true” ]] && return 0; printf “%s %s\n” “${DIM}·${RST}” “$”; }
have()  { command -v “$1” >/dev/null 2>&1; }
ts()    { date -u +%Y-%m-%dT%H:%M:%SZ; }

usage() {
sed -n ‘2,160p’ “$0” | sed ‘s/^# {0,1}//’
exit 0
}

––––– env detect –––––

IS_CI=false;    [[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=true
IS_KAGGLE=false; [[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=true

––––– defaults –––––

REMOTE=””
JOBS=””
REV=””
ALL=false
FORCE=false
DRY_RUN=false
QUIET=false
NO_RUN_CACHE=false
MAX_RETRIES=3
BACKOFF=4
TIMEOUT=”${DVC_PULL_TIMEOUT:-900}”
STATUS_ONLY=false
JSON=false
JSON_PATH=””

TARGETS=()

––––– parse args –––––

while [[ $# -gt 0 ]]; do
case “$1” in
-r|–remote)        REMOTE=”${2:-}”; shift 2 ;;
-j|–jobs)          JOBS=”${2:-}”; shift 2 ;;
-R|–rev)           REV=”${2:-}”; shift 2 ;;
-a|–all)           ALL=true; shift ;;
-f|–force)         FORCE=true; shift ;;
-n|–dry-run)       DRY_RUN=true; shift ;;
-q|–quiet)         QUIET=true; shift ;;
–no-run-cache)     NO_RUN_CACHE=true; shift ;;
–max-retries)      MAX_RETRIES=”${2:-}”; shift 2 ;;
–backoff)          BACKOFF=”${2:-}”; shift 2 ;;
–timeout)          TIMEOUT=”${2:-}”; shift 2 ;;
–status-only)      STATUS_ONLY=true; shift ;;
–json)             JSON=true; shift ;;
–json-path)        JSON_PATH=”${2:-}”; shift 2 ;;
-h|–help)          usage ;;
–)                 shift; TARGETS+=(”$@”); break ;;
-*)
die “Unknown option: $1 (use -h for help)”
;;
*)
TARGETS+=(”$1”); shift ;;
esac
done

––––– repo root & logging –––––

find_repo_root() {
local d=”$PWD”
while [[ “$d” != “/” ]]; do
[[ -d “$d/.git” || -f “$d/pyproject.toml” || -f “$d/README.md” ]] && { echo “$d”; return; }
d=”$(dirname “$d”)”
done
echo “$PWD”
}

ROOT=”$(find_repo_root)”
cd “$ROOT”

LOG_DIR=”$ROOT/logs/ops”
TS_COMPACT=”$(date -u +%Y%m%d_%H%M%S)”
LOG_FILE=”$LOG_DIR/dvc-pull-${TS_COMPACT}.log”
mkdir -p “$LOG_DIR” “$ROOT/logs”

Tee all output to a log file (always on)

if [[ “$QUIET” == “true” ]]; then
exec >>”$LOG_FILE” 2>&1
else

shellcheck disable=SC2094

exec > >(tee -a “$LOG_FILE”) 2>&1
fi

note “Log : $LOG_FILE”
note “Repo: $ROOT”
note “CI : ${IS_CI}  Kaggle: ${IS_KAGGLE}”

––––– preflight checks –––––

have dvc  || die “dvc not found on PATH. Install DVC (e.g., pip install ‘dvc[s3]’ or similar).”
if ! have git; then
warn “git not found on PATH (continuing, but revision checks will be limited).”
fi
[[ -d “$ROOT/.dvc” ]] || die “No .dvc directory found at repo root ($ROOT). Initialize with ‘dvc init’ or run from project root.”

Safety: inform dirty git tree (non-fatal)

if have git && git rev-parse –is-inside-work-tree >/dev/null 2>&1; then
if ! git diff –quiet || ! git diff –cached –quiet; then
warn “Git working tree is dirty. Consider committing or stashing before pulling large artifacts.”
fi
fi

If user passed a rev, attempt checkout (restore later)

ORIG_REF=””
RESTORE_REF=false
if [[ -n “$REV” ]] && have git && git rev-parse –is-inside-work-tree >/dev/null 2>&1; then
ORIG_REF=”$(git rev-parse –abbrev-ref HEAD 2>/dev/null || git rev-parse HEAD 2>/dev/null || true)”
note “Checking out revision: ${BOLD}${REV}${RST}”
git fetch –all –tags –prune >/dev/null 2>&1 || warn “git fetch failed or remote missing (continuing).”
git checkout –quiet “$REV” || die “Failed to checkout rev ‘$REV’”
RESTORE_REF=true
fi

restore_ref() {
if $RESTORE_REF && [[ -n “$ORIG_REF” ]]; then
note “Restoring to original ref: ${BOLD}${ORIG_REF}${RST}”
git checkout –quiet “$ORIG_REF” || warn “Failed to restore original ref ($ORIG_REF).”
fi
}
trap restore_ref EXIT

Infer remote if not provided (prefer first configured remote)

if [[ -z “$REMOTE” ]]; then
set +e
REMOTE=”$(dvc remote list 2>/dev/null | awk ‘{print $1}’ | head -n1)”
set -e
[[ -n “$REMOTE” ]] && note “Auto-selected DVC remote: ${BOLD}${REMOTE}${RST}” || warn “No DVC remote configured. Will rely on default.”
fi

––––– status summary –––––

note “DVC status (cache vs workspace)…”
if ! dvc status -c ${REMOTE:+-r “$REMOTE”} || ! dvc status; then
warn “dvc status encountered issues (continuing).”
fi

––––– build dvc pull command –––––

DVC_ARGS=()
$ALL            && DVC_ARGS+=(”-a”)
[[ -n “$JOBS”   ]] && DVC_ARGS+=(”-j” “$JOBS”)
[[ -n “$REMOTE” ]] && DVC_ARGS+=(”-r” “$REMOTE”)
$QUIET          && DVC_ARGS+=(”-q”)
$NO_RUN_CACHE   && DVC_ARGS+=(”–no-run-cache”)
$FORCE          && DVC_ARGS+=(”–force”)

––––– dry-run / status-only –––––

if $DRY_RUN || $STATUS_ONLY; then
if $STATUS_ONLY; then
note “Status-only: showing remote listing or cache differences.”
else
note “Dry-run: showing listing instead of pulling…”
fi

Prefer status -c vs remote; fallback to remote list

if ! dvc status -c ${REMOTE:+-r “$REMOTE”}; then
if [[ -n “$REMOTE” ]]; then
printf “%s\n” “dvc list $REMOTE ${REV:+–rev “$REV”} .”
dvc list “$REMOTE” ${REV:+–rev “$REV”} . || warn “Remote may not support listing.”
else
warn “No remote selected; cannot list.”
fi
fi

JSON summary (if requested)

if $JSON || [[ -n “${JSON_PATH:-}” ]]; then
payload=$(printf ‘{“ok”:true,“mode”:”%s”,“remote”:”%s”,“rev”:”%s”,“jobs”:”%s”,“log”:”%s”,“targets”:”%s”}’ 
“$([[ $STATUS_ONLY == true ]] && echo status || echo dry-run)” 
“${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}” “$LOG_FILE” “${TARGETS[*]:-}”)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; }
$JSON && printf “%s\n” “$payload”
fi

Append structured CLI-style audit to v50_debug_log.md

mkdir -p “$ROOT/logs”
{
cfg_hash=”-”
[[ -f “$ROOT/run_hash_summary_v50.json” ]] && cfg_hash=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “$ROOT/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)” || true
[[ -z “$cfg_hash” ]] && cfg_hash=”-”
printf ‘[%s] cmd=dvc-pull git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=“mode=%s;remote=%s;rev=%s;jobs=%s;status=ok”\n’ 
“$(ts)” “$(git rev-parse –short HEAD 2>/dev/null || echo nogit)” “$cfg_hash” 
“$([[ $STATUS_ONLY == true ]] && echo status || echo dry-run)” “${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}”
} >> “$ROOT/logs/v50_debug_log.md” || true

Also append JSONL audit

{
printf ‘{“ts”:”%s”,“action”:“dvc-pull”,“status”:“ok”,“mode”:”%s”,“remote”:”%s”,“rev”:”%s”,“jobs”:”%s”,“targets”:”%s”,“log”:”%s”}\n’ 
“$(ts)” “$([[ $STATUS_ONLY == true ]] && echo status || echo dry-run)” “${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}” “${TARGETS[*]:-}” “$LOG_FILE”
} >> “$LOG_DIR/dvc-pull.audit.jsonl” || true
exit 0
fi

––––– retry + timeout –––––

retry() {
local tries=”$1”; shift
local backoff=”$1”; shift
local attempt=1
until “$@”; do
local exit_code=$?
if (( attempt >= tries )); then
return “$exit_code”
fi
warn “Command failed (attempt $attempt/$tries). Retrying in ${backoff}s…”
sleep “$backoff”
backoff=$(( backoff * 2 ))
attempt=$(( attempt + 1 ))
done
}

run_to() {
local t=”$1”; shift
if have timeout; then
timeout –preserve-status –signal=TERM “$t” “$@”
else
“$@”
fi
}

––––– pull –––––

note “Starting dvc pull…”
printf “%s “ “dvc pull ${DVC_ARGS[]} ${TARGETS[]:-}”; echo

if ! retry “$MAX_RETRIES” “$BACKOFF” run_to “$TIMEOUT” dvc pull “${DVC_ARGS[@]}” “${TARGETS[@]}”; then

JSON (error)

if $JSON || [[ -n “${JSON_PATH:-}” ]]; then
payload=$(printf ‘{“ok”:false,“error”:“dvc pull failed”,“remote”:”%s”,“rev”:”%s”,“log”:”%s”}’ “${REMOTE:-default}” “${REV:-current}” “$LOG_FILE”)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; }
$JSON && printf “%s\n” “$payload”
fi

Audit lines

{
cfg_hash=”-”
[[ -f “$ROOT/run_hash_summary_v50.json” ]] && cfg_hash=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “$ROOT/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)” || true
[[ -z “$cfg_hash” ]] && cfg_hash=”-”
printf ‘[%s] cmd=dvc-pull git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=“mode=pull;remote=%s;rev=%s;jobs=%s;status=fail”\n’ 
“$(ts)” “$(git rev-parse –short HEAD 2>/dev/null || echo nogit)” “$cfg_hash” 
“${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}”
} >> “$ROOT/logs/v50_debug_log.md” || true
{
printf ‘{“ts”:”%s”,“action”:“dvc-pull”,“status”:“fail”,“remote”:”%s”,“rev”:”%s”,“jobs”:”%s”,“targets”:”%s”,“log”:”%s”}\n’ 
“$(ts)” “${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}” “${TARGETS[*]:-}” “$LOG_FILE”
} >> “$LOG_DIR/dvc-pull.audit.jsonl” || true
die “dvc pull failed after $MAX_RETRIES attempt(s). See log: $LOG_FILE”
fi

––––– post-checks –––––

note “Verifying workspace integrity (dvc doctor)…”
if ! dvc doctor >/dev/null 2>&1; then
warn “dvc doctor reported issues (see above for details).”
fi

Summarize cache usage (non-fatal)

note “Cache GC preview (workspace only, dry-run)…”
if ! dvc gc –workspace –dry-run >/dev/null 2>&1; then
warn “Unable to compute cache gc preview (continuing).”
fi

info “DVC pull completed successfully.”
note “Tip: consider committing updated .dvc.lock / .dvc files if they changed.”

––––– JSON summary (success) –––––

if $JSON || [[ -n “${JSON_PATH:-}” ]]; then
if have jq; then
jq -nc 
–arg ts “$(ts)” 
–arg action “dvc-pull” 
–arg remote “${REMOTE:-default}” 
–arg rev “${REV:-current}” 
–arg jobs “${JOBS:-auto}” 
–arg all “$ALL” 
–arg force “$FORCE” 
–arg no_run_cache “$NO_RUN_CACHE” 
–arg retries “$MAX_RETRIES” 
–arg backoff_init “$BACKOFF” 
–arg timeout “$TIMEOUT” 
–arg log “$LOG_FILE” 
–arg targets “${TARGETS[]:-}” 
‘{ok:true, ts:$ts, action:$action,
params:{remote:$remote, rev:$rev, jobs:$jobs, all:$all, force:$force, no_run_cache:$no_run_cache,
retries:$retries, backoff_initial:$backoff_init, timeout:$timeout, targets:$targets},
log:$log}’
else
payload=$(printf ‘{“ok”:true,“remote”:”%s”,“rev”:”%s”,“jobs”:”%s”,“log”:”%s”}’ “${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}” “$LOG_FILE”)
printf “%s\n” “$payload”
fi
[[ -n “${JSON_PATH:-}” ]] && { # if both JSON and JSON_PATH, write the jq/stdout result again to file
if have jq; then
jq -nc 
–arg ts “$(ts)” 
–arg action “dvc-pull” 
–arg remote “${REMOTE:-default}” 
–arg rev “${REV:-current}” 
–arg jobs “${JOBS:-auto}” 
–arg all “$ALL” 
–arg force “$FORCE” 
–arg no_run_cache “$NO_RUN_CACHE” 
–arg retries “$MAX_RETRIES” 
–arg backoff_init “$BACKOFF” 
–arg timeout “$TIMEOUT” 
–arg log “$LOG_FILE” 
–arg targets “${TARGETS[]:-}” 
‘{ok:true, ts:$ts, action:$action,
params:{remote:$remote, rev:$rev, jobs:$jobs, all:$all, force:$force, no_run_cache:$no_run_cache,
retries:$retries, backoff_initial:$backoff_init, timeout:$timeout, targets:$targets},
log:$log}’ > “$JSON_PATH”
else
printf ‘%s\n’ “$payload” > “$JSON_PATH”
fi
}
fi

––––– audit lines (success) –––––

{
cfg_hash=”-”
[[ -f “$ROOT/run_hash_summary_v50.json” ]] && cfg_hash=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “$ROOT/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)” || true
[[ -z “$cfg_hash” ]] && cfg_hash=”-”
printf ‘[%s] cmd=dvc-pull git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=“mode=pull;remote=%s;rev=%s;jobs=%s;status=ok”\n’ 
“$(ts)” “$(git rev-parse –short HEAD 2>/dev/null || echo nogit)” “$cfg_hash” 
“${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}”
} >> “$ROOT/logs/v50_debug_log.md” || true

{
printf ‘{“ts”:”%s”,“action”:“dvc-pull”,“status”:“ok”,“remote”:”%s”,“rev”:”%s”,“jobs”:”%s”,“targets”:”%s”,“log”:”%s”}\n’ 
“$(ts)” “${REMOTE:-default}” “${REV:-current}” “${JOBS:-auto}” “${TARGETS[*]:-}” “$LOG_FILE”
} >> “$LOG_DIR/dvc-pull.audit.jsonl” || true

exit 0