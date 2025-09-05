#!/usr/bin/env bash

==============================================================================

🛰️ SpectraMind V50 — make-submission.sh (Mission-grade, upgraded ultimate)

——————————————————————————

Purpose:

Orchestrate a safe, reproducible submission flow:

1) selftest  →  2) predict  →  3) validate  →  4) bundle  →  (5) optional Kaggle submit



Principles:

• CLI-first (wraps the spectramind Typer CLI)

• Hydra-safe (no hard-coded params; pass via –config/–overrides/–extra)

• Reproducible (write a rich run manifest; record config hash, git SHA, tool versions)

• Safe-by-default (DRY-RUN skips final bundle & submit unless disabled)

• CI-hardened (deterministic, fail-fast, structured logs & exit codes)



Usage:

./bin/make-submission.sh [flags]



Common flags:

–dry-run | –no-dry-run       Keep / skip the final bundle + submit     (default: –dry-run)

–open                         Open the bundle (file/dir) or submissions/ when done

–tag                  Version tag for bundle & manifest

–config                 Hydra config (e.g., configs/config_v50.yaml)

–overrides “”     Quoted Hydra overrides (e.g., ‘+training.epochs=1 model=v50’)

–extra “”           Extra raw flags passed to spectramind subcommands

–pred-out <path.csv>          Predictions CSV path                       (default: outputs/predictions.csv)

–bundle-out       Bundle output (dir or .zip)                (default: submissions/bundle.zip)

–skip-selftest                Skip step 1 (selftest)

–skip-validate                Skip step 3 (validate)

–skip-bundle                  Skip step 4 (bundle) even if –no-dry-run

–timeout-sec N                Kill long-running sub-steps after N seconds (best-effort)

–manifest                     Also print the manifest path to stdout



Kaggle:

–kaggle-submit                Submit to Kaggle (requires kaggle CLI & –kaggle-comp)

–kaggle-comp            Kaggle competition slug (e.g., neurips-2025-ariel)

–kaggle-msg “”           Kaggle submission message

–kaggle-gzip                  Gzip CSV before submit (CSV → CSV.GZ)

–kaggle-json            Write kaggle-submit JSON summary to  (passes through)

–kaggle-poll N                Poll submissions page N times post-submit (best-effort)



Environment overrides:

SPECTRAMIND_CLI    (default: ‘spectramind’)

LOG_FILE           (default: logs/v50_debug_log.md)

KAGGLE_COMP        (default Kaggle slug if not given by flag)

KAGGLE_MSG         (default Kaggle message if not given by flag)



Logging:

• Appends a single structured line to $LOG_FILE (timestamp + cmd + git + cfg_hash + paths).

• Writes a rich JSON manifest under outputs/manifests/ with timings, sizes, hashes, env info.



Exit codes:

0  success

1  generic failure

2  usage / invalid arguments

3  self-test failed

4  validation failed

5  bundling failed

6  kaggle submit failed

==============================================================================

set -Eeuo pipefail
IFS=$’\n\t’

––––– Colors –––––

if [[ -t 1 ]]; then
BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; YLW=$’\033[33m’; CYN=$’\033[36m’; RST=$’\033[0m’
else
BOLD=’’; DIM=’’; RED=’’; GRN=’’; YLW=’’; CYN=’’; RST=’’
fi

die()   { printf “${RED}ERROR:${RST} %s\n” “$” >&2; exit 1; }
fail()  { printf “${RED}%s${RST}\n” “$” >&2; }
info()  { printf “${CYN}%s${RST}\n” “$”; }
ok()    { printf “${GRN}%s${RST}\n” “$”; }
warn()  { printf “${YLW}%s${RST}\n” “$*”; }
have()  { command -v “$1” >/dev/null 2>&1; }

––––– Defaults –––––

DRY_RUN=true
OPEN_AFTER=false
TAG=””
CONFIG=“configs/config_v50.yaml”
OVERRIDES=””
EXTRA=””
PRED_CSV=“outputs/predictions.csv”
BUNDLE_OUT=“submissions/bundle.zip”
SKIP_SELFTEST=false
SKIP_VALIDATE=false
SKIP_BUNDLE=false
TIMEOUT_SEC=0
KAGGLE_DO_SUBMIT=false
KAGGLE_COMP=”${KAGGLE_COMP:-}”
KAGGLE_MSG=”${KAGGLE_MSG:-}”
KAGGLE_GZIP=false
KAGGLE_JSON=””
KAGGLE_POLL=0
EMIT_MANIFEST_STDOUT=false

CLI=”${SPECTRAMIND_CLI:-spectramind}”
LOG_FILE=”${LOG_FILE:-logs/v50_debug_log.md}”

––––– Usage –––––

usage() {
sed -n ‘1,/^# ==============================================================================/{p}’ “$0” | sed ‘s/^# {0,1}//’
exit 0
}

––––– Utilities –––––

iso_ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
git_sha() { git rev-parse –short HEAD 2>/dev/null || echo “nogit”; }
hostname_s() { hostname 2>/dev/null || uname -n 2>/dev/null || echo “unknown-host”; }

open_path() {
local path=”$1”
if have xdg-open; then xdg-open “$path” >/dev/null 2>&1 || true
elif have open; then open “$path” >/dev/null 2>&1 || true
fi
}

bytes_of() {
local f=”$1”; stat -c%s – “$f” 2>/dev/null || stat -f%z – “$f” 2>/dev/null || echo 0
}

mb_of() {
local b=”${1:-0}”
echo $(( (b + 10241024 - 1) / (10241024) ))
}

sha256_of() {
local f=”$1”
if have sha256sum; then sha256sum – “$f” 2>/dev/null | awk ‘{print $1}’; 
elif have shasum; then shasum -a 256 – “$f” 2>/dev/null | awk ‘{print $1}’; 
else echo “”; fi
}

json_escape_py() { python3 - <<‘PY’ “$1”; import json,sys; print(json.dumps(sys.argv[1] if len(sys.argv)>1 else “”)); PY
}

cfg_hash() {

Best-effort: CLI may expose a config-hash printer.

if “$CLI” –help 2>/dev/null | grep -qiE – “–print-config-hash|hash-config”; then
if “$CLI” –print-config-hash >/dev/null 2>&1; then “$CLI” –print-config-hash 2>/dev/null || echo “-”
else “$CLI” hash-config 2>/dev/null || echo “-”
fi
else
echo “-”
fi
}

spectramind_version() {
if “$CLI” –version >/dev/null 2>&1; then “$CLI” –version 2>/dev/null | head -n1 || echo “unknown”
else echo “unknown”
fi
}

python_version() { python3 -V 2>/dev/null || echo “python3 (unknown)”; }

log_line() {
mkdir -p “$(dirname “$LOG_FILE”)”
printf ‘%s cmd=%s git=%s cfg_hash=%s tag=%s pred=%s bundle=%s notes=”%s”\n’ 
“$(iso_ts)” “make-submission” “$(git_sha)” “$1” “${TAG:-”-”}” “$2” “$3” “competition=${KAGGLE_COMP:-”-”}” 
>> “$LOG_FILE”
}

run_step() {

run_step  <cmd…>   (honors TIMEOUT_SEC if > 0)

local label=”$1”; shift
local start end rc
start=”$(date +%s)”
info “▶ ${label}”
set -x
if (( TIMEOUT_SEC > 0 )) && have timeout; then
timeout –preserve-status –signal=TERM “${TIMEOUT_SEC}” “$@” || rc=$?
else
“$@” || rc=$?
fi
set +x
rc=”${rc:-0}”
end=”$(date +%s)”
echo “${rc}|$(( end - start ))”
}

csv_quick_sanity() {

Basic CSV sanity: non-empty, header exists, sample consistent column count.

local f=”$1”
[[ -s “$f” ]] || die “Prediction CSV is empty: $f”
local header cols delim=”,”
header=”$(head -n 1 – “$f” || true)”
[[ -n “$header” ]] || die “Prediction CSV has empty header: $f”
if [[ “$header” == $’\t’ ]]; then delim=$’\t’; fi
cols=”$(awk -F”$delim” ‘NR==1{print NF; exit}’ “$f” || echo 0)”
if ! [[ “$cols” =~ ^[0-9]+$ ]] || (( cols < 2 )); then
warn “CSV header column count suspicious: $cols”
fi
local bad
bad=”$(awk -F”$delim” -v C=”$cols” ‘NR>1 && NR<=101 && NF!=C{print NR “:” NF; exit}’ “$f” || true)”
[[ -z “$bad” ]] || warn “CSV sample inconsistent columns near: $bad (expected $cols)”
}

––––– getopt parsing (robust) –––––

if have getopt; then
PARSED=$(getopt -o h –long help,dry-run,no-dry-run,open,tag:,config:,overrides:,extra:,pred-out:,bundle-out:,skip-selftest,skip-validate,skip-bundle,timeout-sec:,kaggle-submit,kaggle-comp:,kaggle-msg:,kaggle-gzip,kaggle-json:,kaggle-poll:,manifest – “$@”) || { usage; }
eval set – “$PARSED”
while true; do
case “$1” in
-h|–help) usage ;;
–dry-run) DRY_RUN=true; shift ;;
–no-dry-run) DRY_RUN=false; shift ;;
–open) OPEN_AFTER=true; shift ;;
–tag) TAG=”${2:-}”; shift 2 ;;
–config) CONFIG=”${2:-}”; shift 2 ;;
–overrides) OVERRIDES=”${2:-}”; shift 2 ;;
–extra) EXTRA=”${2:-}”; shift 2 ;;
–pred-out) PRED_CSV=”${2:-}”; shift 2 ;;
–bundle-out) BUNDLE_OUT=”${2:-}”; shift 2 ;;
–skip-selftest) SKIP_SELFTEST=true; shift ;;
–skip-validate) SKIP_VALIDATE=true; shift ;;
–skip-bundle) SKIP_BUNDLE=true; shift ;;
–timeout-sec) TIMEOUT_SEC=”${2:-0}”; shift 2 ;;
–kaggle-submit) KAGGLE_DO_SUBMIT=true; shift ;;
–kaggle-comp) KAGGLE_COMP=”${2:-}”; shift 2 ;;
–kaggle-msg) KAGGLE_MSG=”${2:-}”; shift 2 ;;
–kaggle-gzip) KAGGLE_GZIP=true; shift ;;
–kaggle-json) KAGGLE_JSON=”${2:-}”; shift 2 ;;
–kaggle-poll) KAGGLE_POLL=”${2:-0}”; shift 2 ;;
–manifest) EMIT_MANIFEST_STDOUT=true; shift ;;
–) shift; break ;;
*) die “Unknown option: $1” ;;
esac
done
else

Minimal fallback parser

while [ $# -gt 0 ]; do
case “$1” in
-h|–help) usage ;;
–dry-run) DRY_RUN=true ;;
–no-dry-run) DRY_RUN=false ;;
–open) OPEN_AFTER=true ;;
–tag) TAG=”${2:-}”; shift ;;
–config) CONFIG=”${2:-}”; shift ;;
–overrides) OVERRIDES=”${2:-}”; shift ;;
–extra) EXTRA=”${2:-}”; shift ;;
–pred-out) PRED_CSV=”${2:-}”; shift ;;
–bundle-out) BUNDLE_OUT=”${2:-}”; shift ;;
–skip-selftest) SKIP_SELFTEST=true ;;
–skip-validate) SKIP_VALIDATE=true ;;
–skip-bundle) SKIP_BUNDLE=true ;;
–timeout-sec) TIMEOUT_SEC=”${2:-0}”; shift ;;
–kaggle-submit) KAGGLE_DO_SUBMIT=true ;;
–kaggle-comp) KAGGLE_COMP=”${2:-}”; shift ;;
–kaggle-msg) KAGGLE_MSG=”${2:-}”; shift ;;
–kaggle-gzip) KAGGLE_GZIP=true ;;
–kaggle-json) KAGGLE_JSON=”${2:-}”; shift ;;
–kaggle-poll) KAGGLE_POLL=”${2:-0}”; shift ;;
–manifest) EMIT_MANIFEST_STDOUT=true ;;
*) die “Unknown option: $1” ;;
esac
shift
done
fi

––––– Setup & context –––––

mkdir -p “outputs” “$(dirname “$PRED_CSV”)” “$(dirname “$BUNDLE_OUT”)” “submissions” “$(dirname “$LOG_FILE”)”

RUN_TS=”$(iso_ts)”
GIT_SHA=”$(git_sha)”
RUN_HOST=”$(hostname_s)”
RUN_ID=”${RUN_TS}-${GIT_SHA}”

trap ‘fail “[make-submission] ❌ Failed at $(iso_ts) (RUN_ID=${RUN_ID})”; exit 1’ ERR

printf ‘%s\n’ “[make-submission] ========================================================”
printf ‘%s\n’ “[make-submission] Start   : $(iso_ts)”
printf ‘%s\n’ “[make-submission] RUN_ID  : ${RUN_ID}”
printf ‘%s\n’ “[make-submission] Host    : ${RUN_HOST}”
printf ‘%s\n’ “[make-submission] CLI     : ${CLI}”
printf ‘%s\n’ “[make-submission] DRYRUN  : ${DRY_RUN}”
[[ -n “$TAG” ]]       && printf ‘%s\n’ “[make-submission] TAG     : ${TAG}”
printf ‘%s\n’         “[make-submission] CONFIG  : ${CONFIG}”
[[ -n “$OVERRIDES” ]] && printf ‘%s\n’ “[make-submission] OVERR   : ${OVERRIDES}”
[[ -n “$EXTRA” ]]     && printf ‘%s\n’ “[make-submission] EXTRA   : ${EXTRA}”
[[ -n “$KAGGLE_COMP” ]] && printf ‘%s\n’ “[make-submission] KGL-CMP : ${KAGGLE_COMP}”

Tooling display (best-effort)

printf ‘%s\n’ “[make-submission] Py      : $(python_version)”
printf ‘%s\n’ “[make-submission] SM-CLI  : $(spectramind_version)”

Compute a config hash (best-effort)

CFG_HASH=”$(cfg_hash)”
[[ -n “$CFG_HASH” ]] && printf ‘%s\n’ “[make-submission] CFGHASH : ${CFG_HASH}”

––––– Guards –––––

command -v “$CLI” >/dev/null 2>&1 || die “Missing CLI: ${CLI}”
if [[ ! -f “$CONFIG” ]]; then
warn “Config not found: $CONFIG (continuing; CLI may resolve defaults)”
fi

––––– 1) Selftest –––––

SELFTEST_RC=0
SELFTEST_SEC=0
if ! $SKIP_SELFTEST; then
out=”$(run_step “Selftest (fast)” “$CLI” test –fast)”
SELFTEST_RC=”${out%%|}”; SELFTEST_SEC=”${out##|}”
if [[ “$SELFTEST_RC” -ne 0 ]]; then
fail  “Selftest failed (${SELFTEST_SEC}s)”
log_line “$CFG_HASH” “-”
exit 3
fi
ok    “Selftest OK (${SELFTEST_SEC}s)”
else
warn  “Skipping selftest per flag”
fi

––––– 2) Predict –––––

PRED_RC=0
PRED_SEC=0
mkdir -p “$(dirname “$PRED_CSV”)”
predict_args=( “$CLI” predict –config “$CONFIG” –out-csv “$PRED_CSV” )
[[ -n “$EXTRA” ]]     && predict_args+=( $EXTRA )
[[ -n “$OVERRIDES” ]] && predict_args+=( $OVERRIDES )

out=”$(run_step “Predict → $PRED_CSV” “${predict_args[@]}”)”
PRED_RC=”${out%%|}”; PRED_SEC=”${out##|}”
if [[ “$PRED_RC” -ne 0 ]]; then
fail “Predict failed (${PRED_SEC}s)”
log_line “$CFG_HASH” “-”
exit 1
fi
[[ -s “$PRED_CSV” ]] || { fail “Prediction CSV not produced: $PRED_CSV”; log_line “$CFG_HASH” “-”; exit 1; }
csv_quick_sanity “$PRED_CSV”
ok “Predictions ready (${PRED_SEC}s) → $PRED_CSV”

––––– 3) Validate –––––

VAL_RC=0
VAL_SEC=0
if ! $SKIP_VALIDATE; then
validate_args=( “$CLI” validate –input “$PRED_CSV” )
[[ -n “$EXTRA” ]] && validate_args+=( $EXTRA )
out=”$(run_step “Validate predictions” “${validate_args[@]}”)”
VAL_RC=”${out%%|}”; VAL_SEC=”${out##|}”
if [[ “$VAL_RC” -ne 0 ]]; then
fail “Validation failed (${VAL_SEC}s)”
log_line “$CFG_HASH” “$PRED_CSV”
exit 4
fi
ok “Validation OK (${VAL_SEC}s)”
else
warn “Skipping validation per flag”
fi

––––– 4) Bundle –––––

BUNDLE_RC=0
BUNDLE_SEC=0
BUNDLE_SAFE=”-”
if $DRY_RUN; then
warn  “Dry-run enabled — skipping bundle. Would run:”
printf ’  %q’ “$CLI” bundle –pred “$PRED_CSV” –out “$BUNDLE_OUT”; printf ‘\n’
[[ -n “$TAG”   ]] && printf ’  %q’ –tag “$TAG” && printf ‘\n’
[[ -n “$EXTRA” ]] && printf ’  EXTRA: %s\n’ “$EXTRA”
elif $SKIP_BUNDLE; then
warn “Bundle explicitly skipped by flag”
else
bundle_args=( “$CLI” bundle –pred “$PRED_CSV” –out “$BUNDLE_OUT” )
[[ -n “$TAG”   ]] && bundle_args+=( –tag “$TAG” )
[[ -n “$EXTRA” ]] && bundle_args+=( $EXTRA )
out=”$(run_step “Bundle → $BUNDLE_OUT” “${bundle_args[@]}”)”
BUNDLE_RC=”${out%%|}”; BUNDLE_SEC=”${out##|}”
if [[ “$BUNDLE_RC” -ne 0 ]]; then
fail “Bundling failed (${BUNDLE_SEC}s)”
log_line “$CFG_HASH” “$PRED_CSV” “-”
exit 5
fi
if [[ -f “$BUNDLE_OUT” ]]; then
[[ -s “$BUNDLE_OUT” ]] || { fail “Bundle appears empty: $BUNDLE_OUT”; log_line “$CFG_HASH” “$PRED_CSV” “$BUNDLE_OUT”; exit 5; }
fi
BUNDLE_SAFE=”$BUNDLE_OUT”
ok “Bundle complete (${BUNDLE_SEC}s)”
fi

––––– 5) Optional Kaggle submission –––––

KAGGLE_RC=0
KAGGLE_SEC=0
KAGGLE_JSON_LOCAL=””
if $KAGGLE_DO_SUBMIT; then
if $DRY_RUN; then
warn “Dry-run: Kaggle submission suppressed.”
else
if [[ -x “bin/kaggle-submit.sh” ]]; then
# Prefer our hardened helper if available
kaggle_submit_cmd=( “bin/kaggle-submit.sh” –comp “${KAGGLE_COMP:?–kaggle-comp required}” –file “$PRED_CSV” –message “${KAGGLE_MSG:-SpectraMind V50 submit ($RUN_ID)}” –yes )
$KAGGLE_GZIP   && kaggle_submit_cmd+=( –gzip )
[[ -n “$KAGGLE_JSON” ]] && { kaggle_submit_cmd+=( –json “$KAGGLE_JSON” ); KAGGLE_JSON_LOCAL=”$KAGGLE_JSON”; } || kaggle_submit_cmd+=( –json )
(( KAGGLE_POLL > 0 )) && kaggle_submit_cmd+=( –poll “$KAGGLE_POLL” )
out=”$(run_step “Kaggle submit (helper)” “${kaggle_submit_cmd[@]}”)”
KAGGLE_RC=”${out%%|}”; KAGGLE_SEC=”${out##|}”
else
# Fallback to Kaggle CLI
command -v kaggle >/dev/null 2>&1 || { fail “kaggle CLI not found”; log_line “$CFG_HASH” “$PRED_CSV” “$BUNDLE_SAFE”; exit 6; }
local_msg=”${KAGGLE_MSG:-SpectraMind V50 submit ($RUN_ID)}”
submit_file=”$PRED_CSV”
if $KAGGLE_GZIP; then
tmp_gz=”$(mktemp –suffix “.csv.gz” 2>/dev/null || mktemp “${TMPDIR:-/tmp}/smgz.XXXXXX.csv.gz”)”
gzip -c “$PRED_CSV” > “$tmp_gz”
submit_file=”$tmp_gz”
fi
out=”$(run_step “Kaggle submit (direct)” kaggle competitions submit -c “${KAGGLE_COMP:?–kaggle-comp required}” -f “$submit_file” -m “$local_msg”)”
KAGGLE_RC=”${out%%|}”; KAGGLE_SEC=”${out##|}”
[[ “${submit_file}” == *.gz && -f “${submit_file}” ]] && rm -f – “${submit_file}” || true
fi
if [[ “$KAGGLE_RC” -ne 0 ]]; then
fail “Kaggle submit failed (${KAGGLE_SEC}s)”
log_line “$CFG_HASH” “$PRED_CSV” “$BUNDLE_SAFE”
exit 6
fi
ok “Kaggle submit issued (${KAGGLE_SEC}s)”
fi
fi

––––– Structured CLI log line –––––

log_line “$CFG_HASH” “$PRED_CSV” “${BUNDLE_SAFE:-”-”}”

––––– Rich Manifest –––––

MANIFEST_DIR=“outputs/manifests”; mkdir -p “$MANIFEST_DIR”
MANIFEST_PATH=”$MANIFEST_DIR/run_manifest_${RUN_ID}.json”

Gather file stats

PRED_BYTES=”$(bytes_of “$PRED_CSV”)”
PRED_MB=”$(mb_of “$PRED_BYTES”)”
PRED_SHA256=”$(sha256_of “$PRED_CSV”)”
if [[ -f “$BUNDLE_OUT” ]]; then
BUNDLE_BYTES=”$(bytes_of “$BUNDLE_OUT”)”
BUNDLE_MB=”$(mb_of “$BUNDLE_BYTES”)”
BUNDLE_SHA256=”$(sha256_of “$BUNDLE_OUT”)”
else
BUNDLE_BYTES=0; BUNDLE_MB=0; BUNDLE_SHA256=””
fi

JSON string escapers

CFG_JSON=$(json_escape_py “$CONFIG”)
OVR_JSON=$(json_escape_py “$OVERRIDES”)
EXT_JSON=$(json_escape_py “$EXTRA”)
TAG_JSON=$(json_escape_py “$TAG”)
KMSG_JSON=$(json_escape_py “$KAGGLE_MSG”)
KCOMP_JSON=$(json_escape_py “$KAGGLE_COMP”)
KJSON_PATH=$(json_escape_py “$KAGGLE_JSON_LOCAL”)

cat > “$MANIFEST_PATH” <<JSON
{
“run”: {
“id”: “$(echo “$RUN_ID”)”,
“timestamp_utc”: “$(echo “$RUN_TS”)”,
“host”: “$(echo “$RUN_HOST”)”,
“git_sha”: “$(echo “$GIT_SHA”)”,
“cfg_hash”: “$(echo “$CFG_HASH”)”,
“dry_run”: $( $DRY_RUN && echo true || echo false ),
“tag”: $TAG_JSON
},
“tools”: {
“spectramind_cli”: “$(spectramind_version)”,
“python”: “$(python_version)”
},
“inputs”: {
“config”: $CFG_JSON,
“overrides”: $OVR_JSON,
“extra_args”: $EXT_JSON
},
“artifacts”: {
“pred_csv”: {
“path”: “$(echo “$PRED_CSV”)”,
“bytes”: $PRED_BYTES,
“mb”: $PRED_MB,
“sha256”: “$(echo “$PRED_SHA256”)”
},
“bundle”: {
“path”: “$(echo “$BUNDLE_OUT”)”,
“bytes”: $BUNDLE_BYTES,
“mb”: $BUNDLE_MB,
“sha256”: “$(echo “$BUNDLE_SHA256”)”,
“skipped”: $( $DRY_RUN || $SKIP_BUNDLE && echo true || echo false )
}
},
“timings_sec”: {
“selftest”: $SELFTEST_SEC,
“predict”: $PRED_SEC,
“validate”: $VAL_SEC,
“bundle”: $BUNDLE_SEC,
“kaggle”: $KAGGLE_SEC
},
“kaggle”: {
“will_submit”: $( $KAGGLE_DO_SUBMIT && echo true || echo false ),
“submitted”: $( (! $DRY_RUN && $KAGGLE_DO_SUBMIT && [[ “$KAGGLE_RC” -eq 0 ]]) && echo true || echo false ),
“gzip”: $( $KAGGLE_GZIP && echo true || echo false ),
“competition”: $KCOMP_JSON,
“message”: $KMSG_JSON,
“json_summary_path”: $KJSON_PATH,
“poll_count”: $KAGGLE_POLL
}
}
JSON
ok “Manifest: $MANIFEST_PATH”

––––– Optional concise manifest path to stdout –––––

if $EMIT_MANIFEST_STDOUT; then
printf ‘%s\n’ “$MANIFEST_PATH”
fi

––––– Optional: open dir/bundle –––––

if $OPEN_AFTER; then
if $DRY_RUN || $SKIP_BUNDLE; then
open_path “$(dirname “$BUNDLE_OUT”)”
else
if [[ -f “$BUNDLE_OUT” ]]; then open_path “$BUNDLE_OUT”; else open_path “$(dirname “$BUNDLE_OUT”)”; fi
fi
fi

ok   “[make-submission] Completed at $(iso_ts)  (RUN_ID=${RUN_ID})”
printf ‘%s\n’ “[make-submission] ========================================================”
exit 0