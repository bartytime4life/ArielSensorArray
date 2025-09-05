#!/usr/bin/env bash

==============================================================================

bin/kaggle-submit.sh — Safe, ergonomic Kaggle submission helper (ultimate)

——————————————————————————

Features

• DRY-RUN BY DEFAULT (use –yes to actually submit)

• Auto-detects submission file (multiple common locations or $KAGGLE_FILE)

• Validates: Kaggle CLI & auth, competition slug, file existence/size/type

• Optional gzip (and auto-compress .csv → .csv.gz) with temp file cleanup

• CSV sanity checks: header present, row/column counts, newline & NUL checks

• Nice logs, optional retries, optional polling of recent submissions

• Emits rich JSON summary (SHA256, rows/cols, env info) for CI dashboards

• Works locally and in Kaggle kernels (best-effort detection)

• Validate-only mode to fail-fast in CI without submitting



Usage

bin/kaggle-submit.sh [–comp SLUG] [–file PATH] [–message “msg”] [–yes]

[–retries N] [–sleep SEC] [–poll N] [–poll-sleep S]

[–open|–no-open] [–gzip] [–json [PATH]]

[–quiet] [–validate-only] [–max-mb N]



Examples

bin/kaggle-submit.sh –yes

bin/kaggle-submit.sh –comp neurips-2025-ariel \

–file outputs/predictions/submission.csv \

–message “V50 run #42” –gzip –yes –json submit.json



Notes

• Requires: kaggle CLI + valid auth (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR)

• Defaults: COMPETITION=$KAGGLE_COMP or neurips-2025-ariel

MESSAGE=$KAGGLE_MSG or “SpectraMind V50 auto-submit”

FILE    from $KAGGLE_FILE or auto-detected candidates

==============================================================================

set -Eeuo pipefail

—– Colors —————————————————————

if [[ -t 1 ]]; then
BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; YLW=$’\033[33m’; CYN=$’\033[36m’; RST=$’\033[0m’
else
BOLD=’’; DIM=’’; RED=’’; GRN=’’; YLW=’’; CYN=’’; RST=’’
fi

log()   { printf “%b\n” “${}”; }
info()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; log “${CYN}::${RST} ${}”; }
ok()    { [[ “${QUIET:-0}” -eq 1 ]] && return 0; log “${GRN}✓${RST} ${}”; }
warn()  { log “${YLW}⚠${RST} ${}”; }
err()   { log “${RED}✗${RST} ${}”; }
die()   { err “$”; exit 1; }
have()  { command -v “$1” >/dev/null 2>&1; }

—– Defaults ———————————————————––

COMPETITION=”${KAGGLE_COMP:-neurips-2025-ariel}”
SUBMIT_FILE=”${KAGGLE_FILE:-}”
MESSAGE=”${KAGGLE_MSG:-SpectraMind V50 auto-submit}”
YES=0
RETRIES=”${KAGGLE_RETRIES:-0}”
SLEEP=”${KAGGLE_SLEEP:-15}”
OPEN=1
GZIP=0
QUIET=0
JSON=0
JSON_PATH=””
VALIDATE_ONLY=0
POLL=0
POLL_SLEEP=15
MAX_MB=512  # soft cap warning (Kaggle allows big files but we warn by default)
TMP_GZ=””

—– Cleanup temp artifacts –––––––––––––––––––––––

cleanup() {
if [[ -n “${TMP_GZ}” && -f “${TMP_GZ}” ]]; then
rm -f – “${TMP_GZ}” || true
fi
}
trap cleanup EXIT INT TERM

—– Helpers –––––––––––––––––––––––––––––––

usage() {
cat <<EOF
${BOLD}kaggle-submit.sh${RST} — submit a CSV to a Kaggle competition (safe by default)

${BOLD}Options${RST}
–comp SLUG          Kaggle competition slug (default: ${COMPETITION})
–file PATH          Path to submission CSV (auto-detected if omitted)
–message TEXT       Submission message (default: “${MESSAGE}”)
–yes                Actually submit (DRY-RUN by default)
–retries N          Retries if Kaggle CLI transiently fails (default: ${RETRIES})
–sleep SEC          Sleep between retries (default: ${SLEEP})
–gzip               Gzip the submission (.csv → .csv.gz) before upload
–open | –no-open   Open competition submissions page after submit (default: open)
–json [PATH]        Emit JSON summary to stdout, or to PATH if provided
–validate-only      Validate environment + file and exit 0 (no submit)
–poll N             After submit, poll submissions list N times (best-effort)
–poll-sleep SEC     Sleep between polls (default: ${POLL_SLEEP})
–max-mb N           Warn if file size exceeds N MB (default: ${MAX_MB})
–quiet              Minimal logs
-h, –help           Show help

${BOLD}Auto-detect CSV (first match wins)${RST}
• predictions/submission.csv
• outputs/predictions/submission.csv
• outputs/submission.csv
• outputs/submission/submission.csv
• submission.csv

Examples:
bin/kaggle-submit.sh –yes
bin/kaggle-submit.sh –comp neurips-2025-ariel –file outputs/predictions/submission.csv –message “V50 run #42” –yes –json submit.json
EOF
}

detect_submit_file() {
local candidates=(
“predictions/submission.csv”
“outputs/predictions/submission.csv”
“outputs/submission.csv”
“outputs/submission/submission.csv”
“submission.csv”
)
for f in “${candidates[@]}”; do
if [[ -f “$f” ]]; then SUBMIT_FILE=”$f”; return 0; fi
done
return 1
}

kaggle_authenticated() {
if [[ -n “${KAGGLE_CONFIG_DIR:-}” ]]; then
[[ -f “${KAGGLE_CONFIG_DIR%/}/kaggle.json” ]]
else
[[ -f “$HOME/.kaggle/kaggle.json” ]]
fi
}

json_escape() {

Minimal JSON string escaper for double quotes and backslashes

shellcheck disable=SC2001

sed -e ‘s/\/\\/g’ -e ‘s/”/\”/g’
}

emit_json() {
local ok_flag=”$1”
local attempts=”$2”
local dry_run_flag
dry_run_flag=”$([[ $YES -eq 1 ]] && echo false || echo true)”

local ts host cwd sha rows cols fsize_bytes ftype sha_short
ts=”$(date -u +”%Y-%m-%dT%H:%M:%SZ”)”
host=”$(hostname || echo unknown)”
cwd=”$(pwd)”
sha=”$( { have sha256sum && sha256sum – “${SUBMIT_FILE}”; } 2>/dev/null | awk ‘{print $1}’ )”
sha_short=”${sha:0:12}”
rows=”${CSV_ROWS:-0}”
cols=”${CSV_COLS:-0}”
fsize_bytes=”${FILE_BYTES:-0}”
ftype=”${FILE_TYPE:-unknown}”

Build JSON string

local json
json=$(
printf ‘{’
printf ’“ok”: %s, ’          “${ok_flag}”
printf ’“attempts”: %s, ’     “${attempts:-0}”
printf ’“dry_run”: %s, ’      “${dry_run_flag}”
printf ’“timestamp”: “%s”, ’  “${ts}”
printf ’“competition”: “%s”, ’ “$(printf “%s” “${COMPETITION}” | json_escape)”
printf ’“file”: “%s”, ’       “$(printf “%s” “${SUBMIT_FILE}” | json_escape)”
printf ’“file_bytes”: %s, ’   “${fsize_bytes}”
printf ’“file_type”: “%s”, ’  “${ftype}”
printf ’“sha256”: “%s”, ’     “${sha}”
printf ’“sha256_short”: “%s”, ’ “${sha_short}”
printf ’“message”: “%s”, ’    “$(printf “%s” “${MESSAGE}” | json_escape)”
printf ’“csv_rows”: %s, ’     “${rows}”
printf ’“csv_cols”: %s, ’     “${cols}”
printf ’“kaggle_env”: %s, ’   “$([[ $IS_KAGGLE -eq 1 ]] && echo true || echo false)”
printf ’“hostname”: “%s”, ’   “$(printf “%s” “${host}” | json_escape)”
printf ‘“cwd”: “%s”’          “$(printf “%s” “${cwd}” | json_escape)”
printf ‘}\n’
)

if [[ $JSON -eq 1 ]]; then
if [[ -n “${JSON_PATH}” ]]; then
printf “%s” “${json}” > “${JSON_PATH}”
ok “Wrote JSON summary → ${BOLD}${JSON_PATH}${RST}”
else
printf “%s\n” “${json}”
fi
fi
}

print_recent_submissions() {
[[ $QUIET -eq 1 ]] && return 0
info “Recent submissions (top 6):”
if ! kaggle competitions submissions -c “$COMPETITION” -v 2>/dev/null | head -n 7; then
warn “Could not list recent submissions (permission or CLI issue).”
fi
}

poll_recent_submissions() {
local i=0
while (( i < POLL )); do
sleep “${POLL_SLEEP}”
print_recent_submissions
i=$((i+1))
done
}

open_competition_page() {
local url=“https://www.kaggle.com/competitions/${COMPETITION}/submissions”
[[ $OPEN -eq 1 ]] || return 0
if have xdg-open; then xdg-open “$url” >/dev/null 2>&1 || true
elif have open; then open “$url” >/dev/null 2>&1 || true
fi
}

submit_once() {
local -r comp=”$1” file=”$2” msg=”$3”
if [[ $QUIET -eq 1 ]]; then
kaggle competitions submit -c “$comp” -f “$file” -m “$msg” >/dev/null
else
kaggle competitions submit -c “$comp” -f “$file” -m “$msg”
fi
}

bytes_to_mb() {

prints integer MB; avoids bc dependency

local b=”${1:-0}”
echo $(( (b + 10241024 - 1) / (10241024) ))
}

csv_sanity_checks() {

Basic CSV validations: header present, consistent column count (best-effort),

no NUL bytes, ends with newline. Stores CSV_ROWS/CSV_COLS globals.

local f=”$1”

NUL byte check

if LC_ALL=C tr -d ‘\n’ < “$f” | grep -q $’\x00’; then
die “File contains NUL bytes (not a plain-text CSV): ${f}”
fi

Has header?

local header
header=”$(head -n 1 – “$f” || true)”
if [[ -z “$header” ]]; then
die “CSV seems to have an empty header row.”
fi

Column count of header (comma-separated). Support both comma/tab (just in case).

local delim=”,”
if [[ “$header” == $’\t’ ]]; then delim=$’\t’; fi
local cols
cols=”$(awk -F”$delim” ‘NR==1{print NF; exit}’ “$f”)”
if ! [[ “$cols” =~ ^[0-9]+$ ]] || (( cols < 2 )); then
warn “Header column count looks suspicious: ‘${cols}’”
fi

Row count

local rows
if have wc; then
rows=”$(wc -l < “$f” | tr -d ’ ’)”
else
rows=”$(awk ‘END{print NR}’ “$f”)”
fi
if ! [[ “$rows” =~ ^[0-9]+$ ]] || (( rows < 2 )); then
warn “CSV has very few rows (${rows}). Did you generate predictions?”
fi

Last line ends with newline?

if [[ -n “$(tail -c 1 – “$f” || true)” ]]; then
warn “CSV does not end with a newline. Kaggle usually accepts it, but it’s best to add one.”
fi

Quick consistency sample: check first 100 lines share the same NF

local bad_sample
bad_sample=”$(awk -F”$delim” -v C=”$cols” ‘NR>1 && NR<=101 && NF!=C{print NR “:” NF; exit}’ “$f” || true)”
if [[ -n “$bad_sample” ]]; then
warn “CSV sample shows inconsistent column counts near: ${bad_sample} (expected ${cols})”
fi

CSV_ROWS=”${rows}”
CSV_COLS=”${cols}”
}

file_type_of() {

Prefer ‘file’ if present; fallback to extension

local f=”$1” t=“unknown”
if have file; then
t=”$(file -b –mime-type – “$f” 2>/dev/null || echo “unknown”)”
else
case “$f” in
*.csv)   t=“text/csv” ;;
*.gz)    t=“application/gzip” ;;
*)       t=“unknown” ;;
esac
fi
printf “%s” “${t}”
}

—– Parse args ———————————————————–

while [[ $# -gt 0 ]]; do
case “$1” in
–comp)         COMPETITION=”${2:?}”; shift 2 ;;
–file)         SUBMIT_FILE=”${2:?}”; shift 2 ;;
–message)      MESSAGE=”${2:?}”; shift 2 ;;
–yes)          YES=1; shift ;;
–retries)      RETRIES=”${2:-0}”; shift 2 ;;
–sleep)        SLEEP=”${2:-15}”; shift 2 ;;
–open)         OPEN=1; shift ;;
–no-open)      OPEN=0; shift ;;
–gzip)         GZIP=1; shift ;;
–json)         JSON=1; JSON_PATH=”${2:-}”; if [[ -n “${JSON_PATH}” && “${JSON_PATH}” == –* ]]; then JSON_PATH=””; fi; [[ -n “${JSON_PATH}” ]] && shift 2 || shift 1 ;;
–quiet)        QUIET=1; shift ;;
–validate-only)VALIDATE_ONLY=1; shift ;;
–poll)         POLL=”${2:-0}”; shift 2 ;;
–poll-sleep)   POLL_SLEEP=”${2:-15}”; shift 2 ;;
–max-mb)       MAX_MB=”${2:-512}”; shift 2 ;;
-h|–help)      usage; exit 0 ;;
*)              err “Unknown option: $1”; usage; exit 2 ;;
esac
done

—– Kaggle kernel detection (best-effort) ———————————

IS_KAGGLE=0
if [[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]]; then
IS_KAGGLE=1
info “Kaggle environment detected.”
fi

—– Validations –––––––––––––––––––––––––––––

[[ -n “$COMPETITION” ]] || die “Competition slug is empty (use –comp).”

have kaggle || die “kaggle CLI not found. Install via ‘pip install kaggle’ and auth.”
kaggle_authenticated || die “Kaggle auth not found (~/.kaggle/kaggle.json or $KAGGLE_CONFIG_DIR). See Kaggle > Account > Create API Token.”

Determine file if not provided

if [[ -z “$SUBMIT_FILE” ]]; then
if ! detect_submit_file; then
die “No submission file found. Provide –file PATH or ensure submission.csv exists in a common location.”
fi
fi
[[ -f “$SUBMIT_FILE” ]] || die “File not found: $SUBMIT_FILE”
[[ -s “$SUBMIT_FILE” ]] || die “Submission file is empty: $SUBMIT_FILE”

Compute file info

FILE_BYTES=”$(stat -c%s – “$SUBMIT_FILE” 2>/dev/null || stat -f%z – “$SUBMIT_FILE” 2>/dev/null || echo 0)”
FILE_MB=”$(bytes_to_mb “${FILE_BYTES}”)”
FILE_TYPE=”$(file_type_of “${SUBMIT_FILE}”)”

if (( FILE_MB > MAX_MB )); then
warn “File size ${FILE_MB} MB exceeds soft cap (${MAX_MB} MB). Kaggle may still accept it; consider gzip.”
fi

Optional gzip

ORIG_FILE=”$SUBMIT_FILE”
if [[ $GZIP -eq 1 ]]; then
if [[ “$SUBMIT_FILE” =~ .csv$ ]]; then
TMP_GZ=”$(mktemp –suffix “.csv.gz” 2>/dev/null || mktemp “${TMPDIR:-/tmp}/kaggle.XXXXXX.csv.gz”)”
info “Gzipping submission → ${BOLD}${TMP_GZ}${RST}”
gzip -c “$SUBMIT_FILE” > “$TMP_GZ”
SUBMIT_FILE=”$TMP_GZ”
FILE_BYTES=”$(stat -c%s – “$SUBMIT_FILE” 2>/dev/null || stat -f%z – “$SUBMIT_FILE” 2>/dev/null || echo 0)”
FILE_MB=”$(bytes_to_mb “${FILE_BYTES}”)”
FILE_TYPE=”$(file_type_of “${SUBMIT_FILE}”)”
else
warn “Skipping –gzip: file does not end with .csv (${SUBMIT_FILE}).”
fi
fi

Enrich default message if untouched

if [[ “$MESSAGE” == “SpectraMind V50 auto-submit” ]]; then
if [[ -f “VERSION” ]]; then
V=$(sed -n ‘1s/[[:space:]]//gp’ VERSION || true)
[[ -n “$V” ]] && MESSAGE=“SpectraMind V50 ${V}”
elif have git; then
SHA=$(git rev-parse –short HEAD 2>/dev/null || true)
[[ -n “$SHA” ]] && MESSAGE=“SpectraMind V50 ${SHA}”
fi
fi

CSV sanity checks (only on textual CSV; if gz, we skip deep checks)

if [[ “${SUBMIT_FILE}” != *.gz ]]; then
csv_sanity_checks “${SUBMIT_FILE}”
else
info “Skipping CSV content checks for gzip file.”
fi

Show context

if [[ $QUIET -ne 1 ]]; then
info “Competition : ${BOLD}${COMPETITION}${RST}”
info “File       : ${BOLD}${SUBMIT_FILE}${RST} (${FILE_TYPE}, ${FILE_MB} MB)”
info “Message    : ${BOLD}${MESSAGE}${RST}”
info “Mode       : ${BOLD}$([[ $YES -eq 1 ]] && echo “SUBMIT” || echo “DRY-RUN”)${RST}”
if [[ “${SUBMIT_FILE}” != *.gz ]]; then
info “CSV stats  : rows=${CSV_ROWS:-?} cols=${CSV_COLS:-?}”
fi
print_recent_submissions
fi

Validate-only path

if [[ $VALIDATE_ONLY -eq 1 ]]; then
ok “Validation-only mode passed. No submission performed.”
emit_json true 0
exit 0
fi

Dry-run guard

if [[ $YES -ne 1 ]]; then
warn “DRY-RUN only (no submission). Use ${BOLD}–yes${RST} to submit.”
emit_json false 0
exit 0
fi

—– Submit with optional retries ––––––––––––––––––––

attempt=0
while :; do
attempt=$((attempt+1))
info “Submitting (attempt ${attempt})…”
if submit_once “$COMPETITION” “$SUBMIT_FILE” “$MESSAGE”; then
ok “Submitted to ${COMPETITION}.”
open_competition_page
print_recent_submissions
(( POLL > 0 )) && poll_recent_submissions
emit_json true “$attempt”
exit 0
fi
if (( attempt > RETRIES )); then
emit_json false “$attempt”
die “Submission failed after ${RETRIES} retries.”
fi
warn “Submission failed. Retrying in ${SLEEP}s…”
sleep “${SLEEP}”
done

End of file