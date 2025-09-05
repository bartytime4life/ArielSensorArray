#!/usr/bin/env bash

==============================================================================

bin/analyze-log.sh — Parse & summarize SpectraMind V50 CLI log(s) (ultimate)

——————————————————————————

What it does

• Parses logs/v50_debug_log.md (and/or additional inputs) into CSV + Markdown

• Optionally calls spectramind analyze-log first (if available) to refresh

• Accepts multiple files or directories; auto-discovers .md,.log,*.csv in dirs

• Robust filters: –since ISO filter, –tail N, –filter-{cmd,tag,notes} 

• De-duplicate by (cmd,git,cfg_hash,tag,pred,bundle) keeping the most recent

• Summaries by cmd|git_sha|cfg|day|hour (choose one or comma-list)

• Emits JSON summary (stdout or file), is dry-run/CI/Kaggle safe, no interactivity



Usage

bin/analyze-log.sh [options] [–] [LOG_OR_DIR …]



Options

–outdir        Output directory; sets defaults for –md/–csv (default: outputs)

–md           Output Markdown path                         (default: outputs/log_table.md)

–csv          Output CSV path                              (default: outputs/log_table.csv)

–log          Primary input log path                       (default: logs/v50_debug_log.md)

–since         Lower bound (e.g., 2025-08-01 or 2025-08-01T00:00:00Z)

–tail            Only keep last N entries after filtering

–clean              De-duplicate by (cmd,git,cfg_hash,tag,pred,bundle)

–group-by     Summary by one or more of: cmd,git_sha,cfg,day,hour  (comma-separated)

–no-summary         Skip all summary sections in Markdown

–title        Markdown table title (default: “SpectraMind V50 — CLI Calls”)

–md-limit        Limit rows shown in Markdown table (0=all)   (default: 0)

–open               Open generated Markdown (non-Kaggle/CI only)

–no-poetry          Do not use Poetry to invoke spectramind

–no-refresh         Do NOT invoke spectramind analyze-log first

–json               Emit JSON summary to stdout

–json-path    Write JSON summary to this file instead of stdout

–timeout       Timeout per CLI step if timeout exists     (default: 120)

–filter-cmd     Keep rows whose cmd matches regex

–filter-tag     Keep rows whose tag matches regex

–filter-notes   Keep rows whose notes match regex

–dry-run            Print actions, do not write files

–quiet              Less chatty

-h|–help            Show help and exit



Notes

• Works even if the CLI isn’t installed; pure-awk parsing fallback is built in.

• Safe on Kaggle/CI (no open calls unless –open and opener available locally).

• For directories passed as inputs, this script scans recursively for .md,.log,*.csv.

• CSV inputs are expected to follow header: ts,cmd,git_sha,cfg_hash,tag,pred,bundle,notes

(extra columns are ignored; missing are tolerated where possible).

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– Pretty –––––

BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; CYN=$’\033[36m’; YLW=$’\033[33m’; RST=$’\033[0m’
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[ANALYZE]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[ANALYZE]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[ANALYZE]%s %s\n’ “${RED}” “${RST}” “$” >&2; }
die()  { fail “$”; exit 1; }

––––– Defaults –––––

OUTDIR=“outputs”
MD_OUT=””
CSV_OUT=””
LOG_IN=“logs/v50_debug_log.md”
SINCE=””
TAIL_N=””
CLEAN=0
GROUP_BY=””
DO_SUMMARY=1
TITLE=“SpectraMind V50 — CLI Calls”
OPEN_AFTER=0
USE_POETRY=1
REFRESH=1
QUIET=0
JSON_OUT=0
JSON_PATH=””
STEP_TIMEOUT=”${ANALYZE_TIMEOUT:-120}”
DRY=0
MD_LIMIT=0
FILTER_CMD=””
FILTER_TAG=””
FILTER_NOTES=””
EXTRA_INPUTS=()

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; }

––––– Args –––––

while [[ $# -gt 0 ]]; do
case “$1” in
–outdir)        OUTDIR=”${2:?}”; shift 2 ;;
–md)            MD_OUT=”${2:?}”; shift 2 ;;
–csv)           CSV_OUT=”${2:?}”; shift 2 ;;
–log)           LOG_IN=”${2:?}”; shift 2 ;;
–since)         SINCE=”${2:?}”; shift 2 ;;
–tail)          TAIL_N=”${2:?}”; shift 2 ;;
–clean)         CLEAN=1; shift ;;
–group-by)      GROUP_BY=”${2:?}”; shift 2 ;;
–no-summary)    DO_SUMMARY=0; shift ;;
–title)         TITLE=”${2:?}”; shift 2 ;;
–md-limit)      MD_LIMIT=”${2:?}”; shift 2 ;;
–open)          OPEN_AFTER=1; shift ;;
–no-poetry)     USE_POETRY=0; shift ;;
–no-refresh)    REFRESH=0; shift ;;
–json)          JSON_OUT=1; shift ;;
–json-path)     JSON_PATH=”${2:?}”; shift 2 ;;
–timeout)       STEP_TIMEOUT=”${2:?}”; shift 2 ;;
–filter-cmd)    FILTER_CMD=”${2:?}”; shift 2 ;;
–filter-tag)    FILTER_TAG=”${2:?}”; shift 2 ;;
–filter-notes)  FILTER_NOTES=”${2:?}”; shift 2 ;;
–dry-run)       DRY=1; shift ;;
–quiet)         QUIET=1; shift ;;
-h|–help)       usage; exit 0 ;;
–)              shift; while [[ $# -gt 0 ]]; do EXTRA_INPUTS+=(”$1”); shift; done ;;
*)               EXTRA_INPUTS+=(”$1”); shift ;;
esac
done

––––– Defaults depending on OUTDIR –––––

[[ -z “$MD_OUT”  ]] && MD_OUT=”${OUTDIR%/}/log_table.md”
[[ -z “$CSV_OUT” ]] && CSV_OUT=”${OUTDIR%/}/log_table.csv”

––––– Repo root –––––

if command -v git >/dev/null 2>&1; then
if ROOT=$(git rev-parse –show-toplevel 2>/dev/null); then cd “$ROOT”; fi
fi

mkdir -p “$(dirname “$MD_OUT”)” “$(dirname “$CSV_OUT”)” || true

––––– Env detect –––––

IS_KAGGLE=0
[[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=1
IS_CI=0
[[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=1

––––– Helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }
with_timeout() {
if have timeout && [[ “${STEP_TIMEOUT:-0}” -gt 0 ]]; then
timeout –preserve-status –signal=TERM “${STEP_TIMEOUT}” “$@”
else
“$@”
fi
}
norm_newlines() { tr -d ‘\r’; }  # strip CR for CRLF files
is_dir() { [[ -d “$1” ]]; }
is_file() { [[ -f “$1” ]]; }

––––– Pick CLI –––––

CLI_BIN=””
if [[ “$USE_POETRY” -eq 1 ]] && have poetry; then
CLI_BIN=“poetry run spectramind”
elif have spectramind; then
CLI_BIN=“spectramind”
elif have python3; then
CLI_BIN=“python3 -m spectramind”
else
CLI_BIN=””
fi

––––– Optional refresh via CLI –––––

if [[ “$REFRESH” -eq 1 && “$DRY” -eq 0 && -n “$CLI_BIN” ]]; then
say “Refreshing via CLI: `$CLI_BIN analyze-log` (if supported)…”
if with_timeout bash -lc “$CLI_BIN analyze-log –md ‘$MD_OUT’ –csv ‘$CSV_OUT’”; then
say “spectramind analyze-log completed.”
else
warn “CLI refresh unavailable or failed; continuing with native parser.”
fi
fi

––––– Build input list –––––

INPUTS=()
append_path() {
local p=”$1”
if is_dir “$p”; then
while IFS= read -r -d ‘’ f; do INPUTS+=(”$f”); done < <(find “$p” -type f -iname “.md” -o -iname “.log” -o -iname “*.csv” -print0 | sort -z)
elif is_file “$p”; then
INPUTS+=(”$p”)
fi
}
append_path “$LOG_IN”
for X in “${EXTRA_INPUTS[@]:-}”; do append_path “$X”; done

No inputs? Try existing CSV

if [[ ${#INPUTS[@]} -eq 0 ]]; then
if [[ -f “$CSV_OUT” ]]; then
say “No raw inputs; using existing CSV as source.”
INPUTS+=(”$CSV_OUT”)
else
die “No input logs found and no existing CSV present.”
fi
fi

––––– Temp workspace –––––

TMPDIR=”$(mktemp -d -t sm_anlz_XXXX)”
trap ‘rm -rf “$TMPDIR” 2>/dev/null || true’ EXIT
TMP_CSV=”${TMPDIR}/combined.csv”

––––– CSV header –––––

CSV_HEADER=“ts,cmd,git_sha,cfg_hash,tag,pred,bundle,notes”

––––– Emit CSV rows from a Markdown/Log line –––––

Expected schema in .md lines:

[ISO8601] cmd= git= cfg_hash= tag=<tag_or_-> pred=<path_or_-> bundle=<path_or_-> notes=”…”

This awk will:

• tolerate missing keys / extra whitespace

• sanitize commas in notes (→ ‘;’) to keep CSV column count stable

• quote cmd and notes for safety (others are assumed no commas)

parse_md_to_csv() {
awk ’
BEGIN {
OFS=”,”
}
function trim(s){sub(/^[ \t\r\n]+/,””,s); sub(/[ \t\r\n]+$/,””,s); return s}
function get(k, s,  r, x) {
if (k==“notes”) {
r = “notes="[^"]*"”
if (match(s, r, x)) {
val = x[0]
sub(/^notes=”/, “”, val)
sub(/”$/, “”, val)
gsub(/[\r\n]/, “ “, val)
gsub(/,/, “;”, val)   # keep CSV sane
gsub(/”/, “""”, val)
return val
}
return “”
} else {
r = “(^|[[:space:]])” k “=[^[:space:]]+”
if (match(s, r, x)) { split(x[0], a, “=”); return a[2] }
return “”
}
}
/^\[/ {
line = $0
# timestamp: [ISO…]
if (match(line, /^\[([^\]]+)]/, T)) {
ts = T[1]
cmd = trim(get(“cmd”, line))
git = trim(get(“git”, line))
cfg = trim(get(“cfg_hash”, line))
tag = trim(get(“tag”, line))
pred = trim(get(“pred”, line))
bundle = trim(get(“bundle”, line))
notes = get(“notes”, line)
# CSV: quote cmd + notes, other fields assumed no commas
gsub(/”/,”""”, cmd)
printf “%s,"%s",%s,%s,%s,%s,%s,"%s"\n”, ts, cmd, git, cfg, tag, pred, bundle, notes
}
}
’
}

––––– Normalize & ingest inputs into CSV –––––

say “Collecting inputs (${#INPUTS[@]} file/s)…”
{
echo “$CSV_HEADER”
for f in “${INPUTS[@]}”; do
case “${f##*.}” in
csv|CSV)
say “ ingest CSV: $f”
# Skip header; pass through first 8 columns; sanitize CR
norm_newlines < “$f” 
| awk -F, ‘NR==1{next} {
# If more than 8 columns, keep first 8. If less, pad.
out=””;
for(i=1;i<=8;i++){
if(i<=NF){ out = out ((i==1)?””:”,”) $i }
else { out = out ((i==1)?””:”,”) “” }
}
print out
}’
;;
md|MD|log|LOG|txt|TXT)
say “ ingest MD/LOG: $f”
norm_newlines < “$f” | parse_md_to_csv
;;
*)
say “ skip (unknown type): $f”
;;
esac
done
} > “$TMP_CSV”

––––– Empty check –––––

if [[ “$(wc -l < “$TMP_CSV”)” -le 1 ]]; then
die “No rows parsed from inputs.”
fi

––––– Filter by –since (ISO string compare is OK) –––––

if [[ -n “$SINCE” ]]; then
say “Filtering rows since ${SINCE}…”
awk -F, -v since=”$SINCE” ‘NR==1{print;next} { if($1>=since) print }’ “$TMP_CSV” > “${TMPDIR}/flt.csv”
mv “${TMPDIR}/flt.csv” “$TMP_CSV”
fi

––––– Additional regex filters –––––

if [[ -n “$FILTER_CMD” || -n “$FILTER_TAG” || -n “$FILTER_NOTES” ]]; then
say “Applying regex filters…”
awk -F, -v rc=”${FILTER_CMD:-}” -v rt=”${FILTER_TAG:-}” -v rn=”${FILTER_NOTES:-}” ’
BEGIN {
has_rc = (length(rc)>0)
has_rt = (length(rt)>0)
has_rn = (length(rn)>0)
}
NR==1 { print; next }
{
cmd=$2; tag=$5; notes=$8
# strip surrounding quotes for cmd/notes if any
gsub(/^”/,””,cmd); gsub(/”$/,””,cmd)
gsub(/^”/,””,notes); gsub(/”$/,””,notes)
ok=1
if (has_rc && cmd !~ rc) ok=0
if (has_rt && tag !~ rt) ok=0
if (has_rn && notes !~ rn) ok=0
if (ok) print
}
’ “$TMP_CSV” > “${TMPDIR}/flt2.csv”
mv “${TMPDIR}/flt2.csv” “$TMP_CSV”
fi

––––– Tail N –––––

if [[ -n “${TAIL_N:-}” ]]; then
say “Keeping last $TAIL_N rows…”
(head -n1 “$TMP_CSV” && tail -n +2 “$TMP_CSV” | tail -n “$TAIL_N”) > “${TMPDIR}/tail.csv”
mv “${TMPDIR}/tail.csv” “$TMP_CSV”
fi

––––– De-dup by (cmd,git,cfg,tag,pred,bundle) keeping most recent –––––

if [[ “$CLEAN” -eq 1 ]]; then
say “De-duplicating rows by (cmd,git,cfg_hash,tag,pred,bundle)…”
awk -F, ’
NR==1 { header=$0; next }
{
key=$2 FS $3 FS $4 FS $5 FS $6 FS $7
last[key]=NR
line[NR]=$0
}
END{
print header
for (i=2;i<=NR;i++){
key_field = “”
# reconstruct key for row i
split(line[i],f,/,/)
# safer: rebuild with current FS
split(line[i], col, FS)
key = col[2] FS col[3] FS col[4] FS col[5] FS col[6] FS col[7]
if (last[key]==i) print line[i]
}
}
’ “$TMP_CSV” > “${TMPDIR}/dedup.csv”
mv “${TMPDIR}/dedup.csv” “$TMP_CSV”
fi

––––– Write CSV –––––

if [[ $DRY -eq 1 ]]; then
say “[dry-run] Would write CSV → $CSV_OUT”
else
say “Writing CSV → $CSV_OUT”
cp “$TMP_CSV” “$CSV_OUT”
fi

––––– Markdown table –––––

if [[ $DRY -eq 1 ]]; then
say “[dry-run] Would write Markdown → $MD_OUT”
else
say “Generating Markdown → $MD_OUT”
NOW_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ || true)
{
printf “# %s\n\n” “$TITLE”
printf “Generated: %s\n\n” “$NOW_UTC”
echo “| time | cmd | git_sha | cfg_hash | tag | pred | bundle | notes |”
echo “|—|—|—|—|—|—|—|—|”
ROWS_WRITTEN=0
LIMIT=”${MD_LIMIT:-0}”
tail -n +2 “$CSV_OUT” 
| awk -F, ’
{
t=$1; c=$2; g=$3; cf=$4; tg=$5; pr=$6; bu=$7; no=$8
gsub(/^”/,””,c); gsub(/”$/,””,c);
gsub(/^”/,””,no); gsub(/”$/,””,no);
# sanitize pipes to keep Markdown table intact
gsub(/|/,”/”,c); gsub(/|/,”/”,no); gsub(/|/,”/”,pr); gsub(/|/,”/”,bu);
# short git hash
if (length(g)>12) { g=substr(g,1,12) “…” }
printf “| %s | %s | %s | %s | %s | %s | %s | %s |\n”, t, c, g, cf, tg, pr, bu, no
}
’
| while IFS= read -r line; do
if [[ “$LIMIT” -gt 0 && “$ROWS_WRITTEN” -ge “$LIMIT” ]]; then
remaining=$(( $(wc -l < “$CSV_OUT”) - 1 - ROWS_WRITTEN ))
printf “\n_… (%d more rows not shown — see CSV)_\n” “$remaining”
break
fi
echo “$line”
ROWS_WRITTEN=$(( ROWS_WRITTEN + 1 ))
done

if [[ "$DO_SUMMARY" -eq 1 && -n "${GROUP_BY:-}" ]]; then
  IFS=',' read -r -a GB_ARR <<< "$GROUP_BY"
  for GB in "${GB_ARR[@]}"; do
    GB_TRIM="${GB//[[:space:]]/}"
    [[ -z "$GB_TRIM" ]] && continue
    echo
    printf "## Summary by \`%s\`\n\n" "$GB_TRIM"
    echo "| ${GB_TRIM} | count |"
    echo "|---|---:|"
    case "$GB_TRIM" in
      cmd)
        tail -n +2 "$CSV_OUT" \
          | awk -F, '{ c=$2; gsub(/^"/,"",c); gsub(/"$/,"",c); cnt[c]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
          | sort -t"|" -k3,3nr
        ;;
      git_sha)
        tail -n +2 "$CSV_OUT" \
          | awk -F, '{ k=$3; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
          | sort -t"|" -k3,3nr
        ;;
      cfg)
        tail -n +2 "$CSV_OUT" \
          | awk -F, '{ k=$4; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
          | sort -t"|" -k3,3nr
        ;;
      day)
        tail -n +2 "$CSV_OUT" \
          | awk -F, '{ d=substr($1,1,10); cnt[d]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
          | sort -t"|" -k3,3nr
        ;;
      hour)
        tail -n +2 "$CSV_OUT" \
          | awk -F, '{
              # extract hour from ISO: YYYY-MM-DDThh:mm:ssZ / local variants
              if (match($1, /T([0-9][0-9]):[0-9][0-9]/, H)) { h=H[1] } else { h="??" }
              cnt[h]++
            } END {
              for(k in cnt) printf("| %s | %d |\n", k, cnt[k])
            }' \
          | sort -t"|" -k2,2   | sort -t"|" -k3,3nr
        ;;
      *)
        echo "| (unsupported key) | 0 |"
        ;;
    esac
  done
fi

} > “$MD_OUT”
fi

––––– Open –––––

if [[ “$OPEN_AFTER” -eq 1 ]]; then
if [[ “$IS_KAGGLE” -eq 1 || “$IS_CI” -eq 1 ]]; then
warn “Skipping –open on Kaggle/CI.”
else
if have xdg-open; then
say “Opening $MD_OUT…”; xdg-open “$MD_OUT” >/dev/null 2>&1 || warn “Failed to open with xdg-open”
elif have open; then
say “Opening $MD_OUT…”; open “$MD_OUT”     >/dev/null 2>&1 || warn “Failed to open with open”
elif have code; then
say “Opening $MD_OUT in VS Code…”; code -r “$MD_OUT” >/dev/null 2>&1 || warn “Failed to open with code”
else
warn “No opener (xdg-open/open/code) found; skipping viewer.”
fi
fi
fi

––––– JSON summary –––––

emit_json () {
local src_csv=”$1”
local rows cmds gits cfgs first_ts last_ts
rows=$(( $(wc -l < “$src_csv” 2>/dev/null || echo 0) - 1 ))
(( rows < 0 )) && rows=0
cmds=$(tail -n +2 “$src_csv” | awk -F, ‘{ c=$2; gsub(/^”/,””,c); gsub(/”$/,””,c); if(c!=””) cc[c]++ } END{ n=0; for(k in cc)n++; print n }’)
gits=$(tail -n +2 “$src_csv” | awk -F, ‘{ if($3!=””) gg[$3]++ } END{ n=0; for(k in gg)n++; print n }’)
cfgs=$(tail -n +2 “$src_csv” | awk -F, ‘{ if($4!=””) cf[$4]++ } END{ n=0; for(k in cf)n++; print n }’)
first_ts=$(tail -n +2 “$src_csv” | awk -F, ‘NR==1{f=$1} END{print f}’)
last_ts=$(tail -n +2 “$src_csv”  | awk -F, ‘END{print $1}’)
printf ‘{’
printf ‘“ok”:true’
printf ‘,“csv”:”%s”’ “$CSV_OUT”
printf ‘,“md”:”%s”’ “$MD_OUT”
printf ‘,“rows”:%d’ “$rows”
printf ‘,“since”:”%s”’ “${SINCE:-}”
printf ‘,“tail”:”%s”’ “${TAIL_N:-}”
printf ‘,“clean”:%s’ “$([[ $CLEAN -eq 1 ]] && echo true || echo false)”
printf ‘,“group_by”:”%s”’ “${GROUP_BY:-}”
printf ‘,“distinct”:{“cmd”:%s,“git_sha”:%s,“cfg”:%s}’ “${cmds:-0}” “${gits:-0}” “${cfgs:-0}”
printf ‘,“window”:{“first”:”%s”,“last”:”%s”}’ “${first_ts:-}” “${last_ts:-}”
printf ‘}\n’
}

if [[ $JSON_OUT -eq 1 || -n “${JSON_PATH:-}” ]]; then
JSON_PAYLOAD=”$(emit_json “$TMP_CSV”)”
if [[ -n “${JSON_PATH:-}” ]]; then
if [[ $DRY -eq 1 ]]; then
say “[dry-run] Would write JSON → $JSON_PATH”
else
printf “%s” “$JSON_PAYLOAD” > “$JSON_PATH”
say “Wrote JSON → $JSON_PATH”
fi
fi
if [[ $JSON_OUT -eq 1 ]]; then
printf “%s” “$JSON_PAYLOAD”
fi
fi

printf “%s✔%s Wrote %s and %s\n” “${GRN}” “${RST}” “$CSV_OUT” “$MD_OUT”
exit 0