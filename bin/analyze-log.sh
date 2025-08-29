#!/usr/bin/env bash
# ==============================================================================
# bin/analyze-log.sh — Parse & summarize SpectraMind V50 CLI log(s) (upgraded)
# ------------------------------------------------------------------------------
# What it does
#   • Parses logs/v50_debug_log.md (and/or additional inputs) into CSV + Markdown
#   • Optionally calls `spectramind analyze-log` first (if available) to refresh
#   • Supports multiple inputs, --since ISO filter, tail N, de-duplication
#   • Summaries by cmd|git_sha|cfg (and optional by day)
#   • Emits JSON summary (optional), dry-run, and is CI/Kaggle safe
#
# Usage
#   bin/analyze-log.sh [options] [--] [LOGPATH ...]
#
# Options
#   --md <path>          Output Markdown path                  (default: outputs/log_table.md)
#   --csv <path>         Output CSV path                       (default: outputs/log_table.csv)
#   --log <path>         Primary input log path                (default: logs/v50_debug_log.md)
#   --since <date>       ISO8601 lower bound (e.g., 2025-08-01 or 2025-08-01T00:00:00Z)
#   --tail <N>           Only keep last N entries after filtering
#   --clean              De-duplicate by (cmd,git,cfg_hash,tag,pred,bundle)
#   --group-by <key>     Group summary by: cmd|git_sha|cfg|day
#   --open               Open generated Markdown (non-Kaggle/CI only)
#   --no-poetry          Do not use Poetry to invoke spectramind
#   --no-refresh         Do NOT invoke `spectramind analyze-log` first
#   --json               Emit JSON summary to stdout
#   --timeout <sec>      Timeout per CLI step if `timeout` exists (default: 120)
#   --dry-run            Print actions, do not write files
#   --quiet              Less chatty
#   -h|--help            Show help and exit
#
# Notes
#   • Works even if the CLI isn’t installed; pure-awk parsing fallback is built in.
#   • Safe on Kaggle/CI (no open calls unless --open and opener available locally).
#   • Additional LOGPATH args (files) can be provided; they will be concatenated.
# ==============================================================================

set -Eeuo pipefail
IFS=$'\n\t'

# ---------- Pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; YLW=$'\033[33m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[ANALYZE]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[ANALYZE]%s %s\n' "${YLW}" "${RST}" "$*" >&2; }
fail() { printf '%s[ANALYZE]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- Defaults ----------
MD_OUT="outputs/log_table.md"
CSV_OUT="outputs/log_table.csv"
LOG_IN="logs/v50_debug_log.md"
SINCE=""
TAIL_N=""
CLEAN=0
GROUP_BY=""
OPEN_AFTER=0
USE_POETRY=1
REFRESH=1
QUIET=0
JSON_OUT=0
STEP_TIMEOUT="${ANALYZE_TIMEOUT:-120}"
DRY=0
EXTRA_INPUTS=()

usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------- Args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --md)           MD_OUT="${2:?}"; shift 2 ;;
    --csv)          CSV_OUT="${2:?}"; shift 2 ;;
    --log)          LOG_IN="${2:?}"; shift 2 ;;
    --since)        SINCE="${2:?}"; shift 2 ;;
    --tail)         TAIL_N="${2:?}"; shift 2 ;;
    --clean)        CLEAN=1; shift ;;
    --group-by)     GROUP_BY="${2:?}"; shift 2 ;;
    --open)         OPEN_AFTER=1; shift ;;
    --no-poetry)    USE_POETRY=0; shift ;;
    --no-refresh)   REFRESH=0; shift ;;
    --json)         JSON_OUT=1; shift ;;
    --timeout)      STEP_TIMEOUT="${2:?}"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    --quiet)        QUIET=1; shift ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; while [[ $# -gt 0 ]]; do EXTRA_INPUTS+=("$1"); shift; done ;;
    *)              EXTRA_INPUTS+=("$1"); shift ;;
  esac
done

# ---------- Repo root ----------
if command -v git >/dev/null 2>&1; then
  if ROOT=$(git rev-parse --show-toplevel 2>/dev/null); then cd "$ROOT"; fi
fi

mkdir -p "$(dirname "$MD_OUT")" "$(dirname "$CSV_OUT")"

# ---------- Env detect ----------
IS_KAGGLE=0
[[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1
IS_CI=0
[[ -n "${GITHUB_ACTIONS:-}" || -n "${CI:-}" ]] && IS_CI=1

# ---------- Helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
with_timeout() {
  if have timeout && [[ "${STEP_TIMEOUT:-0}" -gt 0 ]]; then
    timeout --preserve-status --signal=TERM "${STEP_TIMEOUT}" "$@"
  else
    "$@"
  fi
}
iso_to_epoch() { date -d "$1" +%s 2>/dev/null || echo 0; }

# ---------- Pick CLI ----------
CLI_BIN=""
if [[ "$USE_POETRY" -eq 1 ]] && have poetry; then
  CLI_BIN="poetry run spectramind"
elif have spectramind; then
  CLI_BIN="spectramind"
elif have python3; then
  CLI_BIN="python3 -m spectramind"
else
  CLI_BIN=""
fi

# ---------- Optional refresh via CLI ----------
if [[ "$REFRESH" -eq 1 && "$DRY" -eq 0 && -n "$CLI_BIN" ]]; then
  say "Refreshing via CLI: \`$CLI_BIN analyze-log\` (if supported)…"
  if with_timeout bash -lc "$CLI_BIN analyze-log --md '$MD_OUT' --csv '$CSV_OUT'"; then
    say "spectramind analyze-log completed."
  else
    warn "CLI refresh unavailable or failed; falling back to native parser."
  fi
fi

# ---------- Combine inputs ----------
TMP_COMBINED="$(mktemp -t sm_anlz_all_XXXX).log"
trap 'rm -f "$TMP_COMBINED" "$TMP_CSV" "$TMP2" 2>/dev/null || true' EXIT

append_if_exists() { [[ -f "$1" ]] && cat "$1" >> "$TMP_COMBINED"; }

append_if_exists "$LOG_IN"
for L in "${EXTRA_INPUTS[@]}"; do append_if_exists "$L"; done

if [[ ! -s "$TMP_COMBINED" ]]; then
  if [[ -f "$CSV_OUT" ]]; then
    say "No raw logs found; existing CSV present. Using CSV as source."
    USE_EXISTING_CSV=1
  else
    fail "No input logs found and no existing CSV present."
    exit 1
  fi
else
  USE_EXISTING_CSV=0
fi

# ---------- Convert logs → CSV (direct parser for single-line schema) ----------
TMP_CSV="$(mktemp -t sm_anlz_csv_XXXX).csv"
if [[ "$USE_EXISTING_CSV" -eq 1 ]]; then
  cp "$CSV_OUT" "$TMP_CSV"
else
  # Schema produced by our submission/repair scripts:
  # [ISO8601] cmd=<script> git=<sha> cfg_hash=<hash> tag=<tag_or_-> pred=<path_or_-> bundle=<path_or_-> notes="...".
  # We parse tolerant of missing keys and extra spaces.
  say "Parsing structured log lines → CSV…"
  {
    echo "ts,cmd,git_sha,cfg_hash,tag,pred,bundle,notes"
    awk -v since="$SINCE" '
      function trim(s){sub(/^[ \t\r\n]+/,"",s);sub(/[ \t\r\n]+$/,"",s);return s}
      function val(k, s,   r,x) {
        # find k=VALUE (VALUE up to next space, unless quoted notes="...")
        r = "[[:space:]]" k "=[^ \n]*";
        if (k=="notes") {
          r = "notes=\"[^\"]*\"";
          if (match(s, r, x)) { gsub(/^notes="/,"",x[0]); gsub(/"$/,"",x[0]); return x[0] }
          return ""
        } else {
          if (match(s, r, x)) { split(x[0],a,"="); return a[2] }
          return ""
        }
      }
      function iso_ge(a,b,  A,B){ if(b=="") return 1; A=a; B=b; return (A>=B)?1:0 }
      /^[[]/ {
        line=$0
        # timestamp: [ISO]
        if (match(line, /^\[([^\]]+)\]/, T)) {
          ts = T[1]
          if (!iso_ge(ts,since)) next
          # Extract fields
          cmd = trim(val("cmd", line))
          git = trim(val("git", line))
          cfg = trim(val("cfg_hash", line))
          tag = trim(val("tag", line))
          pred = trim(val("pred", line))
          bundle = trim(val("bundle", line))
          notes = val("notes", line)
          # CSV-escape fields
          gsub(/"/,"\"\"",cmd); gsub(/"/,"\"\"",notes)
          print ts ",\"" cmd "\"," git "," cfg "," tag "," pred "," bundle ",\"" notes "\""
        }
      }
    ' "$TMP_COMBINED"
  } > "$TMP_CSV"
fi

# ---------- Filter by --since ----------
if [[ -n "$SINCE" ]]; then
  say "Filtering rows since ${SINCE}…"
  TMP2="$(mktemp -t sm_anlz_since_XXXX).csv"
  # Keep header + rows where ts >= SINCE (string compare works for ISO8601)
  awk -F, -v since="$SINCE" 'NR==1{print;next} { if($1>=since) print }' "$TMP_CSV" > "$TMP2"
  mv "$TMP2" "$TMP_CSV"
fi

# ---------- Tail N ----------
if [[ -n "${TAIL_N:-}" ]]; then
  say "Keeping last $TAIL_N rows…"
  TMP2="$(mktemp -t sm_anlz_tail_XXXX).csv"
  (head -n1 "$TMP_CSV" && tail -n +2 "$TMP_CSV" | tail -n "$TAIL_N") > "$TMP2"
  mv "$TMP2" "$TMP_CSV"
fi

# ---------- De-dup ----------
if [[ "$CLEAN" -eq 1 ]]; then
  say "De-duplicating rows by (cmd,git,cfg_hash,tag,pred,bundle)…"
  TMP2="$(mktemp -t sm_anlz_dedup_XXXX).csv"
  awk -F, '
    NR==1 { print; next }
    {
      key=$2 FS $3 FS $4 FS $5 FS $6 FS $7
      rows[NR]=$0
      keys[NR]=key
    }
    END{
      # iterate backwards to prefer most recent
      for(i=NR;i>=2;i--){
        if(!seen[keys[i]]++){
          stack[++n]=i
        }
      }
      print $0 > "/dev/stderr" # ensure NR available for END (no-op)
      # print header already printed
      for(j=n;j>=1;j--){
        print rows[ stack[j] ]
      }
    }
  ' "$TMP_CSV" 2>/dev/null | (read -r _; printf "%s\n" "$(head -n1 "$TMP_CSV")"; cat) > "$TMP2" || true
  # If the awk trick fails, fall back to a simple uniq (preserves order)
  if [[ ! -s "$TMP2" ]]; then
    (head -n1 "$TMP_CSV"; tail -n +2 "$TMP_CSV" | awk '!seen[$0]++') > "$TMP2"
  fi
  mv "$TMP2" "$TMP_CSV"
fi

# ---------- Write CSV ----------
if [[ $DRY -eq 1 ]]; then
  say "[dry-run] Would write CSV → $CSV_OUT"
else
  say "Writing CSV → $CSV_OUT"
  cp "$TMP_CSV" "$CSV_OUT"
fi

# ---------- Markdown table ----------
if [[ $DRY -eq 1 ]]; then
  say "[dry-run] Would write Markdown → $MD_OUT"
else
  say "Generating Markdown → $MD_OUT"
  {
    echo "# SpectraMind V50 — CLI Calls"
    echo
    NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "_Generated: ${NOW}_"
    echo
    echo "| time | cmd | git_sha | cfg_hash | tag | pred | bundle | notes |"
    echo "|---|---|---|---|---|---|---|---|"
    tail -n +2 "$CSV_OUT" | awk -F, '
      {
        t=$1; c=$2; g=$3; cf=$4; tg=$5; pr=$6; bu=$7; no=$8
        gsub(/^"/,"",c); gsub(/"$/,"",c);
        gsub(/\|/,"/",c); gsub(/\|/,"/",no); gsub(/\|/,"/",pr); gsub(/\|/,"/",bu);
        printf "| %s | %s | %s | %s | %s | %s | %s | %s |\n", t, c, g, cf, tg, pr, bu, no
      }
    '
    # Optional summary
    if [[ -n "${GROUP_BY}" ]]; then
      echo
      echo "## Summary by \`${GROUP_BY}\`"
      echo
      echo "| ${GROUP_BY} | count |"
      echo "|---|---:|"
      case "$GROUP_BY" in
        cmd)
          tail -n +2 "$CSV_OUT" | awk -F, '{ c=$2; gsub(/^"/,"",c); gsub(/"$/,"",c); cnt[c]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
            | sort -t\| -k3,3nr
          ;;
        git_sha)
          tail -n +2 "$CSV_OUT" | awk -F, '{ k=$3; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
            | sort -t\| -k3,3nr
          ;;
        cfg)
          tail -n +2 "$CSV_OUT" | awk -F, '{ k=$4; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
            | sort -t\| -k3,3nr
          ;;
        day)
          tail -n +2 "$CSV_OUT" | awk -F, '{ d=substr($1,1,10); cnt[d]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' \
            | sort -t\| -k3,3nr
          ;;
        *)
          echo "| (unsupported key) | 0 |"
          ;;
      esac
    fi
  } > "$MD_OUT"
fi

# ---------- Open ----------
if [[ "$OPEN_AFTER" -eq 1 ]]; then
  if [[ "$IS_KAGGLE" -eq 1 || "$IS_CI" -eq 1 ]]; then
    warn "Skipping --open on Kaggle/CI."
  else
    if have xdg-open; then
      say "Opening $MD_OUT…"; xdg-open "$MD_OUT" >/dev/null 2>&1 || warn "Failed to open with xdg-open"
    elif have open; then
      say "Opening $MD_OUT…"; open "$MD_OUT"     >/dev/null 2>&1 || warn "Failed to open with open"
    else
      warn "No opener (xdg-open/open) found; skipping viewer."
    fi
  fi
fi

# ---------- JSON summary ----------
if [[ $JSON_OUT -eq 1 ]]; then
  ROWS=$(( $(wc -l < "$TMP_CSV" 2>/dev/null || echo 0) - 1 ))
  (( ROWS < 0 )) && ROWS=0
  # basic stats
  CMDS=$(tail -n +2 "$TMP_CSV" | awk -F, '{ c=$2; gsub(/^"/,"",c); gsub(/"$/,"",c); if(c!="") cc[c]++ } END{ n=0; for(k in cc)n++; print n }')
  GITS=$(tail -n +2 "$TMP_CSV" | awk -F, '{ if($3!="") gg[$3]++ } END{ n=0; for(k in gg)n++; print n }')
  CFGS=$(tail -n +2 "$TMP_CSV" | awk -F, '{ if($4!="") cf[$4]++ } END{ n=0; for(k in cf)n++; print n }')
  printf '{'
  printf '"ok":true,"csv":"%s","md":"%s","rows":%d,' "$CSV_OUT" "$MD_OUT" "$ROWS"
  printf '"since":"%s","tail":"%s","clean":%s,' "${SINCE:-}" "${TAIL_N:-}" "$([[ $CLEAN -eq 1 ]] && echo true || echo false)"
  printf '"group_by":"%s",' "${GROUP_BY:-}"
  printf '"distinct":{"cmd":%s,"git_sha":%s,"cfg":%s}' "${CMDS:-0}" "${GITS:-0}" "${CFGS:-0}"
  printf '}\n'
fi

printf "%s✔%s Wrote %s and %s\n" "${GRN}" "${RST}" "$CSV_OUT" "$MD_OUT"
exit 0