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
#   --clean              De-duplicate by run_hash else (time,cmd)
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

set -euo pipefail

# ---------- Pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[ANALYZE]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[ANALYZE]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
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

# ---------- Args ----------
usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

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

# ---------- Move to repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; exit 1; }
fi

mkdir -p "$(dirname "$MD_OUT")" "$(dirname "$CSV_OUT")"

# ---------- Detect env ----------
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

# ---------- Pick CLI ----------
CLI_BIN=""
if [[ "$USE_POETRY" -eq 1 ]] && command -v poetry >/dev/null 2>&1; then
  CLI_BIN="poetry run spectramind"
elif command -v spectramind >/dev/null 2>&1; then
  CLI_BIN="spectramind"
else
  if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
  CLI_BIN="$PY -m spectramind"
fi

# ---------- Refresh via CLI (optional) ----------
if [[ "$REFRESH" -eq 1 && $DRY -eq 0 ]]; then
  say "Refreshing log CSV/MD via CLI (if available)…"
  if with_timeout bash -lc "$CLI_BIN analyze-log --md '$MD_OUT' --csv '$CSV_OUT'"; then
    say "spectramind analyze-log completed."
  else
    warn "CLI refresh not available; will parse '$LOG_IN' directly."
  fi
fi

# ---------- Collect inputs ----------
TMP_COMBINED="$(mktemp -t sm_analyze_combined_XXXX).md"
trap 'rm -f "$TMP_COMBINED" "$TMP_CSV" "$TMP2" 2>/dev/null || true' EXIT

append_if_exists() {
  local p="$1"
  [[ -f "$p" ]] && cat "$p" >> "$TMP_COMBINED"
}
# Primary log
append_if_exists "$LOG_IN"
# Extra logs (if any)
for L in "${EXTRA_INPUTS[@]}"; do append_if_exists "$L"; done

if [[ ! -s "$TMP_COMBINED" ]]; then
  if [[ -f "$CSV_OUT" ]]; then
    say "No Markdown log found, but existing CSV present; will transform CSV."
    PARSE_MODE="csv"
  else
    fail "No input logs found and no CSV present. Aborting."
  fi
else
  PARSE_MODE="md"
fi

# ---------- Parse Markdown to CSV (fallback path) ----------
TMP_CSV="$(mktemp -t sm_analyze_csv_XXXX).csv"
if [[ "$PARSE_MODE" == "md" ]]; then
  say "Parsing Markdown logs → CSV (temporary)…"
  awk -v since="$SINCE" '
    function iso_ge(a,b){ if(b==""){return 1} return (a>=b)?1:0 }
    BEGIN{
      FS="\n"; RS="⸻"; OFS=","
      print "time,cmd,git_sha,cfg,run_hash"
    }
    {
      block=$0
      gsub(/\r/,"",block)
      # Extract time and cmd from header "YYYY...Z — cmd"
      if (match(block, /([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z)[[:space:]]+—[[:space:]]+([^\n]*)/, H)) {
        t=H[1]; c=H[2]
        if (!iso_ge(t, since)) next
        git=""; cfg=""; rh=""
        # Scan bullets; tolerate variations
        while (match(block, /\n[•*-][^\n]*/, L)) {
          line=L[0]; block=substr(block, RSTART+RLENGTH)
          gsub(/^[\n•*\-[:space:]]+/,"",line)
          if (match(line, /Git SHA:[[:space:]]*([0-9A-Za-z]+)/, M)) git=M[1]
          if (match(line, /(cfg|config|snapshot)/i, _)) cfg=(cfg==""?"snapshot":"snapshot")
          if (match(line, /Run hash:[[:space:]]*([0-9A-Za-z]+)/, R)) rh=R[1]
        }
        gsub(/"/,"\"\"",c)
        printf "%s,\"%s\",%s,%s,%s\n", t, c, (git==""?"":git), (cfg==""?"none":cfg), (rh==""?"":rh)
      }
    }
  ' "$TMP_COMBINED" > "$TMP_CSV"
else
  say "Copying existing CSV for transforms…"
  cp "$CSV_OUT" "$TMP_CSV"
fi

# ---------- Filter: tail N ----------
if [[ -n "${TAIL_N:-}" ]]; then
  say "Keeping last $TAIL_N rows…"
  TMP2="$(mktemp -t sm_analyze_tail_XXXX).csv"
  (head -n1 "$TMP_CSV" && tail -n +2 "$TMP_CSV" | tail -n "$TAIL_N") > "$TMP2"
  mv "$TMP2" "$TMP_CSV"
fi

# ---------- Clean / de-duplicate ----------
if [[ "$CLEAN" -eq 1 ]]; then
  say "De-duplicating rows (prefer most recent by run_hash else time|cmd)…"
  TMP2="$(mktemp -t sm_analyze_dedup_XXXX).csv"
  awk -F, '
    NR==1 { header=$0; next }
    { rows[NR-1]=$0 }
    END{
      print header
      for(i=NR-1;i>=1;i--){
        split(rows[i],a,FS)
        key=a[5]; if (key=="") key=a[1]"|"a[2]
        if(!(seen[key]++)) print rows[i]
      }
    }
  ' "$TMP_CSV" > "$TMP2"
  mv "$TMP2" "$TMP_CSV"
fi

# ---------- Write final CSV_OUT ----------
if [[ $DRY -eq 1 ]]; then
  say "[dry-run] Would write CSV → $CSV_OUT"
else
  say "Writing CSV → $CSV_OUT"
  cp "$TMP_CSV" "$CSV_OUT"
fi

# ---------- Build Markdown table & optional group summary ----------
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
    echo "| time | cmd | git_sha | cfg | run_hash |"
    echo "|---|---|---|---|---|"
    tail -n +2 "$CSV_OUT" | awk -F, '
      {
        t=$1; c=$2; g=$3; cf=$4; rh=$5
        gsub(/^"/,"",c); gsub(/"$/,"",c); gsub(/\|/,"/",c)
        if (g=="") g=""; if (cf=="") cf=""; if (rh=="") rh=""
        printf "| %s | %s | %s | %s | %s |\n", t, c, g, cf, rh
      }
    '
    if [[ -n "${GROUP_BY}" ]]; then
      echo
      echo "## Summary by \`${GROUP_BY}\`"
      echo
      echo "| ${GROUP_BY} | count |"
      echo "|---|---:|"
      case "$GROUP_BY" in
        cmd)
          tail -n +2 "$CSV_OUT" | awk -F, '
            { c=$2; gsub(/^"/,"",c); gsub(/"$/,"",c); cnt[c]++ }
            END{ for(k in cnt){ gsub(/\|/,"/",k); printf("| %s | %d |\n", k, cnt[k]) } }
          ' | sort -t\| -k3,3nr
          ;;
        git_sha)
          tail -n +2 "$CSV_OUT" | awk -F, '{ k=$3; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' | sort -t\| -k3,3nr
          ;;
        cfg)
          tail -n +2 "$CSV_OUT" | awk -F, '{ k=$4; if(k=="") k="(none)"; cnt[k]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }' | sort -t\| -k3,3nr
          ;;
        day)
          tail -n +2 "$CSV_OUT" | awk -F, '
            { d=substr($1,1,10); cnt[d]++ } END{ for(k in cnt) printf("| %s | %d |\n", k, cnt[k]) }
          ' | sort -t\| -k3,3nr
          ;;
        *)
          echo "| (unsupported key) | 0 |"
          ;;
      esac
    fi
  } > "$MD_OUT"
fi

# ---------- Open (optional) ----------
if [[ "$OPEN_AFTER" -eq 1 ]]; then
  if [[ "$IS_KAGGLE" -eq 1 || "$IS_CI" -eq 1 ]]; then
    warn "Skipping --open on Kaggle/CI."
  else
    if command -v xdg-open >/dev/null 2>&1; then
      say "Opening $MD_OUT with xdg-open…"
      xdg-open "$MD_OUT" >/dev/null 2>&1 || warn "Failed to open with xdg-open"
    elif command -v open >/dev/null 2>&1; then
      say "Opening $MD_OUT with open…"
      open "$MD_OUT" >/dev/null 2>&1 || warn "Failed to open with open"
    else
      warn "No opener (xdg-open/open) found; skipping viewer."
    fi
  fi
fi

# ---------- JSON summary ----------
if [[ $JSON_OUT -eq 1 ]]; then
  ROWS="$(wc -l < "$TMP_CSV" 2>/dev/null || echo 0)"
  [[ "$ROWS" -gt 0 ]] && ROWS=$((ROWS-1)) || ROWS=0  # minus header
  printf '{'
  printf '"ok": true, '
  printf '"csv": "%s", "md": "%s", ' "$CSV_OUT" "$MD_OUT"
  printf '"rows": %s, ' "$ROWS"
  printf '"since": "%s", ' "${SINCE:-}"
  printf '"tail": "%s", ' "${TAIL_N:-}"
  printf '"clean": %s, ' "$([[ $CLEAN -eq 1 ]] && echo true || echo false)"
  printf '"group_by": "%s"' "${GROUP_BY:-}"
  printf '}\n'
fi

printf "%s✔%s Wrote %s and %s\n" "${GRN}" "${RST}" "$CSV_OUT" "$MD_OUT"
exit 0
