#!/usr/bin/env bash
# ==============================================================================
# bin/analyze-log.sh — Parse & summarize SpectraMind V50 CLI log(s)
# ------------------------------------------------------------------------------
# What it does
#   • Parses logs/v50_debug_log.md into CSV + Markdown tables
#   • Optionally calls `spectramind analyze-log` first (if available)
#   • Filters by date, tails last N, de‑duplicates, and groups summaries
#   • Can open the generated Markdown in the default viewer
#
# Usage
#   bin/analyze-log.sh [options]
#
# Options
#   --md <path>        Output Markdown path (default: outputs/log_table.md)
#   --csv <path>       Output CSV path      (default: outputs/log_table.csv)
#   --log <path>       Input log path       (default: logs/v50_debug_log.md)
#   --since <date>     Filter ISO8601 date/time lower bound (e.g., 2025-08-01)
#   --tail <N>         Only keep last N entries after filtering
#   --clean            De‑duplicate by Run hash (if present) else (time,cmd)
#   --group-by <key>   Group summary by one of: cmd|git_sha|cfg
#   --open             Open the generated Markdown file afterward
#   --no-poetry        Do not use Poetry to invoke spectramind
#   --no-refresh       Do not invoke `spectramind analyze-log` first
#   --quiet            Less chatty
#   -h|--help          Show this help and exit
#
# Exit codes
#   0 OK, 1 failure, 2 bad usage
#
# Notes
#   • Works even if the CLI isn’t installed; pure-awk parsing fallback is built in.
#   • Safe on Kaggle (no xdg-open / open calls unless --open and available).
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

# ---------- Args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --md)        MD_OUT="${2:?}"; shift ;;
    --csv)       CSV_OUT="${2:?}"; shift ;;
    --log)       LOG_IN="${2:?}"; shift ;;
    --since)     SINCE="${2:?}"; shift ;;
    --tail)      TAIL_N="${2:?}"; shift ;;
    --clean)     CLEAN=1 ;;
    --group-by)  GROUP_BY="${2:?}"; shift ;;
    --open)      OPEN_AFTER=1 ;;
    --no-poetry) USE_POETRY=0 ;;
    --no-refresh)REFRESH=0 ;;
    --quiet)     QUIET=1 ;;
    -h|--help)
      sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) fail "Unknown arg: $1"; exit 2 ;;
  esac
  shift
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
if [[ "$REFRESH" -eq 1 ]]; then
  say "Refreshing log CSV/MD via CLI (if available)…"
  if $CLI_BIN analyze-log --md "$MD_OUT" --csv "$CSV_OUT" >/dev/null 2>&1; then
    say "spectramind analyze-log completed."
  else
    warn "CLI refresh not available; will parse $LOG_IN directly."
  fi
fi

# ---------- Ensure input log exists (try fallback path if needed) ----------
if [[ ! -f "$LOG_IN" ]]; then
  # Try to parse from outputs/log_table.csv if present to reconstruct MD later
  if [[ -f "$CSV_OUT" ]]; then
    say "Input log not found, but existing CSV found; will operate on CSV for transforms."
    PARSE_MODE="csv"
  else
    fail "No input log at $LOG_IN and no existing CSV. Aborting."
    exit 1
  fi
else
  PARSE_MODE="md"
fi

# ---------- AWK parser (MD -> CSV rows) ----------
# Fields we attempt to extract:
#   time, cmd, git_sha, cfg, run_hash (if present), extra_kv JSON-ish (discarded here)
TMP_CSV="$(mktemp -t spectramind_analyze_XXXX).csv"

if [[ "$PARSE_MODE" == "md" ]]; then
  say "Parsing $LOG_IN → CSV (temporary)…"
  awk -v since="$SINCE" -v tailn="$TAIL_N" -v clean="$CLEAN" '
    function iso_ge(a,b,    A,B) {
      # compare ISO8601 timestamps lexicographically (Z only)
      if (b == "") return 1;
      A=a; B=b;
      return (A >= B) ? 1 : 0;
    }
    BEGIN{
      FS="\n"; RS="⸻"
      print "time,cmd,git_sha,cfg,run_hash"
    }
    {
      block=$0
      gsub(/\r/,"",block)
      # find header line with timestamp and command (YYYY-MM-DDTHH:MM:SSZ — cmd)
      if (match(block, /([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z) — (.*)/, H)) {
        t=H[1]; c=H[2]
        # filter by since
        if (!iso_ge(t, since)) next
        # pull bullet lines
        git=""; cfg=""; rh=""
        while (match(block, /•[^\n]*/, L)) {
          line=L[0]; block=substr(block, RSTART+RLENGTH)
          gsub(/^•[[:space:]]*/,"",line)
          if (match(line, /Git SHA:[[:space:]]*([a-zA-Z0-9]+)/, M)) git=M[1]
          if (match(line, /(cfg|config|snapshot)/i, _)) cfg=(cfg==""? "snapshot":"snapshot")
          if (match(line, /Run hash:[[:space:]]*([a-zA-Z0-9]+)/, R)) rh=R[1]
        }
        # basic csv escaping for cmd
        gsub(/"/,"\"\"",c)
        printf "%s,\"%s\",%s,%s,%s\n", t, c, (git==""?"":git), (cfg==""?"none":cfg), (rh==""?"":rh)
      }
    }
  ' "$LOG_IN" > "$TMP_CSV"
else
  # We already have CSV_OUT; copy to temp and optionally filter in place
  say "Copying existing CSV for transforms…"
  cp "$CSV_OUT" "$TMP_CSV"
fi

# ---------- Filter: tail N ----------
if [[ -n "$TAIL_N" ]]; then
  say "Keeping last $TAIL_N rows…"
  (head -n1 "$TMP_CSV" && tail -n +2 "$TMP_CSV" | tail -n "$TAIL_N") > "${TMP_CSV}.tail"
  mv "${TMP_CSV}.tail" "$TMP_CSV"
fi

# ---------- Clean / de‑duplicate ----------
if [[ "$CLEAN" -eq 1 ]]; then
  say "De‑duplicating rows (preferring most recent)…"
  awk -F, '
    NR==1 { print; next }
    {
      key=$5; # run_hash
      if (key=="") key=$1"|"substr($2,2,length($2)-2) # time|cmd (cmd without quotes)
      if (!seen[key]++) rows[++n]=$0
    }
    END{
      print_rows=1
      # We preserved first occurrence; but we want most recent → reverse stable
      # Simpler: re-read array in reverse order writing unseen2
      print_header=0
    }
  ' "$TMP_CSV" > "${TMP_CSV}.dedup_stage"

  # Reverse to keep most recent
  awk -F, '
    NR==1 { header=$0; next }
    { lines[NR-1]=$0 }
    END{
      print header
      for(i=NR-1;i>=1;i--){
        split(lines[i],a,FS)
        key=a[5]; if (key=="") key=a[1]"|"a[2]
        if(!(seen[key]++)) print lines[i]
      }
    }
  ' "${TMP_CSV}.dedup_stage" > "${TMP_CSV}.dedup"
  mv "${TMP_CSV}.dedup" "$TMP_CSV"
  rm -f "${TMP_CSV}.dedup_stage"
fi

# ---------- Write final CSV_OUT ----------
say "Writing CSV → $CSV_OUT"
cp "$TMP_CSV" "$CSV_OUT"

# ---------- Build Markdown table & optional group summary ----------
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
      t=$1; c=$2; g=$3; cfg=$4; rh=$5
      # unquote cmd if quoted
      gsub(/^"/,"",c); gsub(/"$/,"",c)
      gsub(/\|/,"/",c)
      if (g=="") g=""; if (cfg=="") cfg=""; if (rh=="") rh=""
      printf "| %s | %s | %s | %s | %s |\n", t, c, g, cfg, rh
    }
  '
  # Group summary
  if [[ -n "$GROUP_BY" ]]; then
    key="$GROUP_BY"
    echo
    echo "## Summary by \`$key\`"
    echo
    echo "| $key | count |"
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
      *)
        echo "| (unsupported key) | 0 |"
        ;;
    esac
  fi
} > "$MD_OUT"

# ---------- Open (optional) ----------
if [[ "$OPEN_AFTER" -eq 1 ]]; then
  if [[ "$IS_KAGGLE" -eq 1 ]]; then
    warn "Skipping --open on Kaggle."
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

# ---------- Done ----------
printf "%s✔%s Wrote %s and %s\n" "${GRN}" "${RST}" "$CSV_OUT" "$MD_OUT"
exit 0