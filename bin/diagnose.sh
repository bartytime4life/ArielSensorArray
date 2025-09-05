#!/usr/bin/env bash

==============================================================================

bin/diagnose.sh — Generate SpectraMind V50 diagnostics (ultimate, upgraded)

——————————————————————————

What it does

• Runs light or full diagnostics via the SpectraMind CLI

• Produces smoothness HTML and a versioned dashboard report in outputs/diagnostics

• Lets you toggle UMAP/t-SNE, open the report, control verbosity, emit JSON/JSON file

• Adds structured audit entry to logs/v50_debug_log.md (cmd=diagnose)

• Safe for CI/Kaggle (light mode default; avoids opening browser)



Usage

bin/diagnose.sh [options]



Common examples

# Fast, CI/Kaggle-safe dashboard (no UMAP/t-SNE) + smoothness

bin/diagnose.sh –light



# Full dashboard (UMAP + t-SNE) and open the resulting HTML

bin/diagnose.sh –full –open



Options

–outdir        Output directory (default: outputs/diagnostics)

–light              Light mode (skip UMAP/t-SNE; fastest & CI/Kaggle-safe)

–full               Full mode (attempt UMAP + t-SNE; may be slower)

–no-umap            Skip UMAP even in –full

–no-tsne            Skip t-SNE even in –full

–smoothness-only    Only generate the smoothness HTML (skip dashboard)

–dashboard-only     Only generate dashboard (skip smoothness)

–open               Open the newest dashboard HTML (non-Kaggle/CI only)

–no-poetry          Do not use Poetry; call spectramind directly / via python -m

–timeout       Timeout per CLI step (default: 300)

–log          Combined stdout/stderr log (default: logs/diagnostics/diag-.log)

–json               Emit a JSON result summary to stdout

–json-path    Also write JSON summary to file

–quiet              Less verbose console output

-h|–help            Show help and exit



Exit codes

0 OK, 1 failure, 2 bad usage



Notes

• Auto-detects Kaggle/CI and avoids opening files there.

• Works even if Poetry isn’t installed (module fallback).

• Idempotent: repeated runs should not mutate state beyond outputs/diagnostics.

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– Pretty –––––

is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; CYN=”$(tput setaf 6)”; YLW=”$(tput setaf 3)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; CYN=””; YLW=””; RST=””
fi
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[DIAG]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[DIAG]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[DIAG]%s %s\n’ “${RED}” “${RST}” “$*” >&2; }

––––– Defaults –––––

OUTDIR=“outputs/diagnostics”
LIGHT=0
FULL=0
NO_UMAP=0
NO_TSNE=0
SMOOTHNESS_ONLY=0
DASHBOARD_ONLY=0
OPEN_AFTER=0
USE_POETRY=1
TIMEOUT=”${DIAG_TIMEOUT:-300}”
LOG_DIR_DEFAULT=“logs/diagnostics”
TS=”$(date -u +%Y%m%d_%H%M%S)”
LOG_PATH=””
QUIET=0
JSON_OUT=0
JSON_PATH=””

––––– Args –––––

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; }

while [[ $# -gt 0 ]]; do
case “$1” in
–outdir)          OUTDIR=”${2:?}”; shift ;;
–light)           LIGHT=1 ;;
–full)            FULL=1 ;;
–no-umap)         NO_UMAP=1 ;;
–no-tsne)         NO_TSNE=1 ;;
–smoothness-only) SMOOTHNESS_ONLY=1 ;;
–dashboard-only)  DASHBOARD_ONLY=1 ;;
–open)            OPEN_AFTER=1 ;;
–no-poetry)       USE_POETRY=0 ;;
–timeout)         TIMEOUT=”${2:?}”; shift ;;
–log)             LOG_PATH=”${2:?}”; shift ;;
–json)            JSON_OUT=1 ;;
–json-path)       JSON_PATH=”${2:?}”; shift ;;
–quiet)           QUIET=1 ;;
-h|–help)         usage; exit 0 ;;
*)                 fail “Unknown arg: $1”; exit 2 ;;
esac
shift
done

Normalize modes: default to light if neither specified

if [[ “$LIGHT” -eq 0 && “$FULL” -eq 0 ]]; then LIGHT=1; fi

Incompatible single-action toggles

if [[ “$SMOOTHNESS_ONLY” -eq 1 && “$DASHBOARD_ONLY” -eq 1 ]]; then
fail “Choose either –smoothness-only or –dashboard-only (not both).”
exit 2
fi

––––– Repo root –––––

if command -v git >/dev/null 2>&1 && git rev-parse –show-toplevel >/dev/null 2>&1; then
cd “$(git rev-parse –show-toplevel)”
else
SCRIPT_DIR=”$(cd – “$(dirname – “${BASH_SOURCE[0]}”)” && pwd)”
cd “$SCRIPT_DIR/..” || { fail “Cannot locate repo root”; exit 1; }
fi

mkdir -p “$OUTDIR” “$LOG_DIR_DEFAULT” logs || true

––––– Env detect –––––

IS_KAGGLE=0
[[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=1
IS_CI=0
[[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=1

––––– Pick CLI –––––

CLI_BIN=””
if [[ “$USE_POETRY” -eq 1 ]] && command -v poetry >/dev/null 2>&1; then
CLI_BIN=“poetry run spectramind”
elif command -v spectramind >/dev/null 2>&1; then
CLI_BIN=“spectramind”
else
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi
CLI_BIN=”$PY -m spectramind”
fi

say “Outdir: ${BOLD}${OUTDIR}${RST}”
say “Mode  : ${BOLD}$([[ $LIGHT -eq 1 ]] && echo light || echo full)${RST}”
say “CLI   : ${BOLD}${CLI_BIN}${RST}”

––––– Logging –––––

if [[ -z “$LOG_PATH” ]]; then LOG_PATH=”${LOG_DIR_DEFAULT}/diag-${TS}.log”; fi
if [[ “$QUIET” -eq 0 ]]; then

shellcheck disable=SC2094

exec > >(tee -a “${LOG_PATH}”) 2>&1
else
exec >> “${LOG_PATH}” 2>&1
fi
say “Log   : ${BOLD}${LOG_PATH}${RST}”

––––– Helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }

with_timeout() {
if have timeout && [[ “${TIMEOUT:-0}” -gt 0 ]]; then
timeout –preserve-status –signal=TERM “${TIMEOUT}” “$@”
else
“$@”
fi
}

run_step() {

run_step  <command…>

local title=”$1”; shift
[[ “$QUIET” -eq 0 ]] && printf “%s→ %s%s\n” “${DIM}” “${title}” “${RST}”
if ! with_timeout “$@”; then
fail “${title} — command failed or timed out”
return 1
fi
return 0
}

latest_html() { ls -t “${OUTDIR}”/*.html 2>/dev/null | head -n1 || true; }

write_structured_log() {

[ISO] cmd=diagnose git= cfg_hash=<hash|-> tag=_ pred=_ bundle=_ notes=“mode=<light|full>;outdir=<…>;status=<ok|fail>”

local status=”$1”
local ts_iso git_short cfg_hash notes
ts_iso=”$(date -u +%Y-%m-%dT%H:%M:%SZ)”
if command -v git >/dev/null 2>&1; then
git_short=”$(git rev-parse –short HEAD 2>/dev/null || echo nogit)”
else
git_short=“nogit”
fi
cfg_hash=”-”
if [[ -f “run_hash_summary_v50.json” ]]; then
cfg_hash=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ run_hash_summary_v50.json 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)”
[[ -z “$cfg_hash” ]] && cfg_hash=”-”
fi
notes=“mode=$([[ $LIGHT -eq 1 ]] && echo light || echo full);outdir=${OUTDIR};status=${status}”
printf ‘[%s] cmd=diagnose git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=”%s”\n’ 
“$ts_iso” “$git_short” “$cfg_hash” “$notes” >> “logs/v50_debug_log.md”
}

––––– Build argument sets –––––

DASH_ARGS=(diagnose dashboard –outdir “$OUTDIR”)
if [[ “$LIGHT” -eq 1 ]]; then
DASH_ARGS+=(–no-umap –no-tsne)
else
[[ “$NO_UMAP” -eq 1 ]] && DASH_ARGS+=(–no-umap)
[[ “$NO_TSNE” -eq 1 ]] && DASH_ARGS+=(–no-tsne)
fi

FAILED=0
START_ISO=”$(date -u +%Y-%m-%dT%H:%M:%SZ)”

––––– Smoothness –––––

SMOOTH_PATH=””
if [[ “$DASHBOARD_ONLY” -eq 0 ]]; then
say “Generating smoothness diagnostics…”
if run_step “spectramind diagnose smoothness” bash -lc “${CLI_BIN} diagnose smoothness –outdir ‘$OUTDIR’”; then
SMOOTH_PATH=”$(latest_html || true)”
else
FAILED=1
fi
fi

––––– Dashboard –––––

REPORT=””
if [[ “$SMOOTHNESS_ONLY” -eq 0 ]]; then
say “Building diagnostics dashboard…”
if run_step “spectramind ${DASH_ARGS[]}” bash -lc “${CLI_BIN} ${DASH_ARGS[]}”; then
REPORT=”$(latest_html)”
if [[ -z “$REPORT” ]]; then
warn “Dashboard completed but no HTML found in ${OUTDIR}.”
FAILED=1
fi
else
FAILED=1
fi
fi

––––– Optionally open report –––––

if [[ “$OPEN_AFTER” -eq 1 ]]; then
if [[ “$IS_KAGGLE” -eq 1 || “$IS_CI” -eq 1 ]]; then
warn “Skipping –open in CI/Kaggle.”
else
REPORT_TO_OPEN=”$(latest_html)”
if [[ -n “$REPORT_TO_OPEN” ]]; then
if have xdg-open; then xdg-open “$REPORT_TO_OPEN” >/dev/null 2>&1 || warn “Failed to open with xdg-open”
elif have open; then open “$REPORT_TO_OPEN” >/dev/null 2>&1 || warn “Failed to open with open”
else warn “No opener (xdg-open/open) found; skipping viewer.”
fi
else
warn “No dashboard HTML to open.”
fi
fi
fi

––––– Result / Log –––––

END_ISO=”$(date -u +%Y-%m-%dT%H:%M:%SZ)”
if [[ “$FAILED” -eq 0 ]]; then
printf “%s✔%s Diagnostics completed.\n” “${GRN}” “${RST}”
write_structured_log “ok”
OK=true; EC=0
else
printf “%s✘%s Diagnostics encountered issues.\n” “${RED}” “${RST}”
write_structured_log “fail”
OK=false; EC=1
fi

––––– JSON summary –––––

if [[ “$JSON_OUT” -eq 1 || -n “${JSON_PATH:-}” ]]; then
payload=$(
printf ‘{’
printf ’“ok”: %s, ’ “$([[ $OK == true ]] && echo true || echo false)”
printf ’“outdir”: “%s”, ’ “$OUTDIR”
printf ’“light”: %s, ’ “$([[ $LIGHT -eq 1 ]] && echo true || echo false)”
printf ’“full”: %s, ’ “$([[ $FULL -eq 1 ]] && echo true || echo false)”
printf ’“no_umap”: %s, ’ “$([[ $NO_UMAP -eq 1 ]] && echo true || echo false)”
printf ’“no_tsne”: %s, ’ “$([[ $NO_TSNE -eq 1 ]] && echo true || echo false)”
printf ’“smoothness_only”: %s, ’ “$([[ $SMOOTHNESS_ONLY -eq 1 ]] && echo true || echo false)”
printf ’“dashboard_only”: %s, ’ “$([[ $DASHBOARD_ONLY -eq 1 ]] && echo true || echo false)”
printf ’“report”: “%s”, ’ “${REPORT:-}”
printf ’“smoothness_html”: “%s”, ’ “${SMOOTH_PATH:-}”
printf ’“log”: “%s”, ’ “$LOG_PATH”
printf ’“ci”: %s, “kaggle”: %s, ’ “$([[ $IS_CI -eq 1 ]] && echo true || echo false)” “$([[ $IS_KAGGLE -eq 1 ]] && echo true || echo false)”
printf ‘“start”: “%s”, “end”: “%s”’ “$START_ISO” “$END_ISO”
printf ‘}’
)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; say “Wrote JSON → $JSON_PATH”; }
[[ “$JSON_OUT” -eq 1 ]] && printf “%s\n” “$payload”
fi

exit “$EC”