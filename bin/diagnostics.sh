#!/usr/bin/env bash

==============================================================================

🛰️ SpectraMind V50 — bin/diagnostics.sh (ultimate, upgraded)

——————————————————————————

Purpose:

Run rich diagnostics (smoothness, dashboard, optional symbolic overlays)

for a given run/output directory, with manifest + structured audit log.



What it does

• Light or full dashboard via SpectraMind CLI (UMAP/t-SNE toggles)

• Smoothness HTML generation

• Optional symbolic overlays / violation tables

• Optional –source (dir/file) routed through to CLI (if supported)

• Deterministic env toggles for CI parity

• JSON/JSON-path summary, log append to logs/v50_debug_log.md

• CI/Kaggle-safe (no opener there, light defaults if desired)



Usage:

./bin/diagnostics.sh [options]



Common options:

–outdir           Target directory (default: outputs/diagnostics/)

–source          Optional source dir or file to diagnose

–overrides “”   Quoted Hydra overrides (passed to CLI)

–extra “”         Extra args to pass after CLI subcommands

–no-umap               Skip UMAP in dashboard

–no-tsne               Skip t-SNE in dashboard

–symbolic              Include symbolic overlays/violation tables

–open                  Open newest HTML after run (skips in CI/Kaggle)

–manifest              Write JSON manifest into outdir

–timeout          Per-step timeout if timeout exists (default: 300)

–no-poetry             Do not use Poetry; use system spectramind/python -m

–deterministic 0|1     Export deterministic env toggles (default: 1)

–threads            OMP/MKL/BLAS threads (default: 1)

–json                  Emit JSON result summary to stdout

–json-path       Also write JSON summary to file

–quiet                 Less verbose output

-h|–help               Show help



Exit codes

0 OK, 1 failure, 2 usage error, 3 selftest failed



Notes

- Appends a one-line entry to logs/v50_debug_log.md (cmd=diagnostics)

- Idempotent; safe to re-run

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
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[DIAGX]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[DIAGX]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[DIAGX]%s %s\n’ “${RED}” “${RST}” “$*” >&2; }

––––– Defaults –––––

OUTDIR=””
SOURCE=””
OVERRIDES=””
EXTRA=””
NO_UMAP=0
NO_TSNE=0
WITH_SYMBOLIC=0
OPEN_AFTER=0
WRITE_MANIFEST=0
TIMEOUT=”${DIAG_TIMEOUT:-300}”
USE_POETRY=1
DETERMINISTIC=1
THREADS=1
QUIET=0
JSON_OUT=0
JSON_PATH=””

CLI_ENV_BIN=”${SPECTRAMIND_CLI:-}”
LOG_FILE=”${LOG_FILE:-logs/v50_debug_log.md}”

––––– Helpers –––––

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; }
ts()     { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha() { git rev-parse –short HEAD 2>/dev/null || echo “nogit”; }
mkdirp() { mkdir -p “$1” 2>/dev/null || true; }
have()   { command -v “$1” >/dev/null 2>&1; }

open_path() {
local p=”$1”
if have xdg-open; then xdg-open “$p” >/dev/null 2>&1 || true
elif have open; then open “$p” >/dev/null 2>&1 || true
elif have code; then code -r “$p” >/dev/null 2>&1 || true
fi
}

with_timeout() {
if have timeout && [[ “${TIMEOUT:-0}” -gt 0 ]]; then
timeout –preserve-status –signal=TERM “${TIMEOUT}” “$@”
else
“$@”
fi
}

latest_html() { ls -t “${OUTDIR}”/*.html 2>/dev/null | head -n1 || true; }

read_cfg_hash() {
local h=”-”
if [[ -f “run_hash_summary_v50.json” ]]; then
h=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ run_hash_summary_v50.json 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)”
[[ -z “$h” ]] && h=”-”
fi
printf ‘%s’ “$h”
}

write_structured_log() {

[ISO] cmd=diagnostics git= cfg_hash=<hash|-> tag=_ pred=_ bundle=_ notes=“outdir=…;symbolic=…;no_umap=…;no_tsne=…;status=ok|fail”

local status=”$1”
mkdirp “$(dirname “$LOG_FILE”)”
printf ‘[%s] cmd=diagnostics git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=“outdir=%s;symbolic=%s;no_umap=%s;no_tsne=%s;status=%s”%s’ 
“$(ts)” “$(gitsha)” “$(read_cfg_hash)” 
“${OUTDIR}” “$([[ $WITH_SYMBOLIC -eq 1 ]] && echo true || echo false)” 
“$([[ $NO_UMAP -eq 1 ]] && echo true || echo false)” 
“$([[ $NO_TSNE -eq 1 ]] && echo true || echo false)” 
“$status” $’\n’ | tee -a “$LOG_FILE” >/dev/null
}

––––– Args –––––

if have getopt; then
PARSED=$(getopt -o h –long help,outdir:,source:,overrides:,extra:,no-umap,no-tsne,symbolic,open,manifest,timeout:,no-poetry,deterministic:,threads:,json,json-path:,quiet – “$@”) || { usage; exit 2; }
eval set – “$PARSED”
fi

while [[ $# -gt 0 ]]; do
case “${1:-}” in
-h|–help) usage; exit 0 ;;
–outdir) OUTDIR=”${2:?}”; shift 2 ;;
–source) SOURCE=”${2:?}”; shift 2 ;;
–overrides) OVERRIDES=”${2:-}”; shift 2 ;;
–extra) EXTRA=”${2:-}”; shift 2 ;;
–no-umap) NO_UMAP=1; shift ;;
–no-tsne) NO_TSNE=1; shift ;;
–symbolic) WITH_SYMBOLIC=1; shift ;;
–open) OPEN_AFTER=1; shift ;;
–manifest) WRITE_MANIFEST=1; shift ;;
–timeout) TIMEOUT=”${2:?}”; shift 2 ;;
–no-poetry) USE_POETRY=0; shift ;;
–deterministic) DETERMINISTIC=”${2:?}”; shift 2 ;;
–threads) THREADS=”${2:?}”; shift 2 ;;
–json) JSON_OUT=1; shift ;;
–json-path) JSON_PATH=”${2:?}”; shift 2 ;;
–quiet) QUIET=1; shift ;;
–) shift; break ;;
*) break ;;
esac
done

RUN_TS=”$(ts)”
GIT_SHA=”$(gitsha)”
RUN_ID=”${RUN_TS}-${GIT_SHA}”
[[ -n “$OUTDIR” ]] || OUTDIR=“outputs/diagnostics/${RUN_TS}”
mkdirp “$OUTDIR”
mkdirp “logs”

––––– Repo root –––––

if have git && git rev-parse –show-toplevel >/dev/null 2>&1; then
cd “$(git rev-parse –show-toplevel)”
else
SCRIPT_DIR=”$(cd – “$(dirname – “${BASH_SOURCE[0]}”)” && pwd)”
cd “$SCRIPT_DIR/..” || { fail “Cannot locate repo root”; exit 1; }
fi

––––– Env detect –––––

IS_KAGGLE=0; [[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=1
IS_CI=0; [[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=1

––––– Determinism toggles –––––

if [[ “${DETERMINISTIC:-1}” -eq 1 ]]; then
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=”${THREADS}”
export MKL_NUM_THREADS=”${THREADS}”
export OPENBLAS_NUM_THREADS=”${THREADS}”
export NUMEXPR_NUM_THREADS=”${THREADS}”
export MPLBACKEND=Agg
fi

––––– Pick CLI –––––

resolve_cli() {
if [[ -n “$CLI_ENV_BIN” ]]; then printf ‘%s’ “$CLI_ENV_BIN”; return; fi
if [[ “$USE_POETRY” -eq 1 ]] && have poetry; then
printf ‘%s’ “poetry run spectramind”; return
fi
if have spectramind; then printf ‘%s’ “spectramind”; return; fi
if have python3; then printf ‘%s’ “python3 -m spectramind”; return; fi
if have python; then printf ‘%s’ “python -m spectramind”; return; fi
fail “SpectraMind CLI not found. Install or set SPECTRAMIND_CLI.”
exit 2
}
CLI=”$(resolve_cli)”

say “Run ID : ${BOLD}${RUN_ID}${RST}”
say “Outdir : ${BOLD}${OUTDIR}${RST}”
[[ -n “$SOURCE” ]]   && say “Source : ${BOLD}${SOURCE}${RST}”
[[ -n “$OVERRIDES” ]]&& say “Hydra  : ${BOLD}${OVERRIDES}${RST}”
[[ -n “$EXTRA” ]]    && say “Extra  : ${BOLD}${EXTRA}${RST}”
say “CLI    : ${BOLD}${CLI}${RST}”
say “Env    : CI=${IS_CI} Kaggle=${IS_KAGGLE} Deterministic=${DETERMINISTIC} Threads=${THREADS}”

––––– Self-test (fast) –––––

say “▶ Self-test (fast)”
if ! with_timeout bash -lc “$CLI test –fast”; then
fail “Self-test failed”
write_structured_log “fail”
exit 3
fi

Prepare argument arrays (preserve user quoting by appending strings at eval site)

read -r -a EXTRA_ARR <<< “$EXTRA”
read -r -a OVR_ARR   <<< “$OVERRIDES”

Smoothness

say “▶ Diagnostics: smoothness”
SMOOTH_CMD=(bash -lc)
SMOOTH_PAYLOAD=”$CLI diagnose smoothness –outdir ‘${OUTDIR}’”
[[ -n “$SOURCE” ]]    && SMOOTH_PAYLOAD+=” –source ‘$(printf “%q” “$SOURCE”)’”
[[ -n “$OVERRIDES” ]] && SMOOTH_PAYLOAD+=” ${OVERRIDES}”
[[ -n “$EXTRA” ]]     && SMOOTH_PAYLOAD+=” ${EXTRA}”
SMOOTH_CMD+=(”$SMOOTH_PAYLOAD”)
SMOOTH_OK=1
with_timeout “${SMOOTH_CMD[@]}” || { warn “smoothness returned non-zero”; SMOOTH_OK=0; }

Symbolic overlays (optional)

SYM_OK=1
if [[ “$WITH_SYMBOLIC” -eq 1 ]]; then
say “▶ Diagnostics: symbolic overlays”
SYM_CMD=(bash -lc)
SYM_PAYLOAD=”$CLI diagnose symbolic-rank –outdir ‘${OUTDIR}’”
[[ -n “$SOURCE” ]]    && SYM_PAYLOAD+=” –source ‘$(printf “%q” “$SOURCE”)’”
[[ -n “$OVERRIDES” ]] && SYM_PAYLOAD+=” ${OVERRIDES}”
[[ -n “$EXTRA” ]]     && SYM_PAYLOAD+=” ${EXTRA}”
SYM_CMD+=(”$SYM_PAYLOAD”)
with_timeout “${SYM_CMD[@]}” || { warn “symbolic-rank returned non-zero”; SYM_OK=0; }
fi

Dashboard

say “▶ Diagnostics: dashboard”
DASH_CMD=(bash -lc)
DASH_PAYLOAD=”$CLI diagnose dashboard –outdir ‘${OUTDIR}’”
[[ “$NO_UMAP” -eq 1 ]] && DASH_PAYLOAD+=” –no-umap”
[[ “$NO_TSNE” -eq 1 ]] && DASH_PAYLOAD+=” –no-tsne”
[[ -n “$SOURCE” ]]     && DASH_PAYLOAD+=” –source ‘$(printf “%q” “$SOURCE”)’”
[[ -n “$OVERRIDES” ]]  && DASH_PAYLOAD+=” ${OVERRIDES}”
[[ -n “$EXTRA” ]]      && DASH_PAYLOAD+=” ${EXTRA}”
DASH_CMD+=(”$DASH_PAYLOAD”)
DASH_OK=1
with_timeout “${DASH_CMD[@]}” || { warn “dashboard returned non-zero”; DASH_OK=0; }

Summary file

SUMMARY=”${OUTDIR}/diagnostics_summary.txt”
{
echo “Diagnostics summary”
echo “time_utc : ${RUN_TS}”
echo “run_id   : ${RUN_ID}”
echo “git_sha  : ${GIT_SHA}”
echo “cfg_hash : $(read_cfg_hash)”
echo “outdir   : ${OUTDIR}”
echo “source   : ${SOURCE:-n/a}”
echo “symbolic : $([[ $WITH_SYMBOLIC -eq 1 ]] && echo true || echo false)”
echo “flags    : no_umap=$([[ $NO_UMAP -eq 1 ]] && echo true || echo false) no_tsne=$([[ $NO_TSNE -eq 1 ]] && echo true || echo false)”
echo “steps    : smoothness=$([[ $SMOOTH_OK -eq 1 ]] && echo ok || echo fail) symbolic=$([[ $WITH_SYMBOLIC -eq 1 ]] && ([[ $SYM_OK -eq 1 ]] && echo ok || echo fail) || echo n/a) dashboard=$([[ $DASH_OK -eq 1 ]] && echo ok || echo fail)”
} > “$SUMMARY”
say “Summary → ${BOLD}${SUMMARY}${RST}”

Manifest (optional)

if [[ “$WRITE_MANIFEST” -eq 1 ]]; then
MANIFEST=”${OUTDIR}/diagnostics_manifest_${RUN_ID}.json”
python3 - “$MANIFEST” <<‘PY’
import json, os, time, glob
manifest_path = os.sys.argv[1]
outdir = os.path.dirname(manifest_path)
def read_cfg():
p = “run_hash_summary_v50.json”
if os.path.exists(p):
import re
txt = open(p,‘r’,encoding=‘utf-8’,errors=‘ignore’).read()
m = import(‘re’).search(r’“config_hash”\s*:\s*”([^”]+)”’, txt)
return m.group(1) if m else “-”
return “-”
try:
latest_html = sorted(glob.glob(os.path.join(outdir, “*.html”)), key=os.path.getmtime, reverse=True)[0]
except Exception:
latest_html = “”
payload = {
“run_id”: os.environ.get(“RUN_ID”,””),
“ts_utc”: time.strftime(”%Y-%m-%dT%H:%M:%SZ”, time.gmtime()),
“git_sha”: os.environ.get(“GIT_SHA”,””),
“cfg_hash”: read_cfg(),
“outdir”: outdir,
“source”: os.environ.get(“SOURCE”,””),
“symbolic”: os.environ.get(“WITH_SYMBOLIC”,“0”) in (“1”,“true”,“True”),
“steps”: {
“smoothness”: os.environ.get(“SMOOTH_OK”,“1”)==“1”,
“symbolic”: os.environ.get(“SYM_OK”,“1”)==“1” if os.environ.get(“WITH_SYMBOLIC”,“0”) in (“1”,“true”,“True”) else None,
“dashboard”: os.environ.get(“DASH_OK”,“1”)==“1”,
},
“latest_html”: latest_html or None
}
open(manifest_path,“w”,encoding=“utf-8”).write(json.dumps(payload, indent=2))
print(manifest_path)
PY
say “Manifest → ${BOLD}${MANIFEST}${RST}”
fi

Open (optional, local only)

if [[ “$OPEN_AFTER” -eq 1 ]]; then
if [[ “$IS_CI” -eq 1 || “$IS_KAGGLE” -eq 1 ]]; then
warn “Skipping –open in CI/Kaggle.”
else
LHTML=”$(latest_html)”
if [[ -n “$LHTML” ]]; then
say “Opening ${LHTML}”
open_path “$LHTML”
else
warn “No HTML report found in ${OUTDIR}”
fi
fi
fi

Final status

if [[ “$SMOOTH_OK” -eq 1 && “$DASH_OK” -eq 1 && ( “$WITH_SYMBOLIC” -eq 0 || “$SYM_OK” -eq 1 ) ]]; then
printf “%s✔%s Diagnostics completed.\n” “${GRN}” “${RST}”
write_structured_log “ok”
OK=true; EC=0
else
printf “%s✘%s Diagnostics encountered issues.\n” “${RED}” “${RST}”
write_structured_log “fail”
OK=false; EC=1
fi

JSON summary (stdout / file)

if [[ “$JSON_OUT” -eq 1 || -n “${JSON_PATH:-}” ]]; then
LHTML=”$(latest_html)”
payload=$(
printf ‘{’
printf ’“ok”: %s, ’ “$([[ $OK == true ]] && echo true || echo false)”
printf ’“run_id”: “%s”, “git_sha”: “%s”, “cfg_hash”: “%s”, ’ “$RUN_ID” “$GIT_SHA” “$(read_cfg_hash)”
printf ’“outdir”: “%s”, “source”: “%s”, ’ “$OUTDIR” “${SOURCE:-}”
printf ’“symbolic”: %s, ’ “$([[ $WITH_SYMBOLIC -eq 1 ]] && echo true || echo false)”
printf ’“no_umap”: %s, “no_tsne”: %s, ’ “$([[ $NO_UMAP -eq 1 ]] && echo true || echo false)” “$([[ $NO_TSNE -eq 1 ]] && echo true || echo false)”
printf ’“steps”: {“smoothness”: %s, “symbolic”: %s, “dashboard”: %s}, ’ 
“$([[ $SMOOTH_OK -eq 1 ]] && echo true || echo false)” 
“$([[ $WITH_SYMBOLIC -eq 1 ]] && ([[ $SYM_OK -eq 1 ]] && echo true || echo false) || echo null)” 
“$([[ $DASH_OK -eq 1 ]] && echo true || echo false)”
printf ‘“latest_html”: “%s”’ “${LHTML:-}”
printf ‘}’
)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; say “Wrote JSON → ${JSON_PATH}”; }
[[ “$JSON_OUT” -eq 1 ]] && printf “%s\n” “$payload”
fi

exit “$EC”