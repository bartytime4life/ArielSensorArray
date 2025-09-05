#!/usr/bin/env bash

==============================================================================

🛰️ SpectraMind V50 — bin/benchmark.sh (ultimate, upgraded)

——————————————————————————

Purpose:

Run a standardized benchmark pass with logging + reproducibility guardrails.

- Self-test (optional) → train (few epochs) → diagnostics → summary/manifest

- CI/Kaggle safe (non-interactive, optional opening locally)

- Poetry-aware CLI launcher (falls back to spectramind or python -m)

- Deterministic toggles (hash seed, single-thread CPU by default)

- Writes one-line structured entry to logs/v50_debug_log.md



Usage:

./bin/benchmark.sh [options]



Common options:

–profile {cpu|gpu}       Device profile (default: gpu)

–epochs               Training epochs (default: 1)

–seed                 Random seed (default: 42)

–overrides “”     Quoted Hydra overrides (e.g. ‘+training.lr=3e-4 +training.bs=64’)

–extra “”           Extra args passed to CLI (e.g. ‘–no-umap –html v3.html’)

–outdir             Output dir (default: benchmarks/_)

–tag                Tag label for logs/summary (default: “”)

–manifest                Write JSON manifest in outdir

–open-report             Open newest HTML in outdir after run (skips on CI/Kaggle)

–dry-run                 Print planned commands; skip execution



Advanced options:

–no-selftest             Skip CLI self-test

–skip-diagnostics        Skip diagnostics phase

–poetry / –no-poetry    Force enable/disable Poetry launcher detection

–cli                Explicit CLI (e.g., ‘poetry run spectramind’)

–timeout            Timeout per invoked CLI step (default: 600)

–retries              Retries for train step (default: 2)

–sleep              Sleep between retries (default: 5)

–deterministic {0|1}     Export deterministic env toggles (default: 1)

–threads              OMP/MKL threads (default: 1)

–title “”          Title for summary.txt (default shown in file)



Notes:

- Appends structured log line to logs/v50_debug_log.md:

[ISO] cmd=benchmark git= cfg_hash=<hash|-> tag=<tag|-> pred=_ bundle=_ notes=“outdir=<…> epochs=<…> profile=<…>”

- If run_hash_summary_v50.json exists in repo root or OUTDIR, cfg_hash is read.

- Emits /summary.txt and (if –manifest) /benchmark_manifest_<RUN_ID>.json

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– Pretty –––––

BOLD=$’\033[1m’; DIM=$’\033[2m’; RED=$’\033[31m’; GRN=$’\033[32m’; CYN=$’\033[36m’; YLW=$’\033[33m’; RST=$’\033[0m’
say()  { printf ‘%s[BMK]%s %s\n’ “${CYN}” “${RST}” “$”; }
ok()   { printf ‘%s[BMK]%s %s\n’ “${GRN}” “${RST}” “$”; }
warn() { printf ‘%s[BMK]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[BMK]%s %s\n’ “${RED}” “${RST}” “$” >&2; }

––––– Defaults –––––

PROFILE=“gpu”
EPOCHS=1
SEED=42
OVERRIDES=””
EXTRA=””
OUTDIR=””
TAG=””
WRITE_MANIFEST=0
OPEN_AFTER=0
DRY_RUN=0

DO_SELFTEST=1
DO_DIAG=1
FORCE_POETRY=””           # “”, “on”, or “off”
CLI_EXPLICIT=””

STEP_TIMEOUT=600
RETRIES=2
SLEEP_BT=5

DETERMINISTIC=1
THREADS=1
TITLE=“SpectraMind V50 — Benchmark Summary”

CLI + logs

CLI_ENV_BIN=”${SPECTRAMIND_CLI:-}”
LOG_FILE=”${LOG_FILE:-logs/v50_debug_log.md}”

––––– Env detect –––––

IS_KAGGLE=0
[[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=1
IS_CI=0
[[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=1

––––– Helpers –––––

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; exit 0; }
have() { command -v “$1” >/dev/null 2>&1; }
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gitsha() { git rev-parse –short HEAD 2>/dev/null || echo “nogit”; }
repo_root() { git rev-parse –show-toplevel 2>/dev/null || pwd; }
open_path() {
local p=”$1”
if [[ “$IS_KAGGLE” -eq 1 || “$IS_CI” -eq 1 ]]; then return 0; fi
if have xdg-open; then xdg-open “$p” >/dev/null 2>&1 || true
elif have open; then open “$p” >/dev/null 2>&1 || true
elif have code; then code -r “$p” >/dev/null 2>&1 || true
fi
}
with_timeout() {
if have timeout && [[ “${STEP_TIMEOUT:-0}” -gt 0 ]]; then
timeout –preserve-status –signal=TERM “${STEP_TIMEOUT}” “$@”
else
“$@”
fi
}
retry_run() {
local attempts=”$1” sleep_s=”$2”; shift 2
local n=1
until “$@”; do
if (( n >= attempts )); then return 1; fi
warn “Step failed (attempt ${n}/${attempts}), sleeping ${sleep_s}s and retrying…”
sleep “$sleep_s”; ((n++))
done
}
mkdirp() { mkdir -p “$1” 2>/dev/null || true; }

read_cfg_hash() {
local h=”-”
local R=”$(repo_root)”
if [[ -f “${R}/run_hash_summary_v50.json” ]]; then
h=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “${R}/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)”
fi
if [[ “$h” == “-” && -n “${OUTDIR:-}” && -f “${OUTDIR}/run_hash_summary_v50.json” ]]; then
h=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “${OUTDIR}/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)”
fi
[[ -z “$h” ]] && h=”-”
printf ‘%s’ “$h”
}

write_structured_log() {
local note_outdir=”$1” note_epochs=”$2” note_profile=”$3” note_status=”$4”
local ts_iso cmd git cfg tag pred bundle notes
ts_iso=”$(ts)”
cmd=“benchmark”
git=”$(gitsha)”
cfg=”$(read_cfg_hash)”
tag=”${TAG:-}”; [[ -z “$tag” ]] && tag=””
pred=””; bundle=””
notes=“outdir=${note_outdir};epochs=${note_epochs};profile=${note_profile};status=${note_status}”
mkdirp “$(dirname “$LOG_FILE”)”
printf ‘[%s] cmd=%s git=%s cfg_hash=%s tag=%s pred=%s bundle=%s notes=”%s”\n’ 
“$ts_iso” “$cmd” “$git” “$cfg” “$tag” “$pred” “$bundle” “$notes” 
| tee -a “$LOG_FILE” >/dev/null
}

––––– Argparse –––––

if have getopt; then
PARSED=$(getopt -o h –long help,profile:,epochs:,seed:,overrides:,extra:,outdir:,tag:,manifest,open-report,dry-run,no-selftest,skip-diagnostics,poetry,no-poetry,cli:,timeout:,retries:,sleep:,deterministic:,threads:,title: – “$@”) || usage
eval set – “$PARSED”
while true; do
case “$1” in
-h|–help) usage ;;
–profile) PROFILE=”$2”; shift 2 ;;
–epochs) EPOCHS=”$2”; shift 2 ;;
–seed) SEED=”$2”; shift 2 ;;
–overrides) OVERRIDES=”$2”; shift 2 ;;
–extra) EXTRA=”$2”; shift 2 ;;
–outdir) OUTDIR=”$2”; shift 2 ;;
–tag) TAG=”$2”; shift 2 ;;
–manifest) WRITE_MANIFEST=1; shift ;;
–open-report) OPEN_AFTER=1; shift ;;
–dry-run) DRY_RUN=1; shift ;;
–no-selftest) DO_SELFTEST=0; shift ;;
–skip-diagnostics) DO_DIAG=0; shift ;;
–poetry) FORCE_POETRY=“on”; shift ;;
–no-poetry) FORCE_POETRY=“off”; shift ;;
–cli) CLI_EXPLICIT=”$2”; shift 2 ;;
–timeout) STEP_TIMEOUT=”$2”; shift 2 ;;
–retries) RETRIES=”$2”; shift 2 ;;
–sleep) SLEEP_BT=”$2”; shift 2 ;;
–deterministic) DETERMINISTIC=”$2”; shift 2 ;;
–threads) THREADS=”$2”; shift 2 ;;
–title) TITLE=”$2”; shift 2 ;;
–) shift; break ;;
*) fail “Unknown option: $1”; exit 2 ;;
esac
done
else

Fallback minimal parser

while [[ $# -gt 0 ]]; do
case “$1” in
-h|–help) usage ;;
–profile) PROFILE=”$2”; shift 2 ;;
–epochs) EPOCHS=”$2”; shift 2 ;;
–seed) SEED=”$2”; shift 2 ;;
–overrides) OVERRIDES=”$2”; shift 2 ;;
–extra) EXTRA=”$2”; shift 2 ;;
–outdir) OUTDIR=”$2”; shift 2 ;;
–tag) TAG=”$2”; shift 2 ;;
–manifest) WRITE_MANIFEST=1; shift ;;
–open-report) OPEN_AFTER=1; shift ;;
–dry-run) DRY_RUN=1; shift ;;
–no-selftest) DO_SELFTEST=0; shift ;;
–skip-diagnostics) DO_DIAG=0; shift ;;
–poetry) FORCE_POETRY=“on”; shift ;;
–no-poetry) FORCE_POETRY=“off”; shift ;;
–cli) CLI_EXPLICIT=”$2”; shift 2 ;;
–timeout) STEP_TIMEOUT=”$2”; shift 2 ;;
–retries) RETRIES=”$2”; shift 2 ;;
–sleep) SLEEP_BT=”$2”; shift 2 ;;
–deterministic) DETERMINISTIC=”$2”; shift 2 ;;
–threads) THREADS=”$2”; shift 2 ;;
–title) TITLE=”$2”; shift 2 ;;
*) fail “Unknown option: $1”; exit 2 ;;
esac
done
fi

RUN_TS=”$(ts)”
GIT_SHA=”$(gitsha)”
RUN_ID=”${RUN_TS}-${GIT_SHA}”
[[ -n “$OUTDIR” ]] || OUTDIR=“benchmarks/${RUN_TS}_${PROFILE}”
mkdirp “$OUTDIR”
mkdirp “logs”

––––– Determinism toggles –––––

if [[ “$DETERMINISTIC” -eq 1 ]]; then
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=”${THREADS}”
export MKL_NUM_THREADS=”${THREADS}”
export NUMEXPR_NUM_THREADS=”${THREADS}”
export OPENBLAS_NUM_THREADS=”${THREADS}”
export MPLBACKEND=Agg
fi

––––– Resolve CLI launcher –––––

resolve_cli() {
local bin=””
if [[ -n “$CLI_EXPLICIT” ]]; then
bin=”$CLI_EXPLICIT”
elif [[ -n “$CLI_ENV_BIN” ]]; then
bin=”$CLI_ENV_BIN”
else
case “$FORCE_POETRY” in
on)
if have poetry; then bin=“poetry run spectramind”; fi
;;
off)
if have spectramind; then bin=“spectramind”
elif have python3; then bin=“python3 -m spectramind”
fi
;;
*)
if have poetry; then bin=“poetry run spectramind”
elif have spectramind; then bin=“spectramind”
elif have python3; then bin=“python3 -m spectramind”
fi
;;
esac
fi
if [[ -z “$bin” ]]; then
fail “No SpectraMind CLI found (poetry/spectramind/python3). Provide –cli or install.”
exit 1
fi
printf ‘%s’ “$bin”
}

CLI=”$(resolve_cli)”
ok “CLI: $CLI”

––––– Plan –––––

say “Run ID   : ${RUN_ID}”
say “Profile  : ${PROFILE} | Epochs=${EPOCHS} | Seed=${SEED}”
say “Outdir   : ${OUTDIR}”
[[ -n “$TAG” ]] && say “Tag      : ${TAG}”
[[ -n “$OVERRIDES” ]] && say “Overrides: ${OVERRIDES}”
[[ -n “$EXTRA” ]] && say “Extra    : ${EXTRA}”
say “Timeout  : ${STEP_TIMEOUT}s | Retries=${RETRIES}/${SLEEP_BT}s”
[[ “$DO_SELFTEST” -eq 0 ]] && warn “Self-test disabled”
[[ “$DO_DIAG” -eq 0 ]] && warn “Diagnostics phase disabled”
[[ “$DRY_RUN” -eq 1 ]] && warn “Dry-run mode: no commands will execute”

––––– Logging preamble –––––

{
echo “[benchmark] ========================================================”
echo “[benchmark] Start  : ${RUN_TS}”
echo “[benchmark] RUN_ID : ${RUN_ID}”
echo “[benchmark] Profile: ${PROFILE}  Epochs:${EPOCHS}  Seed:${SEED}  Tag:${TAG}”
[[ -n “$OVERRIDES” ]] && echo “[benchmark] Overrides: ${OVERRIDES}”
[[ -n “$EXTRA” ]] && echo “[benchmark] Extra    : ${EXTRA}”
} | tee -a “$LOG_FILE” >/dev/null

––––– Trap for failure to emit structured log –––––

on_fail() {
fail “❌ Failed at $(ts) (RUN_ID=${RUN_ID})”
write_structured_log “$OUTDIR” “$EPOCHS” “$PROFILE” “fail”
exit 1
}
trap on_fail ERR

––––– Build arrays from quoted strings –––––

Safely split OVERRIDES/EXTRA into arrays (word-safely, respecting quotes)

read -r -a OV_ARR <<< “$OVERRIDES”
read -r -a EX_ARR <<< “$EXTRA”

––––– Self-test –––––

if [[ “$DO_SELFTEST” -eq 1 ]]; then
say “▶ Self-test (fast)”
if [[ “$DRY_RUN” -eq 1 ]]; then
warn “[dry-run] $CLI test –fast”
else
with_timeout bash -lc “$CLI test –fast”
fi
fi

––––– Train –––––

say “▶ Train (${EPOCHS} epochs, seed=${SEED}, device=${PROFILE})”
TRAIN_CMD=(bash -lc)
TRAIN_PAYLOAD=”$CLI train +training.epochs=${EPOCHS} +training.seed=${SEED} –device ${PROFILE} –outdir ‘${OUTDIR}’”

Append overrides/extras textually to preserve quoting

if [[ ${#OV_ARR[@]} -gt 0 ]]; then TRAIN_PAYLOAD+=” ${OVERRIDES}”; fi
if [[ ${#EX_ARR[@]} -gt 0 ]]; then TRAIN_PAYLOAD+=” ${EXTRA}”; fi
TRAIN_CMD+=(”$TRAIN_PAYLOAD”)

if [[ “$DRY_RUN” -eq 1 ]]; then
warn “[dry-run] ${TRAIN_CMD[*]}”
else
if ! retry_run “$RETRIES” “$SLEEP_BT” with_timeout “${TRAIN_CMD[@]}”; then
fail “Training step failed after ${RETRIES} attempt(s).”
exit 1
fi
fi

––––– Diagnostics –––––

if [[ “$DO_DIAG” -eq 1 ]]; then
say “▶ Diagnostics (smoothness + dashboard)”
DIAG1=(bash -lc); DIAG2=(bash -lc)
D1_PAYLOAD=”$CLI diagnose smoothness –outdir ‘${OUTDIR}’”
D2_PAYLOAD=”$CLI diagnose dashboard –outdir ‘${OUTDIR}’”
if [[ ${#EX_ARR[@]} -gt 0 ]]; then
D1_PAYLOAD+=” ${EXTRA}”
D2_PAYLOAD+=” ${EXTRA}”
fi
DIAG1+=(”$D1_PAYLOAD”)
DIAG2+=(”$D2_PAYLOAD”)
if [[ “$DRY_RUN” -eq 1 ]]; then
warn “[dry-run] ${DIAG1[]}”
warn “[dry-run] ${DIAG2[]}”
else
with_timeout “${DIAG1[@]}” || warn “smoothness diagnostics returned non-zero”
with_timeout “${DIAG2[@]}” || warn “dashboard diagnostics returned non-zero”
fi
fi

––––– Summary –––––

SUMMARY=”${OUTDIR}/summary.txt”
{
echo “${TITLE}”
echo “time_utc : $(ts)”
echo “run_id   : ${RUN_ID}”
echo “git_sha  : ${GIT_SHA}”
echo “cfg_hash : $(read_cfg_hash)”
echo “profile  : ${PROFILE}”
echo “epochs   : ${EPOCHS}”
echo “seed     : ${SEED}”
echo “tag      : ${TAG}”
echo “outdir   : ${OUTDIR}”
} > “$SUMMARY”
ok “Summary → $SUMMARY”

––––– Manifest (optional) –––––

if [[ “$WRITE_MANIFEST” -eq 1 ]]; then
MANIFEST=”${OUTDIR}/benchmark_manifest_${RUN_ID}.json”
python3 - “$MANIFEST” <<‘PY’
import json, os, sys, time, glob
manifest_path = sys.argv[1]
outdir = os.path.dirname(manifest_path)
def read_cfg_hash():
for p in (os.path.join(os.getcwd(), “run_hash_summary_v50.json”),
os.path.join(outdir, “run_hash_summary_v50.json”)):
try:
import re
txt = open(p, ‘r’, encoding=‘utf-8’, errors=‘ignore’).read()
m = re.search(r’“config_hash”\s*:\s*”([^”]+)”’, txt)
if m: return m.group(1)
except Exception:
pass
return “-”
latest_html = “”
try:
latest_html = sorted(glob.glob(os.path.join(outdir, “*.html”)), key=os.path.getmtime, reverse=True)[0]
except Exception:
latest_html = “”
payload = {
“run_id”: os.environ.get(“RUN_ID”,””),
“ts_utc”: time.strftime(”%Y-%m-%dT%H:%M:%SZ”, time.gmtime()),
“git_sha”: os.environ.get(“GIT_SHA”,””),
“cfg_hash”: read_cfg_hash(),
“profile”: os.environ.get(“PROFILE”,””),
“epochs”: int(os.environ.get(“EPOCHS”,“1”)),
“seed”: int(os.environ.get(“SEED”,“42”)),
“tag”: os.environ.get(“TAG”,””),
“outdir”: outdir,
“latest_html”: latest_html or None
}
with open(manifest_path, “w”, encoding=“utf-8”) as f:
json.dump(payload, f, indent=2)
print(manifest_path)
PY
ok “Manifest → $MANIFEST”
fi

––––– Open (optional) –––––

if [[ “$OPEN_AFTER” -eq 1 ]]; then
latest_html=”$(ls -t “${OUTDIR}”/*.html 2>/dev/null | head -n1 || true)”
if [[ -n “$latest_html” ]]; then
say “Opening ${latest_html}”
open_path “$latest_html”
else
warn “No HTML report found in ${OUTDIR}”
fi
fi

––––– Structured log (success) –––––

write_structured_log “$OUTDIR” “$EPOCHS” “$PROFILE” “ok”

––––– Footer –––––

{
echo “[benchmark] Completed at $(ts)  (RUN_ID=${RUN_ID})”
echo “[benchmark] ========================================================”
} | tee -a “$LOG_FILE” >/dev/null

ok “✅ Benchmark complete”
exit 0