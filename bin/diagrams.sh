#!/usr/bin/env bash

==============================================================================

bin/diagrams.sh — Render Mermaid diagrams for SpectraMind V50 (ultimate)

——————————————————————————

Purpose

CI- and dev-friendly wrapper to render Mermaid diagrams (SVG/PNG/PDF) from

*.mmd sources (and optionally from Markdown via a Python exporter).



Upgrades in this version

• Source discovery with include/exclude globs and explicit paths

• Parallel rendering (-j) using xargs -P (portable), with fail-fast exit code

• Cache-aware: skips rendering if outputs are newer than source+theme/css/config

• Prune mode: remove orphaned outputs that no longer have sources

• mmdc resolution order: –mmdc → local node_modules → global → npx fallback

• Extra controls: timeout, custom CSS/theme/bg/width/height/scale, extra args

• JSON summary (counts, formats, outdir, list of rendered/ skipped files)

• Watch mode: npm script → watchexec → entr (best-available)

• Lint/format hooks via package.json (optional)

• Dry-run previews, quiet mode, list-only, deterministic env for CI



Usage

bin/diagrams.sh [options] [–] [PATH …]



Common examples

# Render all .mmd files to SVG (default) under outputs/diagrams/

bin/diagrams.sh



# Render to SVG+PNG with dark theme; extract MD-embedded Mermaid

bin/diagrams.sh –png –theme dark –use-python



# Watch diagrams/ for changes and re-render SVG/PNG

bin/diagrams.sh –watch –png



Options

–outdir DIR         Output directory (default: outputs/diagrams)

–svg                Render SVG (default if nothing selected)

–png                Render PNG

–pdf                Render PDF

–theme NAME         Mermaid theme: default|dark|forest|neutral (default: default)

–bg COLOR           Background: transparent|white|#RRGGBB (default: transparent)

–config FILE        Mermaid CLI config (.mermaidrc.mjs/.json)

–css FILE           Custom CSS file for Mermaid CLI

–width N            Canvas width (px) for PNG/PDF

–height N           Canvas height (px) for PNG/PDF

–scale N            Scale factor for raster outputs

–mmdc PATH          Use a specific mmdc binary (fallback: npx @mermaid-js/mermaid-cli mmdc)

–mmdc-args “ARGS”   Extra args passed verbatim to mmdc

–timeout SEC        Timeout per mmdc render (default: 90)

–use-python         Also run scripts/export_mermaid.py for Markdown-embedded Mermaid

–lint               Run npm run lint if present

–format             Run npm run format if present

–watch              Watch for changes (npm run mmd:watch → watchexec → entr)

–clean              Remove output dir before rendering

–prune              Remove orphaned outputs (no matching source)

–jobs N             Parallel jobs (0/1 = serial; default: 0)

–include GLOB       Include glob(s) (repeatable); default: diagrams//*.mmd, docs//*.mmd

–exclude GLOB       Exclude glob(s) (repeatable)

–list               Only list resolved sources and exit

–list-outputs       List would-be output files for current flags and exit

–force              Ignore mtime cache; force re-render

–json               Emit JSON summary to stdout

–dry-run            Show actions without executing

–quiet              Reduce verbosity

-h, –help           Show help



Notes

• Explicit PATH args (files/dirs) override defaults. Directories are searched for *.mmd.

• Exit code is non-zero if any render fails (even in parallel).

• Sets PUZZLE: PUPPETEER_NO_SANDBOX=1 when running in CI/Kaggle for safety.

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– colors –––––

is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; CYN=”$(tput setaf 6)”; YLW=”$(tput setaf 3)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; CYN=””; YLW=””; RST=””
fi
say()  { [[ “${QUIET:-0}” -eq 1 ]] && return 0; printf ‘%s[DIAGRAMS]%s %s\n’ “${CYN}” “${RST}” “$”; }
warn() { printf ‘%s[DIAGRAMS]%s %s\n’ “${YLW}” “${RST}” “$” >&2; }
fail() { printf ‘%s[DIAGRAMS]%s %s\n’ “${RED}” “${RST}” “$*” >&2; exit 1; }

––––– defaults –––––

OUTDIR=“outputs/diagrams”
DO_SVG=0; DO_PNG=0; DO_PDF=0
THEME=“default”; BG=“transparent”
MMDC_BIN=””; MMDC_CONFIG=””; CSS_FILE=””; MMDC_EXTRA=””
WIDTH=””; HEIGHT=””; SCALE=””
USE_PYTHON=0; DO_LINT=0; DO_FORMAT=0; WATCH=0; CLEAN=0; PRUNE=0
JOBS=”${JOBS:-0}”; DRY=0; QUIET=0; JSON=0
TIMEOUT=”${DIAGRAMS_TIMEOUT:-90}”
INCLUDES=(); EXCLUDES=()
ARGS=()
LIST_ONLY=0; LIST_OUTPUTS=0
FORCE=0

––––– env detection & deterministic knobs –––––

IS_CI=0; [[ -n “${GITHUB_ACTIONS:-}” || -n “${CI:-}” ]] && IS_CI=1
IS_KAGGLE=0; [[ -n “${KAGGLE_URL_BASE:-}” || -n “${KAGGLE_KERNEL_RUN_TYPE:-}” || -d “/kaggle” ]] && IS_KAGGLE=1
if [[ $IS_CI -eq 1 || $IS_KAGGLE -eq 1 ]]; then
export PUPPETEER_NO_SANDBOX=1
export NODE_OPTIONS=”${NODE_OPTIONS:-} –max-old-space-size=4096”
fi

usage() { sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’; }

––––– args –––––

while [[ $# -gt 0 ]]; do
case “$1” in
–outdir)     OUTDIR=”${2:?}”; shift 2 ;;
–svg)        DO_SVG=1; shift ;;
–png)        DO_PNG=1; shift ;;
–pdf)        DO_PDF=1; shift ;;
–theme)      THEME=”${2:?}”; shift 2 ;;
–bg)         BG=”${2:?}”; shift 2 ;;
–config)     MMDC_CONFIG=”${2:?}”; shift 2 ;;
–css)        CSS_FILE=”${2:?}”; shift 2 ;;
–width)      WIDTH=”${2:?}”; shift 2 ;;
–height)     HEIGHT=”${2:?}”; shift 2 ;;
–scale)      SCALE=”${2:?}”; shift 2 ;;
–mmdc)       MMDC_BIN=”${2:?}”; shift 2 ;;
–mmdc-args)  MMDC_EXTRA=”${2:?}”; shift 2 ;;
–timeout)    TIMEOUT=”${2:?}”; shift 2 ;;
–use-python) USE_PYTHON=1; shift ;;
–lint)       DO_LINT=1; shift ;;
–format)     DO_FORMAT=1; shift ;;
–watch)      WATCH=1; shift ;;
–clean)      CLEAN=1; shift ;;
–prune)      PRUNE=1; shift ;;
–jobs)       JOBS=”${2:?}”; shift 2 ;;
–include)    INCLUDES+=(”${2:?}”); shift 2 ;;
–exclude)    EXCLUDES+=(”${2:?}”); shift 2 ;;
–list)       LIST_ONLY=1; shift ;;
–list-outputs) LIST_OUTPUTS=1; shift ;;
–force)      FORCE=1; shift ;;
–json)       JSON=1; shift ;;
–dry-run)    DRY=1; shift ;;
–quiet)      QUIET=1; shift ;;
-h|–help)    usage; exit 0 ;;
–)           shift; while [[ $# -gt 0 ]]; do ARGS+=(”$1”); shift; done ;;
*)            ARGS+=(”$1”); shift ;;
esac
done

Defaults: if no formats selected, render SVG

if [[ $DO_SVG -eq 0 && $DO_PNG -eq 0 && $DO_PDF -eq 0 ]]; then DO_SVG=1; fi

––––– repo root –––––

if command -v git >/dev/null 2>&1; then
if ROOT=$(git rev-parse –show-toplevel 2>/dev/null); then cd “$ROOT”; fi
fi

––––– helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }

run() {
local desc=”$1”; shift
if [[ $DRY -eq 1 ]]; then
say “[dry-run] $desc :: $*”
return 0
fi
[[ $QUIET -eq 0 ]] && printf “%s→ %s%s\n” “${DIM}” “${desc}” “${RST}”
“$@”
}

resolve_mmdc() {
if [[ -n “$MMDC_BIN” ]]; then echo “$MMDC_BIN”; return; fi
if [[ -f “node_modules/.bin/mmdc” ]]; then echo “node_modules/.bin/mmdc”; return; fi
if have mmdc; then echo “mmdc”; return; fi
if have npx; then echo “npx -y @mermaid-js/mermaid-cli mmdc”; return; fi
fail “Mermaid CLI not found. Install via ‘npm i -D @mermaid-js/mermaid-cli’, use ‘npx’, or pass –mmdc .”
}
MMDC_CMD=”$(resolve_mmdc)”

mtime_or_zero() { stat -c %Y “$1” 2>/dev/null || stat -f %m “$1” 2>/dev/null || echo 0; }  # Linux/BSD

newer_than_any() {

newer_than_any    …

local target=”$1”; shift
[[ ! -f “$target” ]] && return 0
local t m
t=”$(mtime_or_zero “$target”)”
for s in “$@”; do
m=”$(mtime_or_zero “$s”)”
if (( m > t )); then return 0; fi
done
return 1
}

––––– npm lint/format (optional) –––––

if [[ $DO_LINT -eq 1 && -f package.json ]]; then
if have jq && jq -e ‘.scripts.lint’ package.json >/dev/null 2>&1; then
run “npm run lint” npm run lint
else
warn “package.json has no ‘lint’ script or jq missing.”
fi
fi
if [[ $DO_FORMAT -eq 1 && -f package.json ]]; then
if have jq && jq -e ‘.scripts.format’ package.json >/dev/null 2>&1; then
run “npm run format” npm run format
else
warn “package.json has no ‘format’ script or jq missing.”
fi
fi

––––– prepare –––––

if [[ $CLEAN -eq 1 && -d “$OUTDIR” ]]; then
run “rm -rf $OUTDIR” rm -rf “$OUTDIR”
fi
mkdir -p “$OUTDIR”

––––– glob discovery –––––

collect_from_explicit() {
local -a paths=()
for p in “${ARGS[@]}”; do
if [[ -d “$p” ]]; then
while IFS= read -r -d ‘’ f; do paths+=(”$f”); done < <(find “$p” -type f -name “*.mmd” -print0 2>/dev/null)
elif [[ -f “$p” ]]; then
[[ “$p” == *.mmd ]] && paths+=(”$p”) || warn “Skipping non-.mmd file: $p”
else
warn “Path not found: $p”
fi
done
printf ‘%s\0’ “${paths[@]}”
}

collect_default() {
local -a add=()
if [[ ${#INCLUDES[@]} -gt 0 ]]; then
# Convert includes to find patterns best-effort
for g in “${INCLUDES[@]}”; do
local parent dirpat namepat
parent=”$(dirname “$g”)”; namepat=”$(basename “$g”)”
[[ -d “$parent” ]] || parent=”.”
while IFS= read -r -d ‘’ f; do add+=(”$f”); done < <(find “$parent” -type f -name “$namepat” -print0 2>/dev/null || true)
done
else
[[ -d diagrams ]] && while IFS= read -r -d ‘’ f; do add+=(”$f”); done < <(find diagrams -type f -name “.mmd” -print0 2>/dev/null || true)
[[ -d docs     ]] && while IFS= read -r -d ‘’ f; do add+=(”$f”); done < <(find docs     -type f -name “.mmd” -print0 2>/dev/null || true)
fi

Excludes are simple glob matches

if [[ ${#EXCLUDES[@]} -gt 0 && ${#add[@]} -gt 0 ]]; then
local keep=()
for f in “${add[@]}”; do
local skip=0
for ex in “${EXCLUDES[@]}”; do
[[ “$f” == $ex ]] && skip=1 && break
done
[[ $skip -eq 0 ]] && keep+=(”$f”)
done
printf ‘%s\0’ “${keep[@]}”
else
printf ‘%s\0’ “${add[@]}”
fi
}

declare -a SOURCES=()
if [[ ${#ARGS[@]} -gt 0 ]]; then
while IFS= read -r -d ‘’ s; do SOURCES+=(”$s”); done < <(collect_from_explicit | sort -zu)
else
while IFS= read -r -d ‘’ s; do SOURCES+=(”$s”); done < <(collect_default | sort -zu)
fi

if [[ $LIST_ONLY -eq 1 ]]; then
printf ‘%s\n’ “${SOURCES[@]}”
exit 0
fi

––––– optional: MD extraction via Python –––––

if [[ $USE_PYTHON -eq 1 ]]; then
if [[ -f “scripts/export_mermaid.py” ]]; then
say “Extracting Mermaid blocks from Markdown via scripts/export_mermaid.py”
run “python scripts/export_mermaid.py” python scripts/export_mermaid.py || warn “export_mermaid.py returned non-zero.”
# Refresh sources
SOURCES=()
if [[ ${#ARGS[@]} -gt 0 ]]; then
while IFS= read -r -d ‘’ s; do SOURCES+=(”$s”); done < <(collect_from_explicit | sort -zu)
else
while IFS= read -r -d ‘’ s; do SOURCES+=(”$s”); done < <(collect_default | sort -zu)
fi
else
warn “scripts/export_mermaid.py not found; skipping Markdown extraction.”
fi
fi

if [[ ${#SOURCES[@]} -eq 0 ]]; then
warn “No .mmd sources found to render.”
fi

––––– compute outputs list for current flags –––––

outputs_for_src() {
local src=”$1”
local rel=”${src#./}”
local base=”${rel%.*}”
local out_sub=”$OUTDIR/$(dirname “$base”)”
local name=”$(basename “$base”)”
[[ $DO_SVG -eq 1 ]] && printf ‘%s/%s.svg\0’ “$out_sub” “$name”
[[ $DO_PNG -eq 1 ]] && printf ‘%s/%s.png\0’ “$out_sub” “$name”
[[ $DO_PDF -eq 1 ]] && printf ‘%s/%s.pdf\0’ “$out_sub” “$name”
}

if [[ $LIST_OUTPUTS -eq 1 ]]; then
for s in “${SOURCES[@]}”; do
while IFS= read -r -d ‘’ o; do printf ‘%s\n’ “$o”; done < <(outputs_for_src “$s”)
done
exit 0
fi

––––– prune orphaned outputs –––––

if [[ $PRUNE -eq 1 ]]; then
say “Pruning orphaned outputs from $OUTDIR…”

Build set of valid output stems (relative sans ext)

TMP_STEMS=”$(mktemp -t mmd_stems_XXXX)”; trap ‘rm -f “$TMP_STEMS” “$TMP_REND” “$TMP_LIST”’ EXIT
for s in “${SOURCES[@]}”; do
rel=”${s#./}”; printf ‘%s\n’ “${rel%.*}” >> “$TMP_STEMS”
done
sort -u -o “$TMP_STEMS” “$TMP_STEMS”

Remove outputs with no corresponding stem

while IFS= read -r -d ‘’ out; do
stem=”${out#$OUTDIR/}”; stem=”${stem%.}”
if ! grep -qxF “$stem” “$TMP_STEMS” 2>/dev/null; then
say “Removing orphaned: $out”
[[ $DRY -eq 1 ]] || rm -f “$out”
fi
done < <(find “$OUTDIR” -type f -name “.svg” -o -name “.png” -o -name “.pdf” -print0 2>/dev/null)
fi

––––– renderers –––––

common composer for mmdc args

compose_common_flags() {
local arr=()
[[ -n “$MMDC_CONFIG” ]] && arr+=(–configFile “$MMDC_CONFIG”)
[[ -n “$CSS_FILE”    ]] && arr+=(–cssFile    “$CSS_FILE”)
[[ -n “$WIDTH”  ]] && arr+=(–width  “$WIDTH”)
[[ -n “$HEIGHT” ]] && arr+=(–height “$HEIGHT”)
[[ -n “$SCALE”  ]] && arr+=(–scale  “$SCALE”)
arr+=(–theme “$THEME” –backgroundColor “$BG”)
printf ’%q ’ “${arr[@]}”
}

RENDERED=()
SKIPPED=()
FAILED=0

render_one() {
local src=”$1”
local rel=”${src#./}”
local base=”${rel%.*}”
local out_sub=”$OUTDIR/$(dirname “$base”)”
local name=”$(basename “$base”)”
mkdir -p “$out_sub”

local deps=(”$src”)
[[ -n “$MMDC_CONFIG” ]] && deps+=(”$MMDC_CONFIG”)
[[ -n “$CSS_FILE”    ]] && deps+=(”$CSS_FILE”)

local common; common=”$(compose_common_flags)”
local cmd=”$MMDC_CMD”
[[ -n “$MMDC_EXTRA” ]] && cmd=”$cmd $MMDC_EXTRA”

helper to run one format

_do() {
local ext=”$1”; shift
local flag=”$1”; shift
local out=”$out_sub/$name.$ext”
if [[ $FORCE -eq 0 ]] && ! newer_than_any “$out” “${deps[@]}”; then
SKIPPED+=(”$out”)
return 0
fi
if [[ $DRY -eq 1 ]]; then
say “[dry-run] $ext $src → $out”
return 0
fi
if command -v timeout >/dev/null 2>&1; then
timeout –preserve-status –signal=TERM “$TIMEOUT” bash -lc “$cmd $flag –input ‘$src’ –output ‘$out’ $common”
else
bash -lc “$cmd $flag –input ‘$src’ –output ‘$out’ $common”
fi
RENDERED+=(”$out”)
}

[[ $DO_SVG -eq 1 ]] && _do “svg” “”                || true
[[ $DO_PNG -eq 1 ]] && _do “png” “–png”           || true
[[ $DO_PDF -eq 1 ]] && _do “pdf” “–pdf”           || true
}

export OUTDIR DO_SVG DO_PNG DO_PDF THEME BG MMDC_CMD MMDC_CONFIG CSS_FILE MMDC_EXTRA TIMEOUT DRY QUIET WIDTH HEIGHT SCALE FORCE
export -f render_one newer_than_any mtime_or_zero compose_common_flags

––––– render all –––––

if [[ ${#SOURCES[@]} -gt 0 ]]; then
say “Rendering ${#SOURCES[@]} file(s) → ${OUTDIR}  formats=[$([[ $DO_SVG -eq 1 ]] && echo svg )$([[ $DO_PNG -eq 1 ]] && echo png )$([[ $DO_PDF -eq 1 ]] && echo pdf )]  theme=${THEME} bg=${BG}”
fi

if [[ “$JOBS” =~ ^[1-9]*$ && $JOBS -gt 1 ]]; then

Using a tmp file to collect exit codes

TMP_REND=”$(mktemp -t mmd_rend_XXXX)”
trap ‘rm -f “$TMP_REND” “$TMP_LIST” 2>/dev/null || true’ EXIT
printf ‘%s\0’ “${SOURCES[@]}” | xargs -0 -n1 -P “$JOBS” bash -lc ‘render_one “$0”’ || FAILED=1
else
for s in “${SOURCES[@]}”; do
if ! render_one “$s”; then
FAILED=1
fi
done
fi

––––– watch mode –––––

if [[ $WATCH -eq 1 ]]; then
say “Watch mode enabled (re-render on change)…”

Preferred: npm run mmd:watch if present

if [[ -f package.json ]] && have jq && jq -e ‘.scripts[“mmd:watch”]’ package.json >/dev/null 2>&1; then
run “npm run mmd:watch” npm run mmd:watch
exit 0
fi

Next: watchexec

if have watchexec; then
say “Using watchexec to watch diagrams//*.mmd and docs//*.mmd”
watchexec -e mmd -w diagrams -w docs – 
bash -lc “bin/diagrams.sh –outdir ‘$OUTDIR’ $([[ $DO_SVG -eq 1 ]] && echo –svg) $([[ $DO_PNG -eq 1 ]] && echo –png) $([[ $DO_PDF -eq 1 ]] && echo –pdf) –theme ‘$THEME’ –bg ‘$BG’ $([[ -n $MMDC_CONFIG ]] && printf – ’–config %q ’ “$MMDC_CONFIG”) $([[ -n $CSS_FILE ]] && printf – ’–css %q ’ “$CSS_FILE”) $([[ -n $MMDC_EXTRA ]] && printf – ’–mmdc-args %q ’ “$MMDC_EXTRA”)”
exit 0
fi

Fallback: entr

if have entr; then
say “Using entr to watch diagrams//*.mmd and docs//.mmd”
{ find diagrams docs -type f -name ’.mmd’ 2>/dev/null || true; } 
| entr -r bash -lc “bin/diagrams.sh –outdir ‘$OUTDIR’ $([[ $DO_SVG -eq 1 ]] && echo –svg) $([[ $DO_PNG -eq 1 ]] && echo –png) $([[ $DO_PDF -eq 1 ]] && echo –pdf) –theme ‘$THEME’ –bg ‘$BG’ $([[ -n $MMDC_CONFIG ]] && printf – ’–config %q ’ “$MMDC_CONFIG”) $([[ -n $CSS_FILE ]] && printf – ’–css %q ’ “$CSS_FILE”) $([[ -n $MMDC_EXTRA ]] && printf – ’–mmdc-args %q ’ “$MMDC_EXTRA”)”
exit 0
fi
warn “No watch tool found (npm script, watchexec, or entr). Exiting.”
fi

––––– JSON summary –––––

if [[ $JSON -eq 1 ]]; then

Build JSON arrays

printf ‘{’
printf ’“ok”: %s, ’ “$([[ $FAILED -eq 0 ]] && echo true || echo false)”
printf ’“count_sources”: %d, ’ “${#SOURCES[@]}”
printf ’“outdir”: “%s”, ’ “$OUTDIR”
printf ‘“formats”: [’; first=1
for f in svg png pdf; do
case “$f” in
svg) [[ $DO_SVG -eq 1 ]] || continue ;;
png) [[ $DO_PNG -eq 1 ]] || continue ;;
pdf) [[ $DO_PDF -eq 1 ]] || continue ;;
esac
[[ $first -eq 1 ]] || printf ‘,’
printf ‘”%s”’ “$f”; first=0
done
printf ’], ’
printf ’“theme”:”%s”,“bg”:”%s”,“timeout”:%s, ’ “$THEME” “$BG” “$TIMEOUT”

Rendered / skipped lists

printf ‘“rendered”: [’; for i in “${!RENDERED[@]}”; do [[ $i -gt 0 ]] && printf ‘,’; printf ‘”%s”’ “${RENDERED[$i]}”; done; printf ’], ’
printf ‘“skipped”: [’; for i in “${!SKIPPED[@]}”;  do [[ $i -gt 0 ]] && printf ‘,’; printf ‘”%s”’ “${SKIPPED[$i]}”;  done; printf ‘]’
printf ‘}\n’
fi

––––– exit –––––

if [[ $FAILED -eq 0 ]]; then
say “${GRN}Diagram rendering complete.${RST}”
exit 0
else
fail “Some diagrams failed to render.”
fi