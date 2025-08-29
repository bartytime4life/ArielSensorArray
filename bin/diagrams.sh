#!/usr/bin/env bash
# ==============================================================================
# bin/diagrams.sh — Render Mermaid diagrams for SpectraMind V50 (upgraded)
# ------------------------------------------------------------------------------
# Purpose
#   CI- and dev-friendly wrapper to render Mermaid diagrams (SVG/PNG/PDF) from
#   *.mmd sources (and optionally from Markdown via a Python exporter).
#
# Highlights (new)
#   • Source discovery with include/exclude globs and explicit paths
#   • Parallel rendering (-j) with sane fallbacks (xargs -P)
#   • Prune mode: remove orphaned outputs that no longer have sources
#   • mmdc timeout, extra CLI args passthrough, CSS/theme/bg width/height/scale
#   • JSON summary (--json) and dry-run previews
#   • Node/mmdc resolution: --mmdc → local node_modules → global → npx fallback
#   • Watch mode: npm script → watchexec → entr (best-available)
#
# Usage
#   bin/diagrams.sh [options] [--] [PATH ...]
#
# Common examples
#   # Render all .mmd files to SVG (default) under outputs/diagrams/
#   bin/diagrams.sh
#
#   # Render to SVG+PNG with dark theme; extract MD-embedded Mermaid
#   bin/diagrams.sh --png --theme dark --use-python
#
#   # Watch diagrams/ for changes and re-render SVG/PNG
#   bin/diagrams.sh --watch --png
#
# Options
#   --outdir DIR         Output directory (default: outputs/diagrams)
#   --svg                Render SVG (default if nothing selected)
#   --png                Render PNG
#   --pdf                Render PDF
#   --theme NAME         Mermaid theme: default|dark|forest|neutral (default: default)
#   --bg COLOR           Background: transparent|white|#RRGGBB (default: transparent)
#   --config FILE        Mermaid CLI config (.mermaidrc.mjs/.json)
#   --css FILE           Custom CSS file for Mermaid CLI
#   --width N            Canvas width (px) for PNG/PDF
#   --height N           Canvas height (px) for PNG/PDF
#   --scale N            Scale factor for raster outputs
#   --mmdc PATH          Use a specific mmdc binary (fallback: npx @mermaid-js/mermaid-cli mmdc)
#   --mmdc-args "ARGS"   Extra args passed verbatim to mmdc
#   --timeout SEC        Timeout per mmdc render (default: 90)
#   --use-python         Also run scripts/export_mermaid.py for Markdown-embedded Mermaid
#   --lint               Run `npm run lint` if present
#   --format             Run `npm run format` if present
#   --watch              Watch for changes (npm run mmd:watch → watchexec → entr)
#   --clean              Remove output dir before rendering
#   --prune              Remove orphaned outputs (no matching source)
#   --jobs N             Parallel jobs (best-effort; default: 0 = auto serial)
#   --include GLOB       Include glob(s) (can repeat); default: diagrams/**/*.mmd, docs/**/*.mmd
#   --exclude GLOB       Exclude glob(s) (can repeat)
#   --list               Only list resolved sources and exit
#   --json               Emit JSON summary to stdout
#   --dry-run            Show actions without executing
#   --quiet              Reduce verbosity
#   -h, --help           Show help
#
# Notes
#   • Explicit PATH args (files/dirs) override defaults. Directories are searched for *.mmd.
#   • Mermaid CLI requires Node; we invoke via npx fallback when not locally installed.
#   • Exit code non-zero if any render fails (even in parallel).
# ==============================================================================

set -euo pipefail

# ---------- colors ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[DIAGRAMS]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[DIAGRAMS]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[DIAGRAMS]%s %s\n' "${RED}" "${RST}" "$*" >&2; exit 1; }

# ---------- defaults ----------
OUTDIR="outputs/diagrams"
DO_SVG=0; DO_PNG=0; DO_PDF=0
THEME="default"; BG="transparent"
MMDC_BIN=""; MMDC_CONFIG=""; CSS_FILE=""; MMDC_EXTRA=""
WIDTH=""; HEIGHT=""; SCALE=""
USE_PYTHON=0; DO_LINT=0; DO_FORMAT=0; WATCH=0; CLEAN=0; PRUNE=0
JOBS="${JOBS:-0}"; DRY=0; QUIET=0; JSON=0
TIMEOUT="${DIAGRAMS_TIMEOUT:-90}"
INCLUDES=(); EXCLUDES=()
ARGS=()

usage() { sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)     OUTDIR="${2:?}"; shift 2 ;;
    --svg)        DO_SVG=1; shift ;;
    --png)        DO_PNG=1; shift ;;
    --pdf)        DO_PDF=1; shift ;;
    --theme)      THEME="${2:?}"; shift 2 ;;
    --bg)         BG="${2:?}"; shift 2 ;;
    --config)     MMDC_CONFIG="${2:?}"; shift 2 ;;
    --css)        CSS_FILE="${2:?}"; shift 2 ;;
    --width)      WIDTH="${2:?}"; shift 2 ;;
    --height)     HEIGHT="${2:?}"; shift 2 ;;
    --scale)      SCALE="${2:?}"; shift 2 ;;
    --mmdc)       MMDC_BIN="${2:?}"; shift 2 ;;
    --mmdc-args)  MMDC_EXTRA="${2:?}"; shift 2 ;;
    --timeout)    TIMEOUT="${2:?}"; shift 2 ;;
    --use-python) USE_PYTHON=1; shift ;;
    --lint)       DO_LINT=1; shift ;;
    --format)     DO_FORMAT=1; shift ;;
    --watch)      WATCH=1; shift ;;
    --clean)      CLEAN=1; shift ;;
    --prune)      PRUNE=1; shift ;;
    --jobs)       JOBS="${2:?}"; shift 2 ;;
    --include)    INCLUDES+=("${2:?}"); shift 2 ;;
    --exclude)    EXCLUDES+=("${2:?}"); shift 2 ;;
    --list)       LIST_ONLY=1; shift ;;
    --json)       JSON=1; shift ;;
    --dry-run)    DRY=1; shift ;;
    --quiet)      QUIET=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do ARGS+=("$1"); shift; done ;;
    *)  ARGS+=("$1"); shift ;;
  esac
done
: "${LIST_ONLY:=0}"

# Defaults: if no formats selected, render SVG
if [[ $DO_SVG -eq 0 && $DO_PNG -eq 0 && $DO_PDF -eq 0 ]]; then DO_SVG=1; fi

# ---------- repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; }
fi

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }

run() {
  local desc="$1"; shift
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] $desc :: $*"
    return 0
  fi
  [[ $QUIET -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "${desc}" "${RST}"
  "$@"
}

resolve_mmdc() {
  if [[ -n "$MMDC_BIN" ]]; then echo "$MMDC_BIN"; return; fi
  if [[ -f "node_modules/.bin/mmdc" ]]; then echo "node_modules/.bin/mmdc"; return; fi
  if have mmdc; then echo "mmdc"; return; fi
  if have npx; then echo "npx -y @mermaid-js/mermaid-cli mmdc"; return; fi
  fail "Mermaid CLI not found. Install via 'npm i -D @mermaid-js/mermaid-cli', use 'npx', or pass --mmdc <path>."
}
MMDC_CMD="$(resolve_mmdc)"

# ---------- npm lint/format (optional) ----------
if [[ $DO_LINT -eq 1 && -f package.json ]]; then
  if jq -e '.scripts.lint' package.json >/dev/null 2>&1; then
    run "npm run lint" npm run lint
  else
    warn "package.json has no 'lint' script."
  fi
fi
if [[ $DO_FORMAT -eq 1 && -f package.json ]]; then
  if jq -e '.scripts.format' package.json >/dev/null 2>&1; then
    run "npm run format" npm run format
  else
    warn "package.json has no 'format' script."
  fi
fi

# ---------- prepare ----------
if [[ $CLEAN -eq 1 ]]; then
  [[ -d "$OUTDIR" ]] && run "rm -rf $OUTDIR" rm -rf "$OUTDIR"
fi
mkdir -p "$OUTDIR"

# ---------- source discovery ----------
collect_sources_from_paths() {
  local -a paths=()
  for p in "${ARGS[@]}"; do
    if [[ -d "$p" ]]; then
      while IFS= read -r -d '' f; do paths+=("$f"); done < <(find "$p" -type f -name "*.mmd" -print0)
    elif [[ -f "$p" ]]; then
      [[ "$p" == *.mmd ]] && paths+=("$p") || warn "Skipping non-.mmd file: $p"
    else
      warn "Path not found: $p"
    fi
  done
  printf '%s\n' "${paths[@]}"
}

collect_sources_default() {
  local -a paths=()
  if [[ ${#INCLUDES[@]} -gt 0 ]]; then
    for g in "${INCLUDES[@]}"; do
      while IFS= read -r -d '' f; do paths+=("$f"); done < <(find . -type f -name "$(basename "$g")" -path "*/$(dirname "$g")/*" -print0 2>/dev/null || true)
    done
  else
    while IFS= read -r -d '' f; do paths+=("$f"); done < <(find diagrams -type f -name "*.mmd" -print0 2>/dev/null || true)
    while IFS= read -r -d '' f; do paths+=("$f"); done < <(find docs     -type f -name "*.mmd" -print0 2>/dev/null || true)
  fi
  # Apply excludes
  if [[ ${#EXCLUDES[@]} -gt 0 ]]; then
    printf '%s\n' "${paths[@]}" | while IFS= read -r f; do
      skip=0
      for ex in "${EXCLUDES[@]}"; do [[ "$f" == $ex ]] && skip=1 && break; done
      [[ $skip -eq 0 ]] && printf '%s\n' "$f"
    done
  else
    printf '%s\n' "${paths[@]}"
  fi
}

collect_sources() {
  if [[ ${#ARGS[@]} -gt 0 ]]; then collect_sources_from_paths | sort -u
  else collect_sources_default | sort -u
  fi
}

SOURCES=()
mapfile -t SOURCES < <(collect_sources)

if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%s\n' "${SOURCES[@]}"
  exit 0
fi

# ---------- optional: MD extraction via Python ----------
if [[ $USE_PYTHON -eq 1 ]]; then
  if [[ -f "scripts/export_mermaid.py" ]]; then
    say "Extracting Mermaid codeblocks from Markdown via scripts/export_mermaid.py"
    run "python scripts/export_mermaid.py" python scripts/export_mermaid.py || warn "export_mermaid.py returned non-zero."
    # Re-scan sources in case exporter wrote *.mmd files
    mapfile -t SOURCES < <(collect_sources)
  else
    warn "scripts/export_mermaid.py not found; skipping Markdown extraction."
  fi
fi

if [[ ${#SOURCES[@]} -eq 0 ]]; then
  warn "No .mmd sources found to render."
fi

# ---------- prune orphaned outputs ----------
if [[ $PRUNE -eq 1 ]]; then
  say "Pruning orphaned outputs from $OUTDIR…"
  while IFS= read -r -d '' out; do
    base_rel="${out#$OUTDIR/}"          # diagrams/foo/bar.svg
    src="${base_rel%.*}.mmd"            # diagrams/foo/bar.mmd
    # Try to locate source anywhere under repo (relative)
    if ! [[ -f "$src" || -f "./$src" ]]; then
      say "Removing orphaned: $out"
      [[ $DRY -eq 1 ]] || rm -f "$out"
    fi
  done < <(find "$OUTDIR" -type f \( -name "*.svg" -o -name "*.png" -o -name "*.pdf" \) -print0)
fi

# ---------- compute outputs / render ----------
render_one() {
  local src="$1"
  local rel="${src#./}"
  local base="${rel%.*}"     # diagrams/foo/bar.mmd -> diagrams/foo/bar
  local subdir="$(dirname "$base")"
  local name="$(basename "$base")"
  local out_sub="$OUTDIR/$subdir"
  mkdir -p "$out_sub"

  local common=()
  [[ -n "$MMDC_CONFIG" ]] && common+=(--configFile "$MMDC_CONFIG")
  [[ -n "$CSS_FILE"    ]] && common+=(--cssFile "$CSS_FILE")
  common+=(--input "$src" --theme "$THEME" --backgroundColor "$BG")
  [[ -n "$WIDTH"  ]] && common+=(--width "$WIDTH")
  [[ -n "$HEIGHT" ]] && common+=(--height "$HEIGHT")
  [[ -n "$SCALE"  ]] && common+=(--scale "$SCALE")

  local cmd="$MMDC_CMD"
  [[ -n "$MMDC_EXTRA" ]] && cmd="$cmd $MMDC_EXTRA"

  # SVG
  if [[ $DO_SVG -eq 1 ]]; then
    local out_svg="$out_sub/$name.svg"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] mmdc SVG $src → $out_svg"
    else
      if command -v timeout >/dev/null 2>&1; then
        timeout --preserve-status --signal=TERM "$TIMEOUT" bash -lc "$cmd --output '$out_svg' ${common[*]}"
      else
        bash -lc "$cmd --output '$out_svg' ${common[*]}"
      fi
    fi
  fi
  # PNG
  if [[ $DO_PNG -eq 1 ]]; then
    local out_png="$out_sub/$name.png"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] mmdc PNG $src → $out_png"
    else
      if command -v timeout >/dev/null 2>&1; then
        timeout --preserve-status --signal=TERM "$TIMEOUT" bash -lc "$cmd --png --output '$out_png' ${common[*]}"
      else
        bash -lc "$cmd --png --output '$out_png' ${common[*]}"
      fi
    fi
  fi
  # PDF
  if [[ $DO_PDF -eq 1 ]]; then
    local out_pdf="$out_sub/$name.pdf"
    if [[ $DRY -eq 1 ]]; then
      say "[dry-run] mmdc PDF $src → $out_pdf"
    else
      if command -v timeout >/dev/null 2>&1; then
        timeout --preserve-status --signal=TERM "$TIMEOUT" bash -lc "$cmd --pdf --output '$out_pdf' ${common[*]}"
      else
        bash -lc "$cmd --pdf --output '$out_pdf' ${common[*]}"
      fi
    fi
  fi
}

# Parallelism (best-effort): if jobs > 1, use xargs -P
render_all() {
  local fails=0
  if [[ ${#SOURCES[@]} -eq 0 ]]; then
    return 0
  fi
  say "Rendering ${#SOURCES[@]} Mermaid file(s) → ${OUTDIR} (theme=${THEME}, bg=${BG}, timeout=${TIMEOUT}s)"
  if [[ "$JOBS" =~ ^[1-9][0-9]*$ && $JOBS -gt 1 ]]; then
    export -f render_one
    export OUTDIR DO_SVG DO_PNG DO_PDF THEME BG MMDC_CMD MMDC_CONFIG CSS_FILE MMDC_EXTRA TIMEOUT DRY QUIET WIDTH HEIGHT SCALE
    printf '%s\0' "${SOURCES[@]}" \
      | xargs -0 -n1 -P "$JOBS" bash -lc 'render_one "$0"' \
      || fails=1
  else
    for s in "${SOURCES[@]}"; do
      render_one "$s" || fails=1
    done
  fi
  return $fails
}

# ---------- list sources or render ----------
if [[ $LIST_ONLY -eq 1 ]]; then
  printf '%s\n' "${SOURCES[@]}"
  exit 0
fi

FAILED=0
render_all || FAILED=1

# ---------- watch mode ----------
if [[ $WATCH -eq 1 ]]; then
  say "Watch mode enabled (re-render on change)…"
  # Preferred: npm run mmd:watch if present
  if [[ -f package.json ]] && jq -e '.scripts["mmd:watch"]' package.json >/dev/null 2>&1; then
    run "npm run mmd:watch" npm run mmd:watch
    exit 0
  fi
  # Next: watchexec
  if have watchexec; then
    say "Using watchexec to watch diagrams/**/*.mmd and docs/**/*.mmd"
    watchexec -e mmd -w diagrams -w docs -- \
      bash -lc "bin/diagrams.sh --outdir '$OUTDIR' $( [[ $DO_SVG -eq 1 ]] && echo --svg ) $( [[ $DO_PNG -eq 1 ]] && echo --png ) $( [[ $DO_PDF -eq 1 ]] && echo --pdf ) --theme '$THEME' --bg '$BG' $( [[ -n $MMDC_CONFIG ]] && echo --config "$MMDC_CONFIG" ) $( [[ -n $CSS_FILE ]] && echo --css "$CSS_FILE" ) $( [[ -n $MMDC_EXTRA ]] && echo --mmdc-args "$MMDC_EXTRA" )"
    exit 0
  fi
  # Fallback: entr
  if have entr; then
    say "Using entr to watch diagrams/**/*.mmd and docs/**/*.mmd"
    { find diagrams docs -type f -name '*.mmd' 2>/dev/null || true; } \
      | entr -r bash -lc "bin/diagrams.sh --outdir '$OUTDIR' $( [[ $DO_SVG -eq 1 ]] && echo --svg ) $( [[ $DO_PNG -eq 1 ]] && echo --png ) $( [[ $DO_PDF -eq 1 ]] && echo --pdf ) --theme '$THEME' --bg '$BG' $( [[ -n $MMDC_CONFIG ]] && echo --config "$MMDC_CONFIG" ) $( [[ -n $CSS_FILE ]] && echo --css "$CSS_FILE" ) $( [[ -n $MMDC_EXTRA ]] && echo --mmdc-args "$MMDC_EXTRA" )"
    exit 0
  fi
  warn "No watch tool found (npm script, watchexec, or entr). Exiting."
fi

# ---------- JSON summary ----------
if [[ $JSON -eq 1 ]]; then
  # Minimal robust JSON without external deps
  printf '{'
  printf '"ok": %s, ' "$([[ $FAILED -eq 0 ]] && echo true || echo false)"
  printf '"count": %s, ' "${#SOURCES[@]}"
  printf '"outdir": "%s", ' "$OUTDIR"
  printf '"formats": "%s", ' "$([[ $DO_SVG -eq 1 ]] && echo svg) $([[ $DO_PNG -eq 1 ]] && echo png) $([[ $DO_PDF -eq 1 ]] && echo pdf)"
  printf '"theme": "%s", "bg": "%s", ' "$THEME" "$BG"
  printf '"timeout": %s' "$TIMEOUT"
  printf '}\n'
fi

[[ $FAILED -eq 0 ]] && say "${GRN}Diagram rendering complete.${RST}" || fail "Some diagrams failed to render."
exit $FAILED