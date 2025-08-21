#!/usr/bin/env bash
# ==============================================================================
# bin/diagrams.sh — Render Mermaid diagrams for SpectraMind V50
# ------------------------------------------------------------------------------
# Purpose
#   A robust, CI- and dev-friendly wrapper to render Mermaid diagrams to SVG/PNG/PDF
#   from *.mmd sources (and optionally from Markdown via a Python exporter).
#
# What it does
#   • Scans for sources (default: diagrams/**/*.mmd, docs/**/*.mmd)
#   • Renders with Mermaid CLI (mmdc) via npx if not installed globally
#   • Preserves subfolder structure under the output directory
#   • Optional: use scripts/export_mermaid.py to extract MD-embedded Mermaid
#   • Optional: npm lint/format hooks, watch mode, theming/background
#   • CI-safe, clear logs, dry-run support
#
# Usage
#   bin/diagrams.sh [options] [--] [PATH ...]
#
# Common examples
#   # Render all .mmd files to SVG (default) under outputs/diagrams/
#   bin/diagrams.sh
#
#   # Render to SVG+PNG with a dark theme, and also extract from Markdown via Python exporter
#   bin/diagrams.sh --png --theme dark --use-python
#
#   # Watch diagrams/ for changes and re-render SVG/PNG
#   bin/diagrams.sh --watch --png
#
# Options
#   --outdir DIR        Output directory (default: outputs/diagrams)
#   --svg               Render SVG (default if nothing selected)
#   --png               Render PNG
#   --pdf               Render PDF
#   --theme NAME        Mermaid theme: default|dark|forest|neutral (default: default)
#   --bg COLOR          Background: transparent|white|#RRGGBB (default: transparent)
#   --config FILE       Mermaid CLI config (e.g., .mermaidrc.mjs or .mermaidrc.json)
#   --mmdc PATH         Use a specific mmdc binary (fallback: npx @mermaid-js/mermaid-cli mmdc)
#   --use-python        Also run scripts/export_mermaid.py for Markdown-embedded mermaid
#   --lint              Run `npm run lint` if present
#   --format            Run `npm run format` if present
#   --watch             Watch for changes (requires entr or watchexec or npm script)
#   --clean             Remove the output directory before rendering
#   --jobs N            Parallel jobs (render multiple diagrams at once) [best-effort]
#   --dry-run           Show actions without executing
#   --quiet             Reduce verbosity
#   -h, --help          Show help
#
# Notes
#   • Sources: explicit PATH args (files/dirs) override defaults. Directories are searched
#     for *.mmd. Markdown export (when --use-python) depends on scripts/export_mermaid.py.
#   • Mermaid CLI (mmdc) requires Node; we invoke via npx when needed (no global install).
#   • For watch mode, we try: npm run mmd:watch → watchexec → entr, in that order.
# ==============================================================================

set -euo pipefail

# ---------- colors ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[DIAGRAMS]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[DIAGRAMS]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[DIAGRAMS]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- defaults ----------
OUTDIR="outputs/diagrams"
DO_SVG=0
DO_PNG=0
DO_PDF=0
THEME="default"
BG="transparent"
MMDC_BIN=""
MMDC_CONFIG=""
USE_PYTHON=0
DO_LINT=0
DO_FORMAT=0
WATCH=0
CLEAN=0
JOBS="${JOBS:-0}"
DRY=0
QUIET=0
ARGS=()

usage() { sed -n '1,120p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --outdir)   OUTDIR="${2:?}"; shift 2 ;;
    --svg)      DO_SVG=1; shift ;;
    --png)      DO_PNG=1; shift ;;
    --pdf)      DO_PDF=1; shift ;;
    --theme)    THEME="${2:?}"; shift 2 ;;
    --bg)       BG="${2:?}"; shift 2 ;;
    --config)   MMDC_CONFIG="${2:?}"; shift 2 ;;
    --mmdc)     MMDC_BIN="${2:?}"; shift 2 ;;
    --use-python) USE_PYTHON=1; shift ;;
    --lint)     DO_LINT=1; shift ;;
    --format)   DO_FORMAT=1; shift ;;
    --watch)    WATCH=1; shift ;;
    --clean)    CLEAN=1; shift ;;
    --jobs)     JOBS="${2:?}"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    --quiet)    QUIET=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    --) shift; while [[ $# -gt 0 ]]; do ARGS+=("$1"); shift; done ;;
    *)  ARGS+=("$1"); shift ;;
  esac
done

# Defaults: if no formats selected, render SVG
if [[ $DO_SVG -eq 0 && $DO_PNG -eq 0 && $DO_PDF -eq 0 ]]; then
  DO_SVG=1
fi

# ---------- repo root ----------
if git_root=$(command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel 2>/dev/null); then
  cd "$git_root"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$SCRIPT_DIR/.." || { fail "Cannot locate repo root"; exit 1; }
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

# ---------- prepare ----------
if [[ $CLEAN -eq 1 ]]; then
  [[ -d "$OUTDIR" ]] && run "rm -rf $OUTDIR" rm -rf "$OUTDIR"
fi
mkdir -p "$OUTDIR"

# ---------- Node/npm checks (for mmdc path resolution) ----------
# mmdc resolution order: explicit --mmdc → local node_modules binary → global mmdc → npx fallback
resolve_mmdc() {
  if [[ -n "$MMDC_BIN" ]]; then
    echo "$MMDC_BIN"
    return
  fi
  if [[ -f "node_modules/.bin/mmdc" ]]; then
    echo "node_modules/.bin/mmdc"
    return
  fi
  if have mmdc; then
    echo "mmdc"
    return
  fi
  # npx fallback (no installation, download-once cache)
  if have npx; then
    echo "npx -y @mermaid-js/mermaid-cli mmdc"
    return
  fi
  fail "Mermaid CLI not found. Install via 'npm i -D @mermaid-js/mermaid-cli' or use npx, or pass --mmdc <path>."
  exit 1
}
MMDC_CMD="$(resolve_mmdc)"

# ---------- optional: npm lint/format ----------
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

# ---------- source discovery ----------
# If ARGS provided: treat files/dirs accordingly; else defaults
collect_sources() {
  local paths=()
  if [[ ${#ARGS[@]} -gt 0 ]]; then
    for p in "${ARGS[@]}"; do
      if [[ -d "$p" ]]; then
        while IFS= read -r -d '' f; do paths+=("$f"); done < <(find "$p" -type f -name "*.mmd" -print0)
      elif [[ -f "$p" ]]; then
        case "$p" in
          *.mmd) paths+=("$p") ;;
          *) warn "Skipping non-.mmd file: $p" ;;
        esac
      else
        warn "Path not found: $p"
      fi
    done
  else
    # Defaults
    while IFS= read -r -d '' f; do paths+=("$f"); done < <(find diagrams -type f -name "*.mmd" -print0 2>/dev/null || true)
    while IFS= read -r -d '' f; do paths+=("$f"); done < <(find docs -type f -name "*.mmd" -print0 2>/dev/null || true)
  fi
  printf '%s\n' "${paths[@]}" | awk 'NF' | sort -u
}
SOURCES=($(collect_sources))

# ---------- optional: extract from Markdown via Python exporter ----------
if [[ $USE_PYTHON -eq 1 ]]; then
  if [[ -f "scripts/export_mermaid.py" ]]; then
    say "Extracting Mermaid codeblocks from Markdown via scripts/export_mermaid.py"
    # The exporter should create SVG/PNG/PDF as appropriate. We pass theme/bg/outdir via env if supported.
    # Fallback: just call it with defaults; users can extend the exporter to support CLI flags.
    run "python scripts/export_mermaid.py" python scripts/export_mermaid.py || warn "export_mermaid.py returned non-zero."
  else
    warn "scripts/export_mermaid.py not found; skipping Markdown extraction."
  fi
fi

# ---------- render pipeline ----------
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  warn "No .mmd sources found to render."
fi

# Compute outputs per source, preserving relative path under OUTDIR
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
  common+=(--input "$src" --theme "$THEME" --backgroundColor "$BG")

  # SVG
  if [[ $DO_SVG -eq 1 ]]; then
    local out_svg="$out_sub/$name.svg"
    run "mmdc SVG $src" bash -lc "$MMDC_CMD --output '$out_svg' ${common[*]}"
  fi
  # PNG
  if [[ $DO_PNG -eq 1 ]]; then
    local out_png="$out_sub/$name.png"
    run "mmdc PNG $src" bash -lc "$MMDC_CMD --png --output '$out_png' ${common[*]}"
  fi
  # PDF
  if [[ $DO_PDF -eq 1 ]]; then
    local out_pdf="$out_sub/$name.pdf"
    run "mmdc PDF $src" bash -lc "$MMDC_CMD --pdf --output '$out_pdf' ${common[*]}"
  fi
}

# Parallelism (best-effort): if jobs > 1, use xargs -P
render_all() {
  if [[ ${#SOURCES[@]} -eq 0 ]]; then
    return 0
  fi
  say "Rendering ${#SOURCES[@]} Mermaid file(s) → ${OUTDIR} (theme=${THEME}, bg=${BG})"
  if [[ "$JOBS" =~ ^[1-9][0-9]*$ && $JOBS -gt 1 ]]; then
    # export functions for subshells
    export -f run render_one
    export OUTDIR DO_SVG DO_PNG DO_PDF THEME BG MMDC_CMD MMDC_CONFIG DRY QUIET
    printf '%s\0' "${SOURCES[@]}" | xargs -0 -n1 -P "$JOBS" bash -lc 'render_one "$0"' || true
  else
    for s in "${SOURCES[@]}"; do render_one "$s"; done
  fi
}

render_all

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
    say "Using watchexec (installed) to watch diagrams/**/*.mmd"
    watchexec -e mmd -w diagrams -w docs -- \
      bash -lc "bin/diagrams.sh --outdir '$OUTDIR' $( [[ $DO_SVG -eq 1 ]] && echo --svg ) $( [[ $DO_PNG -eq 1 ]] && echo --png ) $( [[ $DO_PDF -eq 1 ]] && echo --pdf ) --theme '$THEME' --bg '$BG' $( [[ -n $MMDC_CONFIG ]] && echo --config "$MMDC_CONFIG" )"
    exit 0
  fi
  # Fallback: entr
  if have entr; then
    say "Using entr (installed) to watch diagrams/**/*.mmd"
    find diagrams docs -type f -name '*.mmd' | entr -r bash -lc "bin/diagrams.sh --outdir '$OUTDIR' $( [[ $DO_SVG -eq 1 ]] && echo --svg ) $( [[ $DO_PNG -eq 1 ]] && echo --png ) $( [[ $DO_PDF -eq 1 ]] && echo --pdf ) --theme '$THEME' --bg '$BG' $( [[ -n $MMDC_CONFIG ]] && echo --config "$MMDC_CONFIG" )"
    exit 0
  fi
  warn "No watch tool found (npm script, watchexec, or entr). Exiting."
fi

say "${GRN}Diagram rendering complete.${RST}"
exit 0