#!/usr/bin/env bash
# ==============================================================================
# bin/kaggle-boot.sh — SpectraMind V50 Kaggle bootstrap (deps + PyG + sanity)
# ------------------------------------------------------------------------------
# Purpose
#   One-shot, Kaggle-friendly environment setup from inside a notebook/terminal:
#     1) (Optionally) upgrade pip/setuptools/wheel
#     2) Install a curated requirements file (default: requirements-kaggle.txt)
#     3) Detect Torch & CUDA, install matching torch-geometric wheel
#     4) Optionally install extra pip packages
#     5) Sanity: show torch & CUDA, run spectramind --version if available
#
# Usage
#   # Default: install requirements-kaggle.txt + matching PyG wheel
#   bash bin/kaggle-boot.sh
#
#   # Custom requirements file + skip PyG + extras
#   bash bin/kaggle-boot.sh --req path/to/reqs.txt --no-pyg --extra "polars==1.5.0 einops"
#
# Options
#   --req <file>         Requirements file to install (default: requirements-kaggle.txt)
#   --no-req             Do not install a requirements file
#   --no-upgrade-pip     Skip pip/setuptools/wheel upgrade
#   --no-pyg             Skip torch-geometric install
#   --pyg-version <v>    torch-geometric version (default: 2.5.3)
#   --extra "<pkgs>"     Space-separated extra pip packages to install
#   --index-url <url>    Custom PyPI/simple index URL (optional)
#   --transformers-offline  Set TRANSFORMERS_OFFLINE=1 during install
#   --torch-cuda <tag>   Force CUDA tag (e.g. cpu, cu118, cu121) to override autodetect
#   --dry-run            Print actions only, do not execute
#   --quiet              Less verbose output
#   --json               Emit a JSON summary for the bootstrap step
#   -h|--help            Show help and exit
#
# Notes
#   • Safe in Kaggle notebooks (no sudo; user-level pip installs).
#   • PyG wheel index is inferred from torch.__version__ and torch.version.cuda (or --torch-cuda).
#   • Creates .kaggle_boot_ok sentinel on success.
# ==============================================================================

set -euo pipefail

# ---------- pretty ----------
BOLD=$'\033[1m'; DIM=$'\033[2m'; RED=$'\033[31m'; GRN=$'\033[32m'; CYN=$'\033[36m'; RST=$'\033[0m'
say()  { [[ "${QUIET:-0}" -eq 1 ]] && return 0; printf '%s[KAGGLE-BOOT]%s %s\n' "${CYN}" "${RST}" "$*"; }
warn() { printf '%s[KAGGLE-BOOT]%s %s\n' "${DIM}" "${RST}" "$*" >&2; }
fail() { printf '%s[KAGGLE-BOOT]%s %s\n' "${RED}" "${RST}" "$*" >&2; }

# ---------- defaults ----------
REQ_FILE="requirements-kaggle.txt"
DO_REQ=1
UPGRADE_PIP=1
DO_PYG=1
PYG_VERSION="2.5.3"
EXTRA_PKGS=""
CUSTOM_INDEX=""
SET_TRANSFORMERS_OFFLINE=0
FORCE_TORCH_CUDA=""
DRY=0
QUIET=0
JSON=0

usage() {
  sed -n '1,160p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------- args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --req) REQ_FILE="${2:?}"; shift 2 ;;
    --no-req) DO_REQ=0; shift ;;
    --no-upgrade-pip) UPGRADE_PIP=0; shift ;;
    --no-pyg) DO_PYG=0; shift ;;
    --pyg-version) PYG_VERSION="${2:?}"; shift 2 ;;
    --extra) EXTRA_PKGS="${2:?}"; shift 2 ;;
    --index-url) CUSTOM_INDEX="${2:?}"; shift 2 ;;
    --transformers-offline) SET_TRANSFORMERS_OFFLINE=1; shift ;;
    --torch-cuda) FORCE_TORCH_CUDA="${2:?}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --quiet) QUIET=1; shift ;;
    --json) JSON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) fail "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ---------- helpers ----------
have() { command -v "$1" >/dev/null 2>&1; }
run()  {
  local desc="$1"; shift
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] $desc :: $*"
    return 0
  fi
  [[ $QUIET -eq 0 ]] && printf "%s→ %s%s\n" "${DIM}" "${desc}" "${RST}"
  "$@"
}

export_if() {
  local k="$1" v="$2"
  if [[ $DRY -eq 1 ]]; then
    say "[dry-run] export ${k}=${v}"
  else
    export "${k}=${v}"
  fi
}

json_out() {
  [[ $JSON -eq 1 ]] || return 0
  local ok="$1" torch_v="$2" cuda="$3" file="$4" pyg="$5"
  printf '{ "ok": %s, "torch": "%s", "cuda": "%s", "submission_reqs": "%s", "pyg": "%s" }\n' \
    "$([[ "$ok" == "true" ]] && echo true || echo false)" \
    "${torch_v:-unknown}" "${cuda:-unknown}" "${file:-none}" "${pyg:-skipped}"
}

# ---------- sanity: Kaggle-ish environment ----------
IS_KAGGLE=0
[[ -n "${KAGGLE_URL_BASE:-}" || -n "${KAGGLE_KERNEL_RUN_TYPE:-}" || -d "/kaggle" ]] && IS_KAGGLE=1
say "Kaggle env detected: ${BOLD}${IS_KAGGLE}${RST}"

have python || fail "python not found on PATH"
PY="$(command -v python)"
PIP="$PY -m pip"

# Show base versions (helpful for logs)
run "python --version" "$PY" --version
run "pip --version" $PIP --version || true

# ---------- step 1: optional pip tooling upgrade ----------
if [[ $UPGRADE_PIP -eq 1 ]]; then
  run "pip upgrade tooling" $PIP install -U pip setuptools wheel
else
  warn "Skipping pip/setuptools/wheel upgrade (--no-upgrade-pip)."
fi

# ---------- step 2: install requirements file ----------
INSTALLED_REQS="none"
if [[ $DO_REQ -eq 1 ]]; then
  if [[ ! -f "$REQ_FILE" ]]; then
    fail "Requirements file not found: $REQ_FILE (pass --req <file> or --no-req)"
  fi
  # Transformers offline mode can reduce remote calls in Kaggle (if pre-cached).
  if [[ $SET_TRANSFORMERS_OFFLINE -eq 1 ]]; then
    export_if "TRANSFORMERS_OFFLINE" "1"
  fi

  PIP_INSTALL_ARGS=(install -r "$REQ_FILE")
  [[ -n "$CUSTOM_INDEX" ]] && PIP_INSTALL_ARGS=(install --index-url "$CUSTOM_INDEX" -r "$REQ_FILE")
  run "pip install -r $REQ_FILE" $PIP "${PIP_INSTALL_ARGS[@]}"
  INSTALLED_REQS="$REQ_FILE"
else
  warn "Skipping requirements install (--no-req)."
fi

# ---------- step 3: install torch-geometric (PyG) wheel to match torch/CUDA ----------
TORCH_VERSION=""; CUDA_RAW=""
PYG_STATUS="skipped"

if [[ $DO_PYG -eq 1 ]]; then
  TORCH_META=$($PY - <<'PY'
import json
try:
    import torch
    print(json.dumps({"torch": torch.__version__.split('+')[0], "cuda": (torch.version.cuda or "cpu")}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
PY
  )
  if [[ "$TORCH_META" == *"error"* ]]; then
    warn "Torch not importable; skipping torch-geometric install."
    PYG_STATUS="torch-missing"
  else
    TORCH_VERSION="$(python - <<'PY'
import json,sys
print(json.loads(sys.stdin.read())["torch"])
PY
<<<"$TORCH_META")"

    CUDA_RAW_OVERRIDE="$FORCE_TORCH_CUDA"
    if [[ -n "$CUDA_RAW_OVERRIDE" ]]; then
      say "Forcing CUDA tag via --torch-cuda=${CUDA_RAW_OVERRIDE}"
      CUDA_RAW="$CUDA_RAW_OVERRIDE"
    else
      CUDA_RAW="$(python - <<'PY'
import json,sys
print(json.loads(sys.stdin.read())["cuda"])
PY
<<<"$TORCH_META")"
    fi

    if [[ "$CUDA_RAW" == "cpu" || -z "$CUDA_RAW" ]]; then
      CU_TAG="cpu"
    else
      CU_TAG="cu${CUDA_RAW//./}"
    fi

    PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CU_TAG}.html"
    say "torch=${BOLD}${TORCH_VERSION}${RST}  cuda=${BOLD}${CUDA_RAW}${RST}  → PyG index=${BOLD}${PYG_INDEX}${RST}"

    # Install torch-geometric matching wheel index
    run "pip install torch-geometric==${PYG_VERSION} -f ${PYG_INDEX}" \
      $PIP install "torch-geometric==${PYG_VERSION}" -f "${PYG_INDEX}"
    PYG_STATUS="installed-${PYG_VERSION}"
  fi
else
  warn "Skipping torch-geometric install (--no-pyg)."
  PYG_STATUS="skipped"
fi

# ---------- step 4: optional extra packages ----------
if [[ -n "$EXTRA_PKGS" ]]; then
  run "pip install extras: ${EXTRA_PKGS}" $PIP install ${EXTRA_PKGS}
fi

# ---------- step 5: quick sanity prints ----------
TORCH_REPORT="$("$PY" - <<'PY' 2>/dev/null || true
import json, sys
try:
    import torch
    print(json.dumps({"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
PY
)"
if [[ "$TORCH_REPORT" == *"error"* ]]; then
  warn "Torch report unavailable."
else
  say "Torch report: ${TORCH_REPORT}"
fi

# SpectraMind banner (if available)
if command -v spectramind >/dev/null 2>&1; then
  run "spectramind --version" spectramind --version || true
elif [[ -f "spectramind.py" ]]; then
  run "python spectramind.py --version" "$PY" spectramind.py --version || true
else
  warn "SpectraMind CLI not found (this is fine if not yet installed)."
fi

# ---------- sentinel ----------
if [[ $DRY -eq 0 ]]; then
  touch .kaggle_boot_ok || true
fi

say "${GRN}Kaggle bootstrap complete.${RST}"
[[ $DRY -eq 1 ]] && say "(dry-run: no changes were made)"

json_out true "$TORCH_VERSION" "$CUDA_RAW" "$INSTALLED_REQS" "$PYG_STATUS"