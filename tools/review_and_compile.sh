#!/usr/bin/env bash

##############################################################################

# SpectraMind V50 — Repository Review & Compile Orchestrator (v1.3)
#
# Purpose:
#   One-shot audit + compile for your repo. This script:
#   1) Locates repo root.
#   2) Creates outputs/selftest/ and logs/.
#   3) Generates and runs a Python scanner to find scaffolding/incomplete code:
#      - TODO/FIXME/TBD/HACK/XXX markers
#      - “placeholder”/“stub”/“scaffold”/“boilerplate”
#      - raise NotImplementedError
#      - pass-only def/class blocks
#      - ellipsis-only bodies (…)
#      - stub exceptions (raise Exception("TODO"))
#      - empty README/index docs
#      - “pragma: no cover” hotspots
#      - dead code guards (if False:, return NotImplemented)
#   4) Installs deps (Poetry preferred; pip fallback) with optional extras.
#   5) Runs format/lint/type/tests if available: ruff, black, isort, mypy, pytest.
#   6) Byte-compiles Python sources.
#   7) Builds package artifacts (poetry build / python -m build / setup.py).
#   8) Produces Markdown + JSON reports in outputs/selftest/.
#
# Usage:
#   Save as: tools/review_and_compile.sh
#   chmod +x tools/review_and_compile.sh
#   tools/review_and_compile.sh
#
# Environment flags:
#   STRICT=1          # exit nonzero on any critical scaffolding (NotImplemented, pass-only, ellipsis-only)
#   SKIP_TESTS=1      # skip pytest
#   PYTHON=python3.11 # choose interpreter
#   PIP_EXTRAS=".[dev]"  # pip/poetry extras to install if available
#   POETRY_NO_VENV=1  # if set, force Poetry to use system env (virtualenvs.create false)
#   DISABLE_LINT_FIX=1 # run ruff/black/isort in check-only mode (no modifications)
#   EXTRA_PIP="mypy pytest ruff black isort build" # ensure tooling present when using pip
#
# Notes:
#   - Idempotent, CI-safe, colorized logging.
#   - Writes scanner as tools/scan_scaffolding.py (overwrites each run).
#   - Does not modify tracked code other than optional formatters (unless DISABLE_LINT_FIX=1).
##############################################################################

set -euo pipefail

############################# helpers ########################################

CYAN="$(printf '\033[36m')"; GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"; RED="$(printf '\033[31m')"
BOLD="$(printf '\033[1m')"; RESET="$(printf '\033[0m')"

log()  { printf "%s[review]%s %s\n" "$CYAN" "$RESET" "$1"; }
ok()   { printf "%s[  ok  ]%s %s\n" "$GREEN" "$RESET" "$1"; }
warn() { printf "%s[ warn ]%s %s\n" "$YELLOW" "$RESET" "$1"; }
err()  { printf "%s[ FAIL ]%s %s\n" "$RED" "$RESET" "$1"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

STRICT="${STRICT:-0}"
SKIP_TESTS="${SKIP_TESTS:-0}"
PYTHON_BIN="${PYTHON:-python3}"
PIP_EXTRAS="${PIP_EXTRAS:-}"
POETRY_NO_VENV="${POETRY_NO_VENV:-1}"
DISABLE_LINT_FIX="${DISABLE_LINT_FIX:-0}"
EXTRA_PIP="${EXTRA_PIP:-}"

timestamp_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

############################# root find ######################################

find_repo_root() {
  if has_cmd git && git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel
    return
  fi
  local d="$PWD"
  while [ "$d" != "/" ]; do
    if [ -f "$d/pyproject.toml" ] || [ -f "$d/setup.cfg" ] || [ -f "$d/setup.py" ] || [ -f "$d/README.md" ]; then
      echo "$d"; return
    fi
    d="$(dirname "$d")"
  done
  echo "$PWD"
}

ROOT="$(find_repo_root)"
cd "$ROOT"
log "Repo root: $ROOT"

mkdir -p outputs/selftest logs tools

############################ write the scanner ###############################

SCANNER="tools/scan_scaffolding.py"
cat > "$SCANNER" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Scaffolding/Incomplete Code Scanner (v1.3)

Finds common indicators of incomplete or placeholder code and emits a structured JSON.
Severity categories:
  • hard: NotImplementedError, pass-only blocks, ellipsis-only blocks
  • soft: TODO-like, placeholder-like, pragma: no cover, dead-guards
Also flags nearly-empty README/index docs.

Outputs:
  • outputs/selftest/review_findings.json
"""

from __future__ import annotations
import json, os, re, sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(os.environ.get("REPO_ROOT", ".")).resolve()
OUT_DIR = ROOT / "outputs" / "selftest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "review_findings.json"

# Regex patterns
PATTERNS = {
    "todo_like": re.compile(r"\b(TODO|FIXME|TBD|HACK|XXX)\b", re.IGNORECASE),
    "placeholder_like": re.compile(r"\b(placeholder|stub|scaffold|boilerplate)\b", re.IGNORECASE),
    "not_impl": re.compile(r"\braise\s+NotImplementedError\b"),
    "pass_only": re.compile(
        r"^\s*(?:@[\w.]+\s*\n)?\s*(def|class)\s+\w+[^\n]*:\s(?:\"\"\".?\"\"\"\s)?(?:pass\s*)+$",
        re.DOTALL | re.MULTILINE,
    ),
    "ellipsis_only": re.compile(
        r"^\s*(?:@[\w.]+\s*\n)?\s*(def|class)\s+\w+[^\n]*:\s(?:\"\"\".?\"\"\"\s)?(?:\.\.\.\s*)+$",
        re.DOTALL | re.MULTILINE,
    ),
    "stub_exception": re.compile(r"raise\s+Exception\((?:\"|')\s*(?:TODO|TBD|stub|placeholder)", re.IGNORECASE),
    "pragma_no_cover": re.compile(r"#\s*pragma:\s*no\s*cover", re.IGNORECASE),
    "dead_guard": re.compile(r"^\s*if\s+False\s*:\s*$", re.MULTILINE),
    "return_not_impl": re.compile(r"\breturn\s+NotImplemented\b"),
}

CODE_EXT = {".py"}
DOC_EXT = {".md", ".rst", ".txt"}
SKIP_DIRS = {
    "venv",".venv","env",".git",".mypy_cache",".ruff_cache",".pytest_cache",
    "pycache",".tox",".nox","build","dist",".dvc",".idea",".vscode","outputs"
}

def is_skipped_dir(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)

def scan() -> Dict[str, List[Dict[str, str]]]:
    findings = {
        "todo_like": [],
        "placeholder_like": [],
        "not_impl": [],
        "pass_only": [],
        "ellipsis_only": [],
        "stub_exception": [],
        "pragma_no_cover": [],
        "dead_guard": [],
        "return_not_impl": [],
        "empty_docs": [],
    }
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if is_skipped_dir(p):
                continue
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        suf = p.suffix.lower()
        if suf in CODE_EXT:
            for key, rx in PATTERNS.items():
                if key in {"pass_only", "ellipsis_only"}:
                    for m in rx.finditer(text):
                        findings[key].append({"file": str(p), "match": (m.group(0)[:200]).strip()})
                else:
                    for m in rx.finditer(text):
                        line = text.count("\n", 0, m.start()) + 1
                        findings[key].append({"file": str(p), "line": line, "match": (m.group(0)[:200]).strip()})
        elif suf in DOC_EXT and p.name.lower().startswith(("readme","index","overview")):
            nonspace = sum(1 for ch in text if not ch.isspace())
            if nonspace <= 30:
                findings["empty_docs"].append({"file": str(p), "chars_nonspace": nonspace})
    return findings

def main() -> int:
    f = scan()
    by_cat = {k: len(v) for k, v in f.items()}
    total = sum(by_cat.values())
    hard = by_cat.get("not_impl",0) + by_cat.get("pass_only",0) + by_cat.get("ellipsis_only",0)
    payload = {
        "root": str(ROOT),
        "total_hits": total,
        "by_category": by_cat,
        "hard_count": hard,
        "findings": f,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 1 if hard > 0 else 0

if __name__ == "__main__":
    raise SystemExit(main())
PY
chmod +x "$SCANNER"

############################ install dependencies ############################

use_poetry=0
if has_cmd poetry && [ -f "pyproject.toml" ]; then
  use_poetry=1
  log "Poetry detected"
  poetry --version >/dev/null
  if [ "${POETRY_NO_VENV}" = "1" ]; then
    poetry config virtualenvs.create false
  fi
  if [ -n "${PIP_EXTRAS}" ]; then
    # Attempt to install with extras; if fails, fallback to plain install
    if ! poetry install --no-interaction --no-ansi "${PIP_EXTRAS}"; then
      warn "Poetry install with extras failed; falling back to poetry install"
      poetry install --no-interaction --no-ansi
    fi
  else
    poetry install --no-interaction --no-ansi
  fi
  ok "Dependencies installed via Poetry"
elif [ -f "requirements.txt" ]; then
  log "Using pip with requirements.txt"
  "$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools >/dev/null
  "$PYTHON_BIN" -m pip install -r requirements.txt
  if [ -n "${EXTRA_PIP}" ]; then
    log "Installing extra tooling via pip: ${EXTRA_PIP}"
    "$PYTHON_BIN" -m pip install ${EXTRA_PIP}
  fi
  if [ -n "${PIP_EXTRAS}" ]; then
    log "Installing extras via pip: ${PIP_EXTRAS}"
    set +e
    "$PYTHON_BIN" -m pip install "${PIP_EXTRAS}"
    set -e
  fi
  ok "Dependencies installed via pip"
else
  warn "No pyproject.toml or requirements.txt found; proceeding with current environment"
fi

############################ run scanner #####################################

export REPO_ROOT="$ROOT"
log "Scanning for scaffolding / incomplete code"
set +e
"$PYTHON_BIN" "$SCANNER"
SCAN_RC=$?
set -e
if [ $SCAN_RC -ne 0 ]; then
  warn "Scanner reported hard scaffolding (NotImplemented/pass-only/ellipsis-only)."
  if [ "$STRICT" = "1" ]; then
    err "STRICT=1 set; failing early."
    # Still drop a minimal report before exiting
  fi
fi
ok "Scan finished (rc=$SCAN_RC). See outputs/selftest/review_findings.json"

############################ tool discovery ##################################

have_ruff=0;  has_cmd ruff  && have_ruff=1
have_black=0; has_cmd black && have_black=1
have_isort=0; has_cmd isort && have_isort=1
have_mypy=0;  has_cmd mypy  && have_mypy=1
have_pytest=0;has_cmd pytest&& have_pytest=1
have_build=0; "$PYTHON_BIN" -c "import build" >/dev/null 2>&1 && have_build=1

############################ format / lint / type ############################

if [ $have_ruff -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "ruff check (no fixes)"
    set +e; ruff check .; RC_RUFF=$?; set -e
    if [ $RC_RUFF -ne 0 ]; then warn "ruff check found issues"; fi
  else
    log "ruff check+format (apply fixes)"
    ruff check . || true
    ruff format . || true
  fi
  ok "ruff completed"
else
  warn "ruff not installed"
fi

if [ $have_black -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "black --check"
    set +e; black --check .; RC_BLACK=$?; set -e
    if [ $RC_BLACK -ne 0 ]; then warn "black check found formatting issues"; fi
  else
    log "black (format)"
    black .
  fi
  ok "black completed"
else
  warn "black not installed"
fi

if [ $have_isort -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "isort --check-only"
    set +e; isort --check-only .; RC_ISORT=$?; set -e
    if [ $RC_ISORT -ne 0 ]; then warn "isort check found import order issues"; fi
  else
    log "isort (imports)"
    isort .
  fi
  ok "isort completed"
else
  warn "isort not installed"
fi

if [ $have_mypy -eq 1 ]; then
  log "mypy (type check)"
  set +e
  mypy . > logs/mypy.log 2>&1
  MYPY_RC=$?
  set -e
  if [ $MYPY_RC -ne 0 ]; then
    warn "mypy errors; see logs/mypy.log"
    [ "$STRICT" = "1" ] && { err "STRICT=1 set; failing on mypy errors"; exit 4; }
  else
    ok "mypy passed"
  fi
else
  warn "mypy not installed"
fi

############################ tests ###########################################

if [ "$SKIP_TESTS" = "1" ]; then
  warn "Skipping tests (SKIP_TESTS=1)"
else
  if [ $have_pytest -eq 1 ]; then
    log "pytest"
    set +e
    pytest -q
    PYTEST_RC=$?
    set -e
    if [ $PYTEST_RC -ne 0 ]; then
      warn "pytest failed (rc=$PYTEST_RC)"
      [ "$STRICT" = "1" ] && { err "STRICT=1 set; failing on test errors"; exit 5; }
    else
      ok "pytest passed"
    fi
  else
    warn "pytest not installed"
  fi
fi

############################ byte-compile ####################################

log "Byte-compiling Python sources"
"$PYTHON_BIN" -m compileall -q .
ok "compileall finished"

############################ build ###########################################

BUILD_ART_DIR=""
if [ $use_poetry -eq 1 ]; then
  log "Poetry build"
  poetry build --no-interaction --no-ansi
  BUILD_ART_DIR="dist"
  ok "Poetry build complete (dist/)"
elif [ $have_build -eq 1 ]; then
  log "python -m build"
  "$PYTHON_BIN" -m build
  BUILD_ART_DIR="dist"
  ok "PEP 517 build complete (dist/)"
else
  if [ -f "setup.py" ]; then
    log "Legacy setup.py build"
    "$PYTHON_BIN" setup.py sdist bdist_wheel
    BUILD_ART_DIR="dist"
    ok "setup.py build complete (dist/)"
  else
    warn "No build backend detected; skipping packaging"
  fi
fi

############################ CLI smoke ######################################

CLI_SMOKE_STATUS="skipped"
if [ -f "spectramind.py" ]; then
  log "CLI smoke: spectramind (--version or -h)"
  set +e
  "$PYTHON_BIN" spectramind.py --version >/dev/null 2>&1 || "$PYTHON_BIN" spectramind.py -h >/dev/null 2>&1
  rc=$?
  set -e
  if [ $rc -eq 0 ]; then
    CLI_SMOKE_STATUS="ok"
    ok "CLI smoke passed"
  else
    CLI_SMOKE_STATUS="failed"
    warn "CLI smoke failed (non-fatal)"
  fi
fi

############################ summarize ######################################

REPORT_JSON="outputs/selftest/review_report.json"
REPORT_MD="outputs/selftest/review_report.md"
SCAN_JSON="outputs/selftest/review_findings.json"

RUN_TS="$(timestamp_iso)"
GIT_SHA="unknown"
if has_cmd git && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  GIT_SHA="$(git rev-parse --short HEAD)"
fi

TOTAL_HARD=0
if [ -f "$SCAN_JSON" ]; then
  TOTAL_HARD="$("$PYTHON_BIN" - <<PY
import json,sys
try:
    d=json.load(open("$SCAN_JSON"))
    print(int(d.get("hard_count",0)))
except Exception:
    print(0)
PY
)"
fi

# Write JSON summary
"$PYTHON_BIN" - <<PY
import json, os, sys
from pathlib import Path
scan_path = Path("$SCAN_JSON")
try:
    scanner = json.loads(scan_path.read_text()) if scan_path.exists() else {"total_hits":0,"by_category":{}}
except Exception:
    scanner = {"total_hits":0,"by_category":{}}
payload = {
    "time_utc": "$RUN_TS",
    "git_sha": "$GIT_SHA",
    "root": "$ROOT",
    "scanner": scanner,
    "tools": {
        "poetry": $use_poetry,
        "ruff": $have_ruff,
        "black": $have_black,
        "isort": $have_isort,
        "mypy": $have_mypy,
        "pytest": $have_pytest,
        "pep517_build": $have_build
    },
    "byte_compile": "done",
    "build_artifacts_dir": "$BUILD_ART_DIR",
    "cli_smoke_status": "$CLI_SMOKE_STATUS",
    "strict_mode": $STRICT
}
Path("$REPORT_JSON").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

# Write Markdown summary
{
  echo "# SpectraMind V50 — Review & Compile Report"
  echo
  echo "- Time (UTC): $RUN_TS"
  echo "- Git SHA: $GIT_SHA"
  echo "- Repo root: $ROOT"
  echo "- Strict mode: $STRICT"
  echo
  echo "## Scanner Summary"
  if [ -f "$SCAN_JSON" ]; then
    "$PYTHON_BIN" - <<'PY'
import json, sys
d=json.load(open("outputs/selftest/review_findings.json"))
print(f"- Total findings: {d.get('total_hits',0)}")
bc=d.get('by_category',{})
for key in ["todo_like","placeholder_like","not_impl","pass_only","ellipsis_only","stub_exception","pragma_no_cover","dead_guard","return_not_impl","empty_docs"]:
    print(f"  - {key}: {bc.get(key,0)}")
PY
  else
    echo "No scan JSON available"
  fi
  echo
  echo "## Tooling Detected"
  echo "- Poetry: $use_poetry  | ruff: $have_ruff | black: $have_black | isort: $have_isort | mypy: $have_mypy | pytest: $have_pytest | PEP517 build: $have_build"
  echo
  echo "## Steps Run"
  echo "1. Dependency install — done"
  echo "2. Scaffolding scan — done"
  echo "3. Lint/format/type — completed"
  echo "4. Tests — $( [ $SKIP_TESTS -eq 1 ] && echo "skipped" || echo "attempted" )"
  echo "5. Byte-compile — done"
  if [ -n "$BUILD_ART_DIR" ]; then
    echo "6. Package build — artifacts in '$BUILD_ART_DIR/'"
  else
    echo "6. Package build — skipped (no backend found)"
  fi
  echo "7. CLI smoke — $CLI_SMOKE_STATUS"
  echo
  echo "## Next Actions"
  if [ "${TOTAL_HARD:-0}" -gt 0 ]; then
    echo "- Hard issues detected (NotImplemented / pass-only / ellipsis-only). Address these before release."
  else
    echo "- No hard scaffolding found. Review TODO/placeholder/pragma-no-cover areas for completeness."
  fi
  echo
  echo "Generated by tools/review_and_compile.sh"
} > "$REPORT_MD"

ok "Reports written:"
printf "  - %s\n" "$REPORT_MD" "$REPORT_JSON"

# Final exit logic
if [ "$STRICT" = "1" ] && [ "${TOTAL_HARD:-0}" -gt 0 ]; then
  err "STRICT=1 and hard scaffolding found; exiting nonzero."
  exit 6
fi

ok "Done."

##############################################################################

# End of file

##############################################################################
