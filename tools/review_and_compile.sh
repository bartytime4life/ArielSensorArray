#!/usr/bin/env bash
# SpectraMind V50 — Repository Review & Compile Orchestrator (v1.5)

set -euo pipefail

# ========= helpers =========
CYAN="$(printf '\033[36m')"; GREEN="$(printf '\033[32m')"
YELLOW="$(printf '\033[33m')"; RED="$(printf '\033[31m')"
BOLD="$(printf '\033[1m')"; RESET="$(printf '\033[0m')"

log()  { printf "%s[review]%s %s\n" "$CYAN" "$RESET" "$1"; }
ok()   { printf "%s[  ok  ]%s %s\n" "$GREEN" "$RESET" "$1"; }
warn() { printf "%s[ warn ]%s %s\n" "$YELLOW" "$RESET" "$1"; }
err()  { printf "%s[ FAIL ]%s %s\n" "$RED" "$RESET" "$1"; }
has_cmd() { command -v "$1" >/dev/null 2>&1; }

STRICT="${STRICT:-0}"                 # fail on hard scaffolding / type / test errors
SKIP_TESTS="${SKIP_TESTS:-0}"         # skip pytest
PYTHON_BIN="${PYTHON:-python3}"       # interpreter
POETRY_NO_VENV="${POETRY_NO_VENV:-1}" # poetry virtualenvs.create false when 1
POETRY_EXTRAS="${POETRY_EXTRAS:-}"    # space-separated extras for Poetry (e.g., "tracking viz")
PIP_EXTRAS="${PIP_EXTRAS:-}"          # pip extras string if Poetry unavailable (e.g., ".[dev]")
DISABLE_LINT_FIX="${DISABLE_LINT_FIX:-0}"  # run tooling in check-only mode
EXTRA_PIP="${EXTRA_PIP:-}"            # extra tooling for pip path, e.g. "mypy pytest ruff black isort build"

timestamp_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# ========= find repo root =========
find_repo_root() {
  if has_cmd git && git rev-parse --show-toplevel >/dev/null 2>&1; then
    git rev-parse --show-toplevel; return
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

# ========= write scanner =========
SCANNER="tools/scan_scaffolding.py"
cat > "$SCANNER" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Scaffolding/Incomplete Code Scanner (v1.5)
Detects:
  hard -> NotImplementedError, pass-only blocks, ellipsis-only blocks
  soft -> TODO/FIXME/TBD/HACK/XXX, placeholder/stub/scaffold/boilerplate,
          pragma: no cover, dead guards (if False:), return NotImplemented
Also flags near-empty README/index/overview docs.
Outputs: outputs/selftest/review_findings.json
"""
from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Dict, List

ROOT = Path(os.environ.get("REPO_ROOT", ".")).resolve()
OUT_DIR = ROOT / "outputs" / "selftest"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_OUT = OUT_DIR / "review_findings.json"

PATTERNS = {
    "todo_like": re.compile(r"\b(TODO|FIXME|TBD|HACK|XXX)\b", re.IGNORECASE),
    "placeholder_like": re.compile(r"\b(placeholder|stub|scaffold|boilerplate)\b", re.IGNORECASE),
    "not_impl": re.compile(r"\braise\s+NotImplementedError\b"),
    "pass_only": re.compile(
        r"^\s*(?:@[\w.]+\s*\n)?\s*(def|class)\s+\w+[^\n]*:\s*(?:\"\"\".*?\"\"\"\s*)?(?:pass\s*)+$",
        re.DOTALL | re.MULTILINE,
    ),
    "ellipsis_only": re.compile(
        r"^\s*(?:@[\w.]+\s*\n)?\s*(def|class)\s+\w+[^\n]*:\s*(?:\"\"\".*?\"\"\"\s*)?(?:\.\.\.\s*)+$",
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
    "__pycache__",".tox",".nox","build","dist",".dvc",".idea",".vscode","outputs"
}

def is_skipped_dir(p: Path) -> bool:
    return any(part in SKIP_DIRS for part in p.parts)

def scan() -> Dict[str, List[Dict[str, str]]]:
    findings = {k: [] for k in [
        "todo_like","placeholder_like","not_impl","pass_only","ellipsis_only","stub_exception",
        "pragma_no_cover","dead_guard","return_not_impl","empty_docs"
    ]}
    for p in ROOT.rglob("*"):
        if p.is_dir():
            if is_skipped_dir(p): continue
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        suf = p.suffix.lower()
        if suf in CODE_EXT:
            for key, rx in PATTERNS.items():
                if key in {"pass_only","ellipsis_only"}:
                    for m in rx.finditer(text):
                        findings[key].append({"file": str(p), "match": (m.group(0)[:200]).strip()})
                else:
                    for m in rx.finditer(text):
                        line = text.count("\n", 0, m.start()) + 1
                        findings[key].append({"file": str(p), "line": line, "match": (m.group(0)[:200]).strip()})
        elif suf in DOC_EXT and p.name.lower().startswith(("readme","index","overview")):
            if sum(1 for ch in text if not ch.isspace()) <= 30:
                findings["empty_docs"].append({"file": str(p)})
    return findings

def main() -> int:
    f = scan()
    by_cat = {k: len(v) for k, v in f.items()}
    hard = by_cat.get("not_impl",0) + by_cat.get("pass_only",0) + by_cat.get("ellipsis_only",0)
    payload = {
        "root": str(ROOT),
        "total_hits": sum(by_cat.values()),
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

# ========= install dependencies =========
use_poetry=0
if has_cmd poetry && [ -f "pyproject.toml" ]; then
  use_poetry=1
  log "Poetry detected"
  poetry --version >/dev/null
  [ "${POETRY_NO_VENV}" = "1" ] && poetry config virtualenvs.create false

  if poetry check >/dev/null 2>&1; then ok "pyproject.toml is valid"; else warn "poetry check reported issues"; fi

  # Build extras flags for Poetry
  POETRY_E_FLAGS=()
  if [ -n "${POETRY_EXTRAS}" ]; then
    for ex in ${POETRY_EXTRAS}; do POETRY_E_FLAGS+=("-E" "$ex"); done
    log "Poetry extras: ${POETRY_EXTRAS}"
  fi

  set +e
  if [ "${#POETRY_E_FLAGS[@]}" -gt 0 ]; then
    poetry install --no-interaction --no-ansi "${POETRY_E_FLAGS[@]}"
  else
    poetry install --no-interaction --no-ansi
  fi
  RC_POETRY=$?
  set -e
  if [ $RC_POETRY -ne 0 ]; then
    warn "Poetry install failed (rc=$RC_POETRY), retrying without extras"
    poetry install --no-interaction --no-ansi
  fi
  ok "Dependencies installed via Poetry"
elif [ -f "requirements.txt" ]; then
  log "Using pip with requirements.txt"
  "$PYTHON_BIN" -m pip install --upgrade pip wheel setuptools >/dev/null
  "$PYTHON_BIN" -m pip install -r requirements.txt
  [ -n "${EXTRA_PIP}" ] && { log "Installing extra tooling via pip: ${EXTRA_PIP}"; $PYTHON_BIN -m pip install ${EXTRA_PIP}; }
  if [ -n "${PIP_EXTRAS}" ]; then
    log "Installing pip extras: ${PIP_EXTRAS}"
    set +e; $PYTHON_BIN -m pip install "${PIP_EXTRAS}"; set -e
  fi
  ok "Dependencies installed via pip"
else
  warn "No pyproject.toml or requirements.txt found; continuing with current environment"
fi

# ========= run scanner =========
export REPO_ROOT="$ROOT"
log "Scanning for scaffolding / incomplete code"
set +e
"$PYTHON_BIN" "$SCANNER"
SCAN_RC=$?
set -e
if [ $SCAN_RC -ne 0 ]; then
  warn "Hard scaffolding detected (NotImplemented / pass-only / ellipsis-only)."
  [ "$STRICT" = "1" ] && err "STRICT=1 set; will fail at end if hard issues remain."
fi
ok "Scan complete (rc=$SCAN_RC). See outputs/selftest/review_findings.json"

# ========= tool discovery =========
have_ruff=0;  has_cmd ruff   && have_ruff=1
have_black=0; has_cmd black  && have_black=1
have_isort=0; has_cmd isort  && have_isort=1
have_mypy=0;  has_cmd mypy   && have_mypy=1
have_pytest=0;has_cmd pytest && have_pytest=1
have_build=0; "$PYTHON_BIN" -c "import build" >/dev/null 2>&1 && have_build=1

# ========= format / lint / type =========
if [ $have_ruff -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "ruff check (no fixes)"; set +e; ruff check .; RC_RUFF=$?; set -e; [ $RC_RUFF -ne 0 ] && warn "ruff issues found"
  else
    log "ruff check + format"; ruff check . || true; ruff format . || true
  fi; ok "ruff done"
else warn "ruff not installed"; fi

if [ $have_black -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "black --check"; set +e; black --check .; RC_BLACK=$?; set -e; [ $RC_BLACK -ne 0 ] && warn "black formatting issues"
  else
    log "black (format)"; black .
  fi; ok "black done"
else warn "black not installed"; fi

if [ $have_isort -eq 1 ]; then
  if [ "$DISABLE_LINT_FIX" = "1" ]; then
    log "isort --check-only"; set +e; isort --check-only .; RC_ISORT=$?; set -e; [ $RC_ISORT -ne 0 ] && warn "isort order issues"
  else
    log "isort (imports)"; isort .
  fi; ok "isort done"
else warn "isort not installed"; fi

if [ $have_mypy -eq 1 ]; then
  log "mypy (type check)"
  set +e; mypy . > logs/mypy.log 2>&1; MYPY_RC=$?; set -e
  if [ ${MYPY_RC:-0} -ne 0 ]; then
    warn "mypy errors; see logs/mypy.log"
    [ "$STRICT" = "1" ] && { err "STRICT=1; failing on mypy errors"; exit 4; }
  else ok "mypy passed"; fi
else warn "mypy not installed"; fi

# ========= tests =========
if [ "$SKIP_TESTS" = "1" ]; then
  warn "Skipping tests (SKIP_TESTS=1)"
else
  if [ $have_pytest -eq 1 ]; then
    log "pytest"
    set +e; pytest -q; PYTEST_RC=$?; set -e
    if [ ${PYTEST_RC:-0} -ne 0 ]; then
      warn "pytest failed (rc=$PYTEST_RC)"
      [ "$STRICT" = "1" ] && { err "STRICT=1; failing on test errors"; exit 5; }
    else ok "pytest passed"; fi
  else warn "pytest not installed"; fi
fi

# ========= byte-compile =========
log "Byte-compiling Python sources"
"$PYTHON_BIN" -m compileall -q .
ok "compileall finished"

# ========= build =========
BUILD_ART_DIR=""
if [ $use_poetry -eq 1 ]; then
  log "Poetry build"; poetry build --no-interaction --no-ansi; BUILD_ART_DIR="dist"; ok "Poetry build complete (dist/)"
elif [ $have_build -eq 1 ]; then
  log "PEP 517 build"; "$PYTHON_BIN" -m build; BUILD_ART_DIR="dist"; ok "PEP 517 build complete (dist/)"
else
  if [ -f "setup.py" ]; then
    log "Legacy setup.py build"; "$PYTHON_BIN" setup.py sdist bdist_wheel; BUILD_ART_DIR="dist"; ok "setup.py build complete (dist/)"
  else
    warn "No build backend detected; skipping packaging"
  fi
fi

# ========= CLI smoke =========
CLI_SMOKE_STATUS="skipped"
if [ $use_poetry -eq 1 ]; then
  if poetry run spectramind --version >/dev/null 2>&1 || poetry run spectramind -h >/dev/null 2>&1; then
    CLI_SMOKE_STATUS="ok"; ok "CLI smoke: spectramind"
  elif poetry run asa --version >/dev/null 2>&1 || poetry run asa -h >/dev/null 2>&1; then
    CLI_SMOKE_STATUS="ok"; ok "CLI smoke: asa"
  else
    CLI_SMOKE_STATUS="failed"; warn "CLI smoke failed (poetry run spectramind/asa)"
  fi
else
  if spectramind --version >/dev/null 2>&1 || spectramind -h >/dev/null 2>&1; then
    CLI_SMOKE_STATUS="ok"; ok "CLI smoke: spectramind"
  elif asa --version >/dev/null 2>&1 || asa -h >/dev/null 2>&1; then
    CLI_SMOKE_STATUS="ok"; ok "CLI smoke: asa"
  elif [ -f "spectramind.py" ] && "$PYTHON_BIN" spectramind.py -h >/dev/null 2>&1; then
    CLI_SMOKE_STATUS="ok"; ok "CLI smoke: python spectramind.py"
  else
    CLI_SMOKE_STATUS="failed"; warn "CLI smoke not found"
  fi
fi

# ========= summarize =========
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
import json
try:
  d=json.load(open("$SCAN_JSON"))
  print(int(d.get("hard_count",0)))
except Exception:
  print(0)
PY
)"
fi

# JSON report
"$PYTHON_BIN" - <<PY
import json
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

# Markdown report
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
import json
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

# ========= final exit logic =========
EXIT_CODE=0
[ "$STRICT" = "1" ] && [ "${TOTAL_HARD:-0}" -gt 0 ] && { err "STRICT=1 and hard scaffolding found"; EXIT_CODE=6; }
[ "$STRICT" = "1" ] && { [ "${MYPY_RC:-0}" -ne 0 ] || [ "${PYTEST_RC:-0}" -ne 0 ]; } && { err "STRICT=1 and type/test failures present"; EXIT_CODE=7; }
[ "$EXIT_CODE" -ne 0 ] && exit "$EXIT_CODE"
ok "Done."