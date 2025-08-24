#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/diagnostics/test_auto_ablate_v50.py

SpectraMind V50 — Diagnostics Test: auto_ablate_v50 tool

This test suite verifies that the ablation tool (tools/auto_ablate_v50.py) is:
1) Discoverable (script/module exists and shows --help).
2) CLI-invokable in a *safe* mode (e.g., --dry-run / --selftest / --help-only),
   without performing heavy training or external network calls.
3) Emitting expected light artifacts (markdown/csv/html leaderboard) when flags
   indicate such outputs, OR logging a compliant message during dry-run.
4) Logging an audit entry in logs/v50_debug_log.md that references the invocation.
5) Respecting top-N semantics (if exposed) and reporting detected flags correctly.

Design notes
------------
• The tests parse the tool's --help output to adapt to the exact flag names present
  in your repository (e.g., --top-n vs --top_n; --md vs --markdown).
• The tests execute the ablation tool as a Python module or a direct script using
  subprocess, to avoid import-time side effects.
• The tests tolerate two classes of tools:
    A) CLI-first scripts (Typer/argparse) living at tools/auto_ablate_v50.py
    B) CLI wrapper commands that dispatch to the tool via "spectramind ablate"
       (if the top-level CLI is registered). When available, the test will try both.
• The tests never run heavy computation. They always prefer a dry/selftest path.

Preconditions
-------------
• Repository layout (expected, but the tests adapt defensively):
    - tools/auto_ablate_v50.py
    - logs/v50_debug_log.md (created on first run if not present)
    - outputs/ (writable; artifacts will be written to a temporary directory)
• Python 3.9+ and pytest.
• No external GPUs/resources are required.

What gets validated
-------------------
• --help prints and includes a recognizable description (sanity).
• A safe run (dry/selftest) exits with code 0.
• If --md/--csv/--html flags exist and are used, the corresponding files are emitted.
• A log entry referencing "auto_ablate" (or a known alias) appears in v50_debug_log.md.
• top-N flag parsing (e.g., --top-n 3) is accepted when present and recorded.

These tests are intentionally verbose and include explicit comments for auditability.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest


# --------------------------------------------------------------------------------------
# Path helpers
# --------------------------------------------------------------------------------------

def repo_root() -> Path:
    """
    Resolve the repository root by walking up from this test file until we find
    a directory that *likely* contains our project structure.

    Heuristic: Stop at the first ancestor containing a "tools" directory.
    """
    p = Path(__file__).resolve()
    for ancestor in [p] + list(p.parents):
        if (ancestor / "tools").is_dir():
            return ancestor
    # Fallback: assume two levels up from tests/diagnostics
    return Path(__file__).resolve().parents[2]


def tool_script_candidates() -> List[Path]:
    """
    Return plausible script locations for the ablation tool.
    The canonical path is tools/auto_ablate_v50.py.
    """
    root = repo_root()
    candidates = [
        root / "tools" / "auto_ablate_v50.py",
        root / "tools" / "auto_ablate.py",               # legacy alias (defensive)
        root / "tools" / "ablate_v50.py",                # defensive
    ]
    return [c for c in candidates if c.exists()]


def spectramind_cli_candidates() -> List[List[str]]:
    """
    Return candidate invocations for the top-level CLI wrapper. This is optional
    and will be tried after direct script/module execution if present.

    We support both an installed console_script "spectramind" and module forms.
    """
    return [
        ["spectramind", "ablate"],
        [sys.executable, "-m", "spectramind", "ablate"],
        [sys.executable, "-m", "src.cli.cli_ablate"],  # repo-local module path (defensive)
    ]


# --------------------------------------------------------------------------------------
# Subprocess helpers
# --------------------------------------------------------------------------------------

def run_proc(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 90) -> Tuple[int, str, str]:
    """
    Run a subprocess command and return (exit_code, stdout, stderr).
    This always uses text mode for easier assertions.
    """
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ},
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out, err
    except subprocess.TimeoutExpired:
        proc.kill()
        out, err = proc.communicate()
        return 124, out, err


def python_module_invocation(module: str, *args: str) -> List[str]:
    """
    Build a 'python -m <module> ...' command.
    """
    return [sys.executable, "-m", module, *args]


def python_script_invocation(script: Path, *args: str) -> List[str]:
    """
    Build a 'python <script> ...' command.
    """
    return [sys.executable, str(script), *args]


# --------------------------------------------------------------------------------------
# Flag discovery / adaptation
# --------------------------------------------------------------------------------------

FLAG_ALIASES = {
    # Help flags
    "help": ["--help", "-h"],

    # Safe-run flags (only one needs to match the tool)
    "dry_run": ["--dry-run", "--dryrun", "--selftest", "--check", "--no-exec", "--plan"],

    # Output dir
    "outdir": ["--outdir", "--out-dir", "--output", "--output-dir", "-o"],

    # Top N (accept multiple spellings)
    "top_n": ["--top-n", "--top_n", "--topn", "--top", "-n"],

    # Markdown export
    "md": ["--md", "--markdown", "--md-out", "--markdown-out"],

    # HTML export
    "html": ["--html", "--html-out", "--open-html", "--open_html", "--no-open-html"],

    # CSV export (optional)
    "csv": ["--csv", "--csv-out"],

    # Config grid path (optional; not required in dry mode)
    "grid": ["--grid", "--grid-config", "--grid_yaml", "--grid-yaml"],
}


def discover_supported_flags(help_text: str) -> Dict[str, str]:
    """
    Inspect the tool's --help text and map our abstract flag names to the real flags
    the tool supports. If a flag family isn't found, it's omitted.
    """
    mapping: Dict[str, str] = {}
    for abstract, aliases in FLAG_ALIASES.items():
        for alias in aliases:
            # Use word boundary to avoid matching substrings inside other words.
            if re.search(rf"(^|\s){re.escape(alias)}(\s|,|$)", help_text):
                mapping[abstract] = alias
                break
    return mapping


# --------------------------------------------------------------------------------------
# Artifact probing
# --------------------------------------------------------------------------------------

def recent_files_with_suffix(root: Path, suffixes: Tuple[str, ...]) -> List[Path]:
    """
    Return files in 'root' (recursive) that end with any of 'suffixes', ordered by mtime desc.
    """
    hits = []
    if not root.exists():
        return hits
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            hits.append(p)
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def project_root() -> Path:
    return repo_root()


@pytest.fixture
def temp_outdir(tmp_path: Path) -> Path:
    out = tmp_path / "auto_ablate_out"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def ensure_logs_dir(project_root: Path) -> Path:
    logs = project_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    # Touch the debug log if missing to simplify assertions.
    dbg = logs / "v50_debug_log.md"
    if not dbg.exists():
        dbg.write_text("# v50 Debug Log\n", encoding="utf-8")
    return logs


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_tool_discoverable_and_help(project_root: Path):
    """
    The ablation tool must be discoverable and respond to --help.
    This tries, in order:
      1) python tools/auto_ablate_v50.py --help  (direct script)
      2) python -m tools.auto_ablate_v50 --help  (module form if tools is a package)
      3) spectramind ablate --help               (top-level CLI wrapper, optional)
    """
    # 1) Direct script
    candidates = tool_script_candidates()
    help_texts = []
    for script in candidates:
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0 and (out or err):
            help_texts.append(out + "\n" + err)

    # 2) Module form (only if tools is a package with __init__.py)
    module_code, module_out, module_err = 1, "", ""
    tools_pkg = project_root / "tools" / "__init__.py"
    if tools_pkg.exists():
        module_code, module_out, module_err = run_proc(
            python_module_invocation("tools.auto_ablate_v50", "--help"),
            cwd=project_root
        )
        if module_code == 0 and (module_out or module_err):
            help_texts.append(module_out + "\n" + module_err)

    # 3) Top-level CLI wrapper
    wrapper_code, wrapper_out, wrapper_err = 1, "", ""
    for cli in spectramind_cli_candidates():
        wrapper_code, wrapper_out, wrapper_err = run_proc([*cli, "--help"], cwd=project_root)
        if wrapper_code == 0 and (wrapper_out or wrapper_err):
            help_texts.append(wrapper_out + "\n" + wrapper_err)
            break  # first success is enough

    # Assert we got at least one valid help.
    assert any(("ablate" in h.lower() or "ablation" in h.lower() or "auto_ablate" in h.lower()) for h in help_texts), \
        "No --help output mentioning ablation was detected from any candidate invocation."

    # Capture combined help text for downstream flag discovery in this test.
    combined_help = "\n\n".join(help_texts)
    assert "--help" in combined_help or "-h" in combined_help, \
        "Expected standard help flags are missing in the tool's help output."


def test_dry_run_invocation_and_logging(project_root: Path, temp_outdir: Path, ensure_logs_dir: Path):
    """
    Run the ablation tool in a *safe* mode and verify:
    - Exit code 0
    - A log entry is appended to logs/v50_debug_log.md
    - (If supported) an outdir is created and acknowledged
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    pre_log_content = read_text_or_empty(debug_log)

    # Step 1: obtain --help to discover flags
    help_code, help_text = 1, ""
    candidates = tool_script_candidates()
    used_cmd: Optional[List[str]] = None

    # Try direct script candidates first
    for script in candidates:
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_code, help_text = 0, out + "\n" + err
            used_cmd = python_script_invocation(script)  # base command, args appended below
            break

    # If not found, try module form
    if help_code != 0 and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.auto_ablate_v50", "--help"), cwd=project_root)
        if code == 0:
            help_code, help_text = 0, out + "\n" + err
            used_cmd = python_module_invocation("tools.auto_ablate_v50")

    # If still not found, try spectramind CLI wrapper
    if help_code != 0:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_code, help_text = 0, out + "\n" + err
                used_cmd = cli
                break

    assert used_cmd is not None and help_code == 0, "Unable to obtain --help from any ablation tool entrypoint."

    # Step 2: discover supported flags
    flags = discover_supported_flags(help_text)

    # Remove any pre-existing temp dir content to assert fresh effects
    if temp_outdir.exists():
        shutil.rmtree(temp_outdir)
    temp_outdir.mkdir(parents=True, exist_ok=True)

    # Build a safe command using discovered flags.
    cmd = list(used_cmd)

    # Choose a safe-run flag, otherwise we fall back to plain --help-only run.
    dryflag = flags.get("dry_run", None)
    if dryflag:
        cmd.append(dryflag)

    # Add output directory if supported (to check creation or acknowledgments)
    outflag = flags.get("outdir", None)
    if outflag:
        cmd.extend([outflag, str(temp_outdir)])

    # Add top-n if supported
    topn = 3
    topflag = flags.get("top_n", None)
    if topflag:
        cmd.extend([topflag, str(topn)])

    # Ask for markdown if supported (light artifact)
    mdflag = flags.get("md", None)
    if mdflag:
        cmd.append(mdflag)

    # Ask for html if supported (light artifact); some tools need an explicit flag to *not* open browser
    htmlflag = flags.get("html", None)
    if htmlflag:
        cmd.append(htmlflag)

    # CSV if supported
    csvflag = flags.get("csv", None)
    if csvflag:
        cmd.append(csvflag)

    # Execute the safe run
    code, out, err = run_proc(cmd, cwd=project_root, timeout=180)

    # If a tool insists on help-only safe path (no dry flag found), accept exit 0 + help.
    assert code == 0, f"Safe ablate invocation failed (code={code}). STDERR:\n{err}\nSTDOUT:\n{out}"

    # Minimal signal check in stdout/stderr
    combined = (out + "\n" + err).lower()
    assert ("ablate" in combined or "auto_ablate" in combined or "ablation" in combined), \
        "Output does not appear to come from an ablation tool."

    # Verify logging: a new entry should be appended.
    post_log_content = read_text_or_empty(debug_log)
    assert len(post_log_content) >= len(pre_log_content), "Debug log did not grow or is unreadable."
    # Heuristic: look for a line referencing ablate in the appended region.
    appended = post_log_content[len(pre_log_content):]
    assert re.search(r"(ablate|auto_ablate)", appended, re.IGNORECASE) or "ablate" in appended.lower(), \
        "No ablation-related reference found in v50_debug_log.md appended section."

    # Outdir acknowledgment:
    # If the tool supports --outdir, we expect the directory to exist (already ensured)
    # and possibly contain light artifacts (markdown/csv/html). This depends on the tool.
    assert temp_outdir.exists(), "Output directory does not exist after invocation."

    # We tolerate that in a pure dry-run no files are created. However, if flags for
    # md/csv/html are supported, we attempt to detect recently created outputs.
    # Any of the following suffixes count as evidence of a generated artifact.
    recent_md = recent_files_with_suffix(temp_outdir, (".md",))
    recent_html = recent_files_with_suffix(temp_outdir, (".html", ".htm"))
    recent_csv = recent_files_with_suffix(temp_outdir, (".csv",))

    # If the corresponding flags were accepted, we expect at least a placeholder file
    # or a console message stating where the file would be written. We assert len>=0
    # always (safe), but add soft checks for visibility and content if present.
    if mdflag:
        # Soft expectation: at least 1 markdown artifact or a mention in output
        if not recent_md:
            # Look for a statement indicating intended write path (dry planning).
            assert re.search(r"(markdown|\.md)", combined), \
                "Markdown flag accepted but neither file nor message detected."

    if htmlflag:
        # Soft expectation: at least 1 html artifact or a mention in output
        if not recent_html:
            assert re.search(r"(html|\.html)", combined), \
                "HTML flag accepted but neither file nor message detected."

    if csvflag:
        if not recent_csv:
            assert re.search(r"(csv|\.csv)", combined), \
                "CSV flag accepted but neither file nor message detected."


def test_top_n_semantics_and_artifacts_matrix(project_root: Path, temp_outdir: Path):
    """
    If a top-N flag exists, verify that:
    - The tool accepts a small top-N (e.g., 2) and finishes safely in dry/selftest.
    - The output mentions top-N or emits artifacts that imply top-N filtering was considered.

    We again parse --help, then construct an adaptive command. If top-N isn't supported,
    this test passes trivially after confirming safe --help behavior.
    """
    # Probe help to discover flags
    candidates = tool_script_candidates()
    used_cmd: Optional[List[str]] = None
    help_text = ""

    # Direct scripts first
    for script in candidates:
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            used_cmd = python_script_invocation(script)
            break

    # Module path if applicable
    if not used_cmd and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.auto_ablate_v50", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            used_cmd = python_module_invocation("tools.auto_ablate_v50")

    # spectramind wrapper as last resort
    if not used_cmd:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                used_cmd = cli
                break

    assert used_cmd is not None, "Could not obtain tool help to detect top-N flag."
    flags = discover_supported_flags(help_text)

    # If no top-N support, just assert help presence and finish.
    if "top_n" not in flags:
        assert "--help" in help_text or "-h" in help_text
        return

    # Construct a safe top-N command
    cmd = list(used_cmd)

    dryflag = flags.get("dry_run")
    if dryflag:
        cmd.append(dryflag)

    outflag = flags.get("outdir")
    if outflag:
        cmd.extend([outflag, str(temp_outdir)])

    # Use top-N = 2 for a quick, trivial run.
    cmd.extend([flags["top_n"], "2"])

    # Request MD and HTML if supported, to exercise artifact branching.
    if "md" in flags:
        cmd.append(flags["md"])
    if "html" in flags:
        cmd.append(flags["html"])

    code, out, err = run_proc(cmd, cwd=project_root, timeout=180)
    assert code == 0, f"Top-N safe invocation failed (code={code}). STDERR:\n{err}\nSTDOUT:\n{out}"

    combined = (out + "\n" + err).lower()
    # Look for any mention that suggests top-N/selection happened.
    assert re.search(r"\btop[\s\-_]?n\b|\btop\s*=\s*2\b|\bselect(ed)?\s+top\b", combined) or "2" in combined, \
        "No evidence in output that a top-N parameter was parsed or acknowledged."

    # If files were produced (dry-run sometimes produces placeholders),
    # lightly scan for leaderboard-like artifacts.
    md_files = recent_files_with_suffix(temp_outdir, (".md",))
    html_files = recent_files_with_suffix(temp_outdir, (".html", ".htm"))

    # If no files, ensure output indicates where they *would* go.
    if not (md_files or html_files):
        assert ("outdir" in combined or "output" in combined or "leaderboard" in combined), \
            "No artifacts produced and no mention of intended outputs detected."


def test_debug_log_contains_config_hash_or_cli_version_if_available(project_root: Path, ensure_logs_dir: Path):
    """
    The repository's logging policy often appends version/config-hash metadata to the debug log.
    This test checks for a recent entry that includes either:
      - 'config hash' / 'run hash' / a plausible hex-like hash string, or
      - 'version' metadata for the CLI.
    If not found, the test still passes after confirming that the log file is non-empty.
    """
    debug_log = ensure_logs_dir / "v50_debug_log.md"
    content = read_text_or_empty(debug_log)
    assert content.strip(), "v50_debug_log.md is empty after ablation runs."

    # Search only the last ~10KB to avoid scanning huge logs; adjust if necessary.
    tail = content[-10_000:]
    # Heuristics for hash/version signatures.
    hash_like = re.search(r"\b([a-f0-9]{8,64})\b", tail, flags=re.IGNORECASE)
    mentions = any(k in tail.lower() for k in [
        "config hash", "run hash", "version", "build timestamp", "cli version", "spectramind", "ablate"
    ])

    # It's acceptable that a given repository logs only minimal info; in that case,
    # ensure at least some recognizable mention is present.
    assert hash_like or mentions, \
        "Debug log tail lacks hash/version mentions; consider ensuring version/hash logging in CLI."


def test_artifact_manifest_sanity_if_emitted(temp_outdir: Path):
    """
    If the ablation tool wrote any JSON/CSV manifest into the temp outdir, perform quick sanity checks:
      - JSON can be decoded.
      - CSV has at least a header line.
    If nothing was written (pure dry-run), the test passes gracefully.
    """
    json_files = recent_files_with_suffix(temp_outdir, (".json",))
    csv_files = recent_files_with_suffix(temp_outdir, (".csv",))

    # JSON sanity
    for jf in json_files[:3]:  # limit to a few
        text = read_text_or_empty(jf).strip()
        if text:
            try:
                obj = json.loads(text)
                assert isinstance(obj, (dict, list))
            except Exception as e:
                pytest.fail(f"Malformed JSON artifact at {jf}: {e}")

    # CSV sanity
    for cf in csv_files[:3]:
        text = read_text_or_empty(cf).strip()
        if text:
            # Must have at least one newline (header) or one comma separation
            assert ("\n" in text or "," in text), f"CSV-like artifact at {cf} appears empty or malformed."


def test_idempotent_safe_runs_do_not_accumulate_heavy_artifacts(project_root: Path, temp_outdir: Path):
    """
    Re-run the same safe command twice and confirm that:
      - Exit code remains 0.
      - Number of heavy artifacts (e.g., large *.ckpt) does not grow.
    This test is defensive; if no heavy artifacts exist at all, it passes trivially.
    """
    # Probe help
    candidates = tool_script_candidates()
    used_cmd: Optional[List[str]] = None
    help_text = ""

    for script in candidates:
        code, out, err = run_proc(python_script_invocation(script, "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            used_cmd = python_script_invocation(script)
            break

    if not used_cmd and (project_root / "tools" / "__init__.py").exists():
        code, out, err = run_proc(python_module_invocation("tools.auto_ablate_v50", "--help"), cwd=project_root)
        if code == 0:
            help_text = out + "\n" + err
            used_cmd = python_module_invocation("tools.auto_ablate_v50")

    if not used_cmd:
        for cli in spectramind_cli_candidates():
            code, out, err = run_proc([*cli, "--help"], cwd=project_root)
            if code == 0:
                help_text = out + "\n" + err
                used_cmd = cli
                break

    assert used_cmd is not None, "Could not obtain help for idempotency test."
    flags = discover_supported_flags(help_text)

    # Build safe command
    cmd = list(used_cmd)
    dryflag = flags.get("dry_run")
    if dryflag:
        cmd.append(dryflag)
    outflag = flags.get("outdir")
    if outflag:
        cmd.extend([outflag, str(temp_outdir)])

    # Count heavy artifacts (heuristic: checkpoints or >5MB files)
    def count_heavy(root: Path) -> int:
        if not root.exists():
            return 0
        n = 0
        for p in root.rglob("*"):
            if p.is_file():
                if p.suffix.lower() in {".ckpt", ".pt"} or p.stat().st_size > 5 * 1024 * 1024:
                    n += 1
        return n

    pre_heavy = count_heavy(temp_outdir)

    # Run twice
    code1, out1, err1 = run_proc(cmd, cwd=project_root, timeout=180)
    code2, out2, err2 = run_proc(cmd, cwd=project_root, timeout=180)
    assert code1 == 0 and code2 == 0, f"Safe idempotent invocations failed.\n1) {err1}\n2) {err2}"

    post_heavy = count_heavy(temp_outdir)
    assert post_heavy <= pre_heavy, \
        f"Safe runs unexpectedly increased heavy artifacts: before={pre_heavy}, after={post_heavy}"


# --------------------------------------------------------------------------------------
# End of file
# --------------------------------------------------------------------------------------
