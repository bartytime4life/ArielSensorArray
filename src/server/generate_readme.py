````python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_readme.py — SpectraMind V50
===============================================================================
Purpose
    Auto-generate a versioned, reproducible top-level README.md for the
    SpectraMind V50 repository by composing:
      • Repo metadata (git branch/commit/tag, dirty flag)
      • DVC status (if available)
      • Hydra config overview (defaults and groups)
      • CLI help snapshots (spectramind --help and key subcommands)
      • Dependency snapshot (Poetry or pip)
      • Run hash summary (if present)
      • Mermaid architecture/workflow diagrams
      • Project philosophy and quickstart
      • References (handbooks/documents maintained in repo)

Why
    Keeping README current is essential for NASA-grade reproducibility and
    scientific audit. This script consolidates the live repository state into
    human-readable documentation, and writes a timestamped copy to
    ./README_VERSIONED/ for provenance.

Design Notes
    • Pure Python 3.8+, standard library only (subprocess, json, re, etc.)
    • Soft dependencies: git, dvc, poetry, spectramind CLI (optional)
    • Graceful degradation if tools not found; we still produce README.
    • Do not make network calls; local inspection only.
    • Single-file, self-contained, with extensive inline comments.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
READMES_DIR = ROOT / "README_VERSIONED"
RUN_HASH_FILE = ROOT / "run_hash_summary_v50.json"
PYPROJECT = ROOT / "pyproject.toml"

# Helper: run a command and capture output (text). Returns (code, out, err).
def _run(cmd, timeout=20):
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            shell=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} not found"
    except Exception as e:
        return 1, "", f"{e}"

# Helper: try multiple candidate commands (e.g., poetry vs pip); returns first success
def _first_ok(cmds):
    for cmd in cmds:
        code, out, err = _run(cmd)
        if code == 0 and out:
            return out
    return ""

# Helper: read text file if exists
def _read_text(path: Path, default=""):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return default

# Helper: list directory safely
def _ls(path: Path):
    try:
        return sorted([p for p in path.iterdir()])
    except Exception:
        return []

# --------------------------------------------------------------------------------------
# Inspect: Git
# --------------------------------------------------------------------------------------

def get_git_info():
    """Collect git branch, commit short/long, tag if any, dirty flag."""
    info = {
        "branch": "",
        "commit_short": "",
        "commit_long": "",
        "tag": "",
        "dirty": False,
        "describe": "",
    }
    # Branch
    code, out, _ = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    if code == 0:
        info["branch"] = out
    # Commit (short + long)
    code, out, _ = _run(["git", "rev-parse", "--short", "HEAD"])
    if code == 0:
        info["commit_short"] = out
    code, out, _ = _run(["git", "rev-parse", "HEAD"])
    if code == 0:
        info["commit_long"] = out
    # Tag
    code, out, _ = _run(["git", "describe", "--tags", "--abbrev=0"])
    if code == 0:
        info["tag"] = out
    # Dirty?
    code, out, _ = _run(["git", "status", "--porcelain"])
    if code == 0:
        info["dirty"] = len(out.strip()) > 0
    # Describe
    code, out, _ = _run(["git", "describe", "--always", "--dirty", "--tags"])
    if code == 0:
        info["describe"] = out
    return info

# --------------------------------------------------------------------------------------
# Inspect: DVC
# --------------------------------------------------------------------------------------

def get_dvc_status():
    """Return a compact DVC status summary if DVC repo present."""
    # Check if .dvc directory or dvc.yaml exists
    has_dvc = (ROOT / ".dvc").exists() or (ROOT / "dvc.yaml").exists()
    if not has_dvc:
        return "DVC: not configured."

    lines = []
    code, out, err = _run(["dvc", "status", "-q"])
    if code == 0:
        if not out.strip():
            lines.append("DVC status: up to date ✅")
        else:
            lines.append("DVC status:")
            lines.extend([f"  {ln}" for ln in out.splitlines()])
    else:
        lines.append(f"DVC status: unavailable ({err or 'error'})")

    # Remote list (optional)
    code, out, err = _run(["dvc", "remote", "list"])
    if code == 0 and out:
        lines.append("DVC remotes:")
        lines.extend([f"  {ln}" for ln in out.splitlines()])

    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# Inspect: Hydra config groups and defaults
# --------------------------------------------------------------------------------------

def _read_yaml_defaults(train_yaml: Path) -> list[str]:
    """
    Best-effort parse for Hydra 'defaults' in YAML without importing PyYAML.
    We simply scan lines for a 'defaults:' block and collect the next indented mappings.
    This is intentionally simple yet robust for typical Hydra files.
    """
    if not train_yaml.exists():
        return []
    lines = _read_text(train_yaml).splitlines()
    defaults = []
    in_block = False
    indent = None
    for i, ln in enumerate(lines):
        if not in_block and re.match(r"^\s*defaults\s*:\s*$", ln):
            in_block = True
            # next lines will be part of block until dedent or empty line
            continue
        if in_block:
            if not ln.strip():
                break
            # compute indent
            if indent is None:
                m = re.match(r"^(\s+)-\s+(.*)$", ln)
                if m:
                    indent = len(m.group(1))
                else:
                    # Not a list item; end block
                    break
            # verify indentation level
            m = re.match(rf"^(\s{{{indent}}})-\s+(.*)$", ln)
            if m:
                item = m.group(2).strip()
                defaults.append(item)
            else:
                # Dedent -> end
                break
    return defaults

def get_hydra_overview():
    """
    Summarize configs directory structure and Hydra defaults found in a canonical main file
    such as configs/train.yaml (if present).
    """
    summary = []
    summary.append("### Hydra Config Overview")
    summary.append("")
    if not CONFIGS_DIR.exists():
        summary.append("_configs/ not found._")
        return "\n".join(summary)

    # List groups (subfolders)
    groups = [p.name for p in _ls(CONFIGS_DIR) if p.is_dir()]
    if groups:
        summary.append("**Config groups**:")
        summary.append("")
        summary.append(", ".join(sorted(groups)))
        summary.append("")
    else:
        summary.append("_No subgroups under configs/_")
        summary.append("")

    # Show defaults from a main entrypoint config if any
    main_candidates = [
        CONFIGS_DIR / "train.yaml",
        CONFIGS_DIR / "main.yaml",
        CONFIGS_DIR / "config.yaml",
    ]
    main_file = next((p for p in main_candidates if p.exists()), None)
    if main_file:
        defaults = _read_yaml_defaults(main_file)
        summary.append(f"**Defaults (from `{main_file.relative_to(ROOT)}`)**:")
        summary.append("")
        if defaults:
            for d in defaults:
                summary.append(f"- {d}")
        else:
            summary.append("_No explicit defaults block found._")
    else:
        summary.append("_No main config (train.yaml/main.yaml/config.yaml) found._")

    return "\n".join(summary)

# --------------------------------------------------------------------------------------
# Inspect: CLI help
# --------------------------------------------------------------------------------------

def get_cli_help():
    """
    Capture CLI help from spectramind root and common subcommands.
    """
    lines = []
    code, out, err = _run(["spectramind", "--help"], timeout=15)
    if code == 0:
        lines.append("### CLI — `spectramind --help`")
        lines.append("")
        lines.append("```text")
        lines.append(out)
        lines.append("```")
    else:
        lines.append("### CLI — `spectramind --help`")
        lines.append("")
        lines.append(f"_Unavailable: {err or 'spectramind not found'}_")

    for sub in ["calibrate", "train", "diagnose", "submit"]:
        code, out, err = _run(["spectramind", sub, "--help"], timeout=15)
        lines.append("")
        lines.append(f"#### `spectramind {sub} --help`")
        lines.append("")
        if code == 0:
            lines.append("```text")
            lines.append(out)
            lines.append("```")
        else:
            lines.append(f"_Unavailable: {err or 'spectramind not found'}_")

    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# Inspect: Dependencies (Poetry or pip)
# --------------------------------------------------------------------------------------

def get_dependency_snapshot():
    """
    Attempt to snapshot dependencies.
      1) poetry export --without-hashes
      2) pip freeze
    """
    # Poetry export (preferred if pyproject exists)
    if PYPROJECT.exists():
        out = _first_ok([
            ["poetry", "export", "--format=requirements.txt", "--without-hashes"],
            ["poetry", "export", "--without-hashes"],
        ])
        if out:
            return "### Dependencies (Poetry export)\n\n```text\n" + out + "\n```"

    # Fallback to pip freeze
    out = _first_ok([
        [sys.executable, "-m", "pip", "freeze"]
    ])
    if out:
        return "### Dependencies (pip freeze)\n\n```text\n" + out + "\n```"

    return "### Dependencies\n\n_Unable to determine (no Poetry/pip)._"

# --------------------------------------------------------------------------------------
# Inspect: Run Hash Summary
# --------------------------------------------------------------------------------------

def get_run_hash_summary():
    """
    Read run_hash_summary_v50.json if present (created by pipeline).
    """
    if RUN_HASH_FILE.exists():
        try:
            data = json.loads(RUN_HASH_FILE.read_text(encoding="utf-8"))
            # Pretty-printed subset
            pretty = json.dumps(data, indent=2, ensure_ascii=False)
            return "### Run Hash Summary\n\n```json\n" + pretty + "\n```"
        except Exception as e:
            return f"### Run Hash Summary\n\n_Unable to parse JSON: {e}_"
    return "### Run Hash Summary\n\n_No run hash file found._"

# --------------------------------------------------------------------------------------
# Mermaid diagrams (as constants)
# --------------------------------------------------------------------------------------

MERMAID_ARCH = dedent("""\
```mermaid
flowchart TD
    A0[Raw Telescope Data]:::data --> A1[Calibration Kill Chain]:::stage
    A1 --> A2[Preprocessing & Augmentation]:::stage
    A2 --> A3[FGS1 Mamba Encoder]:::model
    A2 --> A4[AIRS GNN Encoder]:::model
    A3 --> A5[Fusion Cross-Attention]:::fusion
    A4 --> A5
    A5 --> A6[Multi-Scale Decoders (μ, σ)]:::model
    A6 --> A7[Loss Engine (GLL + FFT + Symbolics)]:::loss
    A7 --> A8[Diagnostics + Dashboard]:::diag
    A8 --> A9[Submission Bundle]:::artifact

classDef data fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
classDef stage fill:#ede7f6,stroke:#5e35b1,color:#4527a0;
classDef model fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
classDef fusion fill:#fff3e0,stroke:#ef6c00,color:#e65100;
classDef loss fill:#ffebee,stroke:#c62828,color:#b71c1c;
classDef diag fill:#ede7f6,stroke:#5e35b1,color:#311b92;
classDef artifact fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;
````

""")

MERMAID\_FLOW = dedent("""\\

```mermaid
graph LR
    C0[User CLI (spectramind)] -->|Hydra Compose| C1[Pipeline Stage]
    C1 -->|DVC + Git| C2[Data/Model Versioning]
    C1 -->|Logging| C3[v50_debug_log.md + MLflow]
    C1 -->|Artifacts| C4[HTML Dashboard, Plots, JSON]

classDef cli fill:#e3f2fd,stroke:#1565c0;
classDef repo fill:#ede7f6,stroke:#5e35b1;
classDef log fill:#fff3e0,stroke:#ef6c00;
classDef art fill:#f3e5f5,stroke:#6a1b9a;
```

""")

# --------------------------------------------------------------------------------------

# Compose README content

# --------------------------------------------------------------------------------------

def compose\_readme(git\_info: dict) -> str:
now = \_dt.datetime.now().astimezone()
ts = now\.strftime("%Y-%m-%d %H:%M:%S %Z")
ver\_line = f"{git\_info.get('describe','')}  ·  {ts}"
dirty\_flag = " (dirty)" if git\_info.get("dirty") else ""

````
header = dedent(f"""\
# 🌌 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

> Version: **{git_info.get('commit_short','')}** on **{git_info.get('branch','')}**{dirty_flag} · {ver_line}
""")

mission = dedent("""\
## 0) Mission

**SpectraMind V50** is a **neuro-symbolic, physics-informed AI system** that predicts exoplanet transmission spectra (μ) and uncertainty (σ) across 283 bins from ESA Ariel–simulated data (FGS1 + AIRS).

**Design principles**
- 🚀 **CLI-first, GUI-optional** (Typer + Hydra; GUI is a thin wrapper)
- 🛰 **NASA-grade reproducibility** (DVC + Git + Hydra + MLflow logging)
- 🌌 **Physics & Symbolics** (smoothness, nonnegativity, FFT priors, molecular fingerprints)
- 🧩 **Modularity** (encoders, decoders, constraints, calibration swappable)
- 🏆 **Kaggle-safe** (≤ 9h runtime, ≤ 16 GB GPU, deterministic runs)
""")

arch = "## 1) Architecture Overview\n\n" + MERMAID_ARCH

flow = "## 2) Workflow & Reproducibility\n\n" + MERMAID_FLOW

quickstart = dedent("""\
## 3) Quickstart

### Environment
```bash
git clone https://github.com/your-org/spectramind-v50
cd spectramind-v50
poetry install          # or: pip install -r requirements.txt
dvc pull                # fetch data & model artifacts (if configured)
```

### CLI usage
```bash
# Calibration
spectramind calibrate data=nominal calib=nominal

# Training
spectramind train model=v50 optimizer=adamw trainer=kaggle_safe

# Diagnostics dashboard (UMAP, SHAP, FFT, symbolic overlays)
spectramind diagnose dashboard

# Submission bundle (with self-test)
spectramind submit --selftest
```
""")

hydra = get_hydra_overview()
dvc = get_dvc_status()
cli_help = get_cli_help()
deps = get_dependency_snapshot()
runhash = get_run_hash_summary()

refs = dedent("""\
## 7) References & Handbooks

This repository integrates material from internal handbooks (AI processing/decoding, physics modeling, spectroscopy, GUI engineering, Hydra/YAML, Kaggle operations). See `/docs` or `/handbooks` folders if present.
""")

cite = dedent("""\
## 8) Citation

If this repository is helpful in your work:
```
@misc{SpectraMindV50_NeurIPS2025,
  author = {Barta, A. et al.},
  title = {SpectraMind V50: Neuro-Symbolic Exoplanet Spectroscopy System},
  year = {2025},
  howpublished = {\\url{https://github.com/your-org/spectramind-v50}},
  note = {NeurIPS 2025 Ariel Data Challenge}
}
```
""")

# Compose
parts = [
    header,
    mission,
    arch,
    flow,
    quickstart,
    "## 4) Configuration\n\n" + hydra,
    "## 5) Data/Model Versioning\n\n" + dvc,
    "## 6) CLI Snapshots\n\n" + cli_help,
    deps,
    runhash,
    refs,
    cite,
]
return "\n\n".join(parts).rstrip() + "\n"
````

# --------------------------------------------------------------------------------------

# Write README.md and versioned copy

# --------------------------------------------------------------------------------------

def write\_readmes(content: str, git\_info: dict):
\# Write top-level README.md
(ROOT / "README.md").write\_text(content, encoding="utf-8")

```
# Ensure versioned directory
READMES_DIR.mkdir(parents=True, exist_ok=True)

# Compose filename with timestamp and short commit
now = _dt.datetime.now().astimezone()
stamp = now.strftime("%Y%m%d_%H%M%S")
short = git_info.get("commit_short", "unknown")
fname = f"README_{stamp}_{short}.md"
(READMES_DIR / fname).write_text(content, encoding="utf-8")
```

# --------------------------------------------------------------------------------------

# Main

# --------------------------------------------------------------------------------------

def main():
git\_info = get\_git\_info()
content = compose\_readme(git\_info)
write\_readmes(content, git\_info)
print("✅ README.md generated and versioned copy written to README\_VERSIONED/")

if **name** == "**main**":
main()

```
```
