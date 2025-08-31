# gui/streamlit\_app.py

# =============================================================================

# SpectraMind V50 — Streamlit Dashboard (First Implementation)

# -----------------------------------------------------------------------------

# Purpose:

# A thin, optional GUI that wraps the CLI-first SpectraMind V50 pipeline.

# - Provides buttons to run `spectramind diagnose dashboard` reproducibly.

# - Visualizes artifacts emitted by the CLI (JSON, HTML, PNG plots, logs).

#

# Architecture Notes:

# • The GUI NEVER bypasses the CLI or mutate pipeline logic.

# • All operations are serialized to the CLI; artifacts are then rendered.

# • This keeps NASA-grade reproducibility and CLI-first contracts intact.

#

# Usage:

# 1) Ensure the SpectraMind environment is activated and `spectramind` is on PATH.

# 2) From repo root, run:

# streamlit run gui/streamlit\_app.py

# 3) Use the left sidebar to set paths & options, then press "Run Diagnostics".

#

# Implementation Philosophy:

# • No hidden state: GUI loads and shows files written by the CLI.

# • Auditability: The CLI already appends to logs/v50\_debug\_log.md; we render it.

# • Cross-platform: Uses Python stdlib + Streamlit; avoids OS-specific hacks.

#

# Notes:

# • This is a “first implementation” scaffold focused on core flows.

# • Extend with richer charts, filters, and multi-run comparisons as needed.

# =============================================================================

import os
import sys
import json
import time
import glob
import base64
import shlex
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------

# Page Configuration (sets title, layout)

# -----------------------------------------------------------------------------

st.set\_page\_config(
page\_title="SpectraMind V50 — Diagnostics Dashboard",
page\_icon="🛰️",
layout="wide",
)

# -----------------------------------------------------------------------------

# Utility: Paths & Files

# -----------------------------------------------------------------------------

def repo\_root\_default() -> Path:
"""
Attempt to auto-detect a reasonable repo root:
\- Prefer current working directory.
\- If running from inside `gui/`, step up one directory.
"""
here = Path.cwd()
if (here / "gui").is\_dir() and (here / "configs").is\_dir():
return here
if here.name == "gui" and (here.parent / "configs").is\_dir():
return here.parent
return here

def to\_abs(path\_like: str, base: Path) -> Path:
"""Resolve path\_like relative to base, returning an absolute Path."""
p = Path(path\_like)
return p if p.is\_absolute() else (base / p).resolve()

def newest\_file(pattern: str) -> Optional\[Path]:
"""Return newest file matching the glob pattern, or None."""
paths = \[Path(p) for p in glob.glob(pattern)]
if not paths:
return None
paths.sort(key=lambda p: p.stat().st\_mtime, reverse=True)
return paths\[0]

def find\_report\_html(outputs\_dir: Path) -> Optional\[Path]:
"""
Heuristics to find a diagnostics HTML report produced by CLI:
\- diagnostic\_report\*.html
\- *dashboard*.html
Search order prioritizes most recent match.
"""
candidates = \[]
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_report\*.html"),
str(outputs\_dir / "**" / "*dashboard*.html"),
str(outputs\_dir / "diagnostic\_report\*.html"),
str(outputs\_dir / "*dashboard*.html"),
]:
p = newest\_file(pat)
if p:
candidates.append(p)
return candidates\[0] if candidates else None

def find\_diagnostic\_json(outputs\_dir: Path) -> Optional\[Path]:
"""
Locate a canonical JSON artifact, commonly named diagnostic\_summary.json
(exact name may vary; prefer most recent by heuristic).
"""
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_summary.json"),
str(outputs\_dir / "diagnostic\_summary.json"),
str(outputs\_dir / "**" / "*diagnostic*.json"),
str(outputs\_dir / "*diagnostic*.json"),
]:
p = newest\_file(pat)
if p:
return p
return None

def list\_plot\_images(outputs\_dir: Path) -> List\[Path]:
"""Return a list of likely plot images within outputs\_dir."""
images = \[]
for pat in \[
str(outputs\_dir / "**" / "\*.png"),
str(outputs\_dir / "**" / "*.jpg"),
str(outputs\_dir / "\*\*" / "*.jpeg"),
str(outputs\_dir / "plots" / "*.png"),
str(outputs\_dir / "plots" / "*.jpg"),
str(outputs\_dir / "plots" / "\*.jpeg"),
]:
images.extend(\[Path(p) for p in glob.glob(pat)])
\# Deduplicate while preserving order
seen = set()
unique = \[]
for p in images:
if p not in seen:
unique.append(p)
seen.add(p)
return unique

def tail\_file(path: Path, n: int = 3000, encoding: str = "utf-8") -> str:
"""
Tail up to n bytes from a text file (not lines — bytes, for performance).
Decode as UTF-8 best-effort.
"""
if not path.exists():
return ""
try:
with path.open("rb") as f:
f.seek(0, os.SEEK\_END)
size = f.tell()
start = max(size - n, 0)
f.seek(start)
chunk = f.read()
text = chunk.decode(encoding, errors="replace")
\# If we started in the middle of a line, strip partial first line.
if start > 0:
text = text.split("\n", 1)\[-1]
return text
except Exception as e:
return f"\[Error reading {path.name}]: {e}"

# -----------------------------------------------------------------------------

# Utility: Running CLI Commands

# -----------------------------------------------------------------------------

def run\_cli(cmd: List\[str], cwd: Path) -> Tuple\[int, str, str]:
"""
Execute a CLI command and capture (returncode, stdout, stderr).
Uses text mode for Python 3.7+; robust to long-running jobs.

```
Arguments:
  cmd: tokenized command (e.g., ["spectramind", "diagnose", "dashboard", ...])
  cwd: working directory (repo root)

Returns:
  (returncode, stdout, stderr)
"""
# Render the command into a safe shell-quoted string for auditability display.
rendered = " ".join(shlex.quote(part) for part in cmd)
st.write(f"**Running:** `{rendered}`")
st.write(f"**Working directory:** `{str(cwd)}`")

try:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    rc = proc.returncode
    return rc, stdout, stderr
except FileNotFoundError:
    return 127, "", "Command not found. Is `spectramind` on PATH?"
except Exception as e:
    return 1, "", f"Failed to execute command: {e}"
```

# -----------------------------------------------------------------------------

# Sidebar — Controls & Options

# -----------------------------------------------------------------------------

st.sidebar.title("SpectraMind V50 — Controls")

# Root directory of the repository

repo\_root\_input = st.sidebar.text\_input(
"Repository Root",
value=str(repo\_root\_default()),
help="Path to the SpectraMind V50 repository root (contains configs/, logs/, etc.)",
)
repo\_root = Path(repo\_root\_input).resolve()

# Outputs directory where artifacts will be written (may be overridden by CLI)

outputs\_dir\_input = st.sidebar.text\_input(
"Outputs Directory",
value=str(repo\_root / "outputs"),
help="Where the CLI writes diagnostics artifacts (HTML, JSON, plots)",
)
outputs\_dir = to\_abs(outputs\_dir\_input, repo\_root)

# CLI binary (default: spectramind on PATH)

cli\_binary\_input = st.sidebar.text\_input(
"CLI Executable",
value="spectramind",
help="The SpectraMind CLI executable (e.g., `spectramind`). If not on PATH, provide an absolute path.",
)

# Diagnose dashboard options

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnose — Dashboard Options")

enable\_umap = st.sidebar.checkbox("UMAP Projection", value=True)
enable\_tsne = st.sidebar.checkbox("t-SNE Projection", value=True)
open\_html = st.sidebar.checkbox("Open HTML after run (CLI-side)", value=False)
extra\_args = st.sidebar.text\_input(
"Extra CLI Args",
value="",
help="Optional: Add extra CLI flags (space-separated). Example: `--no-symbolic --fast`",
)

run\_button = st.sidebar.button("Run Diagnostics", type="primary", use\_container\_width=True)

# -----------------------------------------------------------------------------

# Main — Header

# -----------------------------------------------------------------------------

st.title("🛰️ SpectraMind V50 — Diagnostics Dashboard (Streamlit)")
st.write(
"This GUI wraps the CLI-first pipeline. All run logic is executed through the "
"`spectramind` CLI; this dashboard reads and visualizes the artifacts it produces."
)

# Guardrails: show quick environment info

with st.expander("Environment & Paths", expanded=False):
st.write(f"**Repository Root:** `{repo_root}`")
st.write(f"**Outputs Directory:** `{outputs_dir}`")
st.write(f"**CLI Executable:** `{cli_binary_input}`")
st.write(
"Tip: Ensure your Python environment is activated and `spectramind --help` works "
"from this repo root in your terminal."
)

# -----------------------------------------------------------------------------

# Execute CLI if requested

# -----------------------------------------------------------------------------

if run\_button:
\# Build the CLI command reproducibly, with explicit options.
cmd = \[cli\_binary\_input, "diagnose", "dashboard"]

```
# Translate GUI toggles to CLI flags (keeping CLI-first behavior):
# The CLI (by convention) supports `--no-umap` and `--no-tsne` toggles.
if not enable_umap:
    cmd.append("--no-umap")
if not enable_tsne:
    cmd.append("--no-tsne")

# Output directory override (if supported by CLI)
# We pass a normalized path for cross-platform robustness.
cmd.extend(["--outputs.dir", str(outputs_dir)])

# Optionally request CLI to open HTML post-run (no-op if CLI ignores)
if open_html:
    cmd.append("--open-html")

# Inject any extra raw arguments (advanced users)
if extra_args.strip():
    # Attempt a safe split; if user wants raw shell semantics, they can include quotes.
    try:
        parts = shlex.split(extra_args.strip())
    except ValueError:
        parts = extra_args.strip().split()
    cmd.extend(parts)

with st.spinner("Running CLI — this may take a while on first run..."):
    rc, stdout, stderr = run_cli(cmd, cwd=repo_root)

col1, col2 = st.columns(2)
with col1:
    st.subheader("CLI — stdout")
    if stdout.strip():
        st.code(stdout)
    else:
        st.write("_(no stdout output)_")
with col2:
    st.subheader("CLI — stderr")
    if stderr.strip():
        st.code(stderr)
    else:
        st.write("_(no stderr output)_")

if rc == 0:
    st.success("CLI completed successfully.")
else:
    st.error(f"CLI failed with return code: {rc}")
```

# -----------------------------------------------------------------------------

# Artifacts — HTML Report, JSON Metrics, Plot Images, Logs

# -----------------------------------------------------------------------------

st.markdown("---")
st.header("Artifacts")

# 1) HTML Report (embedded)

st.subheader("Diagnostics HTML Report")
report\_path = find\_report\_html(outputs\_dir)
if report\_path and report\_path.exists():
\# Reading the HTML into an iframe for display
try:
html\_text = report\_path.read\_text(encoding="utf-8", errors="replace")
st.components.v1.html(html\_text, height=900, scrolling=True)
st.caption(f"Embedded: {report\_path}")
except Exception as e:
st.warning(f"Unable to embed HTML report. Error: {e}")
st.write(f"**Report Path:** `{report_path}`")
else:
st.info("No diagnostics HTML report found yet. Run the CLI to generate one.")

# 2) JSON Metrics Table

st.subheader("diagnostic\_summary.json")
json\_path = find\_diagnostic\_json(outputs\_dir)
if json\_path and json\_path.exists():
try:
data = json.loads(json\_path.read\_text(encoding="utf-8", errors="replace"))
\# Best-effort flatten into DataFrame if dict-of-dicts; otherwise show raw JSON.
if isinstance(data, dict):
\# If it looks like { "metrics": {...}, "per\_planet": \[{...}, ...], ... }
per\_planet = None
if "per\_planet" in data and isinstance(data\["per\_planet"], list):
per\_planet = pd.DataFrame(data\["per\_planet"])
metrics\_df = None
if "metrics" in data and isinstance(data\["metrics"], dict):
metrics\_df = pd.DataFrame(\[data\["metrics"]])

```
        if metrics_df is not None:
            st.markdown("**Global Metrics**")
            st.dataframe(metrics_df, use_container_width=True)

        if per_planet is not None and not per_planet.empty:
            st.markdown("**Per-Planet Summary**")
            st.dataframe(per_planet, use_container_width=True)

        # For transparency, allow expanding raw JSON.
        with st.expander("Raw JSON"):
            st.json(data)
    elif isinstance(data, list):
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.json(data)
    st.caption(f"Loaded: {json_path}")
except Exception as e:
    st.warning(f"Failed to parse JSON: {e}")
    with st.expander("Raw File"):
        st.code(json_path.read_text(encoding="utf-8", errors="replace"))
```

else:
st.info("No diagnostic\_summary.json found yet. Run the CLI to generate one.")

# 3) Plot Images

st.subheader("Plots (PNG/JPG)")
images = list\_plot\_images(outputs\_dir)
if images:
\# Show up to N images per row
N = 3
rows = (len(images) + N - 1) // N
idx = 0
for \_ in range(rows):
cols = st.columns(N)
for c in cols:
if idx >= len(images):
break
img\_path = images\[idx]
try:
c.image(str(img\_path), caption=str(img\_path.relative\_to(repo\_root)), use\_column\_width=True)
except Exception:
\# Fallback: read bytes and display
img\_bytes = Path(img\_path).read\_bytes()
c.image(img\_bytes, caption=str(img\_path.name), use\_column\_width=True)
idx += 1
st.caption(f"Showing {len(images)} plots.")
else:
st.info("No plot images found yet. When the CLI produces plots, they will appear here.")

# 4) Log Tail

st.subheader("logs/v50\_debug\_log.md (tail)")
log\_path = repo\_root / "logs" / "v50\_debug\_log.md"
if log\_path.exists():
text = tail\_file(log\_path, n=50\_000)  # tail up to \~50 KB for view
if text.strip():
st.code(text)
else:
st.info("Log file is empty.")
st.caption(f"Loaded: {log\_path}")
else:
st.info("No `logs/v50_debug_log.md` found yet. It will appear after CLI runs.")

# -----------------------------------------------------------------------------

# Footer — Hints & Next Steps

# -----------------------------------------------------------------------------

with st.expander("Hints & Next Steps", expanded=False):
st.markdown(
"""

* **Reproducibility**: This GUI does not compute new results; it only runs CLI commands and renders files.
* **Artifacts**: If you customize the output directory or file names in your configs, update the “Outputs Directory” in the sidebar.
* **Advanced**: Use “Extra CLI Args” for experimental flags. The exact options depend on the version of your `spectramind` CLI.
* **Extending**:

  * Add plotly/altair charts for per-planet drilling with tooltips.
  * Wire in symbolic overlays and attention traces from artifacts.
  * Expose Hydra config presets for easy switching in the GUI.
    """
    )

# End of file
