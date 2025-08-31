# gui/streamlit\_app.py

# =============================================================================

# SpectraMind V50 — Streamlit Dashboard (Upgraded First Implementation)

# -----------------------------------------------------------------------------

# Purpose:

# A thin, optional GUI that wraps the CLI-first SpectraMind V50 pipeline.

# - Provides controls to run `spectramind diagnose dashboard` reproducibly.

# - Visualizes artifacts emitted by the CLI (JSON, HTML, plots, logs).

#

# Architecture Notes:

# • The GUI NEVER bypasses the CLI or mutate pipeline logic.

# • All operations are serialized to the CLI; artifacts are then rendered.

# • Keeps NASA-grade reproducibility and CLI-first contracts intact.

#

# Usage:

# 1) Ensure the SpectraMind environment is activated and `spectramind` is on PATH.

# 2) From repo root, run:

# streamlit run gui/streamlit\_app.py

# 3) Use the left sidebar to set paths & options, then press "Run Diagnostics".

#

# Implementation Philosophy:

# • No hidden state: GUI loads and shows files written by the CLI.

# • Auditability: The CLI appends to logs/v50\_debug\_log.md; we render it.

# • Cross-platform: Uses Python stdlib + Streamlit; avoids OS-specific hacks.

#

# This upgraded version adds:

# - Robust repo root detection and CLI presence checks

# - Safer argument handling with clear echoing

# - Live streaming stdout/stderr view during CLI runs

# - Selectable HTML/JSON artifacts (not just latest)

# - Image gallery with grid and download buttons

# - Log tail with auto-refresh control

# - Recent runs table parsed from v50\_debug\_log.md (best-effort)

# - Caching of artifact scans with manual refresh

# - Lightweight dark-mode friendly layout choices

# =============================================================================

import os
import sys
import io
import re
import json
import time
import glob
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterable

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------

# Page Configuration

# -----------------------------------------------------------------------------

st.set\_page\_config(
page\_title="SpectraMind V50 — Diagnostics Dashboard",
page\_icon="🛰️",
layout="wide",
)

# -----------------------------------------------------------------------------

# Utilities — Paths & Files

# -----------------------------------------------------------------------------

def repo\_root\_default() -> Path:
"""
Attempt to auto-detect a reasonable repo root:
\- Prefer current working directory if it looks like repo root.
\- If running from inside `gui/`, step up one directory when needed.
\- Fallback to CWD.
"""
here = Path.cwd()
\# If we see typical repo markers from root
if (here / "configs").is\_dir() or (here / "spectramind.py").exists():
return here
\# If running from gui/
if here.name == "gui" and (here.parent / "configs").is\_dir():
return here.parent
return here

def to\_abs(path\_like: str | os.PathLike, base: Path) -> Path:
"""Resolve path\_like relative to base, returning an absolute Path."""
p = Path(path\_like)
return p if p.is\_absolute() else (base / p).resolve()

def \_newest\_file(pattern: str) -> Optional\[Path]:
"""Return newest file matching the glob pattern, or None."""
paths = \[Path(p) for p in glob.glob(pattern, recursive=True)]
if not paths:
return None
paths.sort(key=lambda p: p.stat().st\_mtime, reverse=True)
return paths\[0]

def find\_report\_html(outputs\_dir: Path) -> Optional\[Path]:
"""
Heuristics to find a diagnostics HTML report produced by CLI:
\- diagnostic\_report\*.html
\- *dashboard*.html
Priority: most recent match.
"""
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_report\*.html"),
str(outputs\_dir / "**" / "*dashboard*.html"),
str(outputs\_dir / "diagnostic\_report\*.html"),
str(outputs\_dir / "*dashboard*.html"),
]:
p = \_newest\_file(pat)
if p:
return p
return None

def find\_diagnostic\_json(outputs\_dir: Path) -> Optional\[Path]:
"""
Locate a JSON artifact, commonly 'diagnostic\_summary.json'.
Prefer most recent by heuristic; fallback to any '*diagnostic*.json'.
"""
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_summary.json"),
str(outputs\_dir / "diagnostic\_summary.json"),
str(outputs\_dir / "**" / "*diagnostic*.json"),
str(outputs\_dir / "*diagnostic*.json"),
]:
p = \_newest\_file(pat)
if p:
return p
return None

def list\_plot\_images(outputs\_dir: Path) -> List\[Path]:
"""
Return a list of likely plot images within outputs\_dir (PNG/JPG/JPEG),
deduped while preserving order, sorted by most recent first.
"""
images: List\[Path] = \[]
for pat in \[
str(outputs\_dir / "**" / "\*.png"),
str(outputs\_dir / "**" / "*.jpg"),
str(outputs\_dir / "\*\*" / "*.jpeg"),
str(outputs\_dir / "plots" / "*.png"),
str(outputs\_dir / "plots" / "*.jpg"),
str(outputs\_dir / "plots" / "\*.jpeg"),
]:
images.extend(\[Path(p) for p in glob.glob(pat, recursive=True)])
\# Unique by path, sorted by mtime desc
unique = sorted({p: None for p in images}.keys(), key=lambda p: p.stat().st\_mtime, reverse=True)
return unique

def list\_reports(outputs\_dir: Path, max\_items: int = 50) -> List\[Path]:
"""List HTML diagnostic reports in outputs\_dir for selection."""
results: List\[Path] = \[]
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_report\*.html"),
str(outputs\_dir / "**" / "*dashboard*.html"),
str(outputs\_dir / "diagnostic\_report\*.html"),
str(outputs\_dir / "*dashboard*.html"),
]:
results.extend(\[Path(p) for p in glob.glob(pat, recursive=True)])
results = sorted(set(results), key=lambda p: p.stat().st\_mtime, reverse=True)
return results\[:max\_items]

def list\_jsons(outputs\_dir: Path, max\_items: int = 50) -> List\[Path]:
"""List JSON diagnostic artifacts in outputs\_dir for selection."""
results: List\[Path] = \[]
for pat in \[
str(outputs\_dir / "**" / "diagnostic\_summary.json"),
str(outputs\_dir / "**" / "*diagnostic*.json"),
str(outputs\_dir / "diagnostic\_summary.json"),
str(outputs\_dir / "*diagnostic*.json"),
]:
results.extend(\[Path(p) for p in glob.glob(pat, recursive=True)])
results = sorted(set(results), key=lambda p: p.stat().st\_mtime, reverse=True)
return results\[:max\_items]

def tail\_file(path: Path, n\_bytes: int = 30000, encoding: str = "utf-8") -> str:
"""
Tail up to n\_bytes from a text file (bytes, not lines for speed).
Decode as UTF-8 best-effort and trim partial first line.
"""
if not path.exists():
return ""
try:
with path.open("rb") as f:
f.seek(0, os.SEEK\_END)
size = f.tell()
start = max(size - n\_bytes, 0)
f.seek(start)
chunk = f.read()
text = chunk.decode(encoding, errors="replace")
if start > 0:
text = text.split("\n", 1)\[-1]
return text
except Exception as e:
return f"\[Error reading {path.name}]: {e}"

# -----------------------------------------------------------------------------

# Utilities — CLI Execution

# -----------------------------------------------------------------------------

def spectramind\_available(cli: str) -> tuple\[bool, str]:
"""
Check whether the spectramind CLI is available.
Returns (available, message). Uses shutil.which and --version as a soft check.
"""
exe = shutil.which(cli)
if not exe:
return False, f"'{cli}' not found on PATH. Provide an absolute path or activate environment."
try:
proc = subprocess.run(\[cli, "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
if proc.returncode == 0:
out = (proc.stdout or "").strip() or (proc.stderr or "").strip()
return True, f"CLI OK: {out}"
else:
return True, "CLI found but '--version' returned a non-zero code. It may still be usable."
except Exception as e:
return True, f"CLI found but version check failed: {e!r}. Proceeding may still work."

def \_iter\_stream(proc: subprocess.Popen) -> Iterable\[tuple\[str, str]]:
"""
Yield incremental (stdout\_line, 'stdout') or (stderr\_line, 'stderr') as they arrive.
Uses non-blocking reads by iterating over pipes line-by-line.
"""
\# Iterate lines until both pipes are exhausted and the process terminates.
\# Use iter(stream.readline, '') to stream text mode lines.
if proc.stdout:
for line in iter(proc.stdout.readline, ""):
if line == "":
break
yield line.rstrip("\n"), "stdout"
if proc.stderr:
for line in iter(proc.stderr.readline, ""):
if line == "":
break
yield line.rstrip("\n"), "stderr"

def run\_cli\_stream(cmd: List\[str], cwd: Path) -> Tuple\[int, str, str]:
"""
Execute a CLI command and stream output live into Streamlit while capturing
the full stdout/stderr buffers. Returns (returncode, stdout, stderr).
"""
rendered = " ".join(shlex.quote(part) for part in cmd)
st.write(f"**Running:** `{rendered}`")
st.write(f"**Working directory:** `{str(cwd)}`")

```
# Placeholders for live streaming
stdout_ph = st.empty()
stderr_ph = st.empty()
std_out_buf: list[str] = []
std_err_buf: list[str] = []

try:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        universal_newlines=True,
        bufsize=1,  # line-buffered
    )
except FileNotFoundError:
    return 127, "", "Command not found. Is `spectramind` on PATH?"
except Exception as e:
    return 1, "", f"Failed to execute command: {e}"

# Stream lines as they arrive
for line, which in _iter_stream(proc):
    if which == "stdout":
        std_out_buf.append(line)
        # Render only last N lines to avoid huge widgets
        stdout_ph.code("\n".join(std_out_buf[-500:]) or "(no stdout)")
    else:
        std_err_buf.append(line)
        stderr_ph.code("\n".join(std_err_buf[-500:]) or "(no stderr)")

# Finalize
rc = proc.wait()
stdout = "\n".join(std_out_buf)
stderr = "\n".join(std_err_buf)
return rc, stdout, stderr
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
enable\_umap = st.sidebar.checkbox("UMAP Projection", value=True, help="If unchecked, passes --no-umap to the CLI.")
enable\_tsne = st.sidebar.checkbox("t-SNE Projection", value=True, help="If unchecked, passes --no-tsne to the CLI.")
open\_html = st.sidebar.checkbox("Open HTML after run (CLI-side)", value=False)
extra\_args = st.sidebar.text\_input(
"Extra CLI Args",
value="",
help="Optional: Add extra CLI flags (space-separated). Example: --no-symbolic --fast",
)

# Behavior switches

st.sidebar.markdown("---")
st.sidebar.subheader("Behavior")
dry\_run = st.sidebar.checkbox("Dry Run (skip CLI, just refresh artifacts)", value=False)
auto\_refresh\_secs = st.sidebar.number\_input(
"Auto-refresh interval (seconds, 0=off)",
min\_value=0,
max\_value=3600,
value=0,
help="If > 0, the page will auto-refresh to update logs and artifacts.",
)

run\_button = st.sidebar.button("Run Diagnostics", type="primary", use\_container\_width=True)
refresh\_scan = st.sidebar.button("Rescan Artifacts", use\_container\_width=True)

# -----------------------------------------------------------------------------

# Main — Header & Environment

# -----------------------------------------------------------------------------

st.title("🛰️ SpectraMind V50 — Diagnostics Dashboard (Streamlit)")

st.write(
"This GUI wraps the CLI-first pipeline. All run logic is executed through the "
"`spectramind` CLI; this dashboard reads and visualizes the artifacts it produces."
)

# Optional auto-refresh

if auto\_refresh\_secs > 0:
st.experimental\_singleton.clear()  # minimal reset for older Streamlit; harmless in current too
st\_autorefresh = st.experimental\_rerun  # alias for clarity (Streamlit auto reruns on timer)
st.experimental\_set\_query\_params(\_=int(time.time()))  # force cache-key change per refresh tick
st.experimental\_memo.clear()  # clear any stale memo
st.experimental\_data\_editor  # no-op to ensure module load; keeps lints quiet

# Show environment info

with st.expander("Environment & Paths", expanded=False):
st.write(f"**Repository Root:** `{repo_root}`")
st.write(f"**Outputs Directory:** `{outputs_dir}`")
st.write(f"**CLI Executable:** `{cli_binary_input}`")

```
ok, msg = spectramind_available(cli_binary_input)
if ok:
    st.success(msg)
else:
    st.warning(msg)

st.write(
    "Tip: Ensure your Python environment is activated and `spectramind --help` works "
    "from this repo root in your terminal."
)
```

# -----------------------------------------------------------------------------

# Execute CLI if requested

# -----------------------------------------------------------------------------

if run\_button and not dry\_run:
\# Build the CLI command reproducibly, with explicit options.
cmd: list\[str] = \[cli\_binary\_input, "diagnose", "dashboard"]

```
# Translate GUI toggles to CLI flags (keeping CLI-first behavior).
if not enable_umap:
    cmd.append("--no-umap")
if not enable_tsne:
    cmd.append("--no-tsne")

# Output directory override (if supported by CLI hydra path)
cmd.extend(["--outputs.dir", str(outputs_dir)])

# Optionally request CLI to open HTML post-run (noop if CLI ignores)
if open_html:
    cmd.append("--open-html")

# Inject any extra raw arguments
if extra_args.strip():
    try:
        parts = shlex.split(extra_args.strip())
    except ValueError:
        parts = extra_args.strip().split()
    cmd.extend(parts)

with st.spinner("Running CLI — this may take a while on first run..."):
    rc, stdout, stderr = run_cli_stream(cmd, cwd=repo_root)

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

# Artifact Scanning Helpers (cached)

# -----------------------------------------------------------------------------

@st.cache\_data(show\_spinner=False)
def scan\_reports(outputs\_dir: str) -> list\[tuple\[str, float]]:
"""
Return list of (path\_str, mtime) for reports.
Cache is invalidated by 'Rescan Artifacts' button via cache clear.
"""
dir\_path = Path(outputs\_dir)
items = list\_reports(dir\_path)
return \[(str(p), p.stat().st\_mtime) for p in items]

@st.cache\_data(show\_spinner=False)
def scan\_jsons(outputs\_dir: str) -> list\[tuple\[str, float]]:
"""Return list of (path\_str, mtime) for JSON artifacts."""
dir\_path = Path(outputs\_dir)
items = list\_jsons(dir\_path)
return \[(str(p), p.stat().st\_mtime) for p in items]

@st.cache\_data(show\_spinner=False)
def scan\_images(outputs\_dir: str, limit: int = 300) -> list\[tuple\[str, float]]:
"""Return list of (path\_str, mtime) for images (limited)."""
dir\_path = Path(outputs\_dir)
items = list\_plot\_images(dir\_path)\[:limit]
return \[(str(p), p.stat().st\_mtime) for p in items]

if refresh\_scan:
scan\_reports.clear()
scan\_jsons.clear()
scan\_images.clear()
st.experimental\_rerun()

# -----------------------------------------------------------------------------

# Artifacts — HTML Report, JSON Metrics, Plot Images, Logs

# -----------------------------------------------------------------------------

st.markdown("---")
st.header("Artifacts")

# --- HTML Report (embedded / selectable) ---

st.subheader("Diagnostics HTML Report")

report\_candidates = scan\_reports(str(outputs\_dir))
report\_label\_to\_path: dict\[str, str] = {}
for p\_str, mtime in report\_candidates:
ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
label = f"{Path(p\_str).name} — {ts}"
report\_label\_to\_path\[label] = p\_str

selected\_report\_label = None
if report\_candidates:
default\_idx = 0
\# Prefer newest file label for default (index 0 due to sorting)
selected\_report\_label = st.selectbox(
"Select report to view",
options=list(report\_label\_to\_path.keys()),
index=default\_idx,
help="Pick which diagnostics HTML to embed below.",
)

report\_path: Optional\[Path] = None
if selected\_report\_label:
report\_path = Path(report\_label\_to\_path\[selected\_report\_label])
else:
\# Fallback to heuristic newest
report\_path = find\_report\_html(outputs\_dir)

if report\_path and report\_path.exists():
try:
html\_text = report\_path.read\_text(encoding="utf-8", errors="replace")
st.components.v1.html(html\_text, height=900, scrolling=True)
st.caption(f"Embedded: {report\_path}")
with open(report\_path, "rb") as f:
st.download\_button("Download HTML report", data=f, file\_name=report\_path.name, mime="text/html")
except Exception as e:
st.warning(f"Unable to embed HTML report. Error: {e}")
st.write(f"**Report Path:** `{report_path}`")
else:
st.info("No diagnostics HTML report found yet. Run the CLI to generate one.")

# --- JSON Metrics Table (selectable) ---

st.subheader("Diagnostic JSON (e.g., diagnostic\_summary.json)")

json\_candidates = scan\_jsons(str(outputs\_dir))
json\_label\_to\_path: dict\[str, str] = {}
for p\_str, mtime in json\_candidates:
ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
label = f"{Path(p\_str).name} — {ts}"
json\_label\_to\_path\[label] = p\_str

selected\_json\_label = None
if json\_candidates:
selected\_json\_label = st.selectbox(
"Select JSON to view",
options=list(json\_label\_to\_path.keys()),
index=0,
help="Pick which diagnostic JSON to display below.",
)

json\_path: Optional\[Path] = None
if selected\_json\_label:
json\_path = Path(json\_label\_to\_path\[selected\_json\_label])
else:
json\_path = find\_diagnostic\_json(outputs\_dir)

if json\_path and json\_path.exists():
try:
raw = json\_path.read\_text(encoding="utf-8", errors="replace")
data = json.loads(raw)

```
    # Best-effort flatten for common SpectraMind shape
    if isinstance(data, dict):
        per_planet = None
        metrics_df = None
        if "per_planet" in data and isinstance(data["per_planet"], list):
            per_planet = pd.DataFrame(data["per_planet"])
        if "metrics" in data and isinstance(data["metrics"], dict):
            metrics_df = pd.DataFrame([data["metrics"]])

        if metrics_df is not None:
            st.markdown("**Global Metrics**")
            st.dataframe(metrics_df, use_container_width=True)

        if per_planet is not None and not per_planet.empty:
            st.markdown("**Per-Planet Summary**")
            st.dataframe(per_planet, use_container_width=True)

        with st.expander("Raw JSON"):
            st.json(data)
    elif isinstance(data, list):
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.json(data)

    st.caption(f"Loaded: {json_path}")
    with open(json_path, "rb") as f:
        st.download_button("Download JSON", data=f, file_name=json_path.name, mime="application/json")
except Exception as e:
    st.warning(f"Failed to parse JSON: {e}")
    with st.expander("Raw File"):
        try:
            st.code(json_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            st.write("Could not open file.")
```

else:
st.info("No diagnostic JSON found yet. Run the CLI to generate one.")

# --- Plot Images (grid with downloads) ---

st.subheader("Plots (PNG/JPG)")

images = scan\_images(str(outputs\_dir))
if images:
\# Show N images per row
N = 3
rows = (len(images) + N - 1) // N
idx = 0
for \_ in range(rows):
cols = st.columns(N)
for c in cols:
if idx >= len(images):
break
img\_path = Path(images\[idx]\[0])
caption = str(img\_path.relative\_to(repo\_root)) if str(img\_path).startswith(str(repo\_root)) else img\_path.name
try:
c.image(str(img\_path), caption=caption, use\_column\_width=True)
except Exception:
\# Fallback: read bytes and display
img\_bytes = img\_path.read\_bytes()
c.image(img\_bytes, caption=caption, use\_column\_width=True)

```
        with open(img_path, "rb") as f:
            c.download_button(
                label="Download",
                data=f.read(),
                file_name=img_path.name,
                mime="image/png" if img_path.suffix.lower() == ".png" else "image/jpeg",
                use_container_width=True,
            )
        idx += 1
st.caption(f"Showing {len(images)} plots.")
```

else:
st.info("No plot images found yet. When the CLI produces plots, they will appear here.")

# --- Log Tail ---

st.subheader("logs/v50\_debug\_log.md (tail)")

log\_path = repo\_root / "logs" / "v50\_debug\_log.md"
if log\_path.exists():
bytes\_to\_tail = st.slider("Bytes to tail", min\_value=1000, max\_value=200000, value=50000, step=1000)
text = tail\_file(log\_path, n\_bytes=bytes\_to\_tail)  # tail up to selected bytes
if text.strip():
st.code(text)
else:
st.info("Log file is empty.")
st.caption(f"Loaded: {log\_path}")
else:
st.info("No `logs/v50_debug_log.md` found yet. It will appear after CLI runs.")

# -----------------------------------------------------------------------------

# Recent Runs (best-effort parse of v50\_debug\_log.md)

# -----------------------------------------------------------------------------

st.markdown("---")
st.header("Recent Runs (from v50\_debug\_log.md)")

def parse\_recent\_runs(text: str, max\_rows: int = 200) -> pd.DataFrame:
"""
Best-effort parse of recent runs from v50\_debug\_log.md.
Expected to find lines like:
\[2025-08-02 12:34:56] spectramind version=X hash=ABC123 ... cmd="spectramind diagnose dashboard ..."
This is heuristic — adjust regex to your actual logging format if needed.
"""
rows: list\[dict] = \[]
\# Example heuristic patterns (tune to exact repo format)
time\_re = r"$(?P<ts>\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})$"
version\_re = r"version=(?P<version>\[^\s]+)"
hash\_re = r"(?\:config|run|hash)=(?P<hash>\[A-Za-z0-9\_-.]+)"
cmd\_re = r'cmd=(?:"|')(?P<cmd>.+?)(?:"|')'
rc\_re = r"rc=(?P<rc>-?\d+)"

```
pattern = re.compile(
    rf"{time_re}.*?(?:{version_re})?.*?(?:{hash_re})?.*?(?:{rc_re})?.*?(?:{cmd_re})?",
    flags=re.IGNORECASE,
)

for m in pattern.finditer(text):
    d = {
        "timestamp": m.groupdict().get("ts"),
        "version": m.groupdict().get("version"),
        "hash": m.groupdict().get("hash"),
        "rc": m.groupdict().get("rc"),
        "cmd": m.groupdict().get("cmd"),
    }
    rows.append(d)
    if len(rows) >= max_rows:
        break

if not rows:
    return pd.DataFrame(columns=["timestamp", "version", "hash", "rc", "cmd"])

df = pd.DataFrame(rows)
# Convert rc to int when possible
with pd.option_context("mode.chained_assignment", None):
    try:
        df["rc"] = pd.to_numeric(df["rc"], errors="coerce").astype("Int64")
    except Exception:
        pass
# Sort by timestamp if parseable
try:
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_parsed", ascending=False).drop(columns=["timestamp_parsed"])
except Exception:
    pass
return df
```

if log\_path.exists():
log\_text = tail\_file(log\_path, n\_bytes=200000)
runs\_df = parse\_recent\_runs(log\_text, max\_rows=200)
if not runs\_df.empty:
st.dataframe(runs\_df, use\_container\_width=True)
\# Export buttons
csv\_bytes = runs\_df.to\_csv(index=False).encode("utf-8")
md\_table = runs\_df.to\_markdown(index=False)
st.download\_button("Download runs.csv", data=csv\_bytes, file\_name="recent\_runs.csv", mime="text/csv")
st.download\_button("Download runs.md", data=md\_table.encode("utf-8"), file\_name="recent\_runs.md", mime="text/markdown")
else:
st.info("No recognizable run entries found in the log (parser is heuristic).")
else:
st.info("Run the CLI to generate logs; recent runs will appear here.")

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

  * Add per-planet drilldowns by reading additional JSON/CSV artifacts emitted by the CLI.
  * Wire in symbolic overlays and attention traces from artifacts (e.g., SHAP, COREL, symbolic violations).
  * Expose Hydra config presets for easy switching in the GUI (e.g., select `configs/*` combos).
  * Add a "Compare Runs" tab that loads multiple HTML/JSON artifacts and summarizes deltas.
    """
    )

# End of file
