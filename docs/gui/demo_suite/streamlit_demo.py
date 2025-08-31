```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Streamlit Demo (Upgraded Thin GUI Wrapper)
------------------------------------------------------------
A minimal, safe, and reproducible GUI that wraps the CLI-first pipeline.

Features
- Runs: spectramind diagnose dashboard [--no-umap] [--no-tsne] [extra hydra args...]
- NO shell passthrough; arguments validated and split safely
- Streams subprocess output live; shows final rc/stdout/stderr
- Auto-embeds the newest diagnostics HTML (diagnostic_report*.html / *dashboard*.html)
- Renders diagnostic_summary.json as rich tables + raw JSON
- Displays any PNG/JPG plots in a responsive grid
- Tails logs/v50_debug_log.md
- Artifact browser with glob + sorting + direct download
- Persistent UI state via st.session_state
- Optional: call the FastAPI backend (if provided) instead of local subprocess

Requirements
  pip install streamlit pandas
Optional
  Have `spectramind` on PATH or set an absolute path in the sidebar.
  For backend mode, run the provided FastAPI app and set BACKEND_URL in this UI.

Usage
  streamlit run streamlit_demo.py
"""
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------------
st.set_page_config(
    page_title="SpectraMind V50 — Streamlit Demo",
    page_icon="🛰️",
    layout="wide",
)

# --------------------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------------------
REPO = Path.cwd()
DEFAULT_OUT = REPO / "outputs"
LOG_PATH = REPO / "logs" / "v50_debug_log.md"

# If you have the FastAPI backend running, set e.g.:
# BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_URL: Optional[str] = None  # or st.secrets.get("backend_url")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@dataclass
class RunResult:
    rc: int
    stdout: str
    stderr: str


def _is_windows() -> bool:
    return platform.system().lower().startswith("win")


def sanitize_extra_args(text: str) -> List[str]:
    """
    Reject obvious shell control tokens and split extra args safely.
    We never use shell=True and we pass argv list to Popen.
    """
    illegal = [";", "&&", "||", "|", "`", "$(", "<(", ">{", "<{"]
    for tok in illegal:
        if tok in text:
            raise ValueError(f"Illegal shell control token detected: {tok}")
    return shlex.split(text.strip()) if text.strip() else []


def stream_subprocess(
    cmd: List[str],
    cwd: Path,
    poll_interval: float = 0.05,
) -> Iterable[str]:
    """
    Start a subprocess and yield stdout/stderr lines as they arrive (merged).
    """
    # On Windows, close_fds=False helps with some environments; keep default on *nix
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        close_fds=not _is_windows(),
    )
    try:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            yield line.rstrip("\n")
        # Drain any remaining
        proc.stdout.close()
    finally:
        # Ensure process ended
        while proc.poll() is None:
            time.sleep(poll_interval)


def run_cli_stream(cmd: List[str], cwd: Path) -> RunResult:
    """
    Run CLI streaming into Streamlit UI; returns rc and aggregated outputs.
    """
    placeholder = st.empty()
    out_lines: List[str] = []
    for line in stream_subprocess(cmd, cwd=cwd):
        out_lines.append(line)
        # Update steadily to keep UI responsive
        placeholder.code("\n".join(out_lines)[:50000] or "(empty)", language="bash")
    # Once finished, capture rc by running a no-op poll
    rc = 0
    try:
        # One more proc to fetch rc without re-running: not possible; thus
        # we’ll re-run cheaply `true` to keep code simple.
        rc = 0  # best-effort; real rc captured below via blocking communicate if needed
    except Exception:
        pass
    # As a fallback, run a blocking call just to get return code and stderr if necessary
    # (This keeps semantics simple; above we already streamed actual output.)
    proc2 = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        close_fds=not _is_windows(),
    )
    stdout2, stderr2 = proc2.communicate()
    rc = proc2.returncode
    # Merge already streamed + final capture for a consolidated record
    stdout_all = "\n".join(out_lines)
    if stdout2 and stdout2 not in stdout_all:
        stdout_all = (stdout_all + "\n" + stdout2).strip()
    return RunResult(rc=rc, stdout=stdout_all, stderr=stderr2 or "")


def newest(glob_pattern: str) -> Optional[Path]:
    files = list(Path().glob(glob_pattern))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def tail_bytes(path: Path, n: int = 50000) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n), os.SEEK_SET)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return text.split("\n", 1)[-1] if size > n else text


def artifact_list(root: Path, pattern: str, sort_desc: bool = True, limit: Optional[int] = None) -> List[Path]:
    paths = list(root.glob(pattern))
    try:
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=sort_desc)
    except Exception:
        paths.sort(key=lambda p: str(p), reverse=sort_desc)
    if limit:
        paths = paths[:limit]
    return paths


def embed_html_file(path: Path, height: int = 900) -> None:
    try:
        html = path.read_text(encoding="utf-8", errors="replace")
        st.components.v1.html(html, height=height, scrolling=True)
    except Exception as e:
        st.warning(f"Failed to embed HTML: {e}")


def call_backend_run(args: List[str], cwd: Path, cli: str = "spectramind") -> RunResult:
    import requests  # lazy import

    payload = {"args": args, "cli": cli, "cwd": str(cwd)}
    resp = requests.post(f"{BACKEND_URL}/api/run", json=payload, timeout=3600)
    if resp.status_code != 200:
        return RunResult(rc=resp.status_code, stdout="", stderr=f"Backend error: {resp.text}")
    data = resp.json()
    return RunResult(
        rc=data.get("returncode", 1),
        stdout=data.get("stdout", ""),
        stderr=data.get("stderr", ""),
    )


# --------------------------------------------------------------------------------------
# Sidebar controls (persistent via session_state)
# --------------------------------------------------------------------------------------
def _ss_get(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


st.sidebar.title("Controls")
repo_root = Path(st.sidebar.text_input("Repository Root", _ss_get("repo_root", str(REPO))))
st.session_state["repo_root"] = str(repo_root)

outputs_dir = Path(st.sidebar.text_input("Outputs Directory", _ss_get("outputs_dir", str(DEFAULT_OUT))))
st.session_state["outputs_dir"] = str(outputs_dir)

cli = st.sidebar.text_input("CLI Executable", _ss_get("cli", "spectramind"))
st.session_state["cli"] = cli

use_backend = st.sidebar.checkbox("Use FastAPI backend (if available)", value=bool(BACKEND_URL))
if use_backend:
    st.sidebar.write(f"Backend URL: `{BACKEND_URL or '(unset)'}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnose Options")
umap = st.sidebar.checkbox("Include UMAP", value=_ss_get("umap", True))
st.session_state["umap"] = umap
tsne = st.sidebar.checkbox("Include t-SNE", value=_ss_get("tsne", True))
st.session_state["tsne"] = tsne
extra = st.sidebar.text_input("Extra Hydra/CLI args", _ss_get("extra", ""))
st.session_state["extra"] = extra

run_btn = st.sidebar.button("Run Diagnose", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Artifacts Browser")
glob_pattern = st.sidebar.text_input("Glob Pattern", _ss_get("glob", "**/*"))
st.session_state["glob"] = glob_pattern
sort_desc = st.sidebar.checkbox("Sort by mtime desc", value=_ss_get("sort_desc", True))
st.session_state["sort_desc"] = sort_desc
limit = st.sidebar.number_input("Limit", min_value=0, max_value=5000, value=_ss_get("limit", 100), step=10)
st.session_state["limit"] = int(limit)

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("🛰️ SpectraMind V50 — Streamlit Demo")
st.caption("Thin GUI wrapping CLI calls; renders artifacts without mutating pipeline.")

# --------------------------------------------------------------------------------------
# Run CLI
# --------------------------------------------------------------------------------------
if run_btn:
    cmd = [cli, "diagnose", "dashboard", "--outputs.dir", str(outputs_dir)]
    if not umap:
        cmd.append("--no-umap")
    if not tsne:
        cmd.append("--no-tsne")

    if extra.strip():
        try:
            cmd.extend(sanitize_extra_args(extra))
        except ValueError as e:
            st.error(f"Invalid extra args: {e}")
            st.stop()

    st.write("**CLI (rendered):**")
    st.code(" ".join(shlex.quote(p) for p in cmd), language="bash")

    with st.spinner("Running CLI…"):
        if use_backend and BACKEND_URL:
            result = call_backend_run(cmd[1:], cwd=repo_root, cli=cmd[0])
            # Show outputs
            st.subheader("stdout")
            st.code(result.stdout or "(empty)")
            st.subheader("stderr")
            st.code(result.stderr or "(empty)")
            if result.rc == 0:
                st.success("CLI completed via backend.")
            else:
                st.error(f"Backend run failed: rc={result.rc}")
        else:
            # Local streaming run
            result = run_cli_stream(cmd, cwd=repo_root)
            st.subheader("Final stdout (aggregated)")
            st.code(result.stdout or "(empty)")
            st.subheader("Final stderr")
            st.code(result.stderr or "(empty)")
            if result.rc == 0:
                st.success("CLI completed.")
            else:
                st.error(f"CLI failed: return code {result.rc}")

# --------------------------------------------------------------------------------------
# Tabs: Report / Summary / Plots / Log / Artifacts
# --------------------------------------------------------------------------------------
tab_report, tab_summary, tab_plots, tab_log, tab_art = st.tabs(
    ["Diagnostics HTML", "diagnostic_summary.json", "Plots", "Log", "Artifacts"]
)

with tab_report:
    st.subheader("Diagnostics HTML Report")
    report = (
        newest(str(outputs_dir / "**" / "diagnostic_report*.html"))
        or newest(str(outputs_dir / "**" / "*dashboard*.html"))
    )
    if report and report.exists():
        embed_html_file(report, height=900)
        st.caption(f"Embedded: {report}")
        st.download_button("Download HTML", data=report.read_bytes(), file_name=report.name)
    else:
        st.info("No diagnostics HTML report found yet.")

with tab_summary:
    st.subheader("diagnostic_summary.json")
    js = newest(str(outputs_dir / "**" / "diagnostic_summary.json"))
    if js and js.exists():
        try:
            raw = js.read_text(encoding="utf-8", errors="replace")
            data = json.loads(raw)
            # Per-planet table
            if isinstance(data, dict) and "per_planet" in data and isinstance(data["per_planet"], list):
                st.markdown("**Per-Planet**")
                df = pd.DataFrame(data["per_planet"])
                st.dataframe(df, use_container_width=True)
            # Global metrics
            if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
                st.markdown("**Global Metrics**")
                st.dataframe(pd.DataFrame([data["metrics"]]), use_container_width=True)
            with st.expander("Raw JSON"):
                st.json(data)
            st.caption(f"Loaded: {js}")
            st.download_button("Download JSON", data=raw.encode("utf-8"), file_name=js.name)
        except Exception as e:
            st.warning(f"Failed to parse JSON: {e}")
    else:
        st.info("No diagnostic_summary.json found yet.")

with tab_plots:
    st.subheader("Plots (PNG/JPG)")
    plots = artifact_list(outputs_dir, "**/*", sort_desc=True, limit=None)
    plots = [p for p in plots if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if plots:
        cols = st.columns(3)
        for i, p in enumerate(plots):
            cols[i % 3].image(str(p), caption=str(p.relative_to(repo_root)), use_column_width=True)
        st.caption(f"Showing {len(plots)} plot(s).")
    else:
        st.info("No plots found.")

with tab_log:
    st.subheader("logs/v50_debug_log.md (tail)")
    if LOG_PATH.exists():
        st.code(tail_bytes(LOG_PATH, 50000) or "(empty)")
        st.caption(f"Loaded: {LOG_PATH}")
        st.download_button("Download Log", data=LOG_PATH.read_bytes(), file_name=LOG_PATH.name)
    else:
        st.info("No log yet; run the CLI first.")

with tab_art:
    st.subheader("Artifacts")
    if glob_pattern.strip():
        files = artifact_list(outputs_dir, glob_pattern, sort_desc=sort_desc, limit=limit or None)
        if files:
            for p in files:
                with st.container():
                    cols = st.columns([6, 2])
                    cols[0].markdown(f"**{p.name}**  \n`{p}`")
                    try:
                        size = p.stat().st_size
                        cols[0].caption(f"{size:,} bytes")
                    except Exception:
                        pass
                    with p.open("rb") as f:
                        cols[1].download_button("Download", data=f.read(), file_name=p.name, use_container_width=True)
            st.caption(f"Matched {len(files)} artifact(s).")
        else:
            st.info("No artifacts matched the glob.")
    else:
        st.info("Enter a glob pattern (e.g., **/*.html).")

# --------------------------------------------------------------------------------------
# Footer / environment
# --------------------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"CLI-first, GUI-optional • Python {sys.version.split()[0]} • "
    f"Repo: {REPO} • OS: {platform.platform()}"
)
```
