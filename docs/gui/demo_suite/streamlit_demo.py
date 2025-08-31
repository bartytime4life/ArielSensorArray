
---

# docs/gui/demo_suite/streamlit_demo.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Streamlit Demo (Thin GUI Wrapper)
- Calls the CLI (spectramind …)
- Renders artifacts: diagnostic_report*.html, diagnostic_summary.json, plots, and log tail
"""
import json
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="SpectraMind V50 — Streamlit Demo", page_icon="🛰️", layout="wide")

# ---------- helpers ----------
REPO = Path.cwd()
DEFAULT_OUT = REPO / "outputs"
LOG_PATH = REPO / "logs" / "v50_debug_log.md"

def run_cli(cmd, cwd: Path):
    st.write("**CLI (rendered):**", "```bash\n" + " ".join(shlex.quote(p) for p in cmd) + "\n```")
    try:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = proc.communicate()
        st.subheader("stdout")
        st.code(stdout or "(empty)")
        st.subheader("stderr")
        st.code(stderr or "(empty)")
        return proc.returncode
    except FileNotFoundError:
        st.error("`spectramind` not found on PATH. Activate your env or provide absolute path.")
        return 127

def newest(glob_pattern: str):
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

# ---------- sidebar ----------
st.sidebar.title("Controls")
repo_root = Path(st.sidebar.text_input("Repository Root", str(REPO)))
outputs_dir = Path(st.sidebar.text_input("Outputs Directory", str(DEFAULT_OUT)))
cli = st.sidebar.text_input("CLI Executable", "spectramind")

st.sidebar.markdown("---")
st.sidebar.subheader("Diagnose Options")
umap = st.sidebar.checkbox("UMAP", value=True)
tsne = st.sidebar.checkbox("t-SNE", value=True)
extra = st.sidebar.text_input("Extra CLI args", "")

run_btn = st.sidebar.button("Run Diagnose", type="primary", use_container_width=True)

st.title("🛰️ SpectraMind V50 — Streamlit Demo")
st.write("Thin GUI wrapping CLI calls; renders artifacts without mutating pipeline.")

# ---------- run CLI ----------
if run_btn:
    cmd = [cli, "diagnose", "dashboard", "--outputs.dir", str(outputs_dir)]
    if not umap:
        cmd.append("--no-umap")
    if not tsne:
        cmd.append("--no-tsne")
    if extra.strip():
        try:
            cmd.extend(shlex.split(extra.strip()))
        except ValueError:
            cmd.extend(extra.strip().split())
    with st.spinner("Running CLI…"):
        rc = run_cli(cmd, cwd=repo_root)
    if rc == 0:
        st.success("CLI completed.")
    else:
        st.error(f"CLI failed: return code {rc}")

st.markdown("---")
st.header("Artifacts")

# HTML report
st.subheader("Diagnostics HTML Report")
report = newest(str(outputs_dir / "**" / "diagnostic_report*.html")) or newest(str(outputs_dir / "**" / "*dashboard*.html"))
if report and report.exists():
    html = report.read_text(encoding="utf-8", errors="replace")
    st.components.v1.html(html, height=900, scrolling=True)
    st.caption(f"Embedded: {report}")
else:
    st.info("No diagnostics HTML report found yet.")

# JSON summary
st.subheader("diagnostic_summary.json")
js = newest(str(outputs_dir / "**" / "diagnostic_summary.json"))
if js and js.exists():
    try:
        data = json.loads(js.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict) and "per_planet" in data and isinstance(data["per_planet"], list):
            st.markdown("**Per-Planet**")
            st.dataframe(pd.DataFrame(data["per_planet"]), use_container_width=True)
        if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
            st.markdown("**Global Metrics**")
            st.dataframe(pd.DataFrame([data["metrics"]]), use_container_width=True)
        with st.expander("Raw JSON"):
            st.json(data)
        st.caption(f"Loaded: {js}")
    except Exception as e:
        st.warning(f"Failed to parse JSON: {e}")
else:
    st.info("No diagnostic_summary.json found yet.")

# Plots
st.subheader("Plots (PNG/JPG)")
plots = list(outputs_dir.glob("**/*.png")) + list(outputs_dir.glob("**/*.jpg")) + list(outputs_dir.glob("**/*.jpeg"))
if plots:
    cols = st.columns(3)
    for i, p in enumerate(plots):
        cols[i % 3].image(str(p), caption=str(p.relative_to(repo_root)), use_column_width=True)
    st.caption(f"Showing {len(plots)} plot(s).")
else:
    st.info("No plots found.")

# Log tail
st.subheader("logs/v50_debug_log.md (tail)")
if LOG_PATH.exists():
    st.code(tail_bytes(LOG_PATH, 50000))
    st.caption(f"Loaded: {LOG_PATH}")
else:
    st.info("No log yet; run the CLI first.")

st.markdown("---")
st.caption("GUI is a thin wrapper around CLI execution and artifact rendering. No hidden state.")
