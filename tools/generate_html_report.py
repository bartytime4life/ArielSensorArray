#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_html_report.py

SpectraMind V50 — Unified Diagnostics HTML Report Generator
===========================================================

Purpose
-------
Render a comprehensive, single‑file HTML report summarizing diagnostics produced by
SpectraMind V50. It ingests the JSON/CSV artifacts created by tools like
`generate_diagnostic_summary.py` and optional image/CSV overlays (UMAP, t‑SNE, SHAP,
symbolic tables), then emits a polished, self‑contained report suitable for CI,
Kaggle, and offline review.

Key Features
------------
• Ingests diagnostics JSON (`diagnostic_summary.json`) and CSV metrics
• Summarizes core metrics (GLL, RMSE, MAE, entropy, z‑score calibration, FFT peak)
• Embeds PNGs via <img> tags; can inline images as base64 (optional) for portability
• Renders Symbolic overlays (losses/influence) and SHAP overlays (if present)
• Pulls recent entries from `logs/v50_debug_log.md` into an “Audit & Traceability” section
• Typer CLI with options for custom title, base64 inlining, assets discovery, and auto‑open

Usage
-----
  python generate_html_report.py \
      --artifacts outputs/diagnostics \
      --title "SpectraMind V50 — Diagnostics" \
      --inline-images \
      --open-html

Outputs
-------
• Single HTML file: <artifacts>/report.html
• Appends an audit line into logs/v50_debug_log.md
"""

from __future__ import annotations

import base64
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

# ---------------------------
# Helpers
# ---------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _read_text(path: Path, max_bytes: int = 300_000) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    if len(data) > max_bytes:
        # Truncate large logs to keep report lean
        data = data[-max_bytes:]
        return "\n...\n" + data.decode("utf-8", errors="ignore")
    return data.decode("utf-8", errors="ignore")

def _find_first(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None

def _encode_image_base64(path: Path) -> Optional[str]:
    try:
        b = path.read_bytes()
        enc = base64.b64encode(b).decode("ascii")
        # very crude content type based on extension
        ext = path.suffix.lower()
        mime = "image/png" if ext in (".png", ".npy") else "image/jpeg" if ext in (".jpg", ".jpeg") else "image/svg+xml" if ext == ".svg" else "image/png"
        return f"data:{mime};base64,{enc}"
    except Exception:
        return None

def _img_tag(path: Optional[Path], inline: bool) -> str:
    if not path or not path.exists():
        return '<div class="muted small">Not available</div>'
    if inline:
        data = _encode_image_base64(path)
        if data:
            return f'<img class="card-img" alt="{path.name}" src="{data}"/>'
    # fall back to relative path for large images or when not inlining
    rel = path.as_posix()
    return f'<img class="card-img" alt="{path.name}" src="{rel}"/>'

def _fmt_metric(value: Any) -> str:
    try:
        v = float(value)
        return f"{v:.6f}"
    except Exception:
        return str(value)

def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )

def _kv_rows(d: Dict[str, Any]) -> str:
    rows = []
    for k, v in d.items():
        rows.append(f"<tr><td>{_escape(str(k))}</td><td>{_escape(_fmt_metric(v))}</td></tr>")
    return "\n".join(rows)

def _symbolic_table(symbolic: Dict[str, Any]) -> str:
    if not symbolic:
        return '<div class="muted small">No symbolic overlays present.</div>'
    html = []
    if "losses" in symbolic and symbolic["losses"]:
        html.append("<h4>Symbolic Losses</h4>")
        try:
            rows = _kv_rows(symbolic["losses"])
            html.append(f'<table class="kv"><thead><tr><th>Rule</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>')
        except Exception:
            html.append("<div class='muted small'>Unable to render losses table.</div>")
    if "influence" in symbolic and symbolic["influence"]:
        html.append("<h4>Symbolic Influence</h4>")
        try:
            rows = _kv_rows(symbolic["influence"])
            html.append(f'<table class="kv"><thead><tr><th>Rule</th><th>Influence</th></tr></thead><tbody>{rows}</tbody></table>')
        except Exception:
            html.append("<div class='muted small'>Unable to render influence table.</div>")
    return "\n".join(html) if html else '<div class="muted small">No symbolic overlays present.</div>'

def _shap_table(shap: Dict[str, Any]) -> str:
    if not shap:
        return '<div class="muted small">No SHAP overlays present.</div>'
    try:
        rows = _kv_rows(shap)
        return f'<table class="kv"><thead><tr><th>Feature</th><th>SHAP</th></tr></thead><tbody>{rows}</tbody></table>'
    except Exception:
        return "<div class='muted small'>Unable to render SHAP table.</div>"

def _recent_audit_lines(log_text: str, max_lines: int = 100) -> str:
    if not log_text.strip():
        return '<div class="muted small">No audit log found.</div>'
    lines = log_text.strip().splitlines()
    snippet = "\n".join(lines[-max_lines:])
    return f"<pre class='code'>{_escape(snippet)}</pre>"

# ---------------------------
# Report Builder
# ---------------------------

def build_html(
    title: str,
    summary: Dict[str, Any],
    artifacts_dir: Path,
    inline_images: bool,
    extra_images: List[Path],
    include_fft_png: Optional[Path],
    debug_log_text: str
) -> str:
    metrics = summary.get("metrics", {})
    symbolic = summary.get("symbolic", {})
    explain = summary.get("explainability", {})
    run_id = summary.get("run_id", "N/A")
    timestamp = summary.get("timestamp", dt.datetime.utcnow().isoformat())

    # Try to auto-find common images (UMAP/TSNE/SHAP plots) in artifacts
    # You can drop images into artifacts_dir and they get picked up if names match.
    candidates = [
        ("GLL Heatmap", artifacts_dir / "plots" / "gll_heatmap.png"),
        ("Calibration: σ vs Residual", artifacts_dir / "plots" / "sigma_vs_residual.png"),
        ("Z-score Histogram", artifacts_dir / "plots" / "zscore_hist.png"),
        ("Entropy per Planet", artifacts_dir / "plots" / "entropy_per_planet.png"),
        ("Calibration Scatter", artifacts_dir / "plots" / "calibration_scatter.png"),
    ]

    if include_fft_png and include_fft_png.exists():
        candidates.append(("FFT Power (mean μ)", include_fft_png))

    # Appendix images (UMAP/TSNE/SHAP etc.)
    appendix_imgs = []
    for p in extra_images:
        if p.exists():
            appendix_imgs.append(p)

    # CSS (inline) for a clean, readable layout
    css = """
:root{
  --bg:#0b1220; --fg:#e6edf3; --muted:#8b96a8; --card:#121a2a; --border:#22304a;
  --accent:#7aa2ff; --good:#10b981; --warn:#f59e0b; --bad:#ef4444;
}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial;}
.container{max-width:1200px;margin:0 auto;padding:24px;}
header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--border);padding-bottom:12px;margin-bottom:20px;}
h1{font-size:20px;margin:0;}
h2{font-size:18px;margin:16px 0 8px 0;}
h3{font-size:16px;margin:16px 0 8px 0;}
h4{font-size:14px;margin:14px 0 6px 0;}
.muted{color:var(--muted);}
.small{font-size:12px;}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin:12px 0;}
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px;}
.card h3{margin-top:0;}
.kv{width:100%;border-collapse:collapse;}
.kv th,.kv td{border-bottom:1px solid var(--border);padding:6px 8px;font-size:13px;text-align:left;}
.kv th{color:var(--muted);font-weight:600;}
.card-img{display:block;width:100%;height:auto;border:1px solid var(--border);border-radius:8px;background:#0c1526;}
.code{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;background:#0c1526;border:1px solid var(--border);border-radius:8px;padding:12px;color:#cbd5e1;}
.footer{margin-top:24px;border-top:1px solid var(--border);padding-top:12px;color:var(--muted);font-size:12px;}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid var(--border);background:#0e1a33;color:#a7bde8;font-size:12px;}
"""

    # Metrics table HTML
    metric_rows = []
    for k in ["gll", "rmse", "mae", "entropy_mean", "zscore_mean", "zscore_var", "fft_peak"]:
        if k in metrics:
            metric_rows.append(f"<tr><td>{k}</td><td>{_fmt_metric(metrics[k])}</td></tr>")
    metrics_table = (
        f'<table class="kv"><thead><tr><th>Metric</th><th>Value</th></tr></thead>'
        f'<tbody>{"".join(metric_rows) if metric_rows else "<tr><td colspan=2>No metrics found</td></tr>"}</tbody></table>'
    )

    # Candidates section (plots)
    plots_html = []
    for title_txt, path in candidates:
        plots_html.append(
            f"""
            <div class="card">
              <h3>{title_txt}</h3>
              {_img_tag(path, inline_images)}
            </div>
            """
        )

    # Symbolic overlays
    symbolic_html = _symbolic_table(symbolic)

    # SHAP overlays
    shap_html = _shap_table(explain.get("shap", {}))

    # Appendix images
    appendix_html = []
    if appendix_imgs:
        for p in appendix_imgs:
            appendix_html.append(
                f"""
                <div class="card">
                  <h3>{_escape(p.name)}</h3>
                  {_img_tag(p, inline_images)}
                </div>
                """
            )
    else:
        appendix_html.append('<div class="muted small">No additional images provided.</div>')

    # Audit log
    audit_html = _recent_audit_lines(debug_log_text, max_lines=120)

    # Final HTML
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{_escape(title)}</title>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<style>{css}</style>
</head>
<body>
  <div class="container">
    <header>
      <h1>{_escape(title)}</h1>
      <div class="small muted">Run: <span class="badge">{_escape(run_id)}</span> • Generated: { _escape(timestamp) }</div>
    </header>

    <section class="card">
      <h2>Executive Summary</h2>
      <div class="row">
        <div class="card">
          <h3>Key Metrics</h3>
          {metrics_table}
        </div>
        <div class="card">
          <h3>Notes</h3>
          <div class="small muted">
            This report aggregates SpectraMind V50 diagnostics from JSON/CSV artifacts. Use it alongside
            the interactive dashboard (if present) for deeper drill‑downs.
          </div>
          <div class="small">Artifacts root: <code>{_escape(artifacts_dir.as_posix())}</code></div>
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Diagnostic Plots</h2>
      <div class="row">
        {"".join(plots_html)}
      </div>
    </section>

    <section class="card">
      <h2>Symbolic Overlays</h2>
      {symbolic_html}
    </section>

    <section class="card">
      <h2>SHAP Overlays</h2>
      {shap_html}
    </section>

    <section class="card">
      <h2>Appendix: Additional Figures</h2>
      <div class="row">
        {"".join(appendix_html)}
      </div>
    </section>

    <section class="card">
      <h2>Audit & Traceability</h2>
      <div class="small muted">Recent entries from <code>logs/v50_debug_log.md</code> (tail):</div>
      {audit_html}
    </section>

    <div class="footer">
      SpectraMind V50 • Diagnostics HTML • {_escape(dt.datetime.utcnow().isoformat())}
    </div>
  </div>
</body>
</html>
"""
    return html

# ---------------------------
# CLI
# ---------------------------

@app.command()
def main(
    artifacts: Path = typer.Option(
        Path("outputs/diagnostics"),
        help="Directory containing diagnostic artifacts (JSON/CSV/plots)."
    ),
    title: str = typer.Option(
        "SpectraMind V50 — Diagnostics Report",
        help="Report title."
    ),
    inline_images: bool = typer.Option(
        False, "--inline-images/--no-inline-images",
        help="Inline images as base64 for single-file portability."
    ),
    fft_png: Optional[Path] = typer.Option(
        None,
        help="Optional explicit path to FFT PNG (if not auto-generated)."
    ),
    extra_image: List[Path] = typer.Option(
        None,
        help="Additional images to include in Appendix (repeatable)."
    ),
    out: Optional[Path] = typer.Option(
        None,
        help="Output HTML path (default: <artifacts>/report.html)"
    ),
    open_html: bool = typer.Option(
        False, "--open-html/--no-open-html",
        help="Open the generated report in a browser."
    ),
):
    """
    Build a single-file HTML report for SpectraMind V50 diagnostics.
    """
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    # Load main JSON
    summary_path = artifacts / "diagnostic_summary.json"
    summary = _read_json(summary_path)
    if not summary:
        console.print(f"[red]Missing or empty[/red] {summary_path.as_posix()} — continuing with partial render.")

    # Try to locate FFT plot if requested/available
    fft_plot = None
    if fft_png:
        fft_plot = fft_png.resolve()
    else:
        # common location
        candidate = artifacts / "plots" / "fft_power_example.png"
        if candidate.exists():
            fft_plot = candidate

    # Optional extra images
    extras: List[Path] = []
    if extra_image:
        for p in extra_image:
            try:
                pp = p.resolve()
                if pp.exists():
                    extras.append(pp)
            except Exception:
                continue

    # Pull audit log tail
    debug_text = _read_text(Path("logs/v50_debug_log.md"))

    html = build_html(
        title=title,
        summary=summary,
        artifacts_dir=artifacts,
        inline_images=inline_images,
        extra_images=extras,
        include_fft_png=fft_plot,
        debug_log_text=debug_text
    )

    # Write file
    out_path = out or (artifacts / "report.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    # Append audit line
    audit = Path("logs/v50_debug_log.md")
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("a", encoding="utf-8") as f:
        f.write(f"\n- [{dt.datetime.utcnow().isoformat()}] generate_html_report: out={out_path.as_posix()} title={title}\n")

    console.print(f"[green]HTML report written:[/green] {out_path.as_posix()}")

    if open_html:
        try:
            import webbrowser
            webbrowser.open(f"file://{out_path.resolve().as_posix()}")
        except Exception:
            console.print("[yellow]Could not open browser automatically.[/yellow]")

if __name__ == "__main__":
    app()
