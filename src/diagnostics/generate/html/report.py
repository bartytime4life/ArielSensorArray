# src/diagnostics/generate/html/report.py

# ==============================================================================

# 🖥️ SpectraMind V50 — Unified Diagnostics HTML Report Generator

# ------------------------------------------------------------------------------

# Responsibilities

# • Read diagnostics artifacts (diagnostic\_summary.json, plots, UMAP/t-SNE HTML).

# • Bundle referenced assets (PNG/HTML) into a versioned report directory.

# • Generate a self-contained HTML report (light CSS) with sections:

# - Overview & KPIs

# - GLL

# - FFT / Smoothness

# - Calibration

# - Symbolic

# - Explainability (UMAP/t-SNE, SHAP overlays)

# - CLI Log / Reproducibility

# • Emit a manifest (checksums, sizes) for reproducibility.

#

# Design notes

# • CLI-first (Typer) & Rich logging; no GUI code. Safe if some assets are missing.

# • Does NOT mutate analytics; it only reads produced artifacts and writes a report.

# • Minimal dependencies (stdlib + numpy/pandas optional for small features).

# ==============================================================================

from **future** import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional niceties

try:
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
except Exception:  # pragma: no cover
Console = None

console = Console(stderr=True, highlight=True) if Console else None

def log\_info(msg: str) -> None:
if console:
console.log(f"\[bold cyan]ℹ\[/] {msg}")
else:
print(f"\[INFO] {msg}", file=sys.stderr)

def log\_warn(msg: str) -> None:
if console:
console.log(f"\[bold yellow]⚠\[/] {msg}")
else:
print(f"\[WARN] {msg}", file=sys.stderr)

def log\_err(msg: str) -> None:
if console:
console.log(f"\[bold red]✖\[/] {msg}")
else:
print(f"\[ERROR] {msg}", file=sys.stderr)

# ------------------------------------------------------------------------------

# Paths & Config

# ------------------------------------------------------------------------------

DEFAULT\_GLOB\_PATTERNS = \[
\# Generic plots
"plots/\*\*/*.png",
"plots/*.png",
\# Calibration
"calibration/*.png",
"calibration/*.csv",
\# Symbolic
"symbolic/*.png",
"symbolic/*.json",
\# Diagnostics CSVs
"diagnostics/*.csv",
"diagnostics/*.json",
\# Smoothness
"smoothness/*.csv",
"smoothness/*.json",
"smoothness/\*.png",
]

EXPLAINABLE\_HTML = \[
"umap.html",
"tsne.html",
"shap\_overlay.html",
"shap\_attention.html",
"shap\_symbolic.html",
]

REPORT\_CSS = """
\:root{
\--bg:#0b1020; --panel:#121a2b; --text:#e9f1ff; --muted:#afbbd6; --accent:#5aa6ff;
\--ok:#21c07a; --warn:#e6c229; --bad:#ff5a78;
}
\*{box-sizing\:border-box}
html,body{margin:0;padding:0;background\:var(--bg);color\:var(--text);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial}
a{color\:var(--accent);text-decoration\:none} a\:hover
.container{max-width:1200px;margin:0 auto;padding:24px}
h1{font-size:28px;margin:8px 0 12px}
h2{font-size:22px;margin:28px 0 8px}
h3{font-size:18px;margin:20px 0 6px}
.panel{background\:var(--panel);border:1px solid #22314d;border-radius:12px;padding:16px;margin:16px 0}
.grid{display\:grid;grid-template-columns\:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.kpi{background:#0f1730;border:1px solid #22314d;border-radius:10px;padding:12px}
.kpi .label{color\:var(--muted);font-size:12px}
.kpi .value{font-size:20px;margin-top:4px}
.figure{display\:flex;flex-direction\:column;gap:6px;margin:10px 0}
.figure img{width:100%;height\:auto;border:1px solid #22314d;border-radius:8px;background:#0c1226}
.code{white-space\:pre-wrap;font-family\:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;font-size:12px;background:#0c1226;border:1px solid #1e2a44;border-radius:8px;padding:12px;color:#cfe3ff}
hr.sep{border:0;border-top:1px solid #22314d;margin:24px 0}
.small{font-size:12px;color\:var(--muted)}
.badge{display\:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid #2a3a5f;background:#0f1730;color:#b7c7ec}
table{width:100%;border-collapse\:collapse;margin:8px 0}
td,th{padding:8px;border:1px solid #22314d} th{text-align\:left;background:#0f1730}
.center{text-align\:center}
"""

REPORT\_JS = """
function toggle(elId){
const el = document.getElementById(elId);
if (!el) return;
el.style.display = (el.style.display==='none' ? 'block' : 'none');
}
"""

# ------------------------------------------------------------------------------

# Helpers

# ------------------------------------------------------------------------------

def now\_ts() -> str:
return time.strftime("%Y-%m-%dT%H-%M-%S")

def sha256\_file(path: Path) -> Optional\[str]:
if not path.exists() or not path.is\_file():
return None
h = hashlib.sha256()
with path.open("rb") as f:
for chunk in iter(lambda: f.read(1024 \* 1024), b""):
h.update(chunk)
return h.hexdigest()

def human\_size(num: int) -> str:
for unit in \["B","KB","MB","GB","TB"]:
if num < 1024:
return f"{num:.1f} {unit}"
num /= 1024.0
return f"{num:.1f} PB"

def copy\_asset(src: Path, dst\_dir: Path) -> Optional\[Path]:
if not src.exists():
return None
dst\_dir.mkdir(parents=True, exist\_ok=True)
dst = dst\_dir / src.name
try:
if src.resolve() == dst.resolve():
return dst
except Exception:
pass
shutil.copy2(src, dst)
return dst

def find\_first(parent: Path, names: List\[str]) -> Optional\[Path]:
for n in names:
p = parent / n
if p.exists():
return p
return None

def bump\_version\_name(out\_dir: Path, prefix: str = "report\_v", ext: str = ".html") -> Path:
"""
Auto-bump report filename like report\_v1.html, report\_v2.html, ...
"""
out\_dir.mkdir(parents=True, exist\_ok=True)
existing = list(out\_dir.glob(f"{prefix}\*{ext}"))
nums = \[]
for p in existing:
m = re.match(rf"^{re.escape(prefix)}(\d+){re.escape(ext)}\$", p.name)
if m:
nums.append(int(m.group(1)))
nxt = (max(nums) + 1) if nums else 1
return out\_dir / f"{prefix}{nxt}{ext}"

# ------------------------------------------------------------------------------

# Core

# ------------------------------------------------------------------------------

@dataclass
class ReportConfig:
artifacts\_dir: Path                       # directory with diagnostic artifacts
output\_dir: Path                          # where to write report + bundle
bundle\_dir: Path                          # directory name for copied assets (relative to output\_dir)
title: str = "SpectraMind V50 — Diagnostics Report"
subtitle: str = "NeurIPS 2025 Ariel Data Challenge"
versioned: bool = True
cli\_log\_path: Optional\[Path] = None       # e.g., v50\_debug\_log.md (optional)
glob\_patterns: Optional\[List\[str]] = None # extra asset globs

def load\_summary(artifacts\_dir: Path) -> Dict:
\# Try canonical diagnostic\_summary.json; else search
candidates = \[
artifacts\_dir / "diagnostic\_summary.json",
artifacts\_dir / "diagnostics" / "diagnostic\_summary.json",
artifacts\_dir / "summary" / "diagnostic\_summary.json",
]
for c in candidates:
if c.exists():
with c.open("r", encoding="utf-8") as f:
return json.load(f)
\# Last resort: first *summary*.json
for p in artifacts\_dir.rglob("\*.json"):
if "summary" in p.name:
with p.open("r", encoding="utf-8") as f:
return json.load(f)
raise FileNotFoundError("diagnostic\_summary.json not found under artifacts directory")

def collect\_assets(cfg: ReportConfig) -> Tuple\[List\[Path], List\[Path]]:
"""
Return (pngs, htmls) discovered under artifacts\_dir using default and user-provided globs.
"""
patterns = list(DEFAULT\_GLOB\_PATTERNS)
if cfg.glob\_patterns:
patterns.extend(cfg.glob\_patterns)
pngs: List\[Path] = \[]
htmls: List\[Path] = \[]

```
for pat in patterns:
    for p in cfg.artifacts_dir.glob(pat):
        if p.suffix.lower() == ".png":
            pngs.append(p)
        elif p.suffix.lower() == ".html":
            htmls.append(p)

# Also include known explainable HTMLs at root if present
for name in EXPLAINABLE_HTML:
    p = cfg.artifacts_dir / name
    if p.exists():
        htmls.append(p)

# De-duplicate while preserving order
def dedup(seq: List[Path]) -> List[Path]:
    seen = set()
    out = []
    for x in seq:
        k = x.resolve()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out

return dedup(pngs), dedup(htmls)
```

def bundle\_assets(pngs: List\[Path], htmls: List\[Path], bundle\_dir: Path) -> Dict\[str, Dict]:
"""
Copy assets into bundle\_dir and return a manifest with checksums & sizes.
"""
bundle\_dir.mkdir(parents=True, exist\_ok=True)
manifest: Dict\[str, Dict] = {"images": {}, "html": {}}

```
for p in pngs:
    dst = copy_asset(p, bundle_dir)
    if dst:
        stat = dst.stat()
        manifest["images"][dst.name] = {
            "src": dst.name,
            "bytes": stat.st_size,
            "size": human_size(stat.st_size),
            "sha256": sha256_file(dst),
        }
for p in htmls:
    dst = copy_asset(p, bundle_dir)
    if dst:
        stat = dst.stat()
        manifest["html"][dst.name] = {
            "src": dst.name,
            "bytes": stat.st_size,
            "size": human_size(stat.st_size),
            "sha256": sha256_file(dst),
        }
return manifest
```

def render\_kpi(label: str, value: Optional\[float | int | str]) -> str:
v = "—" if value is None else (f"{value:.6f}" if isinstance(value, (float, int)) else html.escape(str(value)))
return f""" <div class="kpi"> <div class="label">{html.escape(label)}</div> <div class="value">{v}</div> </div>
"""

def render\_img(bundle\_subdir: str, filename: str, caption: Optional\[str] = None) -> str:
cap = f'<div class="small">{html.escape(caption)}</div>' if caption else ""
src = f"{bundle\_subdir}/{filename}"
return f""" <div class="figure"> <img src="{src}" loading="lazy" alt="{html.escape(filename)}"/>
{cap} </div>
"""

def render\_iframe(bundle\_subdir: str, filename: str, title: str) -> str:
src = f"{bundle\_subdir}/{filename}"
safe\_title = html.escape(title)
return f""" <div class="figure"> <div class="small">{safe\_title}</div> <iframe src="{src}" style="width:100%;height:520px;border:1px solid #22314d;border-radius:8px;background:#0c1226"></iframe> </div>
"""

def build\_html(cfg: ReportConfig, summary: Dict, bundle\_manifest: Dict, out\_html: Path) -> str:
k = summary.get("metrics", {})
\# KPIs (robust to missing)
gll\_mean = k.get("gll", {}).get("mean")
fft\_pm = k.get("fft", {}).get("power\_mean\_avg")
ac\_max = k.get("fft", {}).get("autocorr\_max\_avg")
ent\_planet = k.get("entropy", {}).get("planet\_mean")
ent\_bin = k.get("entropy", {}).get("bin\_mean")
grad\_p95 = k.get("smoothness", {}).get("grad\_p95\_mean")
curv\_p95 = k.get("smoothness", {}).get("curv\_p95\_mean")
tv\_max = k.get("smoothness", {}).get("tv\_max\_mean")
flags = k.get("flags", {})

```
# Section asset picks (best-effort)
images = bundle_manifest.get("images", {})
htmls = bundle_manifest.get("html", {})
img_names = list(images.keys())
html_names = list(htmls.keys())

def first_match(regexes: List[str]) -> Optional[str]:
    for r in regexes:
        prog = re.compile(r, re.I)
        for n in img_names:
            if prog.search(n):
                return n
    return None

def first_html(regexes: List[str]) -> Optional[str]:
    for r in regexes:
        prog = re.compile(r, re.I)
        for n in html_names:
            if prog.search(n):
                return n
    return None

# Heuristic picks
gll_img = first_match([r"gll.*\.png"])
fft_img = first_match([r"fft.*power.*\.png", r"fft.*\.png"])
smooth_grad_img = first_match([r"gradient.*\.png", r"grad.*heatmap.*\.png"])
smooth_curv_img = first_match([r"curv.*\.png"])
smooth_tv_img = first_match([r"tv.*\.png"])
calib_img = first_match([r"calibration.*reliab.*\.png", r"reliab.*\.png"])
shap_img = first_match([r"shap.*symbolic.*\.png", r"shap.*overlay.*\.png"])
symbolic_img = first_match([r"violation.*\.png", r"symbolic.*\.png"])

umap_html = first_html([r"umap\.html"])
tsne_html = first_html([r"tsne\.html"])

# Build HTML
title = html.escape(cfg.title)
subtitle = html.escape(cfg.subtitle)
bundle_subdir = html.escape(cfg.bundle_dir.name)

kpis_html = "\n".join([
    render_kpi("GLL mean", gll_mean),
    render_kpi("FFT power mean (avg)", fft_pm),
    render_kpi("Autocorr max (avg)", ac_max),
    render_kpi("Entropy (planet mean)", ent_planet),
    render_kpi("Entropy (bin mean)", ent_bin),
    render_kpi("grad p95 (mean)", grad_p95),
    render_kpi("curv p95 (mean)", curv_p95),
    render_kpi("tv max (mean)", tv_max),
    render_kpi("Flags (n_grad)", flags.get("n_grad")),
    render_kpi("Flags (n_curv)", flags.get("n_curv")),
    render_kpi("Flags (n_tv)", flags.get("n_tv")),
])

def optional_section(header: str, body_html: str) -> str:
    if not body_html.strip():
        return ""
    return f"""
    <div class="panel">
      <h2>{html.escape(header)}</h2>
      {body_html}
    </div>
    """

# Compose sections
sec_overview = f"""
<div class="panel">
  <h1>{title}</h1>
  <div class="small">{subtitle}</div>
  <div class="grid">{kpis_html}</div>
  <hr class="sep"/>
  <div class="small">Generated: {time.strftime("%Y-%m-%d %H:%M:%S %Z")} • Report: {html.escape(out_html.name)}</div>
</div>
"""

sec_gll = optional_section(
    "GLL",
    (render_img(bundle_subdir, gll_img, "GLL heatmap")
     if gll_img else "<div class='small'>No GLL plots found.</div>")
)

sec_fft = optional_section(
    "FFT & Frequency Diagnostics",
    "\n".join(filter(None, [
        render_img(bundle_subdir, fft_img, "FFT power spectrum") if fft_img else "",
    ]))
)

sec_smooth = optional_section(
    "Smoothness (Grad/Curvature/TV)",
    "\n".join(filter(None, [
        render_img(bundle_subdir, smooth_grad_img, "Gradient heatmap") if smooth_grad_img else "",
        render_img(bundle_subdir, smooth_curv_img, "Curvature heatmap") if smooth_curv_img else "",
        render_img(bundle_subdir, smooth_tv_img, "Total Variation map") if smooth_tv_img else "",
    ]))
)

sec_calib = optional_section(
    "Calibration Checks",
    (render_img(bundle_subdir, calib_img, "Reliability / coverage")
     if calib_img else "<div class='small'>No calibration plots found.</div>")
)

sec_symbolic = optional_section(
    "Symbolic Diagnostics",
    (render_img(bundle_subdir, symbolic_img, "Symbolic rule violations / heatmap")
     if symbolic_img else "<div class='small'>No symbolic plots found.</div>")
)

sec_shap = optional_section(
    "Explainability (SHAP ovelays & fusion)",
    (render_img(bundle_subdir, shap_img, "SHAP overlays")
     if shap_img else "<div class='small'>No SHAP plots found.</div>")
)

sec_proj = optional_section(
    "Latent Projections",
    "\n".join(filter(None, [
        render_iframe(bundle_subdir, umap_html, "UMAP Projection") if umap_html else "",
        render_iframe(bundle_subdir, tsne_html, "t-SNE Projection") if tsne_html else "",
        "<div class='small'>No interactive projections found.</div>" if (not umap_html and not tsne_html) else "",
    ]))
)

# CLI log (optional)
cli_log_html = ""
if cfg.cli_log_path and cfg.cli_log_path.exists():
    try:
        txt = cfg.cli_log_path.read_text(encoding="utf-8")[-8000:]  # tail
        cli_log_html = f"<div class='code'>{html.escape(txt)}</div>"
    except Exception as e:
        cli_log_html = f"<div class='small'>Failed to read CLI log: {html.escape(str(e))}</div>"
sec_cli = optional_section("CLI Log (tail)", cli_log_html) if cli_log_html else ""

# Images list (index)
def images_table() -> str:
    if not images:
        return "<div class='small'>No image assets bundled.</div>"
    rows = "\n".join(
        f"<tr><td>{html.escape(n)}</td><td class='small'>{d['size']}</td><td class='small'>{d['sha256'] or ''}</td></tr>"
        for n, d in images.items()
    )
    return f"""
    <table>
      <thead><tr><th>Image</th><th>Size</th><th>SHA256</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """

sec_assets = optional_section("Bundled Assets", images_table())

# Final page
body = "\n".join([
    sec_overview,
    sec_gll,
    sec_fft,
    sec_smooth,
    sec_calib,
    sec_symbolic,
    sec_shap,
    sec_proj,
    sec_cli,
    sec_assets,
    "<div class='small'>© SpectraMind V50 — CLI-first, Hydra-safe, DVC-tracked diagnostics.</div>",
])

html_doc = f"""<!doctype html>
```

<html lang="en">
<head>
<meta charset="utf-8"/>
<meta http-equiv="x-ua-compatible" content="ie=edge"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>{REPORT_CSS}</style>
<script>{REPORT_JS}</script>
</head>
<body>
  <div class="container">
    {body}
  </div>
</body>
</html>
"""
    return html_doc

def write\_report(cfg: ReportConfig) -> Dict:
cfg.output\_dir.mkdir(parents=True, exist\_ok=True)
summary = load\_summary(cfg.artifacts\_dir)

```
# Choose report filename
if cfg.versioned:
    out_html = bump_version_name(cfg.output_dir, prefix="report_v", ext=".html")
else:
    out_html = cfg.output_dir / "report.html"

# Bundle assets
bundle_dir = cfg.output_dir / cfg.bundle_dir
pngs, htmls = collect_assets(cfg)
manifest = bundle_assets(pngs, htmls, bundle_dir)

# Save manifest.json
manifest_path = cfg.output_dir / "report_manifest.json"
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump({
        "generated_at": now_ts(),
        "artifacts_dir": str(cfg.artifacts_dir),
        "output_dir": str(cfg.output_dir),
        "bundle_dir": cfg.bundle_dir.name,
        "report_file": out_html.name,
        "assets": manifest,
    }, f, indent=2)

# Render HTML
html_doc = build_html(cfg, summary, manifest, out_html)
with out_html.open("w", encoding="utf-8") as f:
    f.write(html_doc)

# Provide small console summary
if console:
    tbl = Table(title="Report Bundle", show_header=True, header_style="bold magenta")
    tbl.add_column("File")
    tbl.add_column("Size", justify="right")
    for n, d in manifest.get("images", {}).items():
        tbl.add_row(n, d.get("size", ""))
    console.print(Panel(tbl, title=f"✓ Wrote {out_html.name}"))

return {
    "report": str(out_html),
    "manifest": str(manifest_path),
    "assets": manifest,
}
```

# ------------------------------------------------------------------------------

# CLI

# ------------------------------------------------------------------------------

def build\_argparser() -> argparse.ArgumentParser:
p = argparse.ArgumentParser(
prog="spectramind-generate-html-report",
description="Bundle diagnostics artifacts and generate a unified HTML report.",
formatter\_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add\_argument(
"--artifacts-dir",
type=Path,
required=True,
help="Path to diagnostics artifacts directory (should contain diagnostic\_summary.json and plots/ etc.).",
)
p.add\_argument(
"--output-dir",
type=Path,
default=Path("artifacts/reports"),
help="Where to write the report and its bundle.",
)
p.add\_argument(
"--bundle-name",
type=str,
default="report\_assets",
help="Subdirectory under output-dir where assets will be copied.",
)
p.add\_argument(
"--no-versioning",
action="store\_true",
help="Disable auto-bumped report\_vN.html naming; write report.html instead.",
)
p.add\_argument(
"--cli-log",
type=Path,
default=None,
help="Optional path to v50\_debug\_log.md to include a tail in the report.",
)
p.add\_argument(
"--glob",
type=str,
nargs="\*",
default=None,
help="Additional glob patterns for assets to include (relative to artifacts-dir).",
)
return p

def main(argv: Optional\[List\[str]] = None) -> int:
args = build\_argparser().parse\_args(argv)
try:
cfg = ReportConfig(
artifacts\_dir=args.artifacts\_dir,
output\_dir=args.output\_dir,
bundle\_dir=Path(args.bundle\_name),
versioned=not args.no\_versioning,
cli\_log\_path=args.cli\_log,
glob\_patterns=args.glob,
)
write\_report(cfg)
log\_info(f"Report generated in {cfg.output\_dir.resolve()}")
return 0
except Exception as e:
log\_err(f"Report generation failed: {e}")
return 1

if **name** == "**main**":
sys.exit(main())
