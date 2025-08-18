#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as _dt
from pathlib import Path
from typing import Tuple

import torch
from jinja2 import Template


HTML_TMPL = Template(
    r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ArielSensorArray — Diagnostics Report</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    h1 { margin-bottom: 0; }
    .sub { color: #666; margin-top: 4px; }
    table { border-collapse: collapse; margin-top: 16px; }
    th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: right; }
    th { background: #f5f5f5; }
    .ok { color: #0a0; }
    .warn { color: #c60; }
    .bad { color: #c00; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .section { margin-top: 24px; }
  </style>
</head>
<body>
  <h1>ArielSensorArray — Diagnostics</h1>
  <div class="sub">Generated: {{ generated_utc }} • Torch: {{ torch_ver }} • Source: <span class="mono">{{ preds_src }}</span></div>

  <div class="section">
    <h2>Summary</h2>
    <table>
      <tr>
        <th>Samples (N)</th><th>Bins</th>
        <th>μ mean</th><th>μ std</th><th>σ mean</th><th>σ std</th>
        <th>Filtered NaN/Inf</th>
      </tr>
      <tr>
        <td>{{ N }}</td>
        <td>{{ B }}</td>
        <td>{{ mu_mean|round(6) }}</td>
        <td>{{ mu_std|round(6) }}</td>
        <td>{{ sg_mean|round(6) }}</td>
        <td>{{ sg_std|round(6) }}</td>
        <td class="{{ 'bad' if filtered_count>0 else 'ok' }}">{{ filtered_count }}</td>
      </tr>
    </table>
  </div>

  <div class="section">
    <h2>Header Check (submission.csv)</h2>
    <div class="mono">{{ header_line if header_line else '(no file)' }}</div>
    <div class="{{ 'ok' if header_ok else 'bad' }}">{{ 'OK' if header_ok else 'Mismatch' }}</div>
  </div>

  <div class="section">
    <h2>Notes</h2>
    <ul>
      <li>σ shown above is {{ 'calibrated' if calibrated else 'raw' }} (source file: {{ preds_src_name }}).</li>
      <li>{{ note }}</li>
    </ul>
  </div>
</body>
</html>
"""
)


def _load_submission_header(path: Path) -> tuple[str, int]:
    if not path.exists():
        return "", 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return "", 0
    return ",".join(header), len(header)


def _finite_stats(x: torch.Tensor) -> Tuple[float, float, int]:
    """
    Return (mean, std, filtered_count) where NaN/Inf are *ignored* (filtered),
    not converted to zero. This prevents skewing summary stats.
    """
    is_finite = torch.isfinite(x)
    filtered = (~is_finite).sum().item()
    if filtered > 0:
        x = x[is_finite]
    if x.numel() == 0:
        return 0.0, 0.0, filtered
    return float(x.mean().item()), float(x.std().item()), filtered


def _load_preds(first: Path, fallback: Path) -> tuple[dict, Path, bool]:
    """
    Load a torch .pt dict containing 'mu' and 'sigma'. Prefer `first` if exists,
    else fall back to `fallback`. Returns (payload, used_path, calibrated_flag).
    """
    used = first if first.exists() else fallback
    if not used.exists():
        raise FileNotFoundError(
            f"No predictions file found. Expected {first} or {fallback}"
        )
    payload = torch.load(used, map_location="cpu")
    if not isinstance(payload, dict) or "mu" not in payload or "sigma" not in payload:
        raise ValueError(f"Bad preds payload in {used}: expected dict with 'mu' and 'sigma'")
    calibrated = used.name.endswith("preds_calibrated.pt")
    return payload, used, calibrated


def _as_2d_float(t: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} is not a torch.Tensor")
    if t.ndim != 2:
        raise ValueError(f"{name} must be 2D (N,B), got shape {tuple(t.shape)}")
    return t.detach().to(dtype=torch.float32, device="cpu")


def build_report(
    submission_csv: str = "outputs/submission.csv",
    preds_pt_primary: str = "outputs/preds_calibrated.pt",
    preds_pt_fallback: str = "outputs/preds.pt",
    out_html: str = "outputs/diagnostics/report.html",
    expected_bins: int | None = None,
) -> str:
    """
    Build diagnostics HTML and return the output path as str.
    """
    # Load predictions (prefer calibrated)
    payload, src_path, calibrated = _load_preds(Path(preds_pt_primary), Path(preds_pt_fallback))
    mu = _as_2d_float(payload["mu"], "mu")
    sg = _as_2d_float(payload["sigma"], "sigma")

    # Shape checks
    if mu.shape != sg.shape:
        raise ValueError(f"mu and sigma shapes differ: {tuple(mu.shape)} vs {tuple(sg.shape)}")
    N, B = mu.shape
    if expected_bins is not None and B != expected_bins:
        raise ValueError(f"Bin width mismatch: got {B}, expected {expected_bins}")

    # Stats (finite-only)
    mu_mean, mu_std, mu_filtered = _finite_stats(mu)
    sg_mean, sg_std, sg_filtered = _finite_stats(sg)
    filtered_count = mu_filtered + sg_filtered

    # Submission header check
    header_line, width = _load_submission_header(Path(submission_csv))
    header_ok = (width == B + 1) and header_line.startswith("planet_id,")

    # Render HTML
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    html = HTML_TMPL.render(
        generated_utc=_dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        torch_ver=torch.__version__,
        preds_src=str(src_path),
        preds_src_name=src_path.name,
        N=N,
        B=B,
        mu_mean=mu_mean,
        mu_std=mu_std,
        sg_mean=sg_mean,
        sg_std=sg_std,
        filtered_count=filtered_count,
        header_line=header_line,
        header_ok=header_ok,
        calibrated=calibrated,
        note="Add ground-truth to enable coverage and residual diagnostics.",
    )
    out.write_text(html, encoding="utf-8")
    return str(out)


# ---- tiny CLI so you can call it directly ---------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a diagnostics HTML report from preds and submission CSV.")
    ap.add_argument("--csv", default="outputs/submission.csv", help="submission.csv path")
    ap.add_argument("--preds", default="outputs/preds_calibrated.pt", help="primary preds path")
    ap.add_argument("--preds-fallback", default="outputs/preds.pt", help="fallback preds path")
    ap.add_argument("--out", default="outputs/diagnostics/report.html", help="output HTML path")
    ap.add_argument("--bins", type=int, default=None, help="expected number of bins (optional)")
    return ap.parse_args()


def _main_cli() -> None:
    a = _parse_args()
    path = build_report(
        submission_csv=a.csv,
        preds_pt_primary=a.preds,
        preds_pt_fallback=a.preds_fallback,
        out_html=a.out,
        expected_bins=a.bins,
    )
    print(f"[report] wrote {path}")


if __name__ == "__main__":
    _main_cli()