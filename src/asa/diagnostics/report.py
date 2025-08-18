from __future__ import annotations

import csv
from pathlib import Path

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
  <div class="sub">Generated: {{ generated }}</div>

  <div class="section">
    <h2>Summary</h2>
    <table>
      <tr><th>Samples (N)</th><th>Bins</th><th>μ mean</th><th>μ std</th><th>σ mean</th><th>σ std</th><th>NaNs</th></tr>
      <tr>
        <td>{{ N }}</td>
        <td>{{ B }}</td>
        <td>{{ mu_mean|round(6) }}</td>
        <td>{{ mu_std|round(6) }}</td>
        <td>{{ sg_mean|round(6) }}</td>
        <td>{{ sg_std|round(6) }}</td>
        <td class="{{ 'bad' if nan_count>0 else 'ok' }}">{{ nan_count }}</td>
      </tr>
    </table>
  </div>

  <div class="section">
    <h2>Header Check (submission.csv)</h2>
    <div class="mono">{{ header_line }}</div>
    <div class="{{ 'ok' if header_ok else 'bad' }}">{{ 'OK' if header_ok else 'Mismatch' }}</div>
  </div>

  <div class="section">
    <h2>Notes</h2>
    <ul>
      <li>σ shown above is {{ 'calibrated' if calibrated else 'raw' }}.</li>
      <li>{{ note }}</li>
    </ul>
  </div>
</body>
</html>
"""
)


def _load_submission(path: str) -> tuple[str, int]:
    p = Path(path)
    if not p.exists():
        return "", 0
    with p.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return "", 0
    return ",".join(header), len(header)


def _safe_stats(x: torch.Tensor) -> tuple[float, float, int]:
    nan_count = torch.isnan(x).sum().item()
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return float(x.mean().item()), float(x.std().item()), int(nan_count)


def build_report(
    submission_csv: str = "outputs/submission.csv",
    preds_pt_primary: str = "outputs/preds_calibrated.pt",
    preds_pt_fallback: str = "outputs/preds.pt",
    out_html: str = "outputs/report.html",
) -> str:
    # Load preds (prefer calibrated if present)
    preds_path = (
        Path(preds_pt_primary) if Path(preds_pt_primary).exists() else Path(preds_pt_fallback)
    )
    calibrated = preds_path.name.endswith("preds_calibrated.pt")
    if not preds_path.exists():
        raise FileNotFoundError(
            "No predictions file found. Expected outputs/preds.pt or preds_calibrated.pt"
        )

    preds: dict[str, torch.Tensor] = torch.load(preds_path, map_location="cpu")
    mu: torch.Tensor = preds["mu"].detach().cpu()  # (N, B)
    sg: torch.Tensor = preds["sigma"].detach().cpu()  # (N, B)

    # Stats
    mu_mean, mu_std, mu_nan = _safe_stats(mu)
    sg_mean, sg_std, sg_nan = _safe_stats(sg)
    nan_count = mu_nan + sg_nan
    N, B = mu.shape

    header_line, width = _load_submission(submission_csv)
    header_ok = width == B + 1 and header_line.startswith("planet_id,")

    html = HTML_TMPL.render(
        generated=torch.__version__,  # quick breadcrumb
        N=N,
        B=B,
        mu_mean=mu_mean,
        mu_std=mu_std,
        sg_mean=sg_mean,
        sg_std=sg_std,
        nan_count=nan_count,
        header_line=header_line,
        header_ok=header_ok,
        calibrated=calibrated,
        note="Add real targets to enable coverage diagnostics and more advanced charts.",
    )
    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    path = build_report()
    print(f"[report] wrote {path}")
