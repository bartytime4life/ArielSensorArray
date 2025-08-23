# 🛠️ ArielSensorArray — Tools Directory

**SpectraMind V50** • *Diagnostics, Calibration, Symbolic Overlays, and CLI Utilities*
Part of the **NeurIPS 2025 Ariel Data Challenge** pipeline — transforming raw **FGS1/AIRS frames** → calibrated spectra → **μ/σ predictions** → symbolic/diagnostic overlays → leaderboard-ready submissions.

---

## 📂 Contents

| Script                                     | Purpose                                                                                            |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `fft_power_compare.py`                     | Compare frequency-domain power spectra (pre/post calibration; across models).                      |
| `check_calibration.py`                     | Evaluate σ calibration: coverage, z-scores, quantile validation, per-bin diagnostics.              |
| `generate_diagnostic_summary.py`           | Aggregates GLL, entropy, RMSE, symbolic overlays, FFT, ∂L/∂μ, SHAP into `diagnostic_summary.json`. |
| `generate_html_report.py`                  | Produces interactive **diagnostics dashboard** (UMAP, t-SNE, symbolic violations, SHAP overlays).  |
| `plot_umap_v50.py`                         | Latent embedding visualization (2D/3D), with symbolic overlays & confidence shading.               |
| `plot_tsne_interactive.py`                 | Interactive t-SNE projection (Plotly) with symbolic & SHAP overlays.                               |
| `shap_overlay.py`                          | SHAP × μ overlays: visualize bin attribution, entropy scoring, export JSON/PNG.                    |
| `shap_attention_overlay.py`                | Fuse SHAP attributions with decoder attention weights.                                             |
| `shap_symbolic_overlay.py`                 | SHAP × symbolic violation overlays for interpretability.                                           |
| `spectral_smoothness_map.py`               | L2/entropy smoothness map across μ spectra, with symbolic penalties.                               |
| `spectral_shap_gradient.py`                | ∂μ/∂input, ∂σ/∂input, ∂GLL/∂input — gradient-based SHAP diagnostics.                               |
| `simulate_lightcurve_from_mu.py`           | Generate synthetic AIRS/FGS1 time series from target μ spectra.                                    |
| `symbolic_influence_map.py`                | ∂L/∂μ symbolic influence maps (per-rule, per-planet).                                              |
| `symbolic_violation_overlay.py`            | Plot symbolic rule violations and overlay them on spectra.                                         |
| `spectral_absorption_overlay_clustered.py` | Clustered spectral overlays aligned with molecular fingerprints.                                   |
| `auto_ablate_v50.py`                       | Config ablation engine: mutates Hydra configs, runs diagnostics, exports leaderboard.              |
| `config_grid_launcher.py`                  | Launch grid of configs with run-hash logging & diagnostics sync.                                   |
| `validate_submission.py`                   | Validate Kaggle submission bundles (CSV schema, hashes, selftest).                                 |
| `generate_dummy_data.py`                   | Create challenge-compatible dummy datasets for testing pipelines.                                  |

---

## 🚀 Usage

All tools are CLI-driven, Hydra-compatible, and audited via `spectramind.py` root CLI.
You can run them either standalone or via the **diagnose/submit** subcommands.

### Example: FFT Power Compare

```bash
spectramind diagnose fft-power \
  --mu outputs/predictions/mu.npy \
  --ref outputs/calibration/fgs1_ref.npy \
  --outdir outputs/fft_power
```

### Example: Calibration Check

```bash
spectramind diagnose check-calibration \
  --mu outputs/predictions/mu.npy \
  --sigma outputs/predictions/sigma.npy \
  --y_true data/labels/validation.npy \
  --outdir outputs/calibration_check
```

### Example: Generate HTML Dashboard

```bash
spectramind diagnose dashboard \
  --summary outputs/diagnostic_summary.json \
  --html-out diagnostics_report_v3.html --open
```

---

## 🔗 Integration

* **Models (`src/models/`)** → produce μ/σ, latents.
* **Symbolic (`src/symbolic/`)** → logic engine, priors, violation predictors.
* **Utils (`src/utils/`)** → logging, reproducibility, selftests.
* **Tools (`tools/`)** → diagnostics, overlays, dashboards.
* **CLI (`spectramind.py`)** → unified entrypoint; calls into tools via subcommands.

See the architecture map in the main repo for full flow.

---

## 📊 Outputs

* `diagnostic_summary.json` — metrics, overlays, symbolic scores.
* `dashboard_vX.html` — interactive diagnostics report (UMAP, SHAP, symbolic violations).
* `fft_power/*.png` — frequency-domain plots.
* `submission.csv` — Kaggle-ready predictions.

---

## ✅ Standards

* Hydra configs for reproducibility
* DVC for data versioning
* CI smoke tests for pipeline integrity
* Audit logs: `logs/v50_debug_log.md`
* Kaggle leaderboard compatibility

---
