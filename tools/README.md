# 🛠️ ArielSensorArray — Tools Directory

**SpectraMind V50** · *Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*

This directory hosts **diagnostics, calibration, visualization, and ablation tools** that extend the core `src/` pipeline. Each tool is:

* **CLI-ready** (Typer/argparse with `--help`)
* **Hydra-compatible** (accepts config-driven parameters)
* **Reproducibility-safe** (logs to `logs/v50_debug_log.md` and syncs hashes)

---

## 📂 File Index (35 tools)

| Script                                     | Purpose                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------- |
| `analyze_fft_autocorr_mu.py`               | FFT + autocorrelation diagnostics on μ spectra with symbolic overlays.             |
| `auto_ablate_v50.py`                       | Symbolic-aware ablation: runs multiple configs, logs metrics, exports leaderboard. |
| `check_calibration.py`                     | Evaluates σ calibration (coverage, z-scores, quantiles, COREL).                    |
| `config_grid_launcher.py`                  | Launch Hydra grid sweeps with reproducibility logging & parallel jobs.             |
| `explain_shap_metadata_v50.py`             | SHAP + metadata + symbolic fusion explainer.                                       |
| `fft_power_compare.py`                     | Compare FFT/Welch power between pre/post calibration (FGS1 & AIRS).                |
| `generate_calibration_preview.py`          | Quicklook plots of bias/dark/flat calibration frames.                              |
| `generate_coverage_heatmap.py`             | Coverage heatmap of uncertainty calibration across bins.                           |
| `generate_diagnostic_summary.py`           | Aggregates per-planet diagnostics (GLL, entropy, SHAP, symbolic).                  |
| `generate_dummy_data.py`                   | Produce synthetic Ariel-like test data for debugging.                              |
| `generate_fft_alignment.py`                | Verify calibration via FFT alignment across lightcurves.                           |
| `generate_fft_autocorr.py`                 | Auto-generate FFT + autocorrelation diagnostic report.                             |
| `generate_fft_symbolic_fusion.py`          | Fuse FFT PCA with symbolic fingerprints for clustering.                            |
| `generate_html_report.py`                  | Full interactive diagnostics dashboard (UMAP, SHAP, symbolic, logs).               |
| `generate_lightcurve_preview.py`           | Quick visualization of raw vs calibrated lightcurves.                              |
| `generate_phase_alignment.py`              | Phase-fold and align lightcurves for inspection.                                   |
| `gll_error_localizer.py`                   | Localize bin-wise GLL errors for spectra.                                          |
| `plot_benchmarks.py`                       | Benchmark plotting utility (compare runs/models).                                  |
| `plot_fft_power_cluster_compare.py`        | Compare FFT power across clusters of planets.                                      |
| `plot_gll_heatmap_per_bin.py`              | Heatmap of GLL scores per wavelength bin.                                          |
| `plot_tsne_interactive.py`                 | Interactive latent-space t-SNE visualization.                                      |
| `plot_umap_fusion_latents_v50.py`          | UMAP visualization with symbolic overlays & fusion labels.                         |
| `plot_umap_v50.py`                         | UMAP projection of latent encodings with symbolic coloring.                        |
| `review_and_compile.sh`                    | Meta-script to review results and compile reports.                                 |
| `shap_attention_overlay.py`                | Overlay SHAP attributions with decoder attention weights.                          |
| `shap_overlay.py`                          | SHAP × μ overlays; supports entropy & anomaly scoring.                             |
| `shap_symbolic_overlay.py`                 | SHAP × symbolic fusion overlays for interpretability.                              |
| `simulate_lightcurve_from_mu.py`           | Generate synthetic lightcurves from μ predictions.                                 |
| `spectral_absorption_overlay_clustered.py` | Cluster spectral overlays by molecular absorption bands.                           |
| `spectral_shap_gradient.py`                | Gradient-based SHAP diagnostics (∂μ/∂input, ∂σ/∂input).                            |
| `spectral_smoothness_map.py`               | Visualize spectral smoothness, violations, symbolic penalties.                     |
| `symbolic_influence_map.py`                | ∂L/∂μ symbolic influence maps across rules/planets.                                |
| `symbolic_rule_table.py`                   | Generate HTML/CSV symbolic rule leaderboard.                                       |
| `symbolic_violation_overlay.py`            | Overlay symbolic violations on predicted spectra.                                  |
| `validate_submission.py`                   | Kaggle submission validator (CSV schema, hashes, runtime tests).                   |

---

## 🔗 Workflow Integration

```
Raw Frames → Calibration → μ/σ Prediction
          ↘ tools/diagnostics ↙
          generate_diagnostic_summary.py → generate_html_report.py → dashboard.html
```

---

## 📊 Architecture Diagram

```mermaid
flowchart TD
    A[Raw Frames] --> B[Calibration]
    B --> C[Prediction (μ, σ)]
    C --> D[Tools/Diagnostics]

    subgraph Tools
      D1[FFT: fft_power_compare.py]
      D2[Calibration: check_calibration.py]
      D3[SHAP: shap_overlay.py]
      D4[Symbolic: symbolic_influence_map.py]
      D5[Latents: plot_umap_v50.py, plot_tsne_interactive.py]
      D6[Clusters: spectral_absorption_overlay_clustered.py]
    end

    D --> D1
    D --> D2
    D --> D3
    D --> D4
    D --> D5
    D --> D6

    D --> E[generate_diagnostic_summary.py]
    E --> F[generate_html_report.py]
    F --> G[Dashboard.html]

    G --> H[validate_submission.py]
    H --> I[Kaggle Leaderboard]
```

---

## ✅ Standards

* **Hydra configs** for reproducibility
* **DVC + lakeFS** for dataset versioning
* **CI smoke tests** validate all tools
* **Logs**: append-only `logs/v50_debug_log.md`
* **Submission compatibility**: Kaggle CSV schema

---
