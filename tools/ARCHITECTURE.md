# 🏗️ ArielSensorArray — Tools Architecture

**SpectraMind V50** · *Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*

This document describes the **architectural role** of the `/tools` layer, showing how each utility connects to the core V50 pipeline — from calibration to μ/σ prediction, symbolic/SHAP overlays, latent analysis, diagnostics dashboards, and final Kaggle submission.

---

## 📚 Overview

The **/tools** layer is the **diagnostics and orchestration plane** for SpectraMind V50:

- **Calibration validation** — FFT/phase/coverage tools verify preprocessing integrity.
- **Prediction analysis** — SHAP, symbolic, and gradient overlays diagnose model behavior on spectra.
- **Latent exploration** — UMAP/t‑SNE utilities make encoders’ representations interpretable.
- **Fusion & clustering** — FFT PCA + symbolic fingerprints + molecular overlays expose scientific structure.
- **Submission assurance** — validators enforce Kaggle schema, hashes, and runtime constraints.

Every tool is:
- **CLI‑ready** (Typer/argparse with `--help`)
- **Hydra‑configurable** (reproducible parameters in YAML)
- **Reproducibility‑safe** (append‑only `logs/v50_debug_log.md`, data hashes, DVC tracking)

---

## 🔄 End‑to‑End Flow (Pipeline → Tools → Dashboard → Submission)

```mermaid
flowchart TD
    A[Raw Telescope Frames] --> B[Calibration Pipeline]
    B --> C[Model Predictions (μ, σ)]
    C --> D[/tools Diagnostics Layer]

    subgraph Tools
      T1[FFT & Phase\nanalyze_fft_autocorr_mu.py\ngenerate_fft_alignment.py\ngenerate_phase_alignment.py]
      T2[Calibration Checks\ncheck_calibration.py\ngenerate_coverage_heatmap.py\ngenerate_calibration_preview.py\ngenerate_lightcurve_preview.py]
      T3[SHAP & Gradients\nshap_overlay.py\nshap_attention_overlay.py\nshap_symbolic_overlay.py\nspectral_shap_gradient.py]
      T4[Symbolic Diagnostics\nsymbolic_influence_map.py\nsymbolic_violation_overlay.py\nsymbolic_rule_table.py]
      T5[Latent Visualizers\nplot_umap_v50.py\nplot_umap_fusion_latents_v50.py\nplot_tsne_interactive.py]
      T6[Fusion & Clusters\nfft_power_compare.py\nplot_fft_power_cluster_compare.py\nspectral_absorption_overlay_clustered.py\ngenerate_fft_symbolic_fusion.py]
      T7[Summaries & Packaging\ngenerate_diagnostic_summary.py\nreview_and_compile.sh]
    end

    D --> T1 & T2 & T3 & T4 & T5 & T6 & T7

    T7 --> R[generate_html_report.py → diagnostics_report.html]
    R --> V[validate_submission.py → Kaggle Leaderboard]
````

---

## 🗺️ “Diagnostics Constellation” (Family Map of Tools)

```mermaid
graph TB
  Core[μ/σ Prediction]:::core

  subgraph "Calibration & FFT"
    CF1[generate_calibration_preview.py]
    CF2[generate_lightcurve_preview.py]
    CF3[generate_phase_alignment.py]
    CF4[generate_fft_alignment.py]
    CF5[analyze_fft_autocorr_mu.py]
    CF6[fft_power_compare.py]
    CF7[plot_fft_power_cluster_compare.py]
    CF8[check_calibration.py]
    CF9[generate_coverage_heatmap.py]
  end

  subgraph "SHAP & Attention"
    SA1[shap_overlay.py]
    SA2[shap_attention_overlay.py]
    SA3[shap_symbolic_overlay.py]
    SA4[spectral_shap_gradient.py]
  end

  subgraph "Symbolic & Rules"
    SY1[symbolic_violation_overlay.py]
    SY2[symbolic_influence_map.py]
    SY3[symbolic_rule_table.py]
    SY4[generate_fft_symbolic_fusion.py]
  end

  subgraph "Latents & Clusters"
    LC1[plot_umap_v50.py]
    LC2[plot_umap_fusion_latents_v50.py]
    LC3[plot_tsne_interactive.py]
    LC4[spectral_absorption_overlay_clustered.py]
  end

  subgraph "Summaries, Reports & Submission"
    PS1[generate_diagnostic_summary.py]
    PS2[generate_html_report.py]
    PS3[validate_submission.py]
    PS4[review_and_compile.sh]
  end

  Core --- CF1 & CF2 & CF3 & CF4 & CF5 & CF6 & CF7 & CF8 & CF9
  Core --- SA1 & SA2 & SA3 & SA4
  Core --- SY1 & SY2 & SY3 & SY4
  Core --- LC1 & LC2 & LC3 & LC4

  CF5 & SA1 & SY1 & LC1 --> PS1
  PS1 --> PS2 --> PS3

  classDef core fill:#0b5fff,stroke:#0b5fff,color:#fff,font-weight:700;
```

---

## 🧩 Architectural Grouping (Responsibilities & Typical Inputs/Outputs)

### 1) Calibration & Previews

* **Purpose:** sanity-check the calibration pipeline, visualize pre/post behavior, verify phase handling.
* **Scripts:** `generate_calibration_preview.py`, `generate_lightcurve_preview.py`, `generate_phase_alignment.py`, `generate_fft_alignment.py`, `check_calibration.py`, `generate_coverage_heatmap.py`.
* **I/O:** reads calibrated frames / summary CSVs; writes PNGs/CSVs for coverage, phase plots.

### 2) FFT & Autocorrelation

* **Purpose:** frequency‑domain checks for instrument/systematic bands; verify suppression after calibration.
* **Scripts:** `analyze_fft_autocorr_mu.py`, `fft_power_compare.py`, `plot_fft_power_cluster_compare.py`, `generate_fft_symbolic_fusion.py`.
* **I/O:** reads μ spectra; writes FFT spectra, Welch plots, fusion CSVs/HTML.

### 3) SHAP & Gradient Explainability

* **Purpose:** explain per‑bin influences, fuse with attention and symbolic overlays, inspect ∂μ/∂input.
* **Scripts:** `shap_overlay.py`, `shap_attention_overlay.py`, `shap_symbolic_overlay.py`, `spectral_shap_gradient.py`.
* **I/O:** reads predictions + features; writes overlay PNG/HTML, JSON summaries.

### 4) Symbolic Diagnostics

* **Purpose:** quantify/visualize rule violations, compute ∂L/∂μ symbolic influence, render rule leaderboards.
* **Scripts:** `symbolic_violation_overlay.py`, `symbolic_influence_map.py`, `symbolic_rule_table.py`.
* **I/O:** reads μ, rules/config; writes violation maps, influence plots, HTML/CSV tables.

### 5) Latent Visualizers

* **Purpose:** project encoder latents (UMAP/t‑SNE), color by symbolic/cluster labels, add planet hyperlinks.
* **Scripts:** `plot_umap_v50.py`, `plot_umap_fusion_latents_v50.py`, `plot_tsne_interactive.py`.
* **I/O:** reads latent arrays + labels; writes interactive HTML and static PNGs.

### 6) Spectral Clustering & Fusion

* **Purpose:** cluster spectra by molecular fingerprints; compare FFT power by cluster; fuse FFT PCA + symbolic.
* **Scripts:** `spectral_absorption_overlay_clustered.py`, `plot_fft_power_cluster_compare.py`, `generate_fft_symbolic_fusion.py`.
* **I/O:** reads μ/metadata; writes cluster CSVs, overlays, UMAP/HTML.

### 7) Summaries, Reports & Submission

* **Purpose:** aggregate diagnostics, build the HTML dashboard, validate submission, batch report compilation.
* **Scripts:** `generate_diagnostic_summary.py`, `generate_html_report.py`, `validate_submission.py`, `review_and_compile.sh`.
* **I/O:** unifies outputs into `diagnostics_report.html`; validates leaderboard bundle.

---

## 🧭 Operating Principles

* **CLI‑first orchestration** — invoke via `spectramind diagnose ...` (or direct tool entrypoints).
* **Hydra configs everywhere** — no magic constants; parameters live in YAML groups.
* **Reproducible artifacts** — DVC/lakeFS for data & model versions; append‑only logbook.
* **CI hooks** — smoke runs ensure every tool stays green before merge.
* **Dashboard as the single pane of glass** — all roads lead to `generate_html_report.py`.

---

## 📊 File Count

* **36 active CLI tools**
* **+2 docs:** `README.md`, `ARCHITECTURE.md`
* **Total in `/tools`: 38 files**

---

## 📝 Conventions & Outputs

* **Inputs:** `outputs/` predictions, latents, rule JSON, config snapshots, calibration products
* **Outputs:** PNG/HTML/CSV/JSON under `outputs/diagnostics/<run_id>/...`
* **Logging:** `logs/v50_debug_log.md` (append‑only), plus optional JSONL events

---

## 🔮 Roadmap Hints

* Symbolic × attention **trace matrix** in dashboard.
* Instance‑level **uncertainty calibration** overlays alongside COREL.
* **Cross‑link** UMAP/t‑SNE points to per‑planet diagnostic mini‑pages.

---

```
```
