# 🏗️ Tests — Diagnostics Architecture

**SpectraMind V50** · *Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge*:contentReference[oaicite:0]{index=0}

The `/tests/diagnostics` directory enforces **scientific correctness and consistency** for all  
diagnostic modules. Unlike `/tests/artifacts` (which protects reproducibility), these tests ensure  
that the **diagnostic engines themselves** (FFT, SHAP, symbolic overlays, calibration checks)  
produce **accurate, interpretable, and stable results**.

---

## 📂 Components

| Test Script                              | Diagnostic Under Test                     | Purpose / Guarantee                                                    |
| ---------------------------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| `test_generate_diagnostic_summary.py`    | `generate_diagnostic_summary.py`           | Verifies full pipeline metrics: GLL, RMSE, entropy, symbolic overlays. |
| `test_generate_html_report.py`           | `generate_html_report.py`                  | Confirms HTML dashboard renders plots/iframes without missing artifacts. |
| `test_plot_umap_v50.py`                  | `plot_umap_v50.py`                         | Ensures UMAP projections generate interactive + static plots correctly. |
| `test_plot_tsne_interactive.py`          | `plot_tsne_interactive.py`                 | Validates t-SNE projections + interactive HTML export.                 |
| `test_fft_power_compare.py`              | `fft_power_compare.py`                     | Checks FFT cluster comparisons + symbolic overlays.                    |
| `test_analyze_fft_autocorr_mu.py`        | `analyze_fft_autocorr_mu.py`               | Validates FFT + autocorrelation analysis for μ spectra.                |
| `test_spectral_smoothness_map.py`        | `spectral_smoothness_map.py`               | Confirms smoothness metric maps produce consistent results.            |
| `test_spectral_shap_gradient.py`         | `spectral_shap_gradient.py`                | Validates ∂μ/∂input + SHAP × gradient overlay diagnostics.             |
| `test_symbolic_violation_overlay.py`     | `symbolic_violation_overlay.py`             | Ensures symbolic rule violation overlays match symbolic engine outputs. |
| `test_symbolic_influence_map.py`         | `symbolic_influence_map.py`                | Confirms per-rule ∂L/∂μ influence maps align with symbolic loss.       |
| `test_neural_logic_graph.py`             | `neural_logic_graph.py`                    | Validates symbolic logic graph rendering + dashboard embedding.        |
| `test_shap_overlay.py`                   | `shap_overlay.py`                          | Ensures SHAP × μ spectrum overlays render + export metadata.           |
| `test_shap_attention_overlay.py`         | `shap_attention_overlay.py`                | Confirms SHAP × attention fusion visualizations.                       |
| `test_shap_symbolic_overlay.py`          | `shap_symbolic_overlay.py`                 | Verifies SHAP × symbolic overlays produce consistent diagnostic JSON.  |
| `test_simulate_lightcurve_from_mu.py`    | `simulate_lightcurve_from_mu.py`           | Ensures synthetic lightcurves match μ spectra & metadata.              |

---

## 🔄 Data Flow

```mermaid
flowchart TD
    subgraph Predictions[Model Outputs]
        MU[μ Spectra]
        SIG[σ (Uncertainty)]
    end

    subgraph Diagnostics[Diagnostics Tools]
        SUMM[Diagnostic Summary]
        DASH[HTML Dashboard]
        UMAP[UMAP Projection]
        TSNE[t-SNE Projection]
        FFT[FFT/Autocorr]
        SHAP[SHAP Overlays]
        SYMB[Symbolic Overlays]
        SMOOTH[Smoothness Map]
        INFL[Symbolic Influence Map]
        NLG[Neural Logic Graph]
    end

    subgraph Tests[/tests/diagnostics]
        T1[test_generate_diagnostic_summary]
        T2[test_generate_html_report]
        T3[test_plot_umap_v50]
        T4[test_plot_tsne_interactive]
        T5[test_fft_power_compare]
        T6[test_analyze_fft_autocorr_mu]
        T7[test_spectral_smoothness_map]
        T8[test_spectral_shap_gradient]
        T9[test_symbolic_violation_overlay]
        T10[test_symbolic_influence_map]
        T11[test_neural_logic_graph]
        T12[test_shap_overlay]
        T13[test_shap_attention_overlay]
        T14[test_shap_symbolic_overlay]
        T15[test_simulate_lightcurve_from_mu]
    end

    MU --> Diagnostics
    SIG --> Diagnostics

    Diagnostics --> Tests
    Tests --> CI[GitHub Actions CI]:::ci

    classDef ci fill=#0f62fe,stroke=#fff,color=#fff
````

---

## 🧪 Test Philosophy

* **Scientific Rigor** — FFT, SHAP, symbolic overlays, and diagnostics must be mathematically correct.
* **Interpretability First** — Every diagnostic visualization must export human-readable + machine-parseable outputs.
* **Integration-Ready** — Each diagnostic test validates compatibility with the HTML dashboard.
* **Fail-Fast** — If a diagnostic plot or metric diverges, the CI pipeline halts to prevent publishing misleading science.

---

## 🚀 Why It Matters

Diagnostics are the **eyes of the pipeline**.
They reveal *why* a model works, where it fails, and how symbolic rules constrain predictions.

The `/tests/diagnostics` suite ensures that SpectraMind V50’s diagnostics are:

* Accurate (FFT/spectral math validated)
* Explainable (SHAP + symbolic overlays reproducible)
* Integrated (plots and JSON embed into the dashboard without breakage)
* CI-Enforced (no unverified diagnostics reach the main branch)

In short: these tests guarantee the **scientific trustworthiness** of every insight SpectraMind publishes.

---

```

---
