# SpectraMind V50 — Ariel Data Challenge 2025 Documentation

**Neuro-symbolic, physics-informed AI pipeline for exoplanet spectroscopy**  
*From raw Ariel frames → calibrated light curves → μ/σ spectra → diagnostics → leaderboard-ready submission*

---

## 🚀 North Star

- **Input:** Ariel FGS1/AIRS telescope frames (simulated challenge data)  
- **Pipeline:** Calibration → Training → Prediction → Diagnostics → Packaging  
- **Output:** Mean (μ) and Uncertainty (σ) spectra across 283 bins  
- **Guarantees:** Reproducibility via Hydra configs, DVC data versioning, CI workflows, and Kaggle integration  

---

## 📂 Documentation Index

### 1. **Getting Started**
- [README](../README.md) — Repository overview and setup
- [Architecture](../ARCHITECTURE.md) — Full system design & component map
- [AI Design & Modeling](../AI%20Design%20and%20Modeling.pdf) — Design notes & references
- [Ubuntu Setup Guide](../Ubuntu%20Science%20AI%20Setup%20Guide.docx) — Local workstation setup

### 2. **Data & Calibration**
- [Calibration Pipeline](../calibration_pipeline.md) — ADC → dark/flat → trace → spectra
- [Spectroscopy Background](../Cosmic%20Fingerprints%20.txt) — Physics of stellar spectra
- [Gravitational Lensing](../Gravitational%20Lensing%20and%20Astronomical%20Observation%20-%20Modeling%20and%20Mitigation.pdf) — Mitigation & modeling

### 3. **Modeling**
- [SpectraMind V50 Technical Plan](../SpectraMind%20V50%20Technical%20Plan%20for%20the%20NeurIPS%C2%A02025%20Ariel%20Data%20Challenge.pdf)
- [Model Configs](../configs/model/) — Hydra-based configuration files
- [Training](../configs/train/) — Train scripts and configs
- [AI Decoding Methods](../AI%20Decoding%20and%20Processing%20Methods.pdf)
- [Scientific Modeling Guide](../Scientific%20Modeling%20and%20Simulation%20-%20A%20Comprehensive%20NASA-Grade%20Guide.pdf)

### 4. **Competition Integration**
- [Kaggle Platform Guide](../Kaggle%20Platform%20-%20Comprehensive%20Technical%20Guide.pdf)  
- [Kaggle Model Comparisons](comparison.md) — Analysis of public Kaggle models & baselines  
- [Competition Data](https://www.kaggle.com/competitions/ariel-data-challenge-2025/data)  

### 5. **Diagnostics & Explainability**
- [Diagnostics Dashboard](../report.html) — Auto-generated HTML report
- [SHAP & Symbolic Overlays](../shap_overlay.md) — Feature importance fused with symbolic logic
- [Generate Diagnostic Summary](../generate_diagnostic_summary.md) — CLI diagnostic metrics  
- [FFT/Autocorrelation Analysis](../analyze_fft_autocorr_mu.md)  
- [Symbolic Influence Maps](../symbolic_influence_map.md)  

### 6. **CLI & Automation**
- [CLI Core (`spectramind.py`)](../spectramind.py) — Unified Typer app  
- [Makefile](../Makefile) — Local dev shortcuts  
- [DVC Pipeline (`dvc.yaml`)](../dvc.yaml) — Reproducible data/model pipeline  
- [CI Workflows](../.github/workflows/) — Automated tests, builds, and diagnostics  
- [Pre-commit Hooks](../.pre-commit-config.yaml)  

### 7. **Advanced References**
- [Patterns, Algorithms & Fractals](../Patterns%2C%20Algorithms%2C%20and%20Fractals%20-%20A%20Cross-Disciplinary%20Technical%20Reference.pdf)  
- [Physics Modeling Using Computers](../Physics%20Modeling%20Using%20Computers%20-%20A%20Comprehensive%20Reference.pdf)  
- [Computational Physics Modeling](../Computational%20Physics%20Modeling%20-%20Mechanics%2C%20Thermodynamics%2C%20Electromagnetism%20%26%20Quantum.pdf)  
- [Radiation Reference](../Radiation%20-%20A%20Comprehensive%20Technical%20Reference.pdf)  

---

## 🧰 Key Tools

- **Hydra Configs:** Flexible parameter management across training/eval/calibration  
- **DVC + lakeFS:** Data and model versioning with experiment tracking  
- **DVCLive:** Live metric logging during training (`dvclive/`)  
- **CI/CD:** GitHub Actions workflows for diagnostics, submissions, and artifact validation  
- **Explainability:** SHAP, Symbolic overlays, t-SNE/UMAP, spectral FFT analysis  
- **Submission Guardrails:** Self-tests, config hash logging, and reproducibility enforcement  

---

## 🏆 Competition Notes

- Public LB metrics can shift when private LB is revealed — beware of “Kaggle shake-up”:contentReference[oaicite:0]{index=0}  
- Medium-depth residual MLPs with symbolic loss constraints may outperform very deep plain MLPs in stability and reproducibility  
- Top solutions combine **domain-informed preprocessing (detrending, jitter correction)** with **ML ensembles**  

---

*Last updated: August 2025 — integrated with SpectraMind V50 repo structure and NeurIPS Ariel Data Challenge requirements.*
