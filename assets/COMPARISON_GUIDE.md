# SpectraMind V50 — Comparison Guide

**Purpose:**  
This document provides the narrative context for `comparison_overview.png`.  
It compares publicly available Kaggle baselines from the **NeurIPS 2025 Ariel Data Challenge** with the **SpectraMind V50 architecture**, highlighting *why* V50 looks the way it does in diagrams, dashboards, and reports.

---

## 📊 Kaggle Baseline Models

### 1. Thang Do Duc — *Residual MLP Baseline* (≈0.329 LB)  
- **Architecture:** residual-style MLP.  
- **Strengths:** clean, reproducible, fast to train.  
- **Limitations:** no explicit σ (uncertainty), minimal physics priors, risk of oversimplification.  
- **Role:** strong *reference design*; helped define V50’s reproducibility stack.

---

### 2. V1ctorious3010 — *80bl-128hd-impact* (≈0.322 LB)  
- **Architecture:** 80 residual blocks, 128 hidden units per layer.  
- **Strengths:** very high capacity, captured subtle signal in spectra.  
- **Limitations:** prone to overfitting; interpretability challenges; compute-heavy.  
- **Role:** highlighted the need for structured encoders (Mamba SSM, GNN) instead of brute force depth.

---

### 3. Fawad Awan — *Spectrum Regressor* (≈0.318 LB)  
- **Architecture:** single multi-output regression head (predicts all bins simultaneously).  
- **Strengths:** stable, interpretable, spectrum-wide coherence.  
- **Limitations:** less leaderboard-competitive; weaker at edge cases.  
- **Role:** reinforced the benefit of spectrum-aware decoders and calibration.

---

## 🚀 SpectraMind V50 — Key Upgrades

**Encoders:**  
- **FGS1 → Mamba SSM** (long-sequence modeling).  
- **AIRS → Graph NN** (λ-graph edges: proximity, molecule, detector regions).  

**Calibration:**  
- Bias/dark/flat/CDS corrections.  
- Trace extraction, background & jitter correction, normalization, phase alignment.  

**Uncertainty (σ):**  
- Explicit decoders for μ and σ.  
- Post-hoc calibration: **temperature scaling + COREL conformal GNN**.  

**Symbolic Logic:**  
- Smoothness, FFT coherence, non-negativity, asymmetry, molecular alignment.  
- Curriculum integration + diagnostics overlays.  

**Reproducibility:**  
- Hydra configs (`configs/*.yaml`).  
- DVC pipelines & remotes.  
- Typer CLI (`spectramind …`).  
- GitHub Actions CI (selftest, diagnostics, mermaid-export).  
- Logs: `logs/v50_debug_log.md`, JSONL event streams.

---

## 📐 Diagram Context

The **`comparison_overview.png`** graphic captures:  
- Left: Kaggle baselines (MLP, 80-block, Spectrum Regressor).  
- Right: SpectraMind V50 design (encoders/decoders, calibration, symbolic overlays, CI).  
- Middle: Key differences (residual depth vs symbolic physics; missing σ vs explicit calibration; ad-hoc vs reproducible pipelines).

---

## 🔗 Cross-References

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — full diagrams + layered stack.  
- **[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md)** — Kaggle runtime integration (training, inference, submission).  
- **`assets/diagrams/`** — Mermaid sources + rendered SVG/PNGs.  
- **`report.html`** — Reproducibility log.  
- **`diagnostics_dashboard.html`** — Interactive symbolic/SHAP/latent dashboard.

---

## ✅ Takeaway

V50 doesn’t discard Kaggle baselines — it **absorbs their lessons**:  
- Residual structure → stable encoders.  
- Spectrum-wide outputs → coherent μ/σ predictions.  
- Overfitting risks → physics-informed symbolic losses.  
- Lack of uncertainty → conformal calibration.  
- Limited reproducibility → Hydra + DVC + CI.

This alignment makes SpectraMind V50 **scientifically rigorous, interpretable, and leaderboard-competitive**.