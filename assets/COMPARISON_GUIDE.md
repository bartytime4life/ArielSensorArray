# 📊 Model Comparison Guide — SpectraMind V50 `/assets/COMPARISON_GUIDE.md`

> **Purpose:**  
> This guide summarizes and contrasts **Kaggle baseline models** for the **NeurIPS 2025 Ariel Data Challenge**, and positions **SpectraMind V50** relative to them.  
> It provides architectural notes, training practices, leaderboard results, and best-practice insights for future development.

---

## 🌍 Context

The NeurIPS 2025 Ariel Data Challenge focuses on recovering **exoplanet atmospheric spectra** from **Ariel telescope simulations** (FGS1 + AIRS data).  
Competitors submit predicted spectra (`μ`) for ~1,100 planets. Performance is measured by **Gaussian Log-Likelihood (GLL)** against hidden test labels [oai_citation:0‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).

SpectraMind V50 is our **neuro-symbolic, physics-informed AI pipeline**, designed to **outperform baselines** by combining:

- **FGS1 Mamba encoder** for long-sequence temporal dynamics.  
- **AIRS GNN encoder** with molecule-aware edges.  
- **Multi-scale μ/σ decoders** with symbolic overlays.  
- **Hydra + CLI + DVC** for reproducibility [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## 🏗️ Models Compared

### 1. **Thang Do Duc — “0.329 LB” Baseline**
- **Architecture:**  
  - Residual MLP (dense layers + batch norm + ReLU).  
  - Skip connections inspired by ResNet.  
- **Training:**  
  - Minimal preprocessing (basic normalization).  
  - No domain-specific augmentation.  
  - Runs in ~56s on Kaggle T4 GPU [oai_citation:2‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
- **Performance:**  
  - Public LB: **0.329** (baseline reference) [oai_citation:3‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
  - Stable across dataset updates, minimal overfitting.  
- **Pros:** Simple, reproducible, easy to fork.  
- **Cons:** No uncertainty (σ), misses fine spectral details.  

---

### 2. **V1ctorious3010 — “80bl-128hd-impact”**
- **Architecture:**  
  - Deep residual MLP (~80 blocks × 128 hidden units).  
  - Heavy use of dropout + batch norm.  
- **Training:**  
  - Long training times; robust scheduling/regularization.  
  - Uses careful normalization to stabilize depth [oai_citation:4‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
- **Performance:**  
  - Public LB: ~**0.45–0.50** (depending on settings).  
  - Stronger at fitting complex patterns than baseline.  
- **Pros:** Demonstrates power of depth + residuals.  
- **Cons:** Risk of overfitting to public LB; no physics constraints.  

---

### 3. **Fawad Awan — “Spectrum Regressor”**
- **Architecture:**  
  - Multi-output PyTorch regressor.  
  - Direct dense mapping from input → output spectra.  
- **Training:**  
  - Likely uses moderate preprocessing + optimizer tuning.  
  - Faster than “80bl” but less trivial than baseline [oai_citation:5‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
- **Performance:**  
  - Competitive LB scores (~0.40–0.45).  
  - Positioned as a practical “middle ground.”  
- **Pros:** Balanced tradeoff of complexity vs reproducibility.  
- **Cons:** Limited interpretability; no uncertainty.  

---

## ⚖️ Comparative Summary

| Model                 | Depth / Capacity | Training Time | Physics-Aware? | Handles σ? | LB Score (Public) | Pros                                   | Cons                          |
|-----------------------|------------------|---------------|----------------|------------|-------------------|----------------------------------------|-------------------------------|
| Thang Do Duc (0.329)  | Low              | ~1 min        | ❌             | ❌         | 0.329             | Simple, reproducible baseline           | Over-smooth, no σ             |
| V1ctorious3010 (80bl) | Very High        | Long          | ❌             | ❌         | ~0.45–0.50        | Deep residual stack, strong capacity    | Overfit risk, less interpretable |
| Fawad Awan (Regressor)| Medium           | Moderate      | ❌             | ❌         | ~0.40–0.45        | Efficient, practical middle solution    | No σ, limited domain features |
| **SpectraMind V50**   | Hybrid (Mamba+GNN)| 9h runtime cap | ✅ (symbolic)  | ✅         | Target >0.55      | Physics-informed, reproducible, CLI-safe | Complex system, heavier infra |

---

## 🧪 Lessons for SpectraMind V50

1. **Baselines prove feasibility** — Even a simple residual MLP captures broad spectral structure [oai_citation:6‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).  
2. **Depth alone is not enough** — “80bl” shows diminishing returns without domain knowledge.  
3. **Balanced regressors are practical** — Fawad Awan’s model is efficient but lacks uncertainty.  
4. **V50 differentiation:**  
   - Adds **uncertainty σ predictions**.  
   - Enforces **symbolic & physical constraints** (smoothness, asymmetry, non-negativity).  
   - Uses **multi-instrument encoders** (FGS1 temporal dynamics + AIRS spectral GNN).  
   - Maintains **full reproducibility** via Hydra configs and DVC snapshots [oai_citation:7‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## 📌 Best Practices for Comparison

- Always compare **public vs. private LB** (avoid leaderboard chasing).  
- Include **cross-validation metrics** to detect overfitting.  
- Measure not just **accuracy**, but also:  
  - Spectral smoothness error.  
  - Symbolic rule violation rate.  
  - Calibration (σ vs. residuals).  
- Document **training configs + commit hashes** for reproducibility [oai_citation:8‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).  

---

## 🔗 References

- **Kaggle Competition (Ariel 2025):** [Link](https://www.kaggle.com/competitions/ariel-data-challenge-2025) [oai_citation:9‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- **Baseline “0.329 LB”:** Thang Do Duc’s fork [oai_citation:10‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- **80bl-128hd-impact:** V1ctorious3010’s notebook [oai_citation:11‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- **Spectrum Regressor:** Fawad Awan’s model [oai_citation:12‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- **SpectraMind V50 Plan & Analysis:** SpectraMind docs [oai_citation:13‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)

---

## ✅ Summary

- Kaggle baselines highlight **ML strength**, but ignore **physics & uncertainty**.  
- SpectraMind V50 is designed to be the **challenge-winning upgrade**, combining:  
  - Strong encoders (Mamba + GNN)  
  - Symbolic physics constraints  
  - σ calibration & uncertainty quantification  
  - Fully reproducible CLI + Hydra + DVC infrastructure  

This guide ensures `/assets/COMPARISON_GUIDE.md` is a single source of truth for **benchmarking SpectraMind V50** against Kaggle baselines.