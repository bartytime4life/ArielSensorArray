# SpectraMind V50 — Kaggle Model Comparison  
*NeurIPS 2025 Ariel Data Challenge*

This document reviews three notable Kaggle models that participated in the NeurIPS 2025 Ariel Data Challenge, focused on retrieving **exoplanet transmission spectra** from Ariel telescope simulations.  

We analyze their **architecture, training pipelines, leaderboard performance, and relevance** to the SpectraMind V50 project, and extract best practices for building a **neuro-symbolic, physics-informed, uncertainty-calibrated system**.

---

## 📊 Models Reviewed
1. **Thang Do Duc — “0.329 LB” Baseline**  
   Fork of V1ctorious3010’s reference MLP model.  
2. **V1ctorious3010 — “80bl-128hd-impact”**  
   Extremely deep residual MLP (~80 blocks × 128 dims).  
3. **Fawad Awan — “Spectrum Regressor”**  
   Multi-output PyTorch regressor with moderate complexity.  

---

## 🔍 Model Details

### 1. Thang Do Duc — “0.329 LB” Baseline
- **Architecture:**  
  Residual fully connected MLP. Dense layers with BatchNorm + ReLU + skip connections (ResNet-style).  
- **Training & Preprocessing:**  
  - Minimal preprocessing (basic normalization).  
  - Trains quickly (~56 seconds on Kaggle T4 GPU).  
- **Performance:**  
  - Public LB ~**0.329**.  
  - Stable across dataset updates (drop to 0.287, later recovered).  
- **Strengths:**  
  - Simple, reproducible, fast.  
  - Strong reference baseline.  
- **Weaknesses:**  
  - No uncertainty (σ).  
  - No physics-informed constraints (smoothness, asymmetry).  

---

### 2. V1ctorious3010 — “80bl-128hd-impact”
- **Architecture:**  
  - ~80 residual blocks × 128 hidden units.  
  - Very deep MLP, relies on BatchNorm + dropout for stability.  
- **Training & Preprocessing:**  
  - Strong normalization & detrending.  
  - Requires multi-GPU for efficient training.  
- **Performance:**  
  - Public LB ~**0.322** (slightly above baseline).  
  - Risk of overfitting due to extreme depth.  
- **Strengths:**  
  - Captures fine spectral detail.  
  - Reproducible (Kaggle Model release with weights).  
- **Weaknesses:**  
  - Heavy compute requirements.  
  - No uncertainty modeling.  
  - Risk of fitting noise.  

---

### 3. Fawad Awan — “Spectrum Regressor”
- **Architecture:**  
  - Multi-output regressor, outputs all spectral bins jointly.  
  - Likely hybrid dense layers with optional conv/recurrent elements.  
- **Training & Preprocessing:**  
  - Simpler preprocessing, normalized spectra.  
  - Focused on robustness and reproducibility.  
- **Performance:**  
  - Public LB ~**0.317–0.320**.  
  - Competitive but not top performing.  
- **Strengths:**  
  - Outputs full spectrum in one shot.  
  - Clean, lighter to train.  
- **Weaknesses:**  
  - Less expressive than deep residual nets.  
  - Still lacks uncertainty modeling.  

---

## 📑 Comparative Summary

| Model                          | Depth/Size     | Score (Public LB) | Strengths                                   | Weaknesses                          |
|--------------------------------|----------------|-------------------|---------------------------------------------|-------------------------------------|
| **Thang Do Duc – Baseline**    | Medium (MLP)   | 0.329             | Simple, reproducible, stable reference       | No uncertainty, limited physics      |
| **V1ctorious – 80bl-128hd**    | Very deep MLP  | 0.322             | Captures fine spectral features              | Risk of overfitting, heavy compute   |
| **Fawad – Spectrum Regressor** | Moderate MLP   | 0.317–0.320       | Multi-output design, lighter training        | Slightly weaker performance          |

---

## 🚀 Best Practices for SpectraMind V50
- Use **residual fully-connected blocks** (baseline + deep model).  
- Add **domain-specific preprocessing**: detrending, jitter correction, smoothing.  
- Incorporate **uncertainty modeling (μ/σ)** to improve calibration.  
- Enforce **physics-informed constraints**:  
  - Non-negativity of spectra.  
  - Smoothness and asymmetry penalties.  
  - FFT-based symbolic regularizers.  
- Use **ensembles of medium-depth models** for generalization over brute-force depth.  
- Maintain **reproducibility** with Hydra configs, DVC data versioning, and deterministic seeding.  

---

## 📚 Sources
- Kaggle notebooks & models analyzed in:  
  - Thang Do Duc — [“0.329 LB” baseline notebook]  
  - V1ctorious3010 — [“80bl-128hd-impact” model card]  
  - Fawad Awan — [“Spectrum Regressor” Kaggle model]  

*(See internal file: `Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf` for detailed notes.)*

---
