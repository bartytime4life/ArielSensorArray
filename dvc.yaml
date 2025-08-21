# Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge

The NeurIPS 2025 Ariel Data Challenge focuses on extracting **exoplanet atmospheric spectra** from simulated telescope data. We compare three notable models from this competition:

- **Thang Do Duc’s “0.329 LB” notebook** (a baseline model originally by user V1ctorious3010).  
- **V1ctorious3010’s “80bl-128hd-impact” model** (an improved deep PyTorch model).  
- **Fawad Awan’s “Spectrum Regressor” model** (multi-output PyTorch regressor).  

For each model, we analyze their **architecture, training pipeline, performance, code quality, and relevance** to the Ariel spectroscopy task. A comparative summary and recommended best practices for our SpectraMind V50 project are also provided.

---

## Model 1: Thang Do Duc’s “0.329 LB” Baseline

- **Architecture:** Residual multilayer perceptron (dense layers with BN + ReLU + skip connections). Inspired by ResNet-style stabilization.  
- **Training & Preprocessing:** Minimal. Basic normalization, little augmentation. Light and fast to train (~56s on Kaggle T4s).  
- **Leaderboard:** Public LB score **0.329**, stable reference baseline. Dropped to 0.287 after dataset update, then stabilized back to 0.329.  
- **Code Quality:** Well-structured Kaggle notebook, ~43 upvotes, reproducible.  
- **Relevance:** Captures core regression but lacks uncertainty, physics-informed losses, or domain-specific tricks:contentReference[oaicite:0]{index=0}.

---

## Model 2: V1ctorious3010’s “80bl-128hd-impact”

- **Architecture:** Extremely deep (~80 residual blocks × 128 hidden dims). Pure dense layers, no transformers/convs. Uses BN + dropout to stabilize.  
- **Training & Preprocessing:** Likely multi-epoch, with normalization, possible detrending, and maybe augmentation. Requires multi-GPU.  
- **Leaderboard:** Public LB **0.322**, an improvement over baseline. Potential risk of overfitting to public split.  
- **Code Quality:** Published as a Kaggle Model with weights + card. Reproducible, but retraining is heavy.  
- **Relevance:** Strong at capturing fine spectral features, but interpretable physics constraints are absent:contentReference[oaicite:1]{index=1}.

---

## Model 3: Fawad Awan’s “Spectrum Regressor”

- **Architecture:** Multi-output regression network. Simpler than Model 2, possibly hybrid dense + conv/recurrent. Predicts full spectrum jointly.  
- **Training & Preprocessing:** Standard Kaggle training with normalization. Emphasis on multi-output consistency.  
- **Leaderboard:** Public LB score around **0.317–0.320** (competitive but slightly below baseline).  
- **Code Quality:** Shared as a Kaggle Model with code, lighter to reproduce.  
- **Relevance:** Pragmatic, easy-to-use baseline extension. Less expressive than Model 2, but cleaner for integration:contentReference[oaicite:2]{index=2}.

---

## Comparative Summary

| Model                          | Depth/Size     | Score (Public LB) | Strengths                                   | Weaknesses                          |
|--------------------------------|----------------|-------------------|---------------------------------------------|-------------------------------------|
| **Thang Do Duc – Baseline**    | Medium (MLP)   | 0.329             | Simple, reproducible, stable reference       | No uncertainty, limited physics      |
| **V1ctorious – 80bl-128hd**    | Very deep MLP  | 0.322             | Captures fine spectral features              | Risk of overfitting, heavy compute   |
| **Fawad – Spectrum Regressor** | Moderate MLP   | 0.317–0.320       | Multi-output design, lighter training        | Slightly weaker performance          |

---

## Best Practices for SpectraMind V50

- Use **residual fully-connected blocks** for stability.  
- Apply **domain-specific preprocessing** (detrending, jitter correction, spectral smoothing).  
- Incorporate **uncertainty modeling (μ/σ)**, absent in all three Kaggle models.  
- Enforce **physics-informed constraints** (non-negativity, smoothness, asymmetry) during training.  
- Prefer **ensembles** of medium-depth models over a single very deep MLP for generalization.  
- Keep code clean, reproducible, and Kaggle-compatible.  

---

*Sources: Kaggle notebooks and model cards analyzed in the “Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge” report:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.*
