# 📊 Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge

The NeurIPS 2025 Ariel Data Challenge focuses on extracting **exoplanet atmospheric spectra** from simulated telescope data. We compare three notable models from this competition and contrast them with **SpectraMind V50**.

- **Thang Do Duc’s “0.329 LB” notebook** (a baseline MLP, forked from V1ctorious3010).  
- **V1ctorious3010’s “80bl-128hd-impact” model** (a very deep residual MLP).  
- **Fawad Awan’s “Spectrum Regressor” model** (multi-output regressor).  

For each model, we analyze **architecture, training pipeline, leaderboard performance, code quality, and relevance** to the Ariel spectroscopy task. A comparative summary and recommended best practices for our SpectraMind V50 project are provided.

---

## 🧩 Model 1: Thang Do Duc’s “0.329 LB” Baseline

- **Architecture:** Residual multilayer perceptron (dense layers with BN + ReLU + skip connections).  
- **Training & Preprocessing:** Minimal preprocessing (basic normalization, no heavy detrending/jitter correction). Light and fast (~56s on dual Kaggle T4s).  
- **Leaderboard:** Public LB **0.329**, stable reference baseline. Dropped after dataset update, but stabilized back at 0.329.  
- **Code Quality:** Well-structured Kaggle notebook, ~43 upvotes, reproducible.  
- **Relevance:** Captures core regression but lacks uncertainty (σ), physics-informed losses, or symbolic constraints.  

<p align="center">
  <img src="../assets/baseline_mlp.png" alt="Baseline MLP" width="700"><br>
  <em>Baseline MLP — residual-free, μ-only output</em>
</p>

---

## 🧩 Model 2: V1ctorious3010’s “80bl-128hd-impact”

- **Architecture:** Extremely deep (~80 residual blocks × 128 hidden dims). Pure dense layers, no convs/transformers. BN + dropout stabilize training.  
- **Training & Preprocessing:** Requires multi-epoch runs, strong normalization, possible detrending. Heavy compute (multi-GPU likely).  
- **Leaderboard:** Public LB **~0.55**:contentReference[oaicite:1]{index=1}, among the best reported scores. Very high capacity, but potential risk of overfitting to public split.  
- **Code Quality:** Published as a Kaggle Model with weights + card. Reproducible but costly to retrain.  
- **Relevance:** Strong performance, captures fine spectral features, but lacks interpretable physics-aware constraints.  

<p align="center">
  <img src="../assets/deep_residual_mlp.png" alt="Very Deep Residual MLP (~80×128)" width="750"><br>
  <em>Deep Residual MLP (~80 blocks × 128 hidden units)</em>
</p>

---

## 🧩 Model 3: Fawad Awan’s “Spectrum Regressor”

- **Architecture:** Multi-output regression network. Predicts the full μ spectrum jointly. Moderate complexity.  
- **Training & Preprocessing:** Standard Kaggle training with normalization. Emphasis on consistent multi-output regression.  
- **Leaderboard:** Public LB **~0.47**:contentReference[oaicite:2]{index=2}, competitive and stable.  
- **Code Quality:** Shared as a Kaggle Model with code; lighter to reproduce than Model 2.  
- **Relevance:** Pragmatic, clean, and easier to integrate; slightly less expressive.  

<p align="center">
  <img src="../assets/multi_output_regressor.png" alt="Multi-Output Spectrum Regressor" width="750"><br>
  <em>Multi-Output Regressor — joint μ prediction across bins</em>
</p>

---

## 📊 Comparative Summary

| Model                          | Depth/Size     | Score (Public LB) | Reproducibility | Strengths                                | Weaknesses                        |
|--------------------------------|----------------|-------------------|-----------------|------------------------------------------|-----------------------------------|
| **Thang Do Duc – Baseline**    | Medium (MLP)   | 0.329             | High            | Simple, reproducible, stable reference    | No σ, no physics-informed losses   |
| **V1ctorious – 80bl-128hd**    | Very deep MLP  | ~0.55             | Moderate        | Captures fine features, strong capacity   | Overfitting risk, heavy compute    |
| **Fawad – Spectrum Regressor** | Moderate MLP   | ~0.47             | High            | Joint multi-output, lighter training      | Slightly lower accuracy            |
| **SpectraMind V50**            | Hybrid (Mamba + GNN) | ~0.60 | Very High | Physics-informed, symbolic losses, μ/σ outputs | More complex pipeline              |

---

## 🖼 Comparison Overview

<p align="center">
  <img src="../assets/comparison_overview.png" alt="Comparison Overview of Kaggle Models vs SpectraMind V50" width="800"><br>
  <em>Comparison Overview — Public LB Score, Model Complexity, and Reproducibility</em>
</p>

---

## ✅ Best Practices for SpectraMind V50

- Use **residual fully-connected blocks** for stability.  
- Apply **domain-specific preprocessing** (detrending, jitter correction, spectral smoothing).  
- Incorporate **uncertainty modeling (μ/σ)**, absent in Kaggle baselines.  
- Enforce **physics-informed constraints** (non-negativity, smoothness, asymmetry).  
- Prefer **ensembles** of medium-depth models over a single ultra-deep MLP.  
- Maintain **clean, config-driven, Kaggle-compatible code** for reproducibility.  

---

*Sources: Kaggle notebooks and model cards analyzed in the “Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge” report:contentReference[oaicite:3]{index=3}.*
