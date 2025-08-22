---

# Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge

The NeurIPS 2025 Ariel Data Challenge focuses on extracting exoplanet atmospheric spectra from simulated telescope data. We compare three notable models:

* **Thang Do Duc’s “0.329 LB” notebook** (a baseline model, forked from V1ctorious3010).
* **V1ctorious3010’s “80bl-128hd-impact” model** (deep PyTorch, very high capacity).
* **Fawad Awan’s “Spectrum Regressor”** (multi-output regression).

---

## Model 1: Thang Do Duc’s “0.329 LB” Baseline

* **Architecture:** Fully connected deep net with residual connections (ResNet-style MLP). Linear + BN + ReLU with identity skips. Moderate depth.
* **Training/Preprocessing:** Minimal; basic normalization; no heavy augmentation. Trains quickly on Kaggle T4 GPUs (\~56s run).
* **Performance:** Public LB \~0.329. Stable but not near SOTA (top \~0.55–0.60). Robust to dataset updates (drop from 0.328→0.287 then stabilized).
* **Code Quality:** Well-structured Kaggle notebook; upvoted 43×; reproducible and clear.
* **Relevance:** Simple multi-output regression; no uncertainty, no physics constraints. Captures basics but oversmooths fine details.

---

## Model 2: V1ctorious3010’s “80bl-128hd-impact”

* **Architecture:** \~80 residual fully connected blocks, each 128 hidden units. Essentially a deep residual MLP for 1D/tabular input. Uses BN + dropout. No attention; brute force depth.
* **Training/Preprocessing:** Likely stronger detrending/normalization. May have used noise jitter augmentation. Needs multi-GPU for training.
* **Performance:** Public LB \~0.322, stronger than baseline. Still below SOTA. Possible overfitting risk due to extreme depth.
* **Code Quality:** Experienced competitor. Released via Kaggle Models (Model Card + code). Weights downloadable. Reproducible but heavy.
* **Relevance:** Can resolve fine spectral details. Risk of fitting noise. No explicit uncertainty or physics constraints. Maximizes predictive muscle.

---

## Model 3: Fawad Awan’s “Spectrum Regressor”

* **Architecture:** Multi-output regression. Moderately deep, possibly hybrid (dense + conv/RNN). Outputs full spectrum in one pass.
* **Training/Preprocessing:** Emphasis on simplicity and reproducibility. Data normalized, possibly detrended.
* **Performance:** Public LB competitive with others (details truncated, but better than baseline in robustness).
* **Code Quality:** Shared on Kaggle Models; clean PyTorch regressor; reproducible.
* **Relevance:** Outputs all wavelengths; flexible and interpretable. Still lacks uncertainty modeling.

---

## Comparative Summary

* **Baseline (0.329 LB):** Fast, simple, reproducible; weak performance, no uncertainty.
* **80bl-128hd (0.322 LB):** Deep capacity, best detail capture, but risk of noise/overfit.
* **Spectrum Regressor:** Balanced approach, outputs entire spectrum, simpler and robust.

**Best Practices for SpectraMind V50:**

* Combine residual depth (from 80bl-128hd) with simplicity/reproducibility (from Spectrum Regressor).
* Add **physics-informed losses** (smoothness, non-negativity).
* Incorporate **uncertainty outputs** (μ/σ).
* Use **ensembling and augmentation** to stabilize.

---
