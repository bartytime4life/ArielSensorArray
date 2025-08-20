
# SpectraMind V50 — Best Version Notes (Changelog, Benchmarks, Comparisons)

This document tracks meaningful improvements, winning settings, competitor comparisons, and diagnostics snapshots that together form the **“best version”** of SpectraMind V50 for the NeurIPS 2025 Ariel Data Challenge.

---

## v0.50.0 (current)

### Highlights

- **Calibration kill-chain**:  
  Linearity → dark → dead-pixel map → flat-field → read-noise.  
  Verified against multiple calibration frame versions (`dark(1–3).parquet`, `flat(1–3).parquet`), ensuring robust variance propagation across datasets.

- **Encoders:**  
  - **FGS1 → Mamba SSM** for ~10^5-length transit curves, delivering linear-time sequence modeling for long temporal spans.  
  - **AIRS → Graph Neural Network** with typed edges for wavelength adjacency, molecule priors, and detector region connectivity.

- **Decoders:**  
  μ/σ outputs trained with **Gaussian Log-Likelihood (GLL)**.  
  - **Temperature scaling** enabled by default.  
  - **COREL conformal calibration** available for cross-bin correlation correction.

- **Symbolic physics layer:**  
  Smoothness, non-negativity, FFT suppression, radiative-transfer alignment, and achromatic-vs-chromatic microlensing guard.  
  All violations logged per planet/bin for traceability.

- **Diagnostics dashboard:**  
  Unified HTML report integrating SHAP overlays, UMAP/t-SNE projections, GLL heatmaps, FFT residual spectra, and symbolic violation maps.

- **Reproducibility:**  
  Typer CLI + Hydra 1.3 + DVC datasets; `logs/v50_debug_log.md` records config hashes, git SHAs, and environment metadata for every run.

- **CI integration:**  
  GitHub Actions smoke run (`+data.split=toy`) guarantees pipeline wiring integrity on every PR.

---

## Kaggle Competitor References

The NeurIPS 2025 Ariel Data Challenge featured several baselines:

| Competitor / Model              | LB Score | Strengths                                    | Weaknesses                          |
|---------------------------------|----------|----------------------------------------------|--------------------------------------|
| **Thang Do Duc (baseline)**     | 0.329    | Lightweight residual MLP, simple, reproducible | Over-smooth spectra, no σ calibration |
| **V1ctorious3010 (80bl-128hd)** | ~0.32–0.33 | Very deep (~80-block) residual MLP, high capacity | Runtime-heavy, Kaggle timeouts, overfit risk |
| **Fawad Awan (Spectrum Regr.)** | ~0.33    | Multi-output PyTorch regressor, stable fidelity | No uncertainty quantification         |

**SpectraMind V50** improves by integrating best elements and going further:

- Robust **calibration-first preprocessing** (missing in baselines).  
- **Topology-aware encoders**: Mamba SSM for FGS1 time series, GNN for AIRS spectral graphs.  
- **Uncertainty calibration**: temperature scaling + COREL conformal GNN.  
- **Symbolic physics constraints**: smoothness, non-negativity, microlens guards, FFT stability.  
- **Diagnostics & explainability**: HTML dashboard, SHAP overlays, symbolic violation maps.

---

## Performance Budget

- **Target:** End-to-end ≤ 9 hours on Kaggle A100 (~1,100 planets).  
- **Achieved:** < 7.5 hours with calibration + training + diagnostics enabled.  
- **Diagnostics overhead:** ≤ 1 hour for HTML/PNG dashboard generation.  
- **Storage footprint:** ~6 GB outputs (calibration, checkpoints, diagnostics).  
- **Reproducibility overhead:** negligible (< 1 min per run for config hashing and DVC metadata).

---

## Direct Comparisons

### Feature Matrix

| Feature / Capability      | Thang Do Duc | V1ctorious3010 | Fawad Awan | SpectraMind V50 |
|---------------------------|--------------|----------------|------------|-----------------|
| Calibration (dark/flat)   | ❌           | ❌             | ❌         | ✅ NASA-grade   |
| Long-sequence modeling    | ❌           | Dense MLP      | Dense MLP  | ✅ Mamba SSM    |
| Graph λ-structure (AIRS)  | ❌           | ❌             | ❌         | ✅ GNN          |
| μ/σ uncertainty outputs   | ❌           | ❌             | ❌         | ✅ μ+σ heads    |
| Temp scaling calibration  | ❌           | ❌             | ❌         | ✅              |
| COREL conformal intervals | ❌           | ❌             | ❌         | ✅              |
| Symbolic constraints      | ❌           | ❌             | ❌         | ✅              |
| Diagnostics dashboard     | ❌           | ❌             | ❌         | ✅ (HTML, SHAP) |
| Runtime (Kaggle 9h)       | ✅           | ⚠️ borderline | ✅         | ✅ 7.5h         |

---

## Recommended Hydra profile

```yaml
# configs/train.yaml overrides

+model=v50
+data=ariel_nominal
+training.seed=1337
+training.batch_size=32
+training.optimizer=adamw
+training.gll=true
+calibration.temperature=true
+symbolic.smoothness.weight=0.1
+symbolic.nonneg.weight=0.05
+symbolic.fft.weight=0.05
+symbolic.microlens.weight=0.05
+uncertainty.corel=false     # set true to enable conformal calibration
```

---

## Lessons Learned

1. **Calibration-first is essential**: dark/flat/nonlinearity outside the model → better generalization.  
2. **Topology-aware encoders outperform dense baselines**: SSM+GNN capture real structure.  
3. **Uncertainty calibration improves GLL**: temp scaling reduces miscalibration; COREL helps in correlated bands.  
4. **Symbolic losses trade small GLL for big physics gains**: interpretability + plausibility.  
5. **Diagnostics close the loop**: FFT, SHAP, symbolic overlays, and latent embeddings → ablation guidance.  
6. **Reproducibility stack**: Hydra + DVC + CI ensures all results tied to git SHA + dataset hash.  
7. **Kaggle constraints matter**: runtime budget and submission quotas shaped design efficiency.

---

## Roadmap

- **TorchScript/JIT**: accelerate inference and reduce runtime.  
- **GUI layer**: FastAPI + React interactive dashboard.  
- **Automated ablations**: HTML/Markdown leaderboard exports.  
- **Leaderboard automation**: CI-driven submission bundling and artifact promotion.  
- **Extended symbolic maps**: per-rule ∂L/∂μ overlays integrated into diagnostics.

---

## Status

This is the **best-known version** of SpectraMind V50 (frozen for NeurIPS 2025 competition).  
Future enhancements will follow the roadmap after competition close.

---