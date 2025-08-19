# SpectraMind V50 — AI Design & Modeling Guide

> Mission: finalize a challenge‑grade architecture that’s physics‑informed, uncertainty‑calibrated, and inference‑efficient for Ariel μ/σ prediction (283 bins), with airtight CLI/Hydra/DVC reproducibility.

---

## 0) Outcomes we optimize for

* **Accuracy & Physics**: robust μ estimates respecting spectral structure and transit physics.
* **Honest σ**: well‑calibrated uncertainty (GLL score benefit).
* **Throughput**: ≤ 9 hr end‑to‑end on \~1,100 planets (Kaggle budget).
* **Reproducibility**: CLI-first runs with config hashing + data versioning.&#x20;

---

## 1) Final model blueprint (FGS1 × AIRS → μ/σ)

### 1.1 Encoders

* **FGS1 (135k+ timesteps)** → **Mamba SSM** (linear‑time sequence model; selective state; long‑context SSM).
  Rationale: avoids ViT quadratic attention and scales to ultra‑long curves, matching transformer accuracy at a fraction of cost.&#x20;

* **AIRS (283 λ bins)** → **Graph Neural Network (GNN)** with edge types:

  * wavelength adjacency (smoothness prior)
  * molecule regions (e.g., H₂O/CO₂/CH₄ groups)
  * detector region ties (shared systematics)
    Use GAT/RGCN; message passing “in‑paints” noisy bins from physically related neighbors.&#x20;

* **Fusion**: concatenate latent summaries (FGS1 SSM pooled reps + AIRS GNN node/readout) → joint head(s).

### 1.2 Decoders

* **μ head**: lightweight MLP; optional spectral regularizers (smoothness/FFT penalties inside loss).
* **σ head**: parallel branch for per‑bin σ (heteroscedastic). Training via **Gaussian Log‑Likelihood (GLL)**.&#x20;

### 1.3 Symbolic / physics priors

* Loss add‑ons: smoothness (L2 of ΔΔμ), asymmetry guardrails, FFT noise suppression, non‑negativity if needed; all toggled via Hydra.&#x20;

---

## 2) Uncertainty stack (multi‑tier)

1. **Aleatoric**: predict per‑bin σ and train with GLL.&#x20;
2. **Epistemic**: **ensembles** or **MC‑Dropout** variance across models/passes.&#x20;
3. **Post‑hoc calibration**:

   * global temperature scaling T for σ (optimize val GLL), and optionally per‑λ scaling vectors.&#x20;
   * instance‑level adaptation (lightweight test‑time scale) where allowed.
4. **COREL conformal GNN**: conformal intervals with graph‑aware residual correlation (coverage guarantees; lifts intervals on correlated bands like H₂O).&#x20;

> Optional frontier: **diffusion decoder** for non‑Gaussian posteriors (sample many spectra → μ/var). Heavier; use selectively.&#x20;

---

## 3) Explainability & diagnostics loop

* **GNNExplainer / edge importances** → which λ groups/relations drove predictions.&#x20;
* **FGS1 SSM attributions** (integrated gradients over time) → ingress/egress vs baseline contributions.&#x20;
* **Global SHAP overlays** for time and wavelength to spot biases;
* **FFT of residuals** to verify jitter/systematic suppression.
* **UMAP/t‑SNE latents** to visualize clusters (e.g., transit phases/metadata).
* All exported via `spectramind diagnose dashboard` HTML and logs.&#x20;

---

## 4) Throughput & runtime engineering

* **Calibration cache**: persist science‑ready tensors (NPY/Parquet) keyed by config hash; invalidate on setting changes (hash). Cuts hours off repeat runs.&#x20;
* **Linear encoders** (SSM) + **sparse GNN** inference.
* **Mixed precision** (AMP), pinned memory, overlap I/O.
* **Batching & prefetch**, avoid Python hotspots; vectorize.
* **DVC stages** ensure incremental recompute only when deps change.&#x20;

---

## 5) Training regimen

* **Curriculum**: start with masked‑autoencoding / denoise (FGS1/AIRS) → contrastive (align time–spectrum) → supervised μ/σ fine‑tune.&#x20;
* **Regularization**: spectral smoothness, label‑noise robust GLL clipping, dropout (and MC at test for epistemic).
* **Hydra sweeps**: learning rates, σ‑loss weight, GNN layer/edge‑type configs, SSM depth/state size.&#x20;
* **Reproducibility**: fixed seeds where feasible; record config & git SHA in logs + outputs.&#x20;

---

## 6) CLI/Hydra/DVC contract

* **Run** everything via Typer CLI (`spectramind ...`), never bypass. **Every action logged** with config hash + dataset hash to `v50_debug_log.md`.&#x20;
* **Hydra**: `configs/` groups (`data/`, `model/`, `training/`, `diagnostics/`, `calibration/`). Multirun for sweeps.&#x20;
* **DVC**: `dvc.yaml` stages for calibrate→train→predict→diagnose; cache artifacts; pin data versions.&#x20;

---

## 7) Minimal config sketch (Hydra)

```yaml
# configs/model/v50.yaml
encoders:
  fgs1:
    type: mamba
    d_model: 256
    n_layers: 8
    dropout: 0.1
  airs:
    type: gnn
    gnn_type: gat
    hidden: 256
    layers: 4
    edge_types: [adjacent, molecule, detector]
fusion:
  type: concat
decoders:
  mu:
    hidden: [256, 128]
  sigma:
    hidden: [256, 128]
loss:
  gll_weight: 1.0
  smooth_l2_weight: 0.1
  fft_suppress_weight: 0.05
uq:
  epistemic: ensemble   # or mc_dropout
  temp_scale: true
  corel: true
```

```yaml
# configs/training/default.yaml
optimizer: adamw
lr: 2.0e-4
batch_size: 16
epochs: 40
amp: true
seed: 1337
early_stop_patience: 6
```

```yaml
# configs/diagnostics/dashboard.yaml
shap: true
fft_residuals: true
umap: true
gnn_explain: true
export_html: true
```

---

## 8) Example CLI runs

```bash
# 1) Calibrate once with cache on
python -m spectramind calibrate data=kaggle calibration.cache=true

# 2) Train V50 with GNN+SSM and uncertainty head
python -m spectramind train model=v50 training=default

# 3) Predict μ/σ + conformalize
python -m spectramind predict model=v50 uq.corel=true --out-csv outputs/submission.csv

# 4) Calibrate uncertainties (temperature scaling on val)
python -m spectramind calibrate-temp uq.temp_scale=true

# 5) Diagnostics dashboard (SHAP/FFT/UMAP, explanations)
python -m spectramind diagnose dashboard diagnostics=dashboard

# 6) Sweep a few hypers
python -m spectramind train -m training.lr=1e-4,2e-4 uq.epistemic=ensemble,mc_dropout
```

All of the above auto‑log config+git SHA and produce versioned artifacts; DVC stages ensure minimal recompute.&#x20;

---

## 9) Acceptance checklist

* [ ] GLL improves after temp scaling on val.&#x20;
* [ ] Conformal coverage within target; intervals expand on correlated bands (H₂O/CO₂) as expected.&#x20;
* [ ] Residual FFT shows no persistent jitter peaks.
* [ ] XAI panels indicate molecule bands, not confounders, drive predictions.&#x20;
* [ ] End‑to‑end ≤ 9 hr on \~1,100 planets using cache + AMP + linear SSM.&#x20;
* [ ] Reproducibility: config hash, DVC data hash, Git SHA recorded for every artifact.&#x20;

---

## 10) References (internal)

SSM for long FGS1; AIRS GNN edges; multi‑tier UQ (GLL, ensembles, temp scaling, COREL, diffusion); XAI & FFT diagnostics; caching & throughput; CLI/Hydra/DVC rigor:

*

GUI optional (thin dashboard); CLI UX best practices:

*

Physics/spectroscopy background for rule design:

*

---
