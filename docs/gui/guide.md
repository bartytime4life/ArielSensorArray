# 🛰️ SpectraMind V50 — AI Design & Modeling Guide

> **Mission:** finalize a challenge-grade architecture that’s physics-informed, uncertainty-calibrated, and inference-efficient for Ariel μ/σ prediction (283 bins), with airtight CLI/Hydra/DVC reproducibility.

```mermaid
flowchart LR
  %% ============= Inputs =============
  subgraph Inputs
    FGS1[FGS1 time series (~135k t)]
    AIRS[AIRS spectrum (283 λ)]
  end

  %% ============= Encoders =============
  FGS1 --> SSM[Mamba SSM\n(linear-time selective state)]
  AIRS --> GNN[GNN (GAT/RGCN)\nEdges: adjacency • molecule • detector]

  %% ============= Fusion & Heads =============
  SSM --> FUSION[Concat / Fusion]
  GNN --> FUSION
  FUSION --> MU[μ decoder (MLP)]
  FUSION --> SIGMA[σ decoder (heteroscedastic • GLL)]

  %% ============= UQ Stack =============
  MU -->|residuals| UQ1[Aleatoric (σ via GLL)]
  SIGMA --> UQ1
  FUSION -.-> UQ2[Epistemic\n(ensembles / MC-Dropout)]
  MU --> CAL[Temp scaling (global / per-λ)]
  SIGMA --> CAL
  CAL --> CONFORMAL[COREL conformal GNN\n(graph-aware intervals)]

  %% ============= Diagnostics / XAI =============
  MU --> DIAG[Diagnostics Dashboard (HTML)]
  SIGMA --> DIAG
  DIAG -->|XAI| XAI[SHAP • GNNExplainer • SSM attributions]
  DIAG -->|Spectral| FFT[FFT of residuals\n(jitter/systematics check)]
  DIAG -->|Latents| PROJ[UMAP / t-SNE]

  %% ============= Reproducibility Plane =============
  subgraph Reproducibility
    CLI[Typer CLI\n(spectramind …)]
    HYDRA[Hydra configs\n(config groups & overrides)]
    DVC[DVC stages & cache]
    LOGS[v50_debug_log.md\n+ config & dataset hashes]
  end

  CLI --> HYDRA --> DVC
  CLI --> LOGS
  DVC -. artifacts .-> DIAG
  CLI -->|calibrate • train • predict • diagnose| FUSION
```

---

## 0) Outcomes We Optimize For

* **Accuracy & Physics**: robust μ estimates respecting spectral structure and transit physics.
* **Honest σ**: well-calibrated uncertainty (improves GLL).
* **Throughput**: ≤ 9 hr end-to-end on \~1,100 planets (Kaggle budget).
* **Reproducibility**: CLI-first, config hashing, DVC-versioned data/artifacts.

---

## 1) Final Model Blueprint (FGS1 × AIRS → μ/σ)

### 1.1 Encoders

* **FGS1 (135k+ timesteps)** → **Mamba SSM** (linear-time selective state; long-context SSM).
  *Rationale:* avoids ViT quadratic attention; scales to ultra-long curves while matching transformer accuracy at a fraction of cost.

* **AIRS (283 λ bins)** → **Graph Neural Network (GNN)** with edge types:
  **(a)** wavelength adjacency (smoothness prior)
  **(b)** molecule regions (H₂O/CO₂/CH₄ groups)
  **(c)** detector region ties (shared systematics)
  *Use GAT/RGCN; message passing “in-paints” noisy bins from physically related neighbors.*

* **Fusion**: concatenate pooled FGS1 SSM latent + AIRS GNN readout → joint head(s).

### 1.2 Decoders

* **μ head**: lightweight MLP; optional spectral regularizers (smoothness/FFT penalties inside loss).
* **σ head**: heteroscedastic per-bin σ; trained via **Gaussian Log-Likelihood (GLL)**.

### 1.3 Symbolic / Physics Priors

* Loss add-ons (Hydra-toggle): smoothness (L2 of ΔΔμ), asymmetry guardrails, FFT noise suppression, non-negativity if needed.

---

## 2) Uncertainty Stack (Multi-Tier)

1. **Aleatoric**: predict per-bin σ and train with GLL.
2. **Epistemic**: **ensembles** or **MC-Dropout** variance across models/passes.
3. **Post-hoc calibration**:

   * global temperature scaling **T** for σ (optimize val GLL), optionally **per-λ** scaling vectors;
   * instance-level adaptation (lightweight test-time scale) where allowed.
4. **COREL conformal GNN**: conformal intervals with graph-aware residual correlation (coverage guarantees; lift intervals on correlated bands like H₂O).

> Optional frontier: **diffusion decoder** for non-Gaussian posteriors (sample many spectra → μ/var). Heavier; use selectively.

---

## 3) Explainability & Diagnostics Loop

* **GNNExplainer / edge importances** → which λ groups/relations drove predictions.
* **FGS1 SSM attributions** (integrated gradients over time) → ingress/egress vs baseline contributions.
* **Global SHAP overlays** for time and wavelength to spot biases.
* **FFT of residuals** to verify jitter/systematic suppression.
* **UMAP/t-SNE latents** to visualize clusters (e.g., transit phases/metadata).
* Exported via `spectramind diagnose dashboard` (HTML + logs).

---

## 4) Throughput & Runtime Engineering

* **Calibration cache**: persist science-ready tensors (NPY/Parquet) keyed by config hash; invalidate on setting changes (hash).
* **Linear encoders** (SSM) + **sparse GNN** inference.
* **Mixed precision** (AMP), pinned memory, overlap I/O.
* **Batching & prefetch**, vectorize, avoid Python hotspots.
* **DVC stages** ensure incremental recompute only when deps change.

---

## 5) Training Regimen

* **Curriculum**: masked-autoencoding / denoise (FGS1/AIRS) → contrastive (align time–spectrum) → supervised μ/σ fine-tune.
* **Regularization**: spectral smoothness, label-noise-robust GLL clipping, dropout (and MC at test for epistemic).
* **Hydra sweeps**: learning rates, σ-loss weight, GNN layer/edge-type configs, SSM depth/state size.
* **Reproducibility**: fixed seeds where feasible; record config & Git SHA in logs + outputs.

---

## 6) CLI/Hydra/DVC Contract

* **Run** everything via Typer CLI (`spectramind ...`), never bypass. **Every action logged** with config hash + dataset hash to `logs/v50_debug_log.md`.
* **Hydra**: `configs/` groups (`data/`, `model/`, `training/`, `diagnostics/`, `calibration/`). Multirun for sweeps.
* **DVC**: `dvc.yaml` stages for calibrate → train → predict → diagnose; cache artifacts; pin data versions.

---

## 7) Minimal Config Sketch (Hydra)

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

## 8) Example CLI Runs

```bash
# 1) Calibrate once with cache on
spectramind calibrate data=kaggle calibration.cache=true

# 2) Train V50 with GNN+SSM and uncertainty head
spectramind train model=v50 training=default

# 3) Predict μ/σ + conformalize
spectramind predict model=v50 uq.corel=true --out-csv outputs/submission.csv

# 4) Calibrate uncertainties (temperature scaling on val)
spectramind calibrate-temp uq.temp_scale=true

# 5) Diagnostics dashboard (SHAP/FFT/UMAP, explanations)
spectramind diagnose dashboard diagnostics=dashboard

# 6) Sweep a few hypers
spectramind train -m training.lr=1e-4,2e-4 uq.epistemic=ensemble,mc_dropout
```

All of the above auto-log config + Git SHA and produce versioned artifacts; DVC stages ensure minimal recompute.

---

## 9) Acceptance Checklist

* [ ] GLL improves after temp scaling on val.
* [ ] Conformal coverage within target; intervals expand on correlated bands (H₂O/CO₂) as expected.
* [ ] Residual FFT shows no persistent jitter peaks.
* [ ] XAI panels indicate molecule bands, not confounders, drive predictions.
* [ ] End-to-end ≤ 9 hr on \~1,100 planets using cache + AMP + linear SSM.
* [ ] Reproducibility: config hash, DVC data hash, Git SHA recorded for every artifact.

---

## 10) DVC Pipeline DAG (Calibrate → Train → Predict → Diagnose)

```mermaid
flowchart TB
  RAW[Raw inputs (FGS1/AIRS)] --> CAL[calibrate]
  CAL --> PKG[package_batches]
  PKG --> TRN[train]
  TRN --> PRED[predict μ/σ]
  PRED --> UQ[conformalize (COREL) & temp-scale]
  UQ --> DIAG[diagnose dashboard]
  classDef stage fill:#eaf5ff,stroke:#2b6cb0,color:#1a365d,stroke-width:1px;
  class CAL,PKG,TRN,PRED,UQ,DIAG stage;
```

*Each stage caches artifacts keyed by config/data hashes; only invalidated stages recompute.*

---

### Notes

* Keep GUI **optional** and **thin**: if you add a dashboard, it should **mirror the CLI**, never bypass it; all actions must write to logs and respect Hydra/DVC.
* Additions (e.g., diffusion decoder, extra symbolic constraints) should be toggled via Hydra and tracked in `v50_debug_log.md` + DVC.

---
