## 📂 Directory Map

```

src/
├── train\_v50.py                 # Main training entrypoint (Hydra, curriculum, checkpoints)
├── predict\_v50.py               # End-to-end inference, calibration, submission builder
│
├── models/                      # Encoders & decoders
│   ├── fgs1\_mamba.py            # FGS1 encoder (Mamba state-space model)
│   ├── airs\_gnn.py              # AIRS encoder (GNN with edge features)
│   └── multi\_scale\_decoder.py   # Joint μ/σ decoder (multi-scale, symbolic fusion ready)
│
├── data/                        # Data ingestion & loaders
│   ├── loaders.py               # Standard loaders for train/val/test splits
│   └── ariel\_mae\_dataset\_v50.py # MAE dataset with symbolic/molecular masking
│
├── losses/                      # Training objectives
│   └── symbolic\_loss.py         # GLL + symbolic constraint penalties (smoothness, priors)
│
├── symbolic/                    # Symbolic logic modules
│   ├── symbolic\_logic\_engine.py       # Rule evaluation (hard/soft, per-rule masks)
│   ├── symbolic\_violation\_predictor.py # Rule violation scoring (rule-based)
│   ├── symbolic\_violation\_predictor\_nn.py # Neural predictor of violations
│   ├── symbolic\_fusion\_predictor.py   # Ensemble symbolic violation integrator
│   ├── symbolic\_influence\_map.py      # ∂L/∂μ symbolic influence visualizer
│   └── molecular\_priors.py            # Physics-informed molecular band priors (H₂O, CO₂, CH₄…)
│
├── diagnostics/                 # Post-hoc diagnostics & visualization
│   ├── generate\_diagnostic\_summary.py # Metrics + overlays → dashboard JSON
│   ├── generate\_html\_report.py        # Full interactive diagnostics dashboard (UMAP, t-SNE, FFT, SHAP)
│   ├── plot\_umap\_v50.py               # UMAP latent visualization (symbolic overlays)
│   ├── plot\_tsne\_interactive.py       # Interactive Plotly t-SNE
│   ├── analyze\_fft\_autocorr\_mu.py     # FFT/autocorrelation on μ spectra
│   ├── spectral\_smoothness\_map.py     # Smoothness penalty diagnostics
│   ├── shap\_overlay.py                # SHAP × μ overlays
│   ├── shap\_attention\_overlay.py      # SHAP × attention fusion overlays
│   └── shap\_symbolic\_overlay.py       # SHAP × symbolic overlays
│
├── utils/                      # Shared utilities
│   ├── reproducibility.py       # Seeds, config/env hashing, DVC/lakeFS sync
│   ├── logging.py               # Rich/JSONL logging
│   ├── pipeline\_consistency\_checker.py # CI self-validation of pipeline consistency
│   └── selftest.py              # Fast/deep mode self-tests (CLI + CI)
│
└── cli/                        # Unified CLI layer (Typer)
├── cli\_core\_v50.py          # Train / predict orchestration
├── cli\_diagnose.py          # Diagnostics dashboard / symbolic analysis
├── cli\_submit.py            # Kaggle submission bundle builder
└── spectramind.py           # Root CLI entrypoint (registers all commands)

````

---

## 🔑 Design Principles

- **Encoders:**  
  • `fgs1_mamba.py` → optimized for long FGS1 time series.  
  • `airs_gnn.py` → graph encoder with wavelength/molecule/region edges.  

- **Decoders:**  
  • `multi_scale_decoder.py` predicts μ and σ jointly, integrates symbolic overlays.  

- **Losses & Symbolic:**  
  • `symbolic_loss.py` enforces smoothness, molecular priors, non-negativity, etc.  
  • Symbolic predictors score rule violations and integrate with SHAP overlays.  

- **Diagnostics:**  
  • FFT/UMAP/t-SNE/smoothness maps/SHAP overlays → aggregated into
    `generate_html_report.py` for CI and dashboard export.  

- **Reproducibility:**  
  • Hydra config snapshots, DVC data hashing, JSONL logs, `selftest.py` CI checks.  

- **CLI Integration:**  
  • Unified `spectramind.py` Typer CLI with subcommands:
    `train`, `predict`, `diagnose`, `submit`, `test`.

---

## 📊 Visual

<details>
<summary><strong>src/ system overview (Mermaid diagram)</strong></summary>

```mermaid
flowchart TD
  subgraph Data[data/]
    LDR[loaders.py]
    MAE[ariel_mae_dataset_v50.py]
  end

  subgraph Models[models/]
    FGS1[fgs1_mamba.py]
    AIRS[airs_gnn.py]
    DEC[multi_scale_decoder.py]
  end

  subgraph Losses[losses/]
    SYM[symbolic_loss.py]
  end

  subgraph Symbolic[symbolic/]
    LOGIC[symbolic_logic_engine.py]
    PRIORS[molecular_priors.py]
    VIOL[symbolic_violation_predictor*.py]
    INFL[symbolic_influence_map.py]
  end

  subgraph Diagnostics[diagnostics/]
    SUMM[generate_diagnostic_summary.py]
    HTML[generate_html_report.py]
    UMAP[plot_umap_v50.py]
    TSNE[plot_tsne_interactive.py]
    FFT[analyze_fft_autocorr_mu.py]
    SMOOTH[spectral_smoothness_map.py]
    SHAP[shap_* overlays]
  end

  subgraph Utils[utils/]
    REPRO[reproducibility.py]
    LOG[logging.py]
    CHECK[pipeline_consistency_checker.py]
    SELF[selftest.py]
  end

  subgraph CLI[cli/]
    CORE[cli_core_v50.py]
    DIAG[cli_diagnose.py]
    SUB[cli_submit.py]
    ROOT[spectramind.py]
  end

  LDR --> FGS1
  LDR --> AIRS
  FGS1 --> DEC
  AIRS --> DEC
  DEC --> SYM
  SYM --> SUMM
  SUMM --> HTML
  SUMM --> UMAP
  SUMM --> TSNE
  SUMM --> FFT
  SUMM --> SHAP
  Utils --> CLI
````

</details>

---

## ✅ Guarantees

* **Full reproducibility**: Hydra configs, DVC, Git commit hashes, env captures.
* **Diagnostics-first**: Every training/inference run emits plots + JSON summaries.
* **Challenge-ready**: Outputs μ/σ for 283 bins, packaged into Kaggle submission format.
* **NASA-grade rigor**: Physics priors (molecular bands, smoothness) enforce plausibility.

---

> This document is auto-aligned with the SpectraMind V50 master plan and will remain synced as new modules are added.

```
