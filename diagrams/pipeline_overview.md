# /diagrams/pipeline_overview.md

# 🛰️ SpectraMind V50 — End-to-End Pipeline Overview

The high-level DAG from raw inputs to final leaderboard package.

```mermaid
flowchart LR
  subgraph CLI[CLI: spectramind]
    S0[User Command]
  end

  subgraph CAL[Calibration]
    C1[ADC Correction]
    C2[Nonlinearity]
    C3[Dark Subtraction]
    C4[Flat Fielding]
    C5[CDS / Trace Extraction]
    C6[Normalization & Phase Align]
  end

  subgraph PREP[Preprocess & Packaging]
    P1[QC & SNR Checks]
    P2[Feature Packaging\n(AIRS, FGS1, metadata)]
  end

  subgraph TRAIN[Training / Inference]
    T1[Hydra Compose\n(configs/*)]
    T2[Trainer Engine]
    T3[Checkpoints]
    T4[Predictions μ, σ]
  end

  subgraph DIAG[Diagnostics & Reports]
    D1[Metrics (GLL/RMSE)]
    D2[Calibration Checks]
    D3[Symbolic Overlays]
    D4[UMAP/t-SNE HTML]
  end

  subgraph SUBMIT[Submission & Artifacts]
    B1[Validate Submission]
    B2[Bundle Package]
    B3[Upload/Leaderboard]
  end

  S0 --> CAL
  CAL --> PREP
  PREP --> T1 --> T2 --> T3
  T2 --> T4 --> DIAG
  DIAG --> B1 --> B2 --> B3

  %% Side channels
  T1 -->|Save resolved YAML| R1[[Run Snapshot]]
  T2 -->|events.jsonl| L1[[Structured Logs]]
  DIAG -->|HTML/PNG| R2[[Diagnostics Report]]
