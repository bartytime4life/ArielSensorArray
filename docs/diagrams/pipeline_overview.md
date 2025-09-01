# 🛰️ SpectraMind V50 — End-to-End Pipeline Overview

This document shows the high-level DAG of the **SpectraMind V50** pipeline — from **raw telescope inputs** to **final leaderboard package**.  
It integrates **calibration, preprocessing, training/inference, diagnostics, and submission**, emphasizing **CLI-first orchestration, Hydra-safe configs, and NASA-grade reproducibility**:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.

---

## 🌐 Pipeline DAG

```mermaid
flowchart LR
  %% ----------------------------
  %% CLI Entry
  %% ----------------------------
  subgraph CLI["CLI: spectramind (Typer + Hydra)"]
    S0["User Command"]
  end

  %% ----------------------------
  %% Calibration Stage
  %% ----------------------------
  subgraph CAL["Calibration"]
    C1["ADC Correction"]
    C2["Nonlinearity"]
    C3["Dark Subtraction"]
    C4["Flat Fielding"]
    C5["CDS / Trace Extraction"]
    C6["Normalization & Phase Align"]
  end

  %% ----------------------------
  %% Preprocess & Packaging
  %% ----------------------------
  subgraph PREP["Preprocess & Packaging"]
    P1["QC & SNR Checks"]
    P2["Feature Packaging<br/>(FGS1, AIRS, metadata)"]
  end

  %% ----------------------------
  %% Training / Inference
  %% ----------------------------
  subgraph TRAIN["Training / Inference"]
    T1["Hydra Compose<br/>(configs/*)"]
    T2["Trainer Engine<br/>(FGS1-Mamba + AIRS-GNN)"]
    T3["Checkpoints (DVC-tracked)"]
    T4["Predictions μ, σ"]
  end

  %% ----------------------------
  %% Diagnostics
  %% ----------------------------
  subgraph DIAG["Diagnostics & Reports"]
    D1["Metrics (GLL, RMSE, MAE)"]
    D2["Calibration Checks"]
    D3["Symbolic Overlays"]
    D4["UMAP/t-SNE/FFT HTML"]
    D5["HTML Dashboard Bundle"]
  end

  %% ----------------------------
  %% Submission Stage
  %% ----------------------------
  subgraph SUBMIT["Submission & Artifacts"]
    B1["Validate Submission"]
    B2["Bundle Package"]
    B3["Upload / Kaggle Leaderboard"]
  end

  %% ----------------------------
  %% Main Flow
  %% ----------------------------
  S0 --> CAL
  CAL --> PREP
  PREP --> T1 --> T2 --> T3
  T2 --> T4 --> DIAG
  DIAG --> B1 --> B2 --> B3

  %% ----------------------------
  %% Side Channels (Artifacts & Logging)
  %% ----------------------------
  T1 -->|Save resolved YAML| R1["Run Snapshot (Hydra)"]
  T2 -->|events.jsonl| L1["Structured Logs"]
  DIAG -->|HTML/PNG| R2["Diagnostics Report"]
  B2 -->|manifest.json + hashes| R3["Reproducibility Records"]
