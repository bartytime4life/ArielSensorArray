
---

# SpectraMind V50 — Master Architecture & Scientific Design

**Neuro‑symbolic, physics‑informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** Deliver a reproducible, explainable, physics‑informed system that ingests Ariel FGS1/AIRS cubes, outputs μ and σ for 283 bins, applies calibration and diagnostics (including **microlensing** corrections), and packages a competition‑valid submission — all within Kaggle’s \~9‑hour runtime envelope.

---

## 0) Purpose & Scope

* **Engineers:** modules, contracts, workflows, budgets, acceptance criteria
* **Scientists:** transit physics, radiative transfer, microlensing handling
* **Ops/MLOps:** CLI‑first, Hydra config, DVC, CI/CD, audit logging

---

## 1) System Overview (Components)

```mermaid
flowchart LR
  subgraph Ingest["Data Ingest"]
    RAW[FGS1/AIRS raw cubes]
  end

  subgraph Calib["Calibration Kill Chain"]
    A1[ADC/Bias/Dark]
    A2[Dead-Pixel Map]
    A3[Flat / Nonlinearity]
    A4[Trace Extraction]
    A5[Wavelength Alignment]
    A6[Normalization]
    ML[Microlensing Pre-Check<br/>• Detect achromatic lift<br/>• Fit Paczyński-like curve<br/>• Divide-out correction<br/>• Chromaticity tests]
  end

  subgraph Model["Modeling Core"]
    FGS1[FGS1 Encoder: Mamba SSM]
    AIRS[AIRS Encoder: GNN<br/>Edges: λ-adjacency, molecules, detector region, achromatic factor node]
    DEC1[μ Head]
    DEC2[σ Head (Temp Scaling + COREL)]
  end

  subgraph Symb["Symbolic Physics Layer"]
    S1[Smoothness (λ)]
    S2[Nonnegativity]
    S3[Limb-Darkening Consistency]
    S4[FFT Jitter Suppression]
    S5[Achromatic vs Chromatic<br/>(Microlens Guard)]
    S6[Radiative Transfer Alignment]
  end

  subgraph Diag["Diagnostics & Explainability"]
    D1[GLL/Entropy/ RMSE]
    D2[SHAP (FGS1 time / AIRS λ)]
    D3[UMAP / t‑SNE Latents]
    D4[FFT Residual Spectra]
    D5[Microlens Audit Panel]
    HTML[Versioned HTML Dashboard]
  end

  subgraph Ops["CLI + Config + Logging"]
    CLI[Typer CLI]
    HYD[Hydra Configs]
    DVC[(DVC/lakeFS)]
    LOG[v50_debug_log.md<br/>JSONL Event Log]
    CI[GitHub Actions CI]
  end

  RAW --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
  A6 --> ML --> FGS1
  A6 --> AIRS
  FGS1 --> DEC1
  FGS1 --> DEC2
  AIRS --> DEC1
  AIRS --> DEC2
  DEC1 -->|losses + penalties| Symb
  DEC2 -->|calibration feedback| Symb
  DEC1 --> D1
  DEC2 --> D1
  DEC1 --> D2
  FGS1 --> D2
  AIRS --> D2
  DEC1 --> D3
  D1 --> D4
  A6 --> D5
  D1 --> HTML
  D2 --> HTML
  D3 --> HTML
  D4 --> HTML
  D5 --> HTML

  CLI --- HYD --- DVC --- LOG --- CI
  CLI --> Ingest
  CLI --> Calib
  CLI --> Model
  CLI --> Symb
  CLI --> Diag
```

---

## 2) Calibration Kill Chain (with Microlensing)

```mermaid
flowchart TD
  R[Raw FGS1/AIRS] --> B[ADC/Bias/Dark]
  B --> DPM[Dead‑Pixel Map/Repair]
  DPM --> FLAT[Flat‑Field / Nonlinearity]
  FLAT --> TRACE[Trace Extraction]
  TRACE --> WALIGN[Wavelength Alignment]
  WALIGN --> NORM[Normalization]
  
  NORM -->|FGS1 Baseline + AIRS Continua| ML[Microlensing Pre‑Check]
  ML -->|Achromatic Detected?| CHK{Achromatic?}
  CHK -- "Yes" --> FIT[Fit Paczyński‑like Magnification μ(t)]
  FIT --> DIV[Divide Out μ(t)]
  DIV --> CHROM[Chromaticity Tests on Bands]
  CHROM --> OK{Pass?}
  OK -- "Yes" --> OUT1[Calibrated Lightcurves/Spectra]
  OK -- "No" --> FLAG[Flag for Review<br/>/Robust fitter variant]

  CHK -- "No" --> OUT1
```

*Outputs feed modeling; microlens audit stores fit parameters & QA plots for diagnostics.*

---

## 3) Modeling Core (Encoders & Decoders)

```mermaid
flowchart LR
  subgraph Encoders
    F[FGS1 Sequence<br/>(Long time series)] --> Mamba[Mamba SSM]
    A[AIRS Spectrum (283 bins)] --> GNN[GNN Encoder<br/>λ-adjacency + molecular + region + achromatic factor]
  end

  subgraph Decoders
    Mamba --> MU[μ Head]
    GNN --> MU
    Mamba --> SIG[σ Head (Temp Scaling)]
    GNN --> SIG
    SIG --> COREL[COREL Conformal Calibrator]
  end

  MU -.-> LOSS[Gaussian Log-Likelihood]
  SIG -.-> LOSS
  LOSS --> PEN[Symbolic Penalties]

  classDef enc fill:#eef,stroke:#557;
  classDef dec fill:#efe,stroke:#575;
  class Mamba,GNN enc
  class MU,SIG,COREL dec
```

---

## 4) Symbolic Physics Layer (Constraint Routing)

```mermaid
flowchart TB
  MU[Predicted μ] --> SMOOTH[Smoothness(λ)]
  MU --> POS[Nonnegativity]
  MU --> LD[Limb-Darkening Consistency]
  MU --> RT[RT Alignment (Voigt-like)]
  MU --> ACHR[Microlensing Guard<br/>(Achromatic vs Chromatic)]
  FGS1[FGS1 Residuals] --> FFTS[FFT Jitter Suppression]

  SMOOTH --> PEN[Weighted Penalty Sum]
  POS --> PEN
  LD --> PEN
  RT --> PEN
  ACHR --> PEN
  FFTS --> PEN

  PEN --> BACK[Backprop to Encoders/Decoders]
```

---

## 5) Diagnostics & HTML Dashboard

```mermaid
flowchart LR
  MU[μ] --> GLL[GLL/Entropy/RMSE Maps]
  SIG[σ] --> GLL
  ENC[Encoder Latents] --> UMAP[UMAP/t‑SNE Plots]
  SHAPF[SHAP (FGS1 time)] --> HTML
  SHAPA[SHAP (AIRS λ)] --> HTML
  FFT[FFT of Residuals] --> HTML
  MLC[Microlens Audit: fits, residuals, chromaticity] --> HTML
  GLL --> HTML
  UMAP --> HTML
```

---

## 6) Reproducibility & CI

```mermaid
flowchart LR
  DEV[Developer CLI Call] --> HYD[Hydra Compose]
  HYD --> RUN[Run with Resolved Config]
  RUN --> LOG[v50_debug_log.md + JSONL]
  RUN --> ART[Artifacts (DVC)]
  ART --> DVC[(DVC/lakeFS Remote)]
  LOG -. includes .-> HASH[Git SHA + Config Hash]
  CI[GitHub Actions] --> TEST[Selftest + Smoke Pipelines]
  TEST --> OK{Pass?}
  OK -- Yes --> MERGE[Merge to main]
  OK -- No --> FIX[Iterate & Fix]
```

---

## 7) Data Flow (Summary)

```mermaid
flowchart LR
  RAW[Raw Cubes] --> CAL[Calibration + Microlens]
  CAL --> ENC[Encoders (Mamba/GNN)]
  ENC --> DEC[μ/σ Heads + COREL]
  DEC --> SUB[Submission CSV/NPZ/ZIP]
  DEC --> DIAG[Diagnostics]
  DIAG --> HTML[HTML Dashboard]
```

---

## 8) Acceptance Criteria

* **Scientific:** physically plausible μ/σ; uncertainty coverage calibrated; microlensing explicitly handled
* **Reproducibility:** submissions tied to Git SHA + config hash + DVC dataset
* **Explainability:** SHAP + symbolic overlays + microlens audit in reports
* **Efficiency:** ≤9 h runtime; diagnostics ≤1 h
* **Quality Gate:** `spectramind selftest` + CI must pass

---

## 9) Roadmap

* TorchScript/JIT inference
* Expanded symbolic influence maps
* GUI Dashboard (React + FastAPI)
* Automated ablations & leaderboard export
* Bayesian joint transit + microlens fit (hard cases)

---

**Status:** V50 architecture frozen for NeurIPS 2025 competition.

---

### Notes

* These Mermaid blocks render on GitHub, GitLab, and modern Markdown viewers.
* If you want **SVG exports** for reports, I can add a tiny script to auto‑render diagrams to `docs/` during CI.
