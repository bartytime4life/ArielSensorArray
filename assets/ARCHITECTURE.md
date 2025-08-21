🛰️ SpectraMind V50 — Architecture (Assets Directory)

This document provides a **visual and structural reference** for the SpectraMind V50 pipeline, CLI, and data flows.  
It is designed to work seamlessly with Mermaid diagrams, GitHub Actions (`mermaid-export.yml`), and the diagnostics dashboard.

---

## 🎯 Mission Context

- **Input:** Ariel **FGS1/AIRS** frames (simulated telescope data)  
- **Pipeline:** Calibration → Training → Prediction → Diagnostics → Submission  
- **Output:** Mean (μ) and Uncertainty (σ) spectra across 283 bins  
- **Guarantees:** Full reproducibility (Hydra configs, DVC, CI workflows, Kaggle integration)

---

## 📂 Repository Components

- **`src/`** → Core modeling code (FGS1 Mamba encoder, AIRS GNN, decoders, symbolic logic)  
- **`configs/`** → Hydra YAML configs (data, model, training, diagnostics)  
- **`data/`** → DVC-tracked datasets (raw, processed, intermediate)  
- **`assets/`** → Dashboards, plots, and documentation (this directory)  
- **`bin/`** → Shell utilities (diagnose.sh, push.sh, kaggle-submit.sh, etc.)  
- **`logs/`** → Append-only log streams (`v50_debug_log.md`, JSONL traces)  
- **`.github/workflows/`** → CI/CD (diagnostics, lint, Kaggle submit, Mermaid export)  

---

## 📊 Pipeline Flow (Mermaid Diagram)

```mermaid
flowchart TD
    A[FGS1 Frames] -->|Calibration| B[Calibrated Lightcurves]
    A2[AIRS Frames] -->|Calibration| B
    B --> C[Encoders: FGS1 Mamba + AIRS GNN]
    C --> D[Multi-Scale Decoders]
    D --> E[μ Spectrum]
    D --> F[σ Uncertainty]
    E --> G[Diagnostics: GLL, FFT, Smoothness]
    F --> G
    G --> H[Symbolic Logic Engine]
    H --> I[Diagnostics Dashboard (HTML)]
    I --> J[Submission Bundle (Kaggle-ready)]


⸻

⚙️ CLI Architecture

graph TD
    CLI[Unified CLI: spectramind.py]
    CLI --> A1[selftest]
    CLI --> A2[calibrate]
    CLI --> A3[train]
    CLI --> A4[predict]
    CLI --> A5[corel-train]
    CLI --> A6[diagnose]
    CLI --> A7[submit]
    CLI --> A8[analyze-log]
    CLI --> A9[check-cli-map]

    A6 --> D1[dashboard]
    A6 --> D2[symbolic-rank]
    A6 --> D3[smoothness]
    A6 --> D4[cluster-overlay]


⸻

🔬 Symbolic + Explainability System

flowchart LR
    MU[μ Spectrum] -->|∂L/∂μ| SIM[Symbolic Influence Map]
    MU -->|SHAP| SHAP[SHAP Overlay]
    MU -->|FFT| FFT[FFT + Autocorr Analysis]

    SIM --> FUSION[Symbolic Fusion Diagnostics]
    SHAP --> FUSION
    FFT --> FUSION

    FUSION --> DASH[Diagnostics Dashboard]


⸻

📈 Data Flow & Reproducibility

sequenceDiagram
    participant User
    participant CLI as spectramind CLI
    participant Hydra as Hydra Config
    participant DVC as DVC
    participant Pipeline as V50 Pipeline
    participant Dashboard as Diagnostics Dashboard

    User->>CLI: spectramind train +overrides
    CLI->>Hydra: Load config_v50.yaml
    CLI->>DVC: Fetch correct dataset snapshot
    CLI->>Pipeline: Run calibration + training
    Pipeline-->>CLI: μ/σ outputs + logs
    CLI->>Dashboard: Generate HTML (report.html)
    Dashboard-->>User: View diagnostics + symbolic overlays


⸻

🧩 Integration Notes
	•	Hydra Configs: Every run is fully parameterized (configs/config_v50.yaml + overrides)
	•	DVC: Ensures reproducible datasets and model artifacts
	•	CI Workflows: Run diagnostics, lint, e2e smoke tests, and Mermaid exports automatically
	•	Assets: This directory provides placeholders so dashboards and reports never break pre-run

⸻

✅ Next Steps
	•	Run spectramind diagnose dashboard to regenerate live dashboards
	•	Update mermaid/*.mmd to extend diagrams
	•	Push changes → GitHub Actions will auto-export .svg and .png versions into artifacts

⸻

SpectraMind V50 — Ariel Data Challenge 2025
Neuro-symbolic, physics-informed AI pipeline for exoplanet spectroscopy

---
