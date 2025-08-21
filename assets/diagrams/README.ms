Here’s the next file: assets/diagrams/pipeline_overview.mmd

%% --------------------------------------------------------------------
%% SpectraMind V50 — Pipeline Overview (Mermaid)
%% Canonical source for the top-level flow diagram used in ARCHITECTURE.md,
%% report.html, and diagnostics_dashboard.html.
%% --------------------------------------------------------------------

flowchart TD
  %% Direction
  %% Top-down keeps the dataflow intuitive from ingestion → submission.
  %% Subgraphs help group pipeline stages for readability.
  %% IDs are stable to keep diffs clean.

  %% =========================
  %% Ingestion & Metadata
  %% =========================
  subgraph ING[Ingestion]
    A1[FGS1 Frames (time×32×32)]:::io
    A2[AIRS Frames (time×32×356)]:::io
    A3[Metadata (planet, star, obs)]:::meta
  end

  %% =========================
  %% Calibration & Photometry
  %% =========================
  subgraph CAL[Calibration & Photometry]
    direction TB
    C0[[Start Calibration]]:::op
    C1[Bias/Dark/Flat/CDS]:::op
    C2[Trace Extraction / Aperture Photometry]:::op
    C3[Background / Jitter Correction]:::op
    C4[Normalization & Phase Alignment]:::op
    C5[(Calibrated Light Curves)]:::data
  end

  %% Wiring ingestion → calibration
  A1 --> C0
  A2 --> C0
  A3 --> C0
  C0 --> C1 --> C2 --> C3 --> C4 --> C5

  %% =========================
  %% Modeling (Encoders → Decoders)
  %% =========================
  subgraph MOD[Modeling]
    direction TB

    %% Encoders
    M0[[Feature Build]]:::op
    M1[FGS1 Encoder<br/>Mamba SSM]:::model
    M2[AIRS Encoder<br/>GNN (wavelength graph)]:::model

    %% Fusion & Decoders
    M3[Latent Fusion]:::model
    M4[μ Decoder (mean spectrum)]:::model
    M5[σ Decoder (uncertainty)]:::model

    %% Outputs
    M6[(Predicted μ, σ)]:::data
  end

  %% Wiring calibration → modeling
  C5 --> M0 --> M1
  C5 --> M2
  M1 --> M3
  M2 --> M3
  M3 --> M4
  M3 --> M5
  M4 --> M6
  M5 --> M6

  %% =========================
  %% Uncertainty Calibration
  %% =========================
  subgraph UQ[Uncertainty Calibration]
    direction TB
    U1[Temperature Scaling]:::calib
    U2[COREL Conformal GNN<br/>(binwise coverage)]:::calib
    U3[(Calibrated μ, σ)]:::data
  end

  M6 --> U1 --> U2 --> U3

  %% =========================
  %% Diagnostics & Explainability
  %% =========================
  subgraph DIA[Diagnostics & Explainability]
    direction TB
    D0[[Start Diagnostics]]:::op
    D1[GLL / RMSE / MAE Metrics]:::metric
    D2[FFT / Smoothness / Asymmetry]:::metric
    D3[SHAP / Attention Traces]:::xai
    D4[Symbolic Logic Overlays<br/>(rules, violations)]:::symbolic
    D5[Latent Projections (UMAP / t‑SNE)]:::viz
    D6[HTML Report & Dashboard]:::viz
    D7[(diagnostic_summary.json)]:::data
  end

  U3 --> D0
  D0 --> D1 --> D2 --> D3 --> D4 --> D5 --> D6 --> D7

  %% =========================
  %% Packaging & Submission
  %% =========================
  subgraph PKG[Packaging & Submission]
    direction TB
    P1[Validator (shape, bins, coverage)]:::op
    P2[Bundle (CSV/ZIP + Report)]:::op
    P3[(Kaggle Submission Artifact)]:::io
  end

  U3 --> P1
  D7 --> P1
  P1 --> P2 --> P3

  %% =========================
  %% Reproducibility & Ops
  %% =========================
  subgraph OPS[Reproducibility & Ops]
    direction TB
    O1[(Hydra Configs<br/>configs/**.yaml)]:::cfg
    O2[(DVC-tracked Data / Models)]:::dvc
    O3[(logs/v50_debug_log.md)]:::log
    O4[(events.jsonl)]:::log
    O5[Typer CLI<br/>spectramind ...]:::cli
    O6[CI (GitHub Actions)<br/>mermaid export / tests]:::ci
  end

  %% Global wiring to Ops artefacts
  A1 & A2 & A3 --> O2
  C5 --> O2
  M6 --> O2
  U3 --> O2
  D7 --> O2
  D6 --> O2

  O1 -. config-> C0
  O1 -. config-> M0
  O1 -. config-> U1
  O1 -. config-> D0
  O5 -. invokes -> C0
  O5 -. invokes -> M0
  O5 -. invokes -> D0
  O5 -. invokes -> P1
  O6 -. runs -> O5
  C0 --> O3
  M0 --> O3
  D0 --> O3
  P1 --> O3
  C0 --> O4
  M0 --> O4
  D0 --> O4
  P1 --> O4

  %% =========================
  %% Class Definitions (Styling)
  %% =========================
  classDef io fill:#eef7ff,stroke:#5b9bd5,stroke-width:1px,color:#0b305b;
  classDef meta fill:#fff7e6,stroke:#f0a500,stroke-width:1px,color:#7a4a00;
  classDef data fill:#f3f3f3,stroke:#9e9e9e,stroke-width:1px,color:#333;
  classDef op fill:#e8f5e9,stroke:#43a047,stroke-width:1px,color:#1b5e20;
  classDef model fill:#ede7f6,stroke:#7e57c2,stroke-width:1px,color:#4527a0;
  classDef calib fill:#ffebee,stroke:#e53935,stroke-width:1px,color:#b71c1c;
  classDef metric fill:#e3f2fd,stroke:#1e88e5,stroke-width:1px,color:#0d47a1;
  classDef xai fill:#fff3e0,stroke:#fb8c00,stroke-width:1px,color:#e65100;
  classDef symbolic fill:#fce4ec,stroke:#d81b60,stroke-width:1px,color:#880e4f;
  classDef viz fill:#e0f7fa,stroke:#00acc1,stroke-width:1px,color:#006064;
  classDef cfg fill:#f1f8e9,stroke:#7cb342,stroke-width:1px,color:#33691e;
  classDef dvc fill:#f5f5f5,stroke:#616161,stroke-width:1px,color:#212121;
  classDef log fill:#fffde7,stroke:#fdd835,stroke-width:1px,color:#8d6e00;
  classDef cli fill:#e0e0e0,stroke:#455a64,stroke-width:1px,color:#263238;
  classDef ci fill:#edeff7,stroke:#5c6bc0,stroke-width:1px,color:#283593;
