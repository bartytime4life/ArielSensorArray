graph TD
  A[FGS1 / AIRS Raw Frames] --> B[Calibration Kill Chain]
  B --> C[Dual Encoder Modeling]
  C --> D[Decoders μ/σ]
  D --> E[Uncertainty Calibration]
  E --> F[Diagnostics & Explainability]
  F --> G[Leaderboard-Ready Submission]
