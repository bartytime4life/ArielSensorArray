sequenceDiagram
  participant U as User
  participant CLI as spectramind
  participant P as Pipeline
  participant D as Diagnostics

  U->>CLI: spectramind train
  CLI->>P: Run Training
  P-->>CLI: Model + Checkpoints

  U->>CLI: spectramind predict
  CLI->>P: Generate μ/σ + Submission
  P-->>CLI: submission.csv

  U->>CLI: spectramind diagnose dashboard
  CLI->>D: Run Diagnostics
  D-->>CLI: HTML Dashboard
