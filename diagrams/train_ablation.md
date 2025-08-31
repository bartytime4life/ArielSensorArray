
# /diagrams/train_ablation.md

# 🧪 Training & Ablation Execution Architecture

Ablation orchestration from CLI → Hydra → Trainer → Leaderboard.

```mermaid
flowchart TD
  U[User] -->|spectramind ablate ...| E0[Typer Entrypoint]
  E0 --> H0[Hydra Compose\n(train/ablation.yaml + overrides)]
  H0 --> A0[Ablation Engine]
  A0 -->|generate run grid| A1{Configs i=1..N}

  subgraph RUN[Parallel/Sequential Runs]
    direction TB
    R1[Trainer(i)] --> R2[Metrics/Artifacts(i)] --> R3[Predictions μ,σ(i)]
  end

  A1 --> RUN
  R2 --> D0[Diagnostics Collation]
  R3 --> D0
  D0 --> L0[Leaderboard\nMD+HTML]
  R2 --> L1[events.jsonl]
  H0 --> S0[Save Resolved Configs]
