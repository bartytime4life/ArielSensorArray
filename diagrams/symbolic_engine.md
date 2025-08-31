# /diagrams/symbolic_engine.md

# ♟️ Symbolic Constraint Engine — Overlay & Feedback Flow

How symbolic rules are evaluated and fused with diagnostics.

```mermaid
flowchart LR
  P0[Predictions μ,σ] --> S0[Symbolic Engine]
  M0[Metadata / Molecule Maps] --> S0
  T0[Templates (H₂O/CO₂/CH₄)] --> S0

  subgraph SYM[Symbolic Processing]
    S1[Per-rule Masks]
    S2[Violation Scores]
    S3[Aggregate Indices]
  end

  S0 --> SYM
  S1 --> O1[Overlay Maps]
  S2 --> O2[Rule Leaderboard]
  S3 --> O3[Global Score]

  O1 --> D0[Diagnostics HTML]
  O2 --> D0
  O3 --> D0

  %% Feedback loop
  O2 --> F0[Targeted Ablations]
  F0 -->|adjust loss weights / toggles| TRN[Trainer]
