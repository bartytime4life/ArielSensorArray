# /diagrams/ci_pipeline.md

# 🤖 CI/CD Smoke & Reproducibility DAG

Minimal path to validate integrity on every commit/PR.

```mermaid
flowchart LR
  G[Git Push/PR] --> A[GitHub Actions Workflow]
  A --> C0[Checkout Repo]
  A --> E0[Set up Python/Poetry]
  A --> D0[Cache/Restore DVC]
  A --> T0[Selftest / Lint / Typecheck]
  A --> S1[CI Fast Train\ntrainer=ci_fast]
  S1 --> M0[Mini Metrics]
  S1 --> O0[Artifacts: logs, snapshots]
  A --> V0[Validate Submission Structure]
  A --> H0[Publish CI Artifacts]

  V0 -->|pass| R0[Status: ✅]
  V0 -->|fail| R1[Status: ❌]
