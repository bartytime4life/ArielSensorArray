# docs/diagrams/calibration_chain.md

# 🔧 Calibration Kill Chain

Instrumental corrections prior to modeling.

```mermaid
flowchart TD
  R0[Raw Frames] --> A0[ADC Correction]
  A0 --> N0[Nonlinearity Correction]
  N0 --> D0[Dark Subtraction]
  D0 --> F0[Flat Fielding]
  F0 --> C0[CDS / Bias Removal]
  C0 --> T0[Trace Extraction]
  T0 --> Z0[Normalization & Phase Alignment]
  Z0 --> Q0[QC / Flags]
  Q0 --> PKG[Package Science-ready Batches]
