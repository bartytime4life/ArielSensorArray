# 🖼️ `/diagrams` — Visual Architecture & Execution Flows

## 0) Purpose & Scope

The **`/diagrams`** directory contains **Mermaid-based diagrams** and other visual assets  
that document the **architecture, data flows, and execution DAGs** of the SpectraMind V50 system  
(NeurIPS 2025 Ariel Data Challenge).  

These diagrams provide **visual transparency** of the pipeline — from calibration, preprocessing,  
and training, through symbolic diagnostics, ablations, and leaderboard reporting:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 1) Why Mermaid?

GitHub natively renders **Mermaid diagrams** embedded in Markdown files,  
allowing us to keep diagrams as **code-as-documentation**.  

* ✅ **Version-controlled** (diagrams evolve alongside configs/code).  
* ✅ **Self-updating** (edit text, GitHub re-renders automatically).  
* ✅ **Multi-format** (flowcharts, sequence diagrams, DAGs, timelines, mindmaps):contentReference[oaicite:2]{index=2}.  
* ✅ **Lightweight** (no binary images checked in, all SVG-rendered).  

---

## 2) Directory Contents

```

/diagrams/
├── pipeline\_overview\.md      # End-to-end DAG from raw data → predictions
├── train\_ablation.md         # Training & ablation config execution DAG
├── ci\_pipeline.md            # CI/CD workflow + smoke test DAG
├── symbolic\_engine.md        # Symbolic constraints & violation mapping
├── calibration\_chain.md      # Calibration kill chain (ADC → CDS → dark → flat → normalize)
├── diagnostics\_dashboard.md  # UMAP/t-SNE + HTML diagnostics report flow
└── README.md                 # You are here

````

Each `.md` file embeds **Mermaid diagrams** inside fenced code blocks  
that render automatically when viewed on GitHub.

---

## 3) Example Diagram Snippets

### Pipeline DAG

```mermaid
flowchart LR
  A0[CLI: spectramind] --> A1[Calibrate]
  A1 --> A2[Preprocess]
  A2 --> A3[Train]
  A3 --> A4[Diagnostics]
  A4 --> A5[Submit Leaderboard Package]
````

### Sequence View (Training Ablation)

```mermaid
sequenceDiagram
  participant U as User
  participant CLI as spectramind
  participant HY as Hydra
  participant TR as Trainer
  participant DG as Diagnostics

  U->>CLI: spectramind ablate ...
  CLI->>HY: compose config
  HY-->>CLI: resolved train/ablation.yaml
  CLI->>TR: run training jobs
  TR-->>DG: metrics, artifacts
  DG-->>U: leaderboard + reports
```

---

## 4) Workflow Integration

* **Configs ↔ Diagrams**: Every `configs/*` architecture doc references a diagram here.
* **CI/CD**: CI workflows validate diagram syntax (Mermaid blocks must render).
* **Docs**: Final exported diagrams are embedded in project docs & reports.

---

## 5) Contribution Guide

1. Add new diagrams as `.md` files in this folder.
2. Always embed diagrams inside fenced code blocks using ` ```mermaid ` syntax.
3. Keep diagrams small, modular, and labeled.
4. Update related `ARCHITECTURE.md` docs to link the new diagram.
5. Test rendering on GitHub before merging.

---

## 6) References

* GitHub Docs: [Creating diagrams with Mermaid](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-diagrams)&#x20;
* SpectraMind V50 Technical Plan
* SpectraMind V50 Project Analysis

---

```
