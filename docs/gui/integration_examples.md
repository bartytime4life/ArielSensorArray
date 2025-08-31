# 🧩 GUI Integration Examples — SpectraMind V50

This guide shows **practical, end-to-end** examples of how a thin GUI wraps the **CLI-first** SpectraMind V50 pipeline, and then **renders artifacts** (JSON/HTML/plots/logs). Every example preserves **Hydra configs**, **Typer CLI**, and **audit logs** to maintain NASA-grade reproducibility:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 0) Legend (What each box means)

```mermaid
flowchart LR
  subgraph Legend
    A[GUI Action]:::gui --> B[CLI Command]:::cli
    B --> C[Hydra Configs (configs/*.yaml)]:::cfg
    B --> D[Artifacts (JSON/HTML/plots)]:::art
    B --> E[Logs (v50_debug_log.md)]:::log
    D --> F[GUI Rendering]:::view
  end
classDef gui fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
classDef cli fill:#ede7f6,stroke:#5e35b1,color:#311b92;
classDef cfg fill:#fff3e0,stroke:#ef6c00,color:#e65100;
classDef art fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
classDef log fill:#fce4ec,stroke:#ad1457,color:#880e4f;
classDef view fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;

    CLI-first guarantees discoverability (--help), composition, and scripting

.

GUI is a thin shell: no hidden state; only reflects configs + artifacts

    .

1) Diagnose Dashboard (UMAP + Symbolic)

flowchart LR
  A[GUI: Toggle UMAP + Symbolic]:::gui --> B[CLI: spectramind diagnose dashboard \n diagnostics.umap.enabled=true symbolic.show=true]:::cli
  B --> C[Hydra: configs/diagnostics/*.yaml]:::cfg
  B --> D[Artifacts: outputs/diag_vX/diagnostic_summary.json \n outputs/diag_vX/diagnostic_report_vX.html \n outputs/diag_vX/plots/*.png]:::art
  B --> E[logs/v50_debug_log.md]:::log
  D --> F[GUI: Embed HTML + charts + tables]:::view

Serialized CLI

spectramind diagnose dashboard \
  diagnostics.umap.enabled=true \
  symbolic.show=true \
  --outputs.dir outputs/diag_vX

2) Training Run with Config Overrides

flowchart LR
  A[GUI: Set epochs=50, batch=64]:::gui --> B[CLI: spectramind train trainer.epochs=50 trainer.batch_size=64]:::cli
  B --> C[Hydra: configs/train.yaml + configs/trainer/*.yaml]:::cfg
  B --> D[Artifacts: outputs/train_vY/metrics.json \n checkpoints/*.pt]:::art
  B --> E[logs/v50_debug_log.md + events.jsonl]:::log
  D --> F[GUI: Live metric charts + checkpoint table]:::view

Serialized CLI

spectramind train trainer.epochs=50 trainer.batch_size=64

3) Calibration → Training → Diagnostics (Chained)

sequenceDiagram
    participant GUI as GUI
    participant CLI as spectramind
    participant HYD as Hydra Configs
    participant ART as Artifacts
    participant LOG as Logs

    GUI->>CLI: calibrate data=nominal
    CLI->>HYD: Compose configs/data/nominal.yaml
    CLI->>ART: outputs/calib_vZ/*.npy
    CLI->>LOG: Append CLI call + config hash

    GUI->>CLI: train model=v50 trainer.gpu=true
    CLI->>HYD: Compose configs/model/v50.yaml + trainer/*
    CLI->>ART: checkpoints/*.pt, metrics.json
    CLI->>LOG: Append CLI call + metrics

    GUI->>CLI: diagnose dashboard
    CLI->>HYD: Compose configs/diagnostics/*
    CLI->>ART: diagnostic_summary.json, report.html
    CLI->>LOG: Append CLI call + summary

    GUI->>GUI: Render report.html + plots + tables

4) Streamlit Wrapper Pattern (Prototype)

# Pseudocode
cmd = ["spectramind","diagnose","dashboard","diagnostics.umap.enabled=true","symbolic.show=true","--outputs.dir", outputs_dir]
run_subprocess(cmd)     # stdout/stderr panel
html = read_text(find("outputs/.../diagnostic_report*.html"))
st.components.v1.html(html, height=900, scrolling=True)
json = json_load("outputs/.../diagnostic_summary.json")
st.dataframe(flatten(json))
log = tail("logs/v50_debug_log.md", 50000)
st.code(log)

Streamlit/Gradio are ideal for rapid, Python-native prototypes in research/Kaggle contexts

.
5) React + FastAPI Contracts (Team Dashboard)

flowchart LR
  A[React UI]:::gui -->|POST /api/run| B[FastAPI]:::cli
  B -->|subprocess| C[spectramind …]:::cli
  C --> D[Hydra configs]:::cfg
  C --> E[Artifacts JSON/HTML/plots]:::art
  C --> F[Logs]:::log
  A -->|GET /api/artifacts| B
  B -->|Serve| A

React’s declarative model makes it easy to render diagnostics as “UI = function(state)”

; state = artifacts JSON.
6) Qt/PySide (Offline Mission Control)

flowchart LR
  A[Qt Button: Diagnose]:::gui --> B[QProcess: spectramind diagnose dashboard]:::cli
  B --> C[Hydra]:::cfg
  B --> D[Artifacts]:::art
  D --> E[QWebEngineView: report.html]:::view
  B --> F[Append logs]:::log

Qt signals/slots (Observer) + QProcess keep GUI thin and reproducible

.
7) Pattern Checklist

GUI action → CLI command (serialized in log)

Parameters only via Hydra configs/overrides

Artifacts rendered as-is (no mutation)

Run hashes & logs captured for audit

Same contract across Streamlit/React/Qt

Why thin GUIs?
They preserve CLI discoverability and Hydra composition, visualize artifacts without hidden state, and map naturally to event-driven & declarative UIs
.
