# 🧭 SpectraMind V50 — Diagrams Index

This page links the core architecture diagrams for the SpectraMind V50 project.  
Each document is **GitHub-ready** and uses Mermaid flowcharts with conservative (ASCII-only) labels to avoid renderer issues.

---

## 📂 Diagram set

1. **CLI Flow**  
   Path: `docs/diagrams/cli-flow.md`  
   What it shows: unified `spectramind` command → subcommands → Hydra compose → run → logs/artifacts.  
   Includes examples for `calibrate`, `train`, `diagnose`, `submit`, `ablate`, `test`, `analyze-log`.

2. **Training & Ablation Execution Architecture**  
   Path: `docs/diagrams/train_ablation.md`  
   What it shows: CLI → Hydra → Ablation Engine → parallel/sequential runs → diagnostics → leaderboard.  
   Outputs: `events.jsonl`, resolved config snapshots, artifacts dir, leaderboard (MD/HTML).

3. **Symbolic Engine Flow**  
   Path: `docs/diagrams/symbolic_engine.md`  
   What it shows: rules parsing → mask/operator build → evaluation on `mu`/optional `sigma` → aggregation, influence maps → reports.  
   Integrations: training loss terms, diagnostics, calibration.

4. **Data & Calibration (optional, if present)**  
   Path: `docs/diagrams/data_calibration.md`  
   What it shows: raw inputs → calibration stages → normalized artifacts for training and diagnostics.

5. **Diagnostics & Reports (optional, if present)**  
   Path: `docs/diagrams/diagnostics.md`  
   What it shows: metrics collation (GLL, RMSE, entropy), plots (heatmaps, overlays), HTML/MD bundling.

---

## 🔧 Renderer-safe Mermaid snippet (reference)

Use this template when adding new diagrams (ASCII labels; no `<br/>` or `\n` inside node text):

```mermaid
flowchart LR;
  A[Start];
  B{Decision};
  C[Step yes];
  D[Step no];
  E[(Artifacts)];
  A --> B;
  B -- yes --> C;
  B -- no  --> D;
  C --> E;
  D --> E;

````

**Tips**

* Keep node text short and ASCII-only.
* Avoid HTML and escape sequences inside node labels.
* Prefer spaces and hyphens instead of symbols like `&` or `/`.

---

## ✅ Contribution checklist for new diagrams

* [ ] File placed under `docs/diagrams/`
* [ ] Title and short intro at top of the file
* [ ] Mermaid fenced block: ` ```mermaid … ``` `
* [ ] ASCII-only labels in nodes and edges
* [ ] No line breaks inside labels; split into multiple nodes if needed
* [ ] Brief section with CLI examples, if applicable
* [ ] Link added to this index (`docs/diagrams/main.md`)

---

## 🧩 Cross-references

* **Hydra usage and config structure** — see project configs (`/configs`) and Hydra guidance.
* **CLI design and UX** — unified Typer CLI with structured logs and artifacts.
* **Reproducibility** — DVC for data/models, saved resolved configs, CI hooks.

---

```
```
