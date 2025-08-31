# 🧩 SpectraMind V50 — GUI Components Architecture

## 0) Purpose & Scope

This document defines the **architecture of the React component layer** under `src/gui/components/`.  
It maps **atomic → structural → visualization → composite** usage, explains integration with the **CLI-first pipeline**, and provides Mermaid diagrams for clarity.

The GUI layer is **optional and thin**: all scientific functionality lives in CLI + Hydra configs:contentReference[oaicite:5]{index=5}.  
Components here **only render CLI-produced artifacts** (JSON, PNG, HTML) and provide ergonomic composition for dashboards, reports, and teaching:contentReference[oaicite:6]{index=6}.

---

## 1) Directory Map

```

src/gui/components/
├─ Button.tsx
├─ Input.tsx
├─ Select.tsx
├─ Tooltip.tsx
├─ Loader.tsx
├─ Tabs.tsx
├─ Modal.tsx
├─ Card.tsx
├─ Panel.tsx
├─ Chart.tsx
├─ Table.tsx
├─ index.ts
├─ \*.test.tsx               # Jest/RTL unit tests
├─ README.md
└─ ARCHITECTURE.md          # ← you are here

````

---

## 2) Component Taxonomy

* **Atomic primitives** — Button, Input, Select, Tooltip, Loader  
* **Structural primitives** — Tabs, Modal, Card, Panel  
* **Data visualization** — Chart, Table  
* **Composite usage** — Panels and Cards composed in app routes (`src/gui/app/diagnostics.tsx`, `reports.tsx`)

---

## 3) Component Hierarchy Diagram

```mermaid
%%--------------------------------------------------------------------
%% SpectraMind V50 — GUI Components Hierarchy
%%--------------------------------------------------------------------
flowchart TD
  subgraph Atomic [Atomic Primitives]
    BTN[Button.tsx]
    INP[Input.tsx]
    SEL[Select.tsx]
    TIP[Tooltip.tsx]
    LDR[Loader.tsx]
  end

  subgraph Structural [Structural Primitives]
    TABS[Tabs.tsx]
    MODAL[Modal.tsx]
    CARD[Card.tsx]
    PANEL[Panel.tsx]
  end

  subgraph DataViz [Data Visualization]
    CHART[Chart.tsx]
    TABLE[Table.tsx]
  end

  BTN --> MODAL
  INP --> PANEL
  SEL --> PANEL
  TIP --> CARD
  LDR --> CARD
  TABS --> PANEL
  CARD --> CHART
  CARD --> TABLE

  subgraph Pages [Example Screens]
    DIAG[Diagnostics View]
    REPT[Reports View]
  end

  PANEL --> DIAG
  CARD --> DIAG
  CHART --> DIAG
  TABLE --> DIAG
  MODAL --> DIAG

  CARD --> REPT
  TABLE --> REPT
  TABS --> REPT
````

---

## 4) Artifact Flow (CLI → GUI)

```mermaid
flowchart LR
  CLI[spectramind diagnose] -->|PNG/JSON/HTML| FS[(./artifacts)]
  FS --> API[/GET /api/diagnostics/plot/]
  API --> CHART[Chart.tsx]
  CHART --> UI[(Viewport)]
```

**Principle:** The GUI **never computes analytics**.
It only renders CLI artifacts (plots, JSON summaries, HTML dashboards).

---

## 5) Design Principles

* **CLI-first, GUI-optional** — reproducibility is enforced at the CLI layer.
* **Declarative + retained-mode React** — components map directly to CLI artifacts.
* **Atomic → composite layering** — aligns with modern GUI engineering (MVVM/MVC patterns).
* **Reproducibility hooks** — components never bypass Hydra/DVC configs; all overrides must flow from CLI runs.
* **Mermaid documentation** — diagrams are rendered natively on GitHub.

---

✅ This `ARCHITECTURE.md` is **self-contained**, GitHub-renderable, and fully aligned with SpectraMind V50’s CLI-first philosophy.

```
