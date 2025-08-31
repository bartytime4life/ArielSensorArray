# ♿ UX & Accessibility — SpectraMind V50 (GUI-Optional Layer) — Upgraded

> Ensure our **thin GUIs** (Streamlit / React / Qt) are **usable, accessible, and audit-safe** while preserving the **CLI-first, Hydra-safe** pipeline.
> Scope: visual design, interaction, accessibility (A11y), i18n/l10n, error messaging, progress, testing, and CI enforcement.

---

## 0) First Principles

1. **CLI-first, GUI-thin**
   • All operations must be reproducible via `spectramind …`.
   • GUIs **only** invoke the CLI and render artifacts (JSON/HTML/plots/logs).
   • No hidden state: parameters live in **Hydra configs** or explicit CLI overrides.

2. **Declarative + Event-Driven**
   • UI is a function of state (diagnostic JSON, run status).
   • Event → invoke CLI → render outputs (bind state to views).

3. **Accessible by Default**
   • Follow WCAG 2.2 AA, platform A11y APIs, and pattern libraries (MVC/MVVM).
   • Test with keyboard + screen reader; ship high-contrast and reduced-motion options.

4. **Human-friendly CLI UX surfaced in the GUI**
   • Show help/affordances, progress, actionable errors, and a visible reaction for every action.

---

## 1) A11y Acceptance Criteria (WCAG-inspired, pragmatic)

### 1.1 Perceivable

* **Contrast**: text/UI ≥ 4.5:1; large text ≥ 3:1. Charts follow the theme (see §6 tokens).
* **Color-independence**: never encode meaning with color alone; add shapes, patterns, labels.
* **Alt text** for images/plots; **captions** for embedded HTML reports.
* **Type & scale**: base 16px with responsive scaling (min 14px at dense breakpoints).
* **Media**: no auto-playing audio; videos require captions.

### 1.2 Operable

* **Keyboard first**: all actions reachable; no key traps; visible focus rings.
* **Focus management**: send focus to results panel after Run; provide “Skip to content” link.
* **Motion**: respect `prefers-reduced-motion`; no flashing (≤ 3/sec).
* **Auto-refresh**: user-controlled (pause/resume/throttle) for logs and galleries.

### 1.3 Understandable

* **Plain-language labels**; tooltips for advanced flags.
* **Consistency**: same components for same concepts (buttons, links, toasts).
* **Validation**: inline, descriptive errors; preserve user input.

### 1.4 Robust

* **Semantics**: proper HTML/ARIA roles; use native controls where possible.
* **Live regions**: announce progress/errors politely (`aria-live="polite|assertive"`).
* **Forms**: programmatic labels; inputs associated with `<label for=…>`.

---

## 2) SpectraMind-Specific UX for Reproducibility & Auditability

### 2.1 Provenance & Traceability

* Echo the **exact CLI command** the GUI will run (copy-as-CLI button).
* Show **Hydra overrides**, **composed config hash**, **git SHA**, timestamp, and **run id**.
* Link each action to `logs/v50_debug_log.md` and render a best-effort “Recent Runs” table.

### 2.2 Error Design (Actionable)

* Map common failures → **plain-language** messages + remediation steps.
* Keep raw `stderr` in a collapsible panel; “Copy error context” button.
* Never fail silently: always show **what happened** and **how to recover**.

### 2.3 Progress & Feedback

* Show a spinner or progress bar within **100 ms** of run initiation.
* Stream stdout/stderr line-by-line; show periodic status headings.
* On completion, surface a toast with **status**, **duration**, and **artifact links**.

---

## 3) Layout, Grids, and Visualization

* **Grid design**: page → section → card → content. Use a 12-col grid; avoid nested scroll containers.
* **Scientific visuals**: consistent axes/units/legends; keep Y ranges comparable across runs when meaningful.
* **Chart A11y**: label datasets; add data tables or “download CSV” alternatives; use `aria-describedby` and captions.
* **Density controls**: “Compact” vs “Comfortable” spacing toggles for large tables.

---

## 4) Keyboard Map (defaults)

| Action                      | Keys           |
| --------------------------- | -------------- |
| Run current form            | **Shift + R**  |
| Open logs panel             | **g l**        |
| Dashboard home              | **g d**        |
| Run launcher                | **g r**        |
| Focus CLI preview           | **.** (period) |
| Copy CLI command            | **c**          |
| Toggle auto-refresh         | **t**          |
| Next / Prev run             | **]** / **\[** |
| Open help/shortcuts overlay | **?**          |

> Ship an in-app “Shortcuts (?)” overlay; avoid conflicting with screen-reader commands.

---

## 5) Internationalization (i18n) & Localization (l10n)

* Externalize all UI strings; default `en`, allow locale switch.
* Date/number formatting respects locale; avoid string concatenation (use placeholders).
* Plan for **RTL** support (logical properties, `dir="auto"` where applicable).
* Keep labels concise to prevent overflow in translated UIs.

---

## 6) Design Tokens (A11y-safe theme)

```yaml
# docs/gui/tokens.yaml (excerpt)
color:
  fg:            "#0b0f14"   # foreground
  fg-muted:      "#2d3748"
  bg:            "#ffffff"
  bg-elev:       "#f7f9fc"
  primary:       "#0b67ff"   # ≥ 4.5:1 on bg
  success:       "#0e8a16"
  warning:       "#b7791f"
  danger:        "#b00020"
  border:        "#d8dee9"
  focus:         "#ffb703"   # visible focus ring
chart:
  # Palette chosen for contrast on light & dark backgrounds
  seq: ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#8ab17d", "#5a189a"]
radius:
  card: 12
  button: 10
space:
  xs: 4
  sm: 8
  md: 12
  lg: 16
  xl: 24
type:
  base: 16    # px
  scale: 1.125
```

---

## 7) Component Patterns (specs)

### 7.1 Run Launcher (Form)

* Fields map 1:1 to Hydra paths (e.g., `--outputs.dir`, `--no-umap`).
* **CLI Preview** shows the exact composed command; **Copy** button.
* **Dry-run** toggle to validate and preview without execution.
* On submit: disable button, show spinner, stream logs.

### 7.2 Artifact Viewers

* **HTML** reports: sandboxed embed + filename + timestamp + “Open in new tab” + “Download”.
* **JSON**: render `metrics` table and `per_planet` table if present; collapsible raw JSON.
* **Plots**: responsive images, alt text like “UMAP of latent space, colored by symbolic rule class.”
* **Empty state**: clearly indicate when no artifacts are present; link to a run action.

### 7.3 Logs

* Tail `logs/v50_debug_log.md` with adjustable byte limit & interval; **Pause/Resume**.
* Search filter; “Download log” and “Copy selection”.
* Highlight error lines (`rc != 0`, keywords like “Traceback”, “ERROR”).

---

## 8) Performance & Reliability

* **Non-blocking**: subprocess streaming (Qt `QProcess`, React via WebSocket/child-process service, Streamlit `Popen` line-buffered).
* **Debounce/throttle** frequent toggles; guard against double-submits.
* **Offline-friendly**: cache the last successful diagnostics; show banners when sources are missing.
* **Low-bandwidth**: lazy-load heavy plots; thumbnail grids; progressive JSON loading.

---

## 9) QA & A11y Testing Matrix

| Layer          | What to test                           | Tooling                                   |
| -------------- | -------------------------------------- | ----------------------------------------- |
| Semantics      | Roles, labels, headings, landmarks     | axe-core / Pa11y / eslint-plugin-jsx-a11y |
| Keyboard       | Tab order, focus rings, skip links     | Playwright/Cypress scripts                |
| Screen readers | NVDA (Win), VoiceOver (macOS)          | Manual pass checklist                     |
| Color/Contrast | Token contrast ≥ AA                    | Storybook A11y addon / figma plugin       |
| Motion         | `prefers-reduced-motion` respected     | CSS audit + manual                        |
| Logs/Streaming | No UI lock; consistent updates         | E2E with fake CLI writer                  |
| i18n           | Locale switch, overflow handling       | Pseudolocalization                        |
| Regression     | Artifact embedding, CLI preview stable | Snapshot tests / golden files             |

---

## 10) Reference Snippets

### 10.1 React — Polite live region for progress

```tsx
<div role="status" aria-live="polite" aria-atomic="true">
  {running ? `Running… ${progressMsg}` : 'Idle'}
</div>
```

### 10.2 Streamlit — Accessible caption & download

```python
st.components.v1.html(html_text, height=900, scrolling=True)
st.caption(f"Report: {report_path.name} — {report_path.stat().st_mtime_ns}")
with open(report_path, "rb") as f:
    st.download_button("Download HTML report", f, file_name=report_path.name, mime="text/html")
```

### 10.3 Qt — Non-blocking CLI runner

```python
proc = QProcess(self)
proc.setProgram("spectramind")
proc.setArguments(["diagnose", "dashboard", "--outputs.dir", outputs_dir])
proc.readyReadStandardOutput.connect(lambda: self.append(proc.readAllStandardOutput().data().decode()))
proc.start()
```

---

## 11) Failure Taxonomy → Messages

| Symptom           | Likely Cause          | User Message (short)            | Remediation                                          |
| ----------------- | --------------------- | ------------------------------- | ---------------------------------------------------- |
| CLI not found     | PATH/env not active   | “`spectramind` not found.”      | “Activate your env or set the CLI path in Settings.” |
| Permission denied | Outputs dir RW issues | “Cannot write outputs.”         | “Choose a writable directory or fix permissions.”    |
| Bad override      | Hydra key typo        | “Unknown config key `foo.bar`.” | “Check `configs/**` or remove the override.”         |
| Long stall        | Heavy job, no output  | “Still running…”                | “Streaming last 20 lines. You can pause logs.”       |
| HTML embed fails  | Corrupted file        | “Couldn’t embed report.”        | “Open in new tab or regenerate via Diagnose.”        |

---

## 12) DO / DON’T

**DO**

* Echo exact CLI commands; keep GUI state minimal; prefer native controls; label everything; fail usefully; throttle refresh.

**DON’T**

* Compute pipeline logic in the GUI; hide overrides; auto-refresh without a pause; rely on color alone; trap focus; swallow errors.

---

## 13) CI Hooks (recommended)

* **A11y lint** (web): `yarn test:a11y` running axe on key routes.
* **Unit**: mock subprocess/IPC; assert command lines and parsing.
* **E2E**: fake CLI writer that emits stdout, creates fixture artifacts, and exits with `rc=0/1`.
* **Snapshots**: artifact table & image gallery (fixed fixtures).
* **Contrast**: automated check on token pairs (`fg`, `bg`, `primary` over `bg`).

---

## 14) Diagram — Thin Integration (for docs)

```mermaid
flowchart LR
  U[User] -->|Key/Click| G[GUI Shell]
  G -->|Invoke| C[spectramind …]
  C --> Y[Hydra Configs]
  C --> O[Artifacts (JSON/HTML/plots)]
  C --> L[Audit Log]
  O --> G
  L --> G
  G -->|Render| V[Accessible Views]
```

---

## 15) References (internal)

* `docs/gui/README.md` — GUI layer goals & rules
* `gui/streamlit_app.py` — reference Streamlit implementation
* `logs/v50_debug_log.md` — authoritative run history
* `configs/**` — Hydra groups & overrides (GUI forms/shortcuts map to these)

---

### ✅ TL;DR

Build GUIs that **call the CLI** and **render its artifacts**. Keep state thin, provenance visible, keyboard paths clear, contrast high, motion optional, and every failure **actionable**.
