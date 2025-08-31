# 🖥️ SpectraMind V50 — Comprehensive Guide to GUI Programming (Upgraded)

> A developer playbook for building **optional**, cross-platform GUIs that *wrap* the SpectraMind V50 CLI (Typer + Hydra) without compromising **NASA-grade reproducibility**. Use this with `docs/gui/README.md` and `gui/streamlit_app.py`.

---

## 0) Non-Negotiables

* **CLI-first, GUI-optional.** Every action must be runnable and auditable via `spectramind …`.
* **No hidden logic.** GUI never computes; it **invokes** CLI, then **renders** artifacts (HTML, PNG, JSON, CSV).
* **Deterministic state.** Inputs = Hydra configs + CLI args; Outputs = files + logs. The GUI only surfaces these.
* **Auditability.** All GUI-triggered runs append to `logs/v50_debug_log.md` and reuse `run_hash_summary_v50.json`.
* **Accessibility.** Keyboard navigation, contrast, reduced motion, i18n hooks, and screen reader labels.

---

## 1) Architectural Patterns (and what we use where)

| Pattern  | What it means here                                                                                         | When to use in V50                                                  | Notes                                            |
| -------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| **MVC**  | Views (widgets) render run artifacts; Controller dispatches `spectramind` calls; Model = configs/artifacts | Simple desktop shells (PySide6)                                     | Clear separation; easy to test controller        |
| **MVP**  | Presenter holds logic; View is dumb                                                                        | Electron/React where Presenter lives in preload/main                | Test View via snapshot; Presenter via unit tests |
| **MVVM** | ViewModel exposes observables: `status`, `stdout`, `stderr`, `artifacts`                                   | **Preferred for dashboards** (React/Streamlit ViewModel-like state) | Reactive updates while streaming logs            |

**V50 default mapping**

* **Web dashboard (recommended):** React/Next.js or Streamlit with a thin *ViewModel* abstraction.
* **Desktop shell (optional):** PySide6 (Qt) with MVC/MVVM hybrid (signals → ViewModel slots).

---

## 2) GUI ⇄ CLI Contract

All GUIs must implement the following **contract**:

### 2.1 Commands & Flags (minimum)

* `spectramind test` — fast integrity checks
* `spectramind diagnose dashboard [--no-umap] [--no-tsne] [--open-html] --outputs.dir <path>`
* `spectramind train …` — optionally available in GUI, but default hidden
* `spectramind predict …`
* `spectramind submit …`

### 2.2 Inputs

* Paths: repo root, outputs dir, data dir (read-only in GUI), config presets
* Hydra overrides: free-form text serialized to CLI args (never interpreted in GUI)
* Feature toggles: UMAP/t-SNE checkboxes → `--no-umap/--no-tsne` flags

### 2.3 Outputs (what GUIs can render)

* **HTML** diagnostics: `diagnostic_report*.html`, `*dashboard*.html`
* **JSON** metrics: `diagnostic_summary.json`, calibration JSONs
* **Images**: `*.png|*.jpg` in `outputs/**` (including `plots/`)
* **Logs**: tail of `logs/v50_debug_log.md`

> **Rule:** If you need a new panel, ship it via CLI artifacts first; never add a GUI-only computation path.

---

## 3) Framework Choices (quick matrix)

| Framework        | Strengths                                         | Trade-offs                                             | V50 stance                                            |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- |
| **Streamlit**    | Fast to build, Python-native, easy file rendering | Limited component control; Python environment coupling | **Adopted** for first GUI; see `gui/streamlit_app.py` |
| **React (web)**  | Total control, best UX, Plotly/ECharts, SSR       | Separate frontend stack, packaging                     | **Recommended** for long-term dashboard               |
| **Electron**     | Desktop wrapper for web UI, native fs/process     | Heavier runtime                                        | Good when offline desktop is required                 |
| **PySide6 (Qt)** | True native desktop, MVVM/QML, perf               | More boilerplate, packaging                            | Optional specialized workstation app                  |
| **Flutter**      | Beautiful cross-platform, good perf               | Dart toolchain                                         | Consider for mobile/tablet control surfaces           |

---

## 4) Core GUI Concepts (SpectraMind-specific)

* **Event loop**: never block; heavy work is **process-spawned** (the CLI).
* **Layouts**: responsive (CSS grid/flex, Streamlit columns); zero fixed px where possible.
* **Data binding**: UI fields ⇄ ViewModel ⇄ serialized CLI args.
* **Streaming**: surface live stdout/stderr while the CLI runs; flush line-by-line.
* **Artifacts browser**: always show latest HTML/JSON/plots with **download** buttons.
* **Accessibility**: roles, labels, skip-links, keyboard focus order; test with Axe.
* **i18n**: wrap strings; keep user text (e.g., error lines) verbatim.

---

## 5) Logging, Reproducibility, and Security

* **Log every click** that dispatches a CLI run (action, args, timestamp) into `logs/v50_debug_log.md` (the CLI already does this; GUI should add its own “GUI dispatch” line too).
* **Immutable source of truth**: logs + artifacts; GUI adds zero derived data unless explicitly saved to `outputs/gui/`.
* **Sandbox external content**: when embedding HTML reports, prefer iframe or sanitized string (Streamlit’s `components.html` is acceptable for trusted, locally-generated HTML).
* **Secrets**: never store tokens in GUI state; use environment variables (read-only display for which variables were present is okay).
* **Permissions**: read from `outputs/`, never write inside `data/` or `configs/` without explicit “export override YAML” features.

---

## 6) Testing Strategy

### 6.1 Unit

* **Python (Streamlit):** isolate helpers (glob scans, tail, CLI spawn). Use `pytest` + `pytest-mock` for `subprocess.Popen`/`run`.
* **Qt (PySide6):** `pytest-qt` for signal/slot, widget events.
* **React/Electron:** React Testing Library + Jest; preload IPC mocked; Playwright for `run-cli` happy path.

### 6.2 Integration / E2E

* **Playwright**: load dashboard, select outputs dir with fixtures, run “fake CLI” that echoes canned stdout and writes dummy artifacts; validate embed.
* **Snapshot**: image diff of gallery panel on a fixed artifact set.

### 6.3 Accessibility

* **axe-core**: automated checks in CI (React); manual keyboard traversal tests for Streamlit/Qt.

---

## 7) Streamlit Reference Implementation (what “good” looks like)

* **Live streaming** stdout/stderr via line-buffered `Popen` (already implemented).
* **Artifact discovery**: glob + newest-first sorting; user-selectable lists.
* **Download buttons** for each artifact.
* **Log tail slider** with adjustable bytes.
* **Recent runs table** parsed heuristically from `v50_debug_log.md`.
* **Dry-run mode** to refresh artifacts without executing CLI.

> See `gui/streamlit_app.py` (upgraded). When you add new CLI artifacts, **extend the artifact scanner**, not the business logic.

---

## 8) Desktop Shell (PySide6) — Minimal MVC Example

```python
# gui/qt_shell.py (sketch)
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QFileDialog
from PySide6.QtCore import QThread, Signal
import subprocess, sys, shlex, pathlib

class Runner(QThread):
    line = Signal(str)
    done = Signal(int)
    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd, self.cwd = cmd, cwd
    def run(self):
        try:
            proc = subprocess.Popen(self.cmd, cwd=self.cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for L in iter(proc.stdout.readline, ""):
                if L == "": break
                self.line.emit(L.rstrip("\n"))
            rc = proc.wait()
            self.done.emit(rc)
        except Exception as e:
            self.line.emit(f"[error] {e}")
            self.done.emit(1)

class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraMind V50 — Desktop Shell")
        self.out = QTextEdit(); self.out.setReadOnly(True)
        self.btn = QPushButton("Run Diagnose Dashboard")
        self.btn.clicked.connect(self.on_run)
        lay = QVBoxLayout(self); lay.addWidget(self.btn); lay.addWidget(self.out)

    def on_run(self):
        repo = QFileDialog.getExistingDirectory(self, "Select Repo Root")
        if not repo: return
        cmd = ["spectramind", "diagnose", "dashboard", "--outputs.dir", str(pathlib.Path(repo)/"outputs")]
        self.run(cmd, repo)

    def run(self, cmd, cwd):
        self.out.append(f"$ {' '.join(shlex.quote(x) for x in cmd)}")
        self.runner = Runner(cmd, cwd)
        self.runner.line.connect(self.out.append)
        self.runner.done.connect(lambda rc: self.out.append(f"[rc={rc}]"))
        self.runner.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    m = Main(); m.show()
    sys.exit(app.exec())
```

---

## 9) Electron + React — IPC Contract Sketch

**main.ts**

```ts
import { app, BrowserWindow, ipcMain } from "electron";
import { spawn } from "child_process";
function createWindow() {
  const win = new BrowserWindow({ width: 1200, height: 800, webPreferences: { preload: __dirname + "/preload.js" }});
  win.loadFile("index.html");
}
ipcMain.handle("spectramind:run", async (_evt, args: string[]) => {
  return new Promise(res => {
    const proc = spawn("spectramind", args, { shell: false });
    let out = "", err = "";
    proc.stdout.on("data", d => out += d.toString());
    proc.stderr.on("data", d => err += d.toString());
    proc.on("close", code => res({ code, out, err }));
  });
});
app.whenReady().then(createWindow);
```

**preload.ts**

```ts
import { contextBridge, ipcRenderer } from "electron";
contextBridge.exposeInMainWorld("spectramind", {
  run: (args: string[]) => ipcRenderer.invoke("spectramind:run", args),
});
```

**renderer (React)**

```tsx
async function runDashboard(outputsDir: string) {
  setRunning(true);
  const { code, out, err } = await (window as any).spectramind.run(["diagnose","dashboard","--outputs.dir", outputsDir]);
  setStdout(out); setStderr(err); setRc(code); setRunning(false);
}
```

---

## 10) UX Guidelines (applies to Streamlit/React/Qt)

* **Single source of truth**: always show the **exact CLI command** executed.
* **State clarity**: idle → running → done; show return code; color-code non-zero rc.
* **Discoverability**: “Rescan Artifacts” button; “Open report in browser”.
* **Scales down**: 12-column grid; collapsible panels; avoid nested scrolls.
* **Keyboard**: focus ring visible; `Tab` order mirrors reading order; Esc to cancel run if feasible.
* **Reduced motion**: respect `prefers-reduced-motion`; limit auto-refresh animations.

---

## 11) Packaging & Distribution

* **Streamlit**: run from repo; for “app” feel, add a wrapper script (`bin/gui-streamlit.sh`) and a desktop shortcut.
* **Qt**: PyInstaller (`onefile=False` preferred for size/perf), sign binaries for macOS/Windows if distributing.
* **Electron**: `electron-builder` targets (win/mac/linux). Bundle a minimal Python env *only if* you need to ship the CLI; otherwise require a preinstalled SpectraMind environment.

---

## 12) CI Hooks (optional but recommended)

* **GUI lint**: `ruff`/`flake8` for Python GUI; ESLint for React.
* **Unit tests**: run headless with mocked CLI (`subprocess` patched to write fixture artifacts).
* **Accessibility**: run Axe on a static build of the React dashboard.
* **Artifact fixtures**: keep a tiny `tests/fixtures/outputs/` set to validate gallery and JSON tables.

---

## 13) Roadmap (repository-aligned)

* [x] **v0** Streamlit MVP: run `diagnose dashboard`, embed HTML, show JSON & plots, tail logs.
* [ ] **v1** Config presets panel: pick Hydra YAML groups; serialize overrides to CLI.
* [ ] **v1** Compare runs: load N JSON/HTML reports; compute delta summaries.
* [ ] **v2** Symbolic overlays: dedicated panels for SHAP × Symbolic heatmaps (read files produced by CLI).
* [ ] **v2** UMAP/t-SNE explorer: link points to planet pages; confidence shading (read CLI HTML).
* [ ] **v3** Electron/React desktop shell (optional), with the same contract.
* [ ] **v3** Remote control mode: thin FastAPI proxy that forwards CLI jobs (for headless servers).

---

## 14) Quick Checklists

### 14.1 Before committing a GUI change

* [ ] No business logic added; only artifact plumbing.
* [ ] Every new feature has a corresponding CLI artifact.
* [ ] “Run” buttons echo exact `spectramind …` command.
* [ ] Logs end up in `logs/v50_debug_log.md` (CLI already does; GUI adds a dispatch line if it shells directly).
* [ ] Keyboard and screen reader paths tested on main panels.

### 14.2 Accessibility (minimum)

* [ ] All actionable elements have accessible names.
* [ ] Focus order is logical; focus not trapped.
* [ ] Color contrast ≥ 4.5:1 for body text.
* [ ] Animations can be turned off; avoid flashing.

---

## 15) References (internal)

* `docs/gui/README.md` — GUI layer purpose & rules
* `gui/streamlit_app.py` — reference implementation (first-class)
* `SpectraMind V50 Technical Plan` — end-to-end pipeline and artifacts
* `docs/configs/*` — Hydra groups & overrides (for GUI presets)
* `logs/v50_debug_log.md` — run history (parsed by GUI)

---

### ✅ TL;DR

Build GUIs that **call the CLI and render its artifacts**. If it isn’t possible from `spectramind`, it doesn’t belong in the GUI. Keep it accessible, testable, and fully auditable.
