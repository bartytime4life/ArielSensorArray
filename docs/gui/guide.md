# SpectraMind V50 — GUI Engineering Guide (Dashboard & Control Panel)

> Goal: add an **optional** GUI (diagnostics dashboard / control panel) that mirrors the CLI, preserves full reproducibility, and stays thin. The CLI remains the source of truth; the GUI consumes logged artifacts and commands the pipeline via the unified `spectramind` app.

---

## 0) Principles

1. **CLI-first, GUI-optional**

   * Every operation must be runnable via `spectramind …` with Hydra configs; the GUI is a *client* that shells out or calls a thin API wrapper around the CLI.
   * Avoid hidden state: write/read artifacts only via the pipeline’s standard outputs, logs, and DVC-tracked files.

2. **Mirror, don’t fork**

   * GUI buttons ≙ exact CLI invocations; help panels show the same flags/semantics as `--help`. UX reflects CLI UX guidelines (progress, helpful errors, visible reaction).

3. **Testability & portability**

   * Decouple rendering (View) from logic (Model); prefer **MVVM**/**MVC** patterns for clear seams and unit-testable view-models (no I/O in the View).

4. **Thin, fast, accessible**

   * Prioritize small, cross‑platform stacks; ensure keyboard navigation, high-contrast themes, and i18n hooks from day one.

---

## 1) Architecture Patterns You’ll Use (in practice)

* **MVC**: Controller transforms user actions into CLI calls; View renders artifact files (plots, HTML). Keeps UI passive when possible.
* **MVVM**: View binds to a ViewModel that exposes `status`, `metrics`, `artifacts`, `run_commands()`; perfect if you pick frameworks with native binding (Qt/QML, SwiftUI, Jetpack Compose).
* **Event-driven loop**: GUI reacts to file updates (e.g., new `outputs/diagnostics/report_v*.html`), streaming logs, and long-run tasks via async watchers.

---

## 2) Framework Choices (trade-offs)

**Pick one**; all can work for a thin, portable dashboard.

### A) Qt (PySide/PyQt, C++ or Python)

* **Pros**: Native performance, mature widgets, excellent MVVM/QML; great for desktop-only tools; Python bindings are solid.
* **Cons**: Larger runtime than minimal web UIs; packaging considerations.
* **Recommend**: PySide6 (Python) + QML for clean MVVM.

### B) Electron (+ React)

* **Pros**: Web skills, huge ecosystem, rapid dev, cross-platform; good for rich HTML diagnostics (embedding existing HTML reports).
* **Cons**: Heavier footprint; ensure long-running tasks are backgrounded.
* **Recommend**: Only if you want web tech and embedded Plotly/HTML artifacts.

### C) Flutter (Dart)

* **Pros**: One codebase for desktop+mobile, reactive UI model, good charts; nice for portable dashboard binaries.
* **Cons**: New language for many; plugin surface still evolving for some OS features.

> **Default path** for SpectraMind V50: **Qt (PySide6 + QML)** if you want a native feel and tight Python interop; **Electron+React** if you want to reuse web assets and embed HTML diagnostics.

---

## 3) Minimal, Working Examples

### 3.1 PySide6 (Qt) — MVVM skeleton

```python
# gui/app.py
import sys, subprocess, json, threading
from pathlib import Path
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLabel

REPO = Path(__file__).resolve().parents[1]
CLI  = ["python", "-m", "spectramind"]  # unified Typer CLI

def run_cli(args, on_line):
    # Non-blocking CLI runner with streaming
    proc = subprocess.Popen(args, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        on_line(line.rstrip())
    proc.wait()
    on_line(f"[exit code {proc.returncode}]")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraMind V50 — Dashboard")
        layout = QVBoxLayout(self)

        self.status = QLabel("Idle")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        self.btn_selftest = QPushButton("Run selftest")
        self.btn_diagnose = QPushButton("Generate Diagnostics")

        self.btn_selftest.clicked.connect(self.on_selftest)
        self.btn_diagnose.clicked.connect(self.on_diagnose)

        layout.addWidget(self.status); layout.addWidget(self.btn_selftest)
        layout.addWidget(self.btn_diagnose); layout.addWidget(self.log)

    def on_selftest(self):
        self.status.setText("Running: spectramind selftest …")
        threading.Thread(target=run_cli, args=([*CLI, "selftest"], self.log.append), daemon=True).start()

    def on_diagnose(self):
        self.status.setText("Running: spectramind diagnose dashboard …")
        # Example: generate dashboard with defaults; adapt Hydra overrides as needed
        threading.Thread(target=run_cli, args=([*CLI, "diagnose", "dashboard"], self.log.append), daemon=True).start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.resize(900, 600); w.show()
    sys.exit(app.exec())
```

* **Pattern**: The View (`MainWindow`) is thin; logic dispatches *exact* CLI commands (no hidden state).
* Use QML (+ models) to evolve toward MVVM bindings (status, artifacts list, run history).

### 3.2 Electron + React — thin shell around CLI

```js
// main.js (Electron main process)
const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const path = require('path');

function createWindow() {
  const win = new BrowserWindow({
    width: 1100, height: 720,
    webPreferences: { preload: path.join(__dirname, 'preload.js') }
  });
  win.loadFile('index.html');
}

ipcMain.handle('run-cli', async (_evt, args) => {
  return await new Promise((resolve) => {
    const child = spawn('python', ['-m', 'spectramind', ...args], { cwd: path.resolve(__dirname, '..') });
    let output = '';
    child.stdout.on('data', d => output += d.toString());
    child.stderr.on('data', d => output += d.toString());
    child.on('close', code => resolve({ code, output }));
  });
});

app.whenReady().then(createWindow);
```

```js
// preload.js (safe bridge)
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('spectramind', {
  run: (args) => ipcRenderer.invoke('run-cli', args),
});
```

```html
<!-- index.html (quick UI; replace with React app) -->
<!doctype html>
<html>
  <body>
    <h1>SpectraMind V50 — Dashboard</h1>
    <button onclick="runSelftest()">Run selftest</button>
    <button onclick="runDiagnose()">Generate Diagnostics</button>
    <pre id="log"></pre>
    <script>
      async function runSelftest() {
        const res = await window.spectramind.run(['selftest']);
        document.getElementById('log').textContent = res.output + `\n[exit ${res.code}]`;
      }
      async function runDiagnose() {
        const res = await window.spectramind.run(['diagnose', 'dashboard']);
        document.getElementById('log').textContent = res.output + `\n[exit ${res.code}]`;
      }
    </script>
  </body>
</html>
```

* **Pattern**: Electron main process spawns the exact CLI, renderer displays logs; scale up with React (components, routing, charts).

---

## 4) What the GUI Should Show

* **Run Controls**: buttons for `selftest`, `calibrate`, `train`, `predict`, `diagnose dashboard`, `submit`.
* **Hydra Overrides Panel**: key/value inputs → `spectramind … key=value`; defaults pre-filled.
* **Artifact Browser**: list latest `outputs/diagnostics/*.html`, plots, `v50_debug_log.md`; click to open.
* **Live Log**: stream CLI output with progress; always show “visible reaction” to actions; format helpful errors.
* **Status Cards**: last run time, config hash, Git SHA, DVC data hash, and links to artifacts.
* **Accessibility**: keyboard shortcuts, high-contrast theme, reduced motion, alt text for images.

---

## 5) How the GUI Calls the System (No Hidden State)

1. Spawn the **Typer CLI** (`python -m spectramind …`) for every action; never bypass it.
2. Read **Hydra-resolved** configs & outputs from `outputs/YYYY-MM-DD/HH-MM-SS/…` for display (not from memory).
3. Show **DVC-tracked** artifacts by reading the working tree; no ad-hoc caches.
4. Append CLI invocations to `v50_debug_log.md` (already standard); surface them in the GUI history.

---

## 6) Testing & CI

* **Unit**:

  * ViewModel functions return the *exact* CLI arg lists for given options.
  * Parsers that read metrics/artifacts from output dirs.
* **Integration**:

  * Headless smoke tests: launch GUI, trigger `selftest` and `diagnose dashboard`, assert artifacts exist.
* **CI**:

  * Build standalone app (Qt/Electron) on Linux runner; run smoke tests with a small toy config.
  * Ensure failures display actionable messages (map stderr to friendly text).

---

## 7) Performance & Packaging Notes

* Qt/PySide: ship a single binary (PyInstaller/Briefcase) + Qt libs; MVVM with QML runs fast & native.
* Electron: keep main process light; offload long tasks; bundle with `electron-builder`; beware size; cache HTML plots.&#x20;
* Streaming logs: line-buffer the process; keep the UI responsive with async threads/workers.

---

## 8) Roadmap (phased)

* **v0 (MVP)**: Controls for `selftest`, `diagnose dashboard`; live log; artifact list & “Open in browser”.
* **v1**: Hydra overrides panel; run history; “rerun with last config”.
* **v2**: Live charts (tailing metrics JSON/CSV); UMAP/SHAP embeds via embedded web views.
* **v3**: Remote mode (optional) — small HTTP sidecar that accepts *only* specific `spectramind` commands, still logging via CLI.

---

## 9) Security & Safety

* Never execute arbitrary shell from the GUI. Only allow predefined commands with sanitized overrides.
* Keep read-only browsing of artifacts by default; writes happen only via CLI.
* Log everything: command, config, start/end, exit code, artifact paths.

---

## 10) Quick Start Recipes

### A) Qt (PySide6) dev loop

```bash
pip install PySide6
python gui/app.py
```

* Package later with PyInstaller (`pyinstaller --onefile gui/app.py` + datas).

### B) Electron dev loop

```bash
cd gui-electron
npm install
npm start
# main.js, preload.js, index.html as above; replace index.html with React SPA later
```

* Package with `electron-builder`.

---

## 11) Why this fits SpectraMind V50

* Matches the **terminal-first** blueprint and keeps full auditability via logs and artifacts (no GUI-only paths).
* Uses **MVVM/MVC** to ensure maintainable, testable UI layers.
* Adheres to CLI UX best practices: clear help, progress, helpful errors, visible reaction.

---

## References

* CLI-first pipeline & audit logging: unified Typer app, Hydra configs, DVC, CI; saved HTML diagnostics.
* GUI architecture & patterns (event-driven, MVC/MVVM, layout, reactive UIs), framework overviews (Qt/Electron/Flutter).
* CLI UX best practices (help, flags, progress, error messaging, readability).

---

## Appendix — “Glue Code” Stubs (ViewModel style)

**Python ViewModel (Qt path)**

```python
# gui/viewmodel.py
from dataclasses import dataclass, field
from typing import List

@dataclass
class RunSpec:
    command: List[str]  # e.g., ["selftest"] or ["diagnose","dashboard"]
    overrides: List[str] = field(default_factory=list)  # e.g., ["data=toy","training.fast=true"]

    def to_args(self) -> List[str]:
        return ["-m", "spectramind"] + self.command + self.overrides
```

**Electron Renderer (React hook)**

```ts
// useSpectramind.ts
import { useState } from 'react';

export function useSpectramind() {
  const [running, setRunning] = useState(false);
  const [log, setLog] = useState('');

  async function run(args: string[]) {
    setRunning(true);
    const res = await (window as any).spectramind.run(args);
    setLog(res.output + `\n[exit ${res.code}]`);
    setRunning(false);
  }
  return { running, log, run };
}```
