\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
SpectraMind V50 — Qt Diagnose Demo (Upgraded)
---------------------------------------------

A thin, GUI-optional wrapper around the CLI-first pipeline.

What this demo does

* Runs:   spectramind diagnose dashboard \[extra hydra args...]
* Streams stdout/stderr live into the UI (non-blocking)
* Finds the newest HTML diagnostics report and previews it
* Lets you pick output directory and persist settings across runs
* Validates the CLI path and shows helpful errors
* Provides cancel support and safeguards against shell injection

Requirements
pip install PyQt5 PyQtWebEngine
Optional
Ensure `spectramind` is on PATH or provide the absolute path to the CLI.
"""
import os
import sys
import shlex
import platform
from pathlib import Path
from typing import List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView

# --------------------------------------------------------------------------------------

# Constants and helpers

# --------------------------------------------------------------------------------------

REPO = Path.cwd()  # assumes we run from repo root; change if embedding elsewhere
DEFAULT\_OUT = REPO / "outputs"
APP\_ORG = "SpectraMind"
APP\_NAME = "V50 Qt Diagnose Demo"
SETTINGS\_KEY = "qt\_diag\_demo"

def is\_executable(path: Path) -> bool:
"""Cross-platform executable check."""
if not path.exists():
return False
if platform.system().lower().startswith("win"):
\# On Windows, extension matters (.exe, .bat, .cmd)
return path.suffix.lower() in (".exe", ".bat", ".cmd") or os.access(str(path), os.X\_OK)
return os.access(str(path), os.X\_OK)

def sanitize\_extra\_args(text: str) -> List\[str]:
"""
Split extra args safely (Hydra flags are allowed). Reject obvious shell control tokens.
We do NOT pass through a shell; QProcess uses execve-like behavior.
"""
\# quick reject for command chaining characters; conservative by design
for token in \[";", "&&", "||", "|", "\`", "\$(", "<(", ">{", "<{"]:
if token in text:
raise ValueError(f"Illegal shell control token detected: {token}")
\# shlex.split gives robust tokenization but we still won't run in a shell
return shlex.split(text.strip()) if text.strip() else \[]

def find\_newest\_report(outputs\_dir: Path) -> Optional\[Path]:
"""
Find the newest diagnostics HTML report under outputs\_dir.
Searches common report patterns produced by the pipeline.
"""
patterns = \[
"**/diagnostic\_report\*.html",
"**/*dashboard*.html",
"\*\*/*diag*.html",
"\*.html",
]
candidates: List\[Path] = \[]
for pat in patterns:
candidates.extend(outputs\_dir.glob(pat))
candidates = \[p for p in candidates if p.is\_file()]
if not candidates:
return None
candidates.sort(key=lambda p: p.stat().st\_mtime, reverse=True)
return candidates\[0]

# --------------------------------------------------------------------------------------

# Worker using QProcess (non-blocking CLI run with live output)

# --------------------------------------------------------------------------------------

class CliRunner(QtCore.QObject):
"""
Runs the CLI via QProcess, streams output via signals, supports cancel.
"""
started = QtCore.pyqtSignal(str)                 # emitted with joined command line
line = QtCore.pyqtSignal(str)                    # stdout/stderr lines
finished = QtCore.pyqtSignal(int, str, str)      # returncode, stdout, stderr (aggregated)
error = QtCore.pyqtSignal(str)                   # human-readable error

```
def __init__(self, parent=None):
    super().__init__(parent)
    self._proc = QtCore.QProcess(self)
    self._proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)  # merge stderr into stdout
    self._proc.readyReadStandardOutput.connect(self._on_ready_read)
    self._proc.finished.connect(self._on_finished)
    self._buffer_stdout: List[str] = []
    self._buffer_stderr: List[str] = []  # merged into stdout, kept for API symmetry
    self._cwd = str(REPO)

    # On Windows, ensure .bat/.cmd discovered; QProcess resolves with PATHEXT
    env = QtCore.QProcessEnvironment.systemEnvironment()
    self._proc.setProcessEnvironment(env)

def set_cwd(self, path: Path) -> None:
    self._cwd = str(path.resolve())

def is_running(self) -> bool:
    return self._proc.state() == QtCore.QProcess.Running

def kill(self) -> None:
    if self.is_running():
        self._proc.kill()

def start(self, cli_path: str, args: List[str]) -> None:
    """Start the process with validated args (no shell)."""
    if self.is_running():
        self.error.emit("A process is already running.")
        return

    # Build the command; QProcess takes program and args separately (safe)
    program = cli_path.strip() or "spectramind"
    program_path = Path(program)

    # If not absolute, rely on PATH resolution; otherwise validate executable
    if program_path.is_absolute():
        if not is_executable(program_path):
            self.error.emit(f"CLI not executable or not found: {program_path}")
            return

    # Join for logging
    joined = " ".join(shlex.quote(program)) + (
        "" if not args else " " + " ".join(shlex.quote(a) for a in args)
    )
    self.started.emit(joined)

    # Reset buffers
    self._buffer_stdout.clear()
    self._buffer_stderr.clear()

    # Set working dir and launch
    self._proc.setWorkingDirectory(self._cwd)
    self._proc.start(program, args)

    if not self._proc.waitForStarted(5000):  # 5s to start
        # capture any immediate error channel output
        msg = self._proc.errorString()
        self.error.emit(f"Failed to start process: {msg}")
        return

# Slots for process events
@QtCore.pyqtSlot()
def _on_ready_read(self) -> None:
    chunk = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
    if not chunk:
        return
    # Emit lines progressively
    for line in chunk.splitlines():
        self._buffer_stdout.append(line)
        self.line.emit(line)

@QtCore.pyqtSlot(int, QtCore.QProcess.ExitStatus)
def _on_finished(self, exit_code: int, _status: QtCore.QProcess.ExitStatus) -> None:
    out = "\n".join(self._buffer_stdout)
    err = "\n".join(self._buffer_stderr)  # empty for merged channels
    self.finished.emit(exit_code, out, err)
```

# --------------------------------------------------------------------------------------

# Main Window UI

# --------------------------------------------------------------------------------------

class MainWindow(QtWidgets.QMainWindow):
def **init**(self):
super().**init**()
self.setWindowTitle("SpectraMind V50 — Qt Diagnose Demo")
self.resize(1400, 900)
self.\_creating\_ui = True

```
    # Persistent settings
    self.settings = QtCore.QSettings(APP_ORG, APP_NAME)
    s = self.settings.value(SETTINGS_KEY, {}, type=dict) or {}

    # Widgets
    self.cliEdit = QtWidgets.QLineEdit(s.get("cli", "spectramind"))
    self.cliBrowse = QtWidgets.QToolButton()
    self.cliBrowse.setText("…")
    self.cliBrowse.setToolTip("Browse for CLI executable")

    self.outEdit = QtWidgets.QLineEdit(s.get("outputs", str(DEFAULT_OUT)))
    self.outBrowse = QtWidgets.QToolButton()
    self.outBrowse.setText("…")
    self.outBrowse.setToolTip("Choose outputs directory")

    self.extraArgs = QtWidgets.QLineEdit(s.get("extra_args", ""))
    self.extraArgs.setPlaceholderText("--data=nominal trainer=ci_fast (optional Hydra overrides)")

    self.runBtn = QtWidgets.QPushButton("Run Diagnose")
    self.cancelBtn = QtWidgets.QPushButton("Cancel")
    self.cancelBtn.setEnabled(False)

    self.stdoutBox = QtWidgets.QPlainTextEdit()
    self.stdoutBox.setReadOnly(True)
    self.stdoutBox.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    self.stdoutBox.setPlaceholderText("CLI output will stream here...")

    self.web = QWebEngineView()
    self.web.setZoomFactor(1.0)

    # Top form
    form = QtWidgets.QFormLayout()
    cliRow = QtWidgets.QHBoxLayout()
    cliRow.addWidget(self.cliEdit)
    cliRow.addWidget(self.cliBrowse)
    form.addRow("CLI:", self._wrap(cliRow))

    outRow = QtWidgets.QHBoxLayout()
    outRow.addWidget(self.outEdit)
    outRow.addWidget(self.outBrowse)
    form.addRow("Outputs:", self._wrap(outRow))

    form.addRow("Extra args:", self.extraArgs)

    # Buttons row
    btnRow = QtWidgets.QHBoxLayout()
    btnRow.addWidget(self.runBtn)
    btnRow.addWidget(self.cancelBtn)
    btnRow.addStretch(1)

    # Left panel (controls + output stream)
    left = QtWidgets.QWidget()
    vleft = QtWidgets.QVBoxLayout(left)
    vleft.addLayout(form)
    vleft.addLayout(btnRow)
    vleft.addWidget(self.stdoutBox, 1)

    # Splitter: left (controls/log) | right (web preview)
    splitter = QtWidgets.QSplitter()
    splitter.addWidget(left)
    splitter.addWidget(self.web)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)
    self.setCentralWidget(splitter)

    # Status bar
    self.status = self.statusBar()
    self.status.showMessage("Ready")

    # Menu
    self._build_menu()

    # Process runner
    self.runner = CliRunner(self)
    self.runner.started.connect(self._on_started)
    self.runner.line.connect(self._on_line)
    self.runner.finished.connect(self._on_finished)
    self.runner.error.connect(self._on_error)

    # Connections
    self.runBtn.clicked.connect(self.on_run)
    self.cancelBtn.clicked.connect(self.on_cancel)
    self.cliBrowse.clicked.connect(self.on_browse_cli)
    self.outBrowse.clicked.connect(self.on_browse_out)
    self._creating_ui = False

# --------- UI builders and helpers ---------

def _build_menu(self) -> None:
    menubar = self.menuBar()

    # File menu
    m_file = menubar.addMenu("&File")
    act_open = QtWidgets.QAction("Open HTML Report…", self)
    act_open.triggered.connect(self.action_open_report)
    m_file.addAction(act_open)

    act_quit = QtWidgets.QAction("Quit", self)
    act_quit.setShortcut(QtGui.QKeySequence.Quit)
    act_quit.triggered.connect(self.close)
    m_file.addAction(act_quit)

    # View menu
    m_view = menubar.addMenu("&View")
    act_zoom_in = QtWidgets.QAction("Zoom In", self)
    act_zoom_in.setShortcut(QtGui.QKeySequence.ZoomIn)
    act_zoom_in.triggered.connect(lambda: self.web.setZoomFactor(self.web.zoomFactor() + 0.1))
    m_view.addAction(act_zoom_in)

    act_zoom_out = QtWidgets.QAction("Zoom Out", self)
    act_zoom_out.setShortcut(QtGui.QKeySequence.ZoomOut)
    act_zoom_out.triggered.connect(lambda: self.web.setZoomFactor(self.web.zoomFactor() - 0.1))
    m_view.addAction(act_zoom_out)

    act_zoom_reset = QtWidgets.QAction("Reset Zoom", self)
    act_zoom_reset.setShortcut("Ctrl+0")
    act_zoom_reset.triggered.connect(lambda: self.web.setZoomFactor(1.0))
    m_view.addAction(act_zoom_reset)

    # Tools menu
    m_tools = menubar.addMenu("&Tools")
    act_open_outdir = QtWidgets.QAction("Open Outputs Folder", self)
    act_open_outdir.triggered.connect(self.action_open_outputs_folder)
    m_tools.addAction(act_open_outdir)

    act_find_report = QtWidgets.QAction("Find Newest Report", self)
    act_find_report.triggered.connect(self.action_find_and_load_report)
    m_tools.addAction(act_find_report)

    # Help menu
    m_help = menubar.addMenu("&Help")
    act_about = QtWidgets.QAction("About", self)
    act_about.triggered.connect(self.action_about)
    m_help.addAction(act_about)

def _wrap(self, layout: QtWidgets.QLayout) -> QtWidgets.QWidget:
    w = QtWidgets.QWidget()
    w.setLayout(layout)
    return w

# --------- Actions ---------

def action_open_report(self):
    html_path, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Open Diagnostics HTML",
        str(Path(self.outEdit.text()).expanduser().resolve()),
        "HTML files (*.html *.htm);;All files (*)",
    )
    if html_path:
        self._load_html(Path(html_path))

def action_open_outputs_folder(self):
    path = Path(self.outEdit.text()).expanduser().resolve()
    if not path.exists():
        QtWidgets.QMessageBox.warning(self, "Outputs", f"Folder does not exist:\n{path}")
        return
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

def action_find_and_load_report(self):
    outputs_dir = Path(self.outEdit.text()).expanduser().resolve()
    report = find_newest_report(outputs_dir)
    if report and report.exists():
        self._load_html(report)
    else:
        QtWidgets.QMessageBox.information(self, "Diagnostics", "No report found under outputs directory.")

def action_about(self):
    QtWidgets.QMessageBox.information(
        self,
        "About",
        "SpectraMind V50 — Qt Diagnose Demo\n\n"
        "A thin GUI wrapper around the CLI-first diagnostics pipeline.\n"
        "Run the dashboard, stream logs, and preview the generated HTML report."
    )

# --------- Run / Cancel ---------

@QtCore.pyqtSlot()
def on_run(self):
    # Persist settings eagerly
    self._save_settings()

    # Prepare CLI and args
    cli = self.cliEdit.text().strip() or "spectramind"
    outputs_dir = Path(self.outEdit.text().strip() or str(DEFAULT_OUT)).expanduser().resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        extras = sanitize_extra_args(self.extraArgs.text())
    except ValueError as e:
        QtWidgets.QMessageBox.critical(self, "Invalid Extra Args", str(e))
        return

    # Construct command: spectramind diagnose dashboard --outputs.dir <dir> [extras...]
    args = ["diagnose", "dashboard", "--outputs.dir", str(outputs_dir), *extras]

    # UI state
    self.runBtn.setEnabled(False)
    self.cancelBtn.setEnabled(True)
    self.stdoutBox.appendPlainText("\n─── RUN ─────────────────────────────────────────────────────────")
    pretty_cmd = " ".join(shlex.quote(x) for x in ([cli] + args))
    self.stdoutBox.appendPlainText(f"Running: {pretty_cmd}")
    self.status.showMessage("Running diagnostics…")

    # Start process
    self.runner.set_cwd(REPO)
    self.runner.start(cli, args)

@QtCore.pyqtSlot()
def on_cancel(self):
    if self.runner.is_running():
        self.runner.kill()
        self.status.showMessage("Cancelled")
        self.stdoutBox.appendPlainText("⚠️ Process cancelled by user.")
    self.runBtn.setEnabled(True)
    self.cancelBtn.setEnabled(False)

# --------- Runner signals ---------

@QtCore.pyqtSlot(str)
def _on_started(self, joined_cmd: str):
    # Stream to UI
    self.stdoutBox.appendPlainText(f"[started] {joined_cmd}")

@QtCore.pyqtSlot(str)
def _on_line(self, line: str):
    # Append live lines
    self.stdoutBox.appendPlainText(line)

@QtCore.pyqtSlot(int, str, str)
def _on_finished(self, rc: int, out: str, err: str):
    self.stdoutBox.appendPlainText(f"[finished] return code: {rc}")
    if err:
        self.stdoutBox.appendPlainText(f"[stderr]\n{err}")
    self.status.showMessage(f"Done (rc={rc})")

    # Re-enable UI
    self.runBtn.setEnabled(True)
    self.cancelBtn.setEnabled(False)

    # Try to load newest report automatically
    outputs_dir = Path(self.outEdit.text()).expanduser().resolve()
    report = find_newest_report(outputs_dir)
    if report and report.exists():
        self._load_html(report)
        self.stdoutBox.appendPlainText(f"[report] Loaded: {report}")
    else:
        self.stdoutBox.appendPlainText("[report] No report found; check CLI output and outputs dir.")

@QtCore.pyqtSlot(str)
def _on_error(self, message: str):
    self.stdoutBox.appendPlainText(f"[error] {message}")
    QtWidgets.QMessageBox.critical(self, "Process Error", message)
    self.runBtn.setEnabled(True)
    self.cancelBtn.setEnabled(False)
    self.status.showMessage("Error")

# --------- File dialogs ---------

@QtCore.pyqtSlot()
def on_browse_cli(self):
    start_dir = str(Path(self.cliEdit.text() or "").expanduser().parent) if self.cliEdit.text() else str(REPO)
    exe_filter = "Executables (*)" if not platform.system().lower().startswith("win") else "Executables (*.exe *.bat *.cmd);;All files (*)"
    path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CLI Executable", start_dir, exe_filter)
    if path:
        self.cliEdit.setText(path)

@QtCore.pyqtSlot()
def on_browse_out(self):
    path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Outputs Directory", str(Path(self.outEdit.text()).expanduser().resolve()))
    if path:
        self.outEdit.setText(path)

# --------- HTML loading and settings ---------

def _load_html(self, path: Path):
    self.web.load(QtCore.QUrl.fromLocalFile(str(path.resolve())))
    self.status.showMessage(f"Loaded report: {path.name}")

def _save_settings(self):
    if self._creating_ui:
        return
    sdict = {
        "cli": self.cliEdit.text().strip(),
        "outputs": self.outEdit.text().strip(),
        "extra_args": self.extraArgs.text().strip(),
    }
    self.settings.setValue(SETTINGS_KEY, sdict)

def closeEvent(self, event: QtGui.QCloseEvent) -> None:
    self._save_settings()
    super().closeEvent(event)
```

# --------------------------------------------------------------------------------------

# Entrypoint

# --------------------------------------------------------------------------------------

def main():
\# High-DPI awareness for crisp UI on retina/4K
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA\_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA\_UseHighDpiPixmaps, True)

```
app = QtWidgets.QApplication(sys.argv)
app.setOrganizationName(APP_ORG)
app.setApplicationName(APP_NAME)
w = MainWindow()
w.show()
sys.exit(app.exec_())
```

if **name** == "**main**":
main()
