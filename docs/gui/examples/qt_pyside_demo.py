```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
docs/gui/examples/qt_pyside_demo.py
===================================

SpectraMind V50 — PySide Demo (Thin GUI Wrapper)

What this demo does
-------------------
• Provides a minimal, accessible GUI around the CLI-first pipeline.
• Invokes `spectramind` via QProcess (non-blocking) with user-selected options.
• Renders the latest diagnostics HTML report inside a Qt WebEngine view.
• Shows live CLI stdout/stderr and can tail logs/v50_debug_log.md.
• NEVER bypasses the pipeline: no hidden state or custom compute in the GUI.

Requirements
------------
    pip install PySide6 PySide6-Qt6-WebEngine

Launch
------
    python docs/gui/examples/qt_pyside_demo.py

Notes
-----
• This GUI is intentionally thin: it calls the CLI and visualizes artifacts only.
• If QtWebEngine is not available in your environment, you can open reports with
  the "Open in Browser" button as a fallback.
"""

from __future__ import annotations

import os
import shlex
import sys
import typing as T
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

# QtWebEngine is in a separate module; available on most platforms via PySide6
try:
    from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore
    HAS_WEBENGINE = True
except Exception:
    HAS_WEBENGINE = False


# --------------------------
# Small filesystem utilities
# --------------------------
def newest(glob_pattern: str) -> Path | None:
    """Return the newest file matching a glob pattern, or None if none found."""
    paths = list(Path().glob(glob_pattern))
    if not paths:
        return None
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths[0]


def find_latest_report(outputs_dir: Path) -> Path | None:
    """Heuristics to find the newest diagnostics report (HTML) in outputs_dir."""
    # Common naming patterns:
    #   outputs/.../diagnostic_report*.html
    #   outputs/.../*dashboard*.html
    for pat in (
        str(outputs_dir / "**" / "diagnostic_report*.html"),
        str(outputs_dir / "**" / "*dashboard*.html"),
    ):
        path = newest(pat)
        if path and path.exists():
            return path
    return None


def tail_bytes(path: Path, n: int = 50000) -> str:
    """Tail n bytes from a text file (UTF-8 best-effort)."""
    if not path.exists():
        return ""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n), os.SEEK_SET)
        data = f.read()
    text = data.decode("utf-8", errors="replace")
    return text.split("\n", 1)[-1] if size > n else text


# --------------------------
# Main Window
# --------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SpectraMind V50 — PySide Demo (Thin Wrapper)")
        self.resize(1280, 840)

        # Defaults
        self.repo_root = Path.cwd()
        self.outputs_dir = self.repo_root / "outputs"
        self.logs_path = self.repo_root / "logs" / "v50_debug_log.md"

        # Top controls
        self.cli_edit = QtWidgets.QLineEdit("spectramind")
        self.repo_edit = QtWidgets.QLineEdit(str(self.repo_root))
        self.out_edit = QtWidgets.QLineEdit(str(self.outputs_dir))

        self.umap_chk = QtWidgets.QCheckBox("UMAP")
        self.umap_chk.setChecked(True)
        self.tsne_chk = QtWidgets.QCheckBox("t-SNE")
        self.tsne_chk.setChecked(True)
        self.extra_edit = QtWidgets.QLineEdit("")
        self.extra_edit.setPlaceholderText("Extra CLI args (optional)")

        self.run_btn = QtWidgets.QPushButton("Run: diagnose dashboard")
        self.run_btn.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))

        # Output panes: stdout / report + log tail controls
        self.stdout_box = QtWidgets.QPlainTextEdit()
        self.stdout_box.setReadOnly(True)
        self.stdout_box.setPlaceholderText("CLI stdout/stderr will appear here...")

        self.open_report_btn = QtWidgets.QPushButton("Open Latest Report (External Browser)")
        self.reload_report_btn = QtWidgets.QPushButton("Reload Report")
        self.log_tail_btn = QtWidgets.QPushButton("Tail Log")
        self.log_bytes_spin = QtWidgets.QSpinBox()
        self.log_bytes_spin.setRange(1000, 2_000_000)
        self.log_bytes_spin.setValue(50000)
        self.log_bytes_spin.setSuffix(" bytes")

        # Web view (if available)
        if HAS_WEBENGINE:
            self.web = QWebEngineView()
            self.web.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.NoContextMenu)
        else:
            self.web = None

        # Process
        self.proc = QtCore.QProcess(self)
        self.proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._on_proc_output)
        self.proc.readyReadStandardError.connect(self._on_proc_output)  # merged anyway
        self.proc.finished.connect(self._on_proc_finished)
        self.proc.errorOccurred.connect(self._on_proc_error)

        # Build UI layout
        self._build_ui()

        # Connect signals
        self.run_btn.clicked.connect(self.on_run)
        self.open_report_btn.clicked.connect(self.on_open_in_browser)
        self.reload_report_btn.clicked.connect(self.on_reload_report)
        self.log_tail_btn.clicked.connect(self.on_tail_log)

        # Shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("Shift+R"), self, activated=self.on_run)
        QtGui.QShortcut(QtGui.QKeySequence("?"), self, activated=self._show_shortcuts_help)

        # Status bar info
        self.statusBar().showMessage("Ready. CLI-thin demo — no hidden state.")

    # --------------------------
    # UI construction
    # --------------------------
    def _build_ui(self) -> None:
        # Top form
        form = QtWidgets.QGridLayout()
        r = 0
        form.addWidget(QtWidgets.QLabel("CLI Executable"), r, 0)
        form.addWidget(self.cli_edit, r, 1, 1, 3)
        r += 1
        form.addWidget(QtWidgets.QLabel("Repository Root"), r, 0)
        form.addWidget(self.repo_edit, r, 1, 1, 3)
        r += 1
        form.addWidget(QtWidgets.QLabel("Outputs Directory"), r, 0)
        form.addWidget(self.out_edit, r, 1, 1, 3)
        r += 1
        form.addWidget(QtWidgets.QLabel("Options"), r, 0)
        form.addWidget(self.umap_chk, r, 1)
        form.addWidget(self.tsne_chk, r, 2)
        form.addWidget(self.extra_edit, r, 3)
        r += 1
        form.addWidget(self.run_btn, r, 0, 1, 4)

        top = QtWidgets.QWidget()
        top.setLayout(form)

        # Left column: stdout + log controls
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.addWidget(QtWidgets.QLabel("CLI Output"))
        left_layout.addWidget(self.stdout_box)

        log_row = QtWidgets.QHBoxLayout()
        log_row.addWidget(self.log_tail_btn)
        log_row.addWidget(self.log_bytes_spin)
        left_layout.addLayout(log_row)

        # Right column: web and report controls
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.addWidget(QtWidgets.QLabel("Diagnostics Report"))

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.reload_report_btn)
        btn_row.addWidget(self.open_report_btn)
        right_layout.addLayout(btn_row)

        if self.web is not None:
            right_layout.addWidget(self.web, 1)
        else:
            fallback = QtWidgets.QTextBrowser()
            fallback.setOpenExternalLinks(True)
            fallback.setHtml(
                "<h3>QtWebEngine not available</h3>"
                "<p>Use 'Open Latest Report (External Browser)' to view diagnostics.</p>"
            )
            right_layout.addWidget(fallback, 1)

        # Splitter for main panes
        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        # Central layout
        central = QtWidgets.QWidget()
        vmain = QtWidgets.QVBoxLayout(central)
        vmain.addWidget(top)
        vmain.addWidget(splitter, 1)
        self.setCentralWidget(central)

    # --------------------------
    # Helpers
    # --------------------------
    def _compose_command(self) -> list[str]:
        """Compose the CLI command from the UI state."""
        cli = self.cli_edit.text().strip() or "spectramind"

        # Update paths from fields
        self.repo_root = Path(self.repo_edit.text().strip() or str(Path.cwd()))
        self.outputs_dir = Path(self.out_edit.text().strip() or str(self.repo_root / "outputs"))

        cmd = [cli, "diagnose", "dashboard", "--outputs.dir", str(self.outputs_dir)]
        if not self.umap_chk.isChecked():
            cmd.append("--no-umap")
        if not self.tsne_chk.isChecked():
            cmd.append("--no-tsne")
        extra = self.extra_edit.text().strip()
        if extra:
            # Attempt safe split (support quotes)
            try:
                cmd.extend(shlex.split(extra))
            except ValueError:
                # fallback basic split
                cmd.extend(extra.split())
        return cmd

    def _append_cli_line(self, line: str) -> None:
        """Append a single line to the CLI output pane."""
        self.stdout_box.moveCursor(QtGui.QTextCursor.End)
        self.stdout_box.insertPlainText(line)
        if not line.endswith("\n"):
            self.stdout_box.insertPlainText("\n")
        self.stdout_box.moveCursor(QtGui.QTextCursor.End)

    def _load_latest_report_in_web(self) -> None:
        """Try to load the newest diagnostics HTML into the web view (if available)."""
        report = find_latest_report(self.outputs_dir)
        if report and report.exists():
            if HAS_WEBENGINE and self.web is not None:
                url = QtCore.QUrl.fromLocalFile(str(report.resolve()))
                self.web.load(url)
                self.statusBar().showMessage(f"Loaded: {report.name}")
            else:
                self.statusBar().showMessage(f"Report ready (open externally): {report}")
        else:
            self.statusBar().showMessage("No diagnostics report found yet.")

    # --------------------------
    # Slots: QProcess callbacks
    # --------------------------
    @QtCore.Slot()
    def _on_proc_output(self) -> None:
        """Handle live process output."""
        data = self.proc.readAllStandardOutput().data().decode("utf-8", errors="replace")
        if data:
            for line in data.splitlines():
                self._append_cli_line(line)

    @QtCore.Slot(int, QtCore.QProcess.ExitStatus)
    def _on_proc_finished(self, exit_code: int, status: QtCore.QProcess.ExitStatus) -> None:
        """Process finished; show status and attempt to load report."""
        if status == QtCore.QProcess.ExitStatus.NormalExit and exit_code == 0:
            self._append_cli_line("[OK] CLI completed.")
            self.statusBar().showMessage("CLI completed successfully.")
        else:
            self._append_cli_line(f"[ERR] CLI exited with code {exit_code}.")
            self.statusBar().showMessage(f"CLI error (exit {exit_code}).")
        # Try to load the latest report (if generated)
        self._load_latest_report_in_web()

    @QtCore.Slot(QtCore.QProcess.ProcessError)
    def _on_proc_error(self, err: QtCore.QProcess.ProcessError) -> None:
        """Process error; notify user."""
        self._append_cli_line(f"[ERR] Process error: {err}")
        self.statusBar().showMessage("Process error.")

    # --------------------------
    # Slots: UI actions
    # --------------------------
    @QtCore.Slot()
    def on_run(self) -> None:
        """Run the CLI command with QProcess (non-blocking)."""
        cmd = self._compose_command()
        self.stdout_box.appendPlainText("")
        self._append_cli_line(">>> Running:")
        self._append_cli_line("    " + " ".join(shlex.quote(p) for p in cmd))

        # Start process in given repo_root
        self.proc.setWorkingDirectory(str(self.repo_root))
        # Run without shell; pass arguments as list to avoid shell injection
        program, args = cmd[0], cmd[1:]
        try:
            self.proc.start(program, args)
            started = self.proc.waitForStarted(5000)
            if not started:
                self._append_cli_line("[ERR] Failed to start process (is CLI on PATH?)")
                self.statusBar().showMessage("Failed to start.")
        except Exception as e:
            self._append_cli_line(f"[ERR] Exception while starting process: {e}")
            self.statusBar().showMessage("Exception while starting process.")

    @QtCore.Slot()
    def on_open_in_browser(self) -> None:
        """Open newest report in the OS default browser."""
        report = find_latest_report(self.outputs_dir)
        if report and report.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(report.resolve())))
            self.statusBar().showMessage(f"Opening: {report.name}")
        else:
            self.statusBar().showMessage("No report found to open.")

    @QtCore.Slot()
    def on_reload_report(self) -> None:
        """Reload the embedded report (no CLI call)."""
        self._load_latest_report_in_web()

    @QtCore.Slot()
    def on_tail_log(self) -> None:
        """Tail logs/v50_debug_log.md and append to CLI output."""
        n = int(self.log_bytes_spin.value())
        text = tail_bytes(self.logs_path, n)
        if text:
            self._append_cli_line(">>> Log tail:")
            for line in text.splitlines():
                self._append_cli_line(line)
        else:
            self._append_cli_line(">>> No log found or log is empty.")

    @QtCore.Slot()
    def _show_shortcuts_help(self) -> None:
        """Overlay with quick keyboard shortcuts."""
        msg = (
            "<b>Shortcuts</b><br>"
            "• <b>Shift+R</b>: Run diagnose dashboard<br>"
            "• <b>?</b>: Show shortcuts<br>"
        )
        QtWidgets.QMessageBox.information(self, "Shortcuts", msg)

    # --------------------------
    # Close event
    # --------------------------
    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        """Ensure the child process is terminated when the window closes."""
        try:
            if self.proc.state() != QtCore.QProcess.ProcessState.NotRunning:
                self.proc.kill()
                self.proc.waitForFinished(1000)
        except Exception:
            pass
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    app = QtWidgets.QApplication(argv or sys.argv)
    app.setApplicationName("SpectraMind V50 — PySide Demo")
    app.setOrganizationName("SpectraMind")
    app.setApplicationVersion("0.1.0")

    # Use a readable default font size for accessibility
    font = app.font()
    if font.pointSize() < 11:
        font.setPointSize(11)
        app.setFont(font)

    # High DPI behavior
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        QtGui.QGuiApplication.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    w = MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```
