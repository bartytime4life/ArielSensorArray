
---

# docs/gui/demo_suite/qt_diag_demo.py

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQt demo: press a button to run `spectramind diagnose dashboard`,
then show the resulting HTML report in a QWebEngineView.

Requirements:
  pip install PyQt5 PyQtWebEngine
"""
import os
import shlex
import subprocess
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView

REPO = Path.cwd()
DEFAULT_OUT = REPO / "outputs"

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpectraMind V50 — Qt Diagnose Demo")
        self.resize(1200, 800)

        self.cliEdit = QtWidgets.QLineEdit("spectramind")
        self.outEdit = QtWidgets.QLineEdit(str(DEFAULT_OUT))
        self.runBtn = QtWidgets.QPushButton("Run Diagnose")
        self.stdoutBox = QtWidgets.QPlainTextEdit(); self.stdoutBox.setReadOnly(True)
        self.web = QWebEngineView()

        form = QtWidgets.QFormLayout()
        form.addRow("CLI:", self.cliEdit)
        form.addRow("Outputs:", self.outEdit)

        top = QtWidgets.QWidget()
        top.setLayout(form)

        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(left)
        v.addWidget(top)
        v.addWidget(self.runBtn)
        v.addWidget(self.stdoutBox)
        splitter.addWidget(left)
        splitter.addWidget(self.web)
        self.setCentralWidget(splitter)

        self.runBtn.clicked.connect(self.on_run)

    def on_run(self):
        cli = self.cliEdit.text().strip() or "spectramind"
        outputs_dir = Path(self.outEdit.text().strip() or str(DEFAULT_OUT))
        cmd = [cli, "diagnose", "dashboard", "--outputs.dir", str(outputs_dir)]
        self.stdoutBox.appendPlainText("Running: " + " ".join(shlex.quote(p) for p in cmd))
        proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate()
        self.stdoutBox.appendPlainText(out or "(no output)")
        # find the newest report and load into webview
        report = max(list(outputs_dir.glob("**/diagnostic_report*.html")) + list(outputs_dir.glob("**/*dashboard*.html")), key=lambda p: p.stat().st_mtime, default=None)
        if report and report.exists():
            self.web.load(QtCore.QUrl.fromLocalFile(str(report.resolve())))
        else:
            self.stdoutBox.appendPlainText("No report found; check CLI output and outputs dir.")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    w = MainWindow()
    w.show()
    app.exec_()
