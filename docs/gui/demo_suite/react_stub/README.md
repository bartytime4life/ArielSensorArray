# React Stub — SpectraMind V50 Dashboard

This stub explains how a React dashboard would bind to the thin FastAPI contract:

## API Contracts

- `POST /api/run`
  ```json
  {
    "cli": "spectramind",
    "args": ["diagnose","dashboard","--outputs.dir","outputs/diag_vX"],
    "cwd": "/path/to/repo"
  }

Response

{ "returncode": 0, "stdout": "...", "stderr": "..." }

    POST /api/artifacts

{ "glob": "outputs/diag_vX/**/*.html" }

Response

    { "files": ["/abs/path/diagnostic_report_vX.html", "..."] }

    GET /api/log?n=50000 → returns a string tail of logs/v50_debug_log.md.

Minimal React Flow (pseudocode)

const run = async () => {
  await fetch("/api/run", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      cli: "spectramind",
      args: ["diagnose","dashboard","--outputs.dir","outputs/diag_vX"],
      cwd: repoRoot
    })
  });
  const htmls = await fetch("/api/artifacts", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ glob: "outputs/diag_vX/**/*.html" })
  }).then(r => r.json());
  setReportPath(htmls.files[0]);
};

useEffect(() => {
  const id = setInterval(async () => {
    const log = await fetch(`/api/log?n=50000`).then(r => r.text());
    setLogText(log);
  }, 1500);
  return () => clearInterval(id);
}, []);

Rendering HTML Report

    Either fetch the HTML content and inject into a sandboxed iframe, or

    Serve static files via the backend and embed with <iframe src="/static/diagnostic_report_vX.html" />.
