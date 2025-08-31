* Clear **API contracts** (FastAPI endpoints)
* A **minimal React pseudocode flow** showing fetch + state binding
* **Rendering strategies** (sandboxed iframe vs static serving)
* Notes on **Hydra config binding**, reproducibility guardrails, and audit logs

---

# React Stub — SpectraMind V50 Dashboard

This stub explains how a **React dashboard** would bind to the thin **FastAPI backend contract**, while preserving SpectraMind’s **CLI-first, GUI-optional** design.

---

## 1. API Contracts (FastAPI)

The backend provides three core endpoints (see `backend.py`):

### `POST /api/run`

```json
{
  "cli": "spectramind",
  "args": ["diagnose", "dashboard", "--outputs.dir", "outputs/diag_vX"],
  "cwd": "/path/to/repo"
}
```

**Response:**

```json
{
  "returncode": 0,
  "stdout": "...",
  "stderr": "...",
  "command": "spectramind diagnose dashboard --outputs.dir outputs/diag_vX",
  "cwd": "/path/to/repo",
  "timestamp": "2025-08-31T13:45:00Z"
}
```

---

### `POST /api/artifacts`

```json
{ "glob": "outputs/diag_vX/**/*.html" }
```

**Response:**

```json
{ "files": ["/abs/path/diagnostic_report_vX.html", "..."] }
```

---

### `GET /api/log?n=50000`

Returns a **string tail** of `logs/v50_debug_log.md` for auditability.

---

## 2. Minimal React Flow (pseudocode)

```tsx
// Example React hook-based pseudocode

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
```

---

## 3. Rendering HTML Report

Two safe options:

1. **Sandboxed iframe injection**

```tsx
<iframe srcDoc={htmlContent} sandbox="allow-same-origin allow-scripts" />
```

2. **Static serving via backend**
   Expose artifacts with FastAPI’s `StaticFiles` and embed:

```html
<iframe src="/static/diagnostic_report_vX.html" />
```

---

## 4. Integration Notes

* **Hydra binding**: CLI args (`--outputs.dir`) are passed exactly as in terminal runs.
* **Reproducibility guardrails**: GUI actions write only to:

  * Hydra configs (`configs/*.yaml`)
  * Artifacts (`outputs/**`)
  * Logs (`logs/v50_debug_log.md`)
* **Auditability**: All `/api/run` calls append to `v50_debug_log.md` with CLI command, timestamp, return code.
* **Optional GUI**: React layer is thin — no hidden state outside CLI/JSON outputs.

---
