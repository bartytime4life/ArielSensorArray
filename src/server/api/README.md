# 🌐 `src/server/api/` — FastAPI Layer for SpectraMind V50

## 0) Purpose & Scope

This directory defines the **API façade** for the SpectraMind V50 system (NeurIPS 2025 Ariel Data Challenge).  
It exposes CLI-produced artifacts (JSON, HTML, PNG) through a thin, secure, and reproducible FastAPI server.

**Philosophy**:
- **CLI-first, GUI-optional**: All analytics and modeling are performed via the Typer/Hydra CLI (`spectramind …`).  
  The API never computes results directly; it only orchestrates and serves artifacts:contentReference[oaicite:2]{index=2}.
- **Reproducibility by construction**: Server responses always point to CLI-generated files tied to config + run hash:contentReference[oaicite:3]{index=3}.
- **Minimal, air-gapped safe**: No external IAM or cloud dependencies. Local header-based authz, optional CORS, static file serving.

---

## 1) Modules

### `diagnostics.py`
- **Endpoints**:
  - `GET /api/diagnostics/health` → report status + artifact dir.
  - `GET /api/diagnostics/summary` → stream `diagnostic_summary.json` (produced by `spectramind diagnose`).
  - `POST /api/diagnostics/run` → optionally trigger `spectramind diagnose dashboard`.
- **Static mount**: `/artifacts` → serves HTML/PNG/JSON diagnostic outputs (e.g., UMAP, SHAP overlays).
- **Config**:
  - `ARTIFACTS_DIR` (default: `./artifacts`)
  - `DIAGNOSTICS_SUMMARY_FILE` (default: `diagnostic_summary.json`)
  - `SPECTRAMIND_CLI` (default: `spectramind`)
  - `DIAGNOSE_SUBCOMMAND` (default: `diagnose dashboard`)
  - `CLI_TIMEOUT_SECONDS` (default: 1800)

### `authz.py`
- Minimal, configurable **authorization layer**.
- Modes:
  - `OFF`: (default) allow all requests — useful for dev/CI.
  - `HEADER_API_KEY`: validate `X-API-Key` or `Authorization: Bearer …`.
- Policy defined via:
  - `AUTHZ_POLICY_FILE` (JSON/YAML with users, roles, scopes)
  - `AUTHZ_API_KEYS_JSON` (inline ENV mapping)
- Provides FastAPI dependencies:
  - `get_current_user()`
  - `require_roles()`, `require_any_role()`
  - `require_scopes()`, `require_any_scope()`

### `main.py`
- **Entrypoint**: `uvicorn src.server.main:app`
- Routes:
  - `/` → redirects to `/docs`
  - `/health` → liveness probe
  - `/version` → version string + run hash (`run_hash_summary_v50.json`):contentReference[oaicite:4]{index=4}
  - `/me` → echo current user (via `authz`)
- Mounts:
  - diagnostics API (`/api/diagnostics/*`)
  - static artifacts (`/artifacts/*`)
- Configurable via ENV: `HOST`, `PORT`, `RELOAD`, `APP_TITLE`, `APP_DESC`, etc.

---

## 2) Integration Pattern

All API calls **map back to CLI operations**:

```mermaid
flowchart TD
  A[GUI Action] --> B[API Endpoint]
  B --> C[Typer CLI (`spectramind ...`)]
  C --> D[Hydra Configs (`/configs/*.yaml`)]
  C --> E[Artifacts (JSON/HTML/plots)]
  C --> F[Logs (`v50_debug_log.md`)]
  E --> G[GUI Rendering]
````

This ensures that **server outputs are always reproducible** and traceable to CLI invocations + config hashes.

---

## 3) Development Notes

* **Local Dev**: run with `uvicorn src.server.main:app --reload --port 8000`.
* **Testing**:

  * Use `pytest` with FastAPI’s `TestClient`.
  * Smoke test authz with inline policy:

    ```bash
    AUTHZ_MODE=HEADER_API_KEY \
    AUTHZ_API_KEYS_JSON='{"abc123":{"id":"dev","roles":["admin"],"scopes":["*"]}}' \
    pytest tests/server/
    ```
* **CI/CD**:

  * The API is validated in GitHub Actions alongside CLI tests.
  * No direct data science occurs here; failures usually indicate mis-wired paths, missing artifacts, or authz misconfig.

---

## 4) Related References

* **SpectraMind V50 Project Analysis** — confirms CLI-first architecture and reproducibility philosophy.
* **Update & Strategy Plan** — recommends keeping the API thin, authz local, and artifacts version-controlled.
* **Mermaid Reference** — diagrams like the above render directly in GitHub Markdown.

---

## 5) Future Extensions

* Add `/api/diagnostics/explain` (wrap SHAP/symbolic explainers).
* Stream CLI logs (via WebSocket) for real-time dashboard integration.
* Optional **GUI embedding** (React app under `/ui/`) mirroring CLI artifacts.
* Extend authz for role-based scope filtering (e.g., only `read` role can access `/artifacts/*`).

---

```
