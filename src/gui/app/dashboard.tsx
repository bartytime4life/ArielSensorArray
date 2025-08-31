// src/gui/app/dashboard.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — Dashboard Page
// ----------------------------------------------------------------------------
// Purpose:
//   • Provide a single-screen overview of SpectraMind V50 diagnostics.
//   • Thin wrapper over CLI artifacts (JSON, PNG, HTML).
//   • Render plots (UMAP, t-SNE, GLL heatmaps, SHAP overlays, calibration).
//   • All data is pulled via FastAPI backend (src/server/api/\*).
//
// Principles:
//   • CLI-first: Dashboard only *renders* what `spectramind diagnose` produced.
//   • Reproducible: No hidden state; all actions map to logged CLI commands.
//   • Modern UX: React + Tailwind + shadcn/ui; responsive, accessible, minimal.
// ----------------------------------------------------------------------------
// Upgrades in this version:
//   • Env-configurable API/artifact bases (VITE\_API\_BASE, VITE\_ARTIFACTS\_BASE).
//   • ETag-aware summary fetch with 304 handling.
//   • Graceful artifact availability checks (HEAD) with inline guidance.
//   • Quick metrics cards (version/hash/time, artifact count).
//   • Accessible loading/empty/error states.
//   • Consistent helpers (urlJoin, encodeRelativePath) across GUI app.
// ============================================================================

import \* as React from "react";
import {
Card,
CardHeader,
CardContent,
CardDescription,
CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Loader2, RefreshCw, ExternalLink } from "lucide-react";

// ---------------------------------------------------------------------------
// Env + URL helpers
// ---------------------------------------------------------------------------

const API\_BASE = (import.meta as any).env?.VITE\_API\_BASE ?? "";
const ARTIFACTS\_BASE = (import.meta as any).env?.VITE\_ARTIFACTS\_BASE ?? "/artifacts";

function urlJoin(base: string, path: string): string {
const b = base.endsWith("/") ? base.slice(0, -1) : base;
const p = path.startsWith("/") ? path.slice(1) : path;
return `${b}/${p}`;
}

function encodeRelativePath(rel: string): string {
return rel
.split("/")
.map((seg) => encodeURIComponent(seg))
.join("/");
}

// ---------------------------------------------------------------------------
// Types (tolerant to partial server JSON shapes)
// ---------------------------------------------------------------------------

type Summary = {
version?: string;
generated\_at?: string | number;
run\_hash?: Record\<string, any>;
metrics?: Record\<string, any>;
artifacts?: Partial<{
umap\_html: string;
tsne\_html: string;
gll\_heatmap\_png: string;
shap\_overlay\_png: string;
calibration\_png: string;
}>;
};

type ArtifactCheck = {
name: string;
relPath: string;
url: string;
available: boolean | null; // null = unknown/pending
reason?: string;
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

async function fetchJSON<T>(pathWithQuery: string, init?: RequestInit): Promise<{ data: T; res: Response }> {
const res = await fetch(urlJoin(API\_BASE, pathWithQuery), init);
if (!res.ok) {
const msg = await (async () => {
try { return await res.text(); } catch { return ""; }
})();
throw new Error(`HTTP ${res.status} ${res.statusText} — ${msg || "error"}`);
}
const ct = res.headers.get("content-type") || "";
if (!ct.includes("application/json")) {
throw new Error(`Unexpected content-type: ${ct}`);
}
return { data: (await res.json()) as T, res };
}

function prettyTime(value?: string | number) {
if (value == null) return "—";
const d = new Date(value);
return Number.isFinite(d.getTime()) ? d.toLocaleString() : "—";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const DashboardPage: React.FC = () => {
// Summary
const \[summary, setSummary] = React.useState\<Summary | null>(null);
const \[etag, setEtag] = React.useState\<string | null>(null);
const \[loading, setLoading] = React.useState<boolean>(true);
const \[error, setError] = React.useState\<string | null>(null);

// Artifact availability checks
const \[artifacts, setArtifacts] = React.useState\<ArtifactCheck\[]>(\[]);

// Tabs
const \[tab, setTab] = React.useState<string>("overview");

// Abort controller for fetches
const abortRef = React.useRef\<AbortController | null>(null);

// Resolve artifact relative paths from summary or sensible defaults
function resolveArtifactPaths(s: Summary | null) {
const art = s?.artifacts ?? {};
return {
umap\_html: art.umap\_html ?? "umap.html",
tsne\_html: art.tsne\_html ?? "tsne.html",
gll\_heatmap\_png: art.gll\_heatmap\_png ?? "gll\_heatmap.png",
shap\_overlay\_png: art.shap\_overlay\_png ?? "shap\_overlay.png",
calibration\_png: art.calibration\_png ?? "calibration\_plot.png",
};
}

// HEAD check for artifact availability
async function checkArtifact(relPath: string): Promise<{ ok: boolean; reason?: string }> {
try {
const res = await fetch(urlJoin(ARTIFACTS\_BASE, encodeRelativePath(relPath)), {
method: "HEAD",
});
if (!res.ok) return { ok: false, reason: `HTTP ${res.status}` };
return { ok: true };
} catch (e: any) {
return { ok: false, reason: e?.message ?? "network error" };
}
}

// Load summary (ETag-aware)
const loadSummary = React.useCallback(async () => {
abortRef.current?.abort();
const ac = new AbortController();
abortRef.current = ac;

```
setLoading(true);
setError(null);
try {
  const headers: HeadersInit = {};
  if (etag) headers["If-None-Match"] = etag;

  const res = await fetch(urlJoin(API_BASE, "api/diagnostics/summary"), {
    headers,
    signal: ac.signal,
  });

  if (res.status === 304) {
    // Not modified — reuse previous summary
    setLoading(false);
    return;
  }
  if (!res.ok) {
    throw new Error(`summary HTTP ${res.status}`);
  }

  const newEtag = res.headers.get("ETag");
  if (newEtag) setEtag(newEtag);

  const data = (await res.json()) as Summary;
  if (!ac.signal.aborted) setSummary(data);
} catch (err: any) {
  if (!ac.signal.aborted) setError(err?.message ?? "failed to load summary");
} finally {
  if (!ac.signal.aborted) setLoading(false);
}
```

}, \[etag]);

// Build artifacts list and check availability
const refreshArtifacts = React.useCallback(async (s: Summary | null) => {
const paths = resolveArtifactPaths(s);
const candidates: ArtifactCheck\[] = \[
{ name: "UMAP", relPath: paths.umap\_html, url: urlJoin(ARTIFACTS\_BASE, encodeRelativePath(paths.umap\_html)), available: null },
{ name: "t-SNE", relPath: paths.tsne\_html, url: urlJoin(ARTIFACTS\_BASE, encodeRelativePath(paths.tsne\_html)), available: null },
{ name: "GLL Heatmap", relPath: paths.gll\_heatmap\_png, url: urlJoin(ARTIFACTS\_BASE, encodeRelativePath(paths.gll\_heatmap\_png)), available: null },
{ name: "SHAP × Symbolic", relPath: paths.shap\_overlay\_png, url: urlJoin(ARTIFACTS\_BASE, encodeRelativePath(paths.shap\_overlay\_png)), available: null },
{ name: "Calibration", relPath: paths.calibration\_png, url: urlJoin(ARTIFACTS\_BASE, encodeRelativePath(paths.calibration\_png)), available: null },
];
setArtifacts(candidates);

```
// Parallel HEAD checks
const results = await Promise.all(
  candidates.map(async (c) => {
    const r = await checkArtifact(c.relPath);
    return { ...c, available: r.ok, reason: r.reason };
  })
);
setArtifacts(results);
```

}, \[]);

// Initial load
React.useEffect(() => {
(async () => {
await loadSummary();
})();
return () => abortRef.current?.abort();
}, \[loadSummary]);

// Refresh artifacts when summary changes
React.useEffect(() => {
refreshArtifacts(summary);
}, \[summary, refreshArtifacts]);

// Quick metrics
const version = summary?.version ?? "unknown";
const runHash =
(summary?.run\_hash && (summary.run\_hash as any).hash) ??
(summary?.metrics && (summary.metrics as any).hash) ??
null;
const generated = prettyTime(summary?.generated\_at);

// Render helpers
function renderArtifactBadge(c: ArtifactCheck) {
if (c.available === true) {
return <Badge variant="secondary">Available</Badge>;
}
if (c.available === false) {
return <Badge variant="outline">Missing</Badge>;
}
return <Badge variant="outline">Checking…</Badge>;
}

function renderUnavailableHint(name: string) {
// Show the exact CLI that typically produces the artifacts
return ( <div className="text-xs text-muted-foreground">
{name} artifact not found. Run: <pre className="mt-1 rounded bg-muted p-2">
spectramind diagnose dashboard </pre> </div>
);
}

// -------------------------------------------------------------------------
// UI
// -------------------------------------------------------------------------
return ( <div className="w-full min-h-screen px-4 py-6 lg:px-8 bg-background">
{/\* Header \*/} <div className="mb-6 flex items-center justify-between"> <div> <h1 className="text-2xl font-semibold tracking-tight">SpectraMind V50 — Dashboard</h1> <p className="text-sm text-muted-foreground">
Thin, read-only overview of CLI-produced diagnostics. Zero analytics run in the browser. </p> </div> <div className="flex gap-2">
\<Button onClick={() => loadSummary()} disabled={loading}>
{loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <RefreshCw className="mr-2 h-4 w-4" />}
Refresh </Button> <Button variant="outline" asChild>
\<a href={urlJoin(API\_BASE, "api/diagnostics/health")} target="\_blank" rel="noreferrer">
API Health <ExternalLink className="ml-2 h-3 w-3" /> </a> </Button> </div> </div>

```
  {error && (
    <div className="mb-4 rounded border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive">
      ❌ Error loading diagnostics summary: {error}
    </div>
  )}

  {loading && !error && (
    <div
      role="status"
      aria-live="polite"
      className="mb-6 flex items-center gap-2 text-sm text-muted-foreground"
    >
      <Loader2 className="h-4 w-4 animate-spin" />
      Loading summary…
    </div>
  )}

  {/* Top cards */}
  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Build Version</CardDescription>
        <CardTitle className="text-lg">{version}</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-muted-foreground space-y-1">
        <div>
          Run hash:{" "}
          {runHash ? <code className="rounded bg-muted px-1 py-0.5">{String(runHash)}</code> : "—"}
        </div>
        <div>Generated: {generated}</div>
      </CardContent>
    </Card>

    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Artifacts</CardDescription>
        <CardTitle className="text-lg">{artifacts.filter((a) => a.available).length}</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-muted-foreground space-y-1">
        <div>Total tracked: {artifacts.length}</div>
        <div className="flex flex-wrap gap-1">
          {artifacts.map((a) => (
            <span key={a.name} className="inline-flex items-center gap-1">
              <span className="uppercase">{a.name}</span>
              {renderArtifactBadge(a)}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>

    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Quick Links</CardDescription>
        <CardTitle className="text-lg">Artifacts</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-2 text-xs">
        {artifacts.map((a) => (
          <a
            key={a.name}
            href={a.url}
            target="_blank"
            rel="noreferrer"
            className="underline text-primary hover:opacity-80"
          >
            {a.url}
          </a>
        ))}
      </CardContent>
    </Card>

    <Card>
      <CardHeader className="pb-2">
        <CardDescription>Reports</CardDescription>
        <CardTitle className="text-lg">/reports</CardTitle>
      </CardHeader>
      <CardContent className="text-xs text-muted-foreground">
        Browse HTML reports and notes discovered under <code>{ARTIFACTS_BASE}</code>.
        <div className="mt-2">
          <Button asChild size="sm" variant="secondary">
            <a href="/reports">Open Reports Browser</a>
          </Button>
        </div>
      </CardContent>
    </Card>
  </div>

  <Separator className="my-6" />

  {/* Tabs with embedded artifacts */}
  <Tabs value={tab} onValueChange={setTab} className="w-full">
    <TabsList>
      <TabsTrigger value="overview">Overview</TabsTrigger>
      <TabsTrigger value="umap">UMAP</TabsTrigger>
      <TabsTrigger value="tsne">t-SNE</TabsTrigger>
      <TabsTrigger value="gll">GLL Heatmap</TabsTrigger>
      <TabsTrigger value="shap">SHAP × Symbolic</TabsTrigger>
      <TabsTrigger value="calibration">Calibration</TabsTrigger>
    </TabsList>

    {/* Overview */}
    <TabsContent value="overview" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>Run Summary</CardTitle>
          <CardDescription>Snapshot of diagnostic_summary.json as provided by the server.</CardDescription>
        </CardHeader>
        <CardContent>
          {summary ? (
            <pre className="max-h-[420px] overflow-auto rounded bg-muted p-3 text-xs">
              {JSON.stringify(summary, null, 2)}
            </pre>
          ) : (
            <div className="text-sm text-muted-foreground">No summary available.</div>
          )}
        </CardContent>
      </Card>
    </TabsContent>

    {/* UMAP */}
    <TabsContent value="umap" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>UMAP Embedding</CardTitle>
          <CardDescription>Interactive HTML rendered by the CLI.</CardDescription>
        </CardHeader>
        <CardContent>
          {(() => {
            const a = artifacts.find((x) => x.name === "UMAP");
            if (!a) return <div className="text-sm text-muted-foreground">Not tracked.</div>;
            if (a.available === false) return renderUnavailableHint("UMAP");
            return (
              <iframe
                src={a.url}
                title="UMAP Plot"
                className="h-[520px] w-full rounded border"
              />
            );
          })()}
        </CardContent>
      </Card>
    </TabsContent>

    {/* t-SNE */}
    <TabsContent value="tsne" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>t-SNE Embedding</CardTitle>
          <CardDescription>Interactive HTML rendered by the CLI.</CardDescription>
        </CardHeader>
        <CardContent>
          {(() => {
            const a = artifacts.find((x) => x.name === "t-SNE");
            if (!a) return <div className="text-sm text-muted-foreground">Not tracked.</div>;
            if (a.available === false) return renderUnavailableHint("t-SNE");
            return (
              <iframe
                src={a.url}
                title="t-SNE Plot"
                className="h-[520px] w-full rounded border"
              />
            );
          })()}
        </CardContent>
      </Card>
    </TabsContent>

    {/* GLL Heatmap */}
    <TabsContent value="gll" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>GLL Heatmap</CardTitle>
          <CardDescription>PNG generated by diagnostics; higher indicates worse GLL.</CardDescription>
        </CardHeader>
        <CardContent>
          {(() => {
            const a = artifacts.find((x) => x.name === "GLL Heatmap");
            if (!a) return <div className="text-sm text-muted-foreground">Not tracked.</div>;
            if (a.available === false) return renderUnavailableHint("GLL Heatmap");
            return (
              <img
                src={a.url}
                alt="GLL Heatmap"
                className="rounded border shadow-sm max-h-[640px]"
              />
            );
          })()}
        </CardContent>
      </Card>
    </TabsContent>

    {/* SHAP × Symbolic */}
    <TabsContent value="shap" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>SHAP × Symbolic Overlay</CardTitle>
          <CardDescription>Overlay plot exported by the CLI for explainability.</CardDescription>
        </CardHeader>
        <CardContent>
          {(() => {
            const a = artifacts.find((x) => x.name === "SHAP × Symbolic");
            if (!a) return <div className="text-sm text-muted-foreground">Not tracked.</div>;
            if (a.available === false) return renderUnavailableHint("SHAP × Symbolic");
            return (
              <img
                src={a.url}
                alt="SHAP × Symbolic Overlay"
                className="rounded border shadow-sm max-h-[640px]"
              />
            );
          })()}
        </CardContent>
      </Card>
    </TabsContent>

    {/* Calibration */}
    <TabsContent value="calibration" className="mt-4">
      <Card>
        <CardHeader>
          <CardTitle>Calibration Diagnostics</CardTitle>
          <CardDescription>σ vs. residuals, coverage, and related plots.</CardDescription>
        </CardHeader>
        <CardContent>
          {(() => {
            const a = artifacts.find((x) => x.name === "Calibration");
            if (!a) return <div className="text-sm text-muted-foreground">Not tracked.</div>;
            if (a.available === false) return renderUnavailableHint("Calibration");
            return (
              <img
                src={a.url}
                alt="Calibration Plot"
                className="rounded border shadow-sm max-h-[640px]"
              />
            );
          })()}
        </CardContent>
      </Card>
    </TabsContent>
  </Tabs>

  <Separator className="my-6" />

  {/* Footer */}
  <div className="flex items-center justify-between">
    <div className="text-xs text-muted-foreground">
      API:{" "}
      <a
        className="underline"
        href={urlJoin(API_BASE, "api/diagnostics/summary")}
        target="_blank"
        rel="noreferrer"
      >
        {urlJoin(API_BASE, "api/diagnostics/summary")}
      </a>{" "}
      ·{" "}
      <a
        className="underline"
        href={urlJoin(API_BASE, "api/diagnostics/list")}
        target="_blank"
        rel="noreferrer"
      >
        {urlJoin(API_BASE, "api/diagnostics/list")}
      </a>
    </div>
    <div className="text-xs text-muted-foreground">
      CLI-first · Reproducible by construction
    </div>
  </div>
</div>
```

);
};

export default DashboardPage;

// ============================================================================
// Notes:
// • Artifacts are served from {ARTIFACTS\_BASE} (default "/artifacts").
// • Summary comes from GET {API\_BASE}/api/diagnostics/summary (ETag-aware).
// • If artifacts are missing, the UI shows the exact CLI to produce them:
//     spectramind diagnose dashboard
// • This component performs no analytics; it renders what the CLI produced.
// ============================================================================
