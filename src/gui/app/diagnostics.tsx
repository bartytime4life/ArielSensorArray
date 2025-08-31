// src/gui/app/diagnostics.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — Diagnostics Dashboard (React + Tailwind + shadcn/ui)
// ----------------------------------------------------------------------------
// Purpose
//   Thin, read-only GUI for inspecting CLI-produced diagnostics.
//   • Fetches /api/diagnostics/summary (ETag-aware, optional manual refresh)
//   • Lists /api/diagnostics/list artifacts (HTML/PNG/CSV/JSON) with filters
//   • Quick links to rendered artifacts under /artifacts/*
//   • Optional charts for quick-look metrics if present in summary JSON
//
// Design
//   • GUI is *stateless* with respect to analytics; server/CLI is the source.
//   • Works in air-gapped environments; no external CDNs or telemetry.
//   • Tailwind + shadcn/ui components; Recharts for simple plots.
//
// Notes
//   • This file assumes your project is configured with:
//       - TailwindCSS
//       - shadcn/ui (installed under "@/components/ui/*")
//       - Recharts
//   • If you don't use shadcn/ui, replace imports with your component library.
//   • Safe against missing/partial fields in diagnostic_summary.json.
// ============================================================================

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  LineChart,
  Line,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";

// -------------------------------------------------------------------------------------
// Types (tolerant to partial server JSON)
// -------------------------------------------------------------------------------------
type Summary = {
  version?: string;
  project_root?: string;
  run_hash?: Record<string, any>;
  version_sha256?: string | null;
  generated_at?: string | number;
  metrics?: Record<string, any>;
  // Example shapes we *may* see; code guards for absence
  planets?: Array<{ id: string; gll?: number; mae?: number; rmse?: number }>;
  histograms?: {
    gll?: Array<{ bin: string | number; count: number }>;
    mae?: Array<{ bin: string | number; count: number }>;
  };
};

type ArtifactItem = {
  path: string;   // relative path under ARTIFACTS_DIR
  size: number;   // bytes
  mtime: number;  // epoch seconds
  ext: string;    // "html" | "png" | "json" | "csv" | "md" | ...
};

type ArtifactListResponse = {
  dir: string;
  count: number;
  total: number;
  items: ArtifactItem[];
};

// -------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------

/** Format bytes into human-readable string. */
function prettySize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let v = bytes / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${units[i]}`;
}

/** Convert unix seconds to ISO string local time. */
function prettyTime(sec: number | undefined): string {
  if (!sec && sec !== 0) return "-";
  const d = new Date(sec * 1000);
  return d.toLocaleString();
}

/** Extract quick metrics defensively from summary. */
function extractQuickMetrics(summary: Summary) {
  const version = summary.version ?? "unknown";
  const hash = (summary.run_hash && (summary.run_hash as any).hash) ?? null;
  const gen = summary.generated_at
    ? new Date(summary.generated_at).toLocaleString()
    : null;

  // Try several shapes to get per-planet GLLs
  let planetRows: Array<{ id: string; gll: number; mae?: number; rmse?: number }> = [];
  if (Array.isArray(summary.planets)) {
    planetRows = summary.planets
      .map((p) => ({
        id: String(p.id),
        gll: typeof p.gll === "number" ? p.gll : NaN,
        mae: typeof p.mae === "number" ? p.mae : undefined,
        rmse: typeof p.rmse === "number" ? p.rmse : undefined,
      }))
      .filter((r) => !Number.isNaN(r.gll));
  } else if (summary.metrics && Array.isArray((summary.metrics as any).planets)) {
    planetRows = (summary.metrics as any).planets
      .map((p: any) => ({
        id: String(p.id),
        gll: typeof p.gll === "number" ? p.gll : NaN,
        mae: typeof p.mae === "number" ? p.mae : undefined,
        rmse: typeof p.rmse === "number" ? p.rmse : undefined,
      }))
      .filter((r: any) => !Number.isNaN(r.gll));
  }

  // Sort by worst GLL if available
  planetRows.sort((a, b) => (b.gll ?? 0) - (a.gll ?? 0));

  // Histogram fallback
  const gllHist =
    summary?.histograms?.gll ??
    (summary.metrics && (summary.metrics as any).gll_hist) ??
    [];

  return { version, hash, gen, planetRows, gllHist };
}

/** Get a color badge variant by extension. */
function extBadgeVariant(ext: string): "default" | "secondary" | "destructive" | "outline" {
  const e = ext.toLowerCase();
  if (e === "html") return "default";
  if (e === "png" || e === "jpg" || e === "jpeg" || e === "svg") return "secondary";
  if (e === "json" || e === "csv" || e === "md") return "outline";
  return "outline";
}

// -------------------------------------------------------------------------------------
// Main Component
// -------------------------------------------------------------------------------------

export default function DiagnosticsPage(): JSX.Element {
  // Summary state (ETag-aware)
  const [summary, setSummary] = React.useState<Summary | null>(null);
  const [summaryEtag, setSummaryEtag] = React.useState<string | null>(null);
  const [summaryLoading, setSummaryLoading] = React.useState<boolean>(false);
  const [summaryError, setSummaryError] = React.useState<string | null>(null);

  // Artifact listing state
  const [artifacts, setArtifacts] = React.useState<ArtifactItem[]>([]);
  const [artifactsTotal, setArtifactsTotal] = React.useState<number>(0);
  const [artifactFilter, setArtifactFilter] = React.useState<string>("all"); // "all" | "html" | "png" | ...
  const [artifactQuery, setArtifactQuery] = React.useState<string>("");
  const [showNewestOnly, setShowNewestOnly] = React.useState<boolean>(true);
  const [artifactLoading, setArtifactLoading] = React.useState<boolean>(false);
  const [artifactError, setArtifactError] = React.useState<string | null>(null);

  // UI state
  const [activeTab, setActiveTab] = React.useState<string>("overview");

  // ---------------------------------------------
  // Fetch summary (ETag-aware)
  // ---------------------------------------------
  const fetchSummary = React.useCallback(async () => {
    setSummaryLoading(true);
    setSummaryError(null);
    try {
      const headers: HeadersInit = {};
      if (summaryEtag) headers["If-None-Match"] = summaryEtag;

      const res = await fetch("/api/diagnostics/summary", { headers });

      if (res.status === 304) {
        // Not modified — keep current summary
        setSummaryLoading(false);
        return;
      }
      if (!res.ok) {
        throw new Error(`summary HTTP ${res.status}`);
      }
      const etag = res.headers.get("ETag");
      if (etag) setSummaryEtag(etag);
      const data = (await res.json()) as Summary;
      setSummary(data);
    } catch (err: any) {
      setSummaryError(err?.message ?? "failed to fetch summary");
    } finally {
      setSummaryLoading(false);
    }
  }, [summaryEtag]);

  // ---------------------------------------------
  // Fetch artifacts list
  // ---------------------------------------------
  const fetchArtifacts = React.useCallback(async () => {
    setArtifactLoading(true);
    setArtifactError(null);
    try {
      // Pass optional extension filter to server to reduce payload (if supported)
      const params = new URLSearchParams();
      if (artifactFilter !== "all") params.set("exts", artifactFilter);
      params.set("limit", "500");

      const res = await fetch(`/api/diagnostics/list?${params.toString()}`);
      if (!res.ok) throw new Error(`artifacts HTTP ${res.status}`);
      const data = (await res.json()) as ArtifactListResponse;

      let items = data.items ?? [];

      // Client-side search filter
      if (artifactQuery.trim()) {
        const q = artifactQuery.toLowerCase();
        items = items.filter((it) => it.path.toLowerCase().includes(q));
      }

      // Newest first if toggled
      if (showNewestOnly) {
        items = [...items].sort((a, b) => b.mtime - a.mtime);
      }

      setArtifacts(items);
      setArtifactsTotal(data.total ?? items.length);
    } catch (err: any) {
      setArtifactError(err?.message ?? "failed to fetch artifacts");
    } finally {
      setArtifactLoading(false);
    }
  }, [artifactFilter, artifactQuery, showNewestOnly]);

  // Initial loads
  React.useEffect(() => {
    fetchSummary();
  }, [fetchSummary]);

  React.useEffect(() => {
    fetchArtifacts();
  }, [fetchArtifacts]);

  // Derived quick metrics
  const quick = React.useMemo(() => extractQuickMetrics(summary || {}), [summary]);

  // Filtered artifacts for display
  const visibleArtifacts = React.useMemo(() => artifacts, [artifacts]);

  // ---------------------------------------------
  // UI
  // ---------------------------------------------
  return (
    <div className="w-full min-h-screen px-4 py-6 lg:px-8 bg-background">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Diagnostics</h1>
          <p className="text-sm text-muted-foreground">
            Read-only view into CLI-produced artifacts and summary metrics. The CLI is the
            source of truth for all analytics.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => fetchArtifacts()}>
            Refresh Artifacts
          </Button>
          <Button onClick={() => fetchSummary()} disabled={summaryLoading}>
            {summaryLoading ? "Refreshing…" : "Refresh Summary"}
          </Button>
        </div>
      </div>

      {/* Top cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Build Version</CardDescription>
            <CardTitle className="text-lg">{quick.version}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              Run hash:{" "}
              {quick.hash ? (
                <code className="rounded bg-muted px-1 py-0.5">{String(quick.hash)}</code>
              ) : (
                <span>—</span>
              )}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              Generated: {quick.gen ?? "—"}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Artifacts</CardDescription>
            <CardTitle className="text-lg">{visibleArtifacts.length}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              Total tracked: {artifactsTotal}
            </div>
            <div className="text-xs text-muted-foreground">
              Newest-first: {showNewestOnly ? "Yes" : "No"}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Worst GLL (top 1)</CardDescription>
            <CardTitle className="text-lg">
              {quick.planetRows[0]?.gll != null
                ? quick.planetRows[0].gll.toFixed(4)
                : "—"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-muted-foreground">
              Planet:{" "}
              {quick.planetRows[0]?.id ? (
                <code className="rounded bg-muted px-1 py-0.5">
                  {quick.planetRows[0].id}
                </code>
              ) : (
                "—"
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Quick Links</CardDescription>
            <CardTitle className="text-lg">Artifacts</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {[
              "umap.html",
              "tsne.html",
              "diagnostic_report_v1.html",
              "symbolic_rule_table.html",
            ].map((name) => (
              <a
                key={name}
                href={`/artifacts/${name}`}
                className="text-xs underline text-primary hover:opacity-80"
                target="_blank"
                rel="noreferrer"
              >
                /artifacts/{name}
              </a>
            ))}
          </CardContent>
        </Card>
      </div>

      <Separator className="my-6" />

      <Tabs defaultValue={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
          <TabsTrigger value="charts">Charts</TabsTrigger>
        </TabsList>

        {/* Overview */}
        <TabsContent value="overview" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Summary</CardTitle>
              <CardDescription>
                Raw JSON snapshot of diagnostic summary (safely truncated in UI).
              </CardDescription>
            </CardHeader>
            <CardContent>
              {summaryError ? (
                <div className="text-sm text-destructive">Error: {summaryError}</div>
              ) : summaryLoading ? (
                <div className="text-sm text-muted-foreground">Loading…</div>
              ) : summary ? (
                <pre className="max-h-[420px] overflow-auto rounded bg-muted p-3 text-xs">
                  {JSON.stringify(summary, null, 2)}
                </pre>
              ) : (
                <div className="text-sm text-muted-foreground">No summary available.</div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Top Planets by GLL</CardTitle>
              <CardDescription>
                Worst (highest) Gaussian log-likelihood first — if provided by summary.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {quick.planetRows.length === 0 ? (
                <div className="text-sm text-muted-foreground">No per-planet GLL found.</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-muted-foreground">
                      <tr className="[&>th]:px-2 [&>th]:py-1 text-left border-b">
                        <th>Planet</th>
                        <th>GLL</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {quick.planetRows.slice(0, 12).map((r) => (
                        <tr key={r.id} className="[&>td]:px-2 [&>td]:py-1 border-b hover:bg-muted/40">
                          <td>
                            <code className="rounded bg-muted px-1 py-0.5">{r.id}</code>
                          </td>
                          <td>{Number.isFinite(r.gll) ? r.gll.toFixed(4) : "—"}</td>
                          <td>{r.mae != null ? r.mae.toFixed(4) : "—"}</td>
                          <td>{r.rmse != null ? r.rmse.toFixed(4) : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Artifacts */}
        <TabsContent value="artifacts" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Artifact Browser</CardTitle>
              <CardDescription>
                Files produced by the CLI (served from <code>/artifacts</code>).
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-1">
                  <Label htmlFor="query">Search</Label>
                  <Input
                    id="query"
                    placeholder="Filter by path (e.g. .html, umap, planet_42)"
                    value={artifactQuery}
                    onChange={(e) => setArtifactQuery(e.currentTarget.value)}
                  />
                </div>

                <div className="space-y-1">
                  <Label>Extension</Label>
                  <Select value={artifactFilter} onValueChange={setArtifactFilter}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="All" />
                    </SelectTrigger>
                    <SelectContent>
                      {["all", "html", "png", "jpg", "svg", "json", "csv", "md"].map((opt) => (
                        <SelectItem key={opt} value={opt}>
                          {opt.toUpperCase()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1">
                  <Label>Newest first</Label>
                  <div className="flex h-10 items-center gap-3 rounded-md border px-3">
                    <Switch checked={showNewestOnly} onCheckedChange={setShowNewestOnly} />
                    <span className="text-sm text-muted-foreground">
                      {showNewestOnly ? "Enabled" : "Disabled"}
                    </span>
                  </div>
                </div>

                <div className="space-y-1">
                  <Label className="invisible">Action</Label>
                  <div className="flex h-10 items-center">
                    <Button variant="secondary" onClick={() => fetchArtifacts()}>
                      Apply
                    </Button>
                  </div>
                </div>
              </div>

              {artifactError ? (
                <div className="text-sm text-destructive">Error: {artifactError}</div>
              ) : artifactLoading ? (
                <div className="text-sm text-muted-foreground">Loading…</div>
              ) : visibleArtifacts.length === 0 ? (
                <div className="text-sm text-muted-foreground">No artifacts found.</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-muted-foreground">
                      <tr className="[&>th]:px-2 [&>th]:py-1 text-left border-b">
                        <th>Path</th>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Modified</th>
                        <th>Open</th>
                      </tr>
                    </thead>
                    <tbody>
                      {visibleArtifacts.map((it) => (
                        <tr key={it.path} className="[&>td]:px-2 [&>td]:py-1 border-b hover:bg-muted/40">
                          <td className="max-w-[560px] truncate">{it.path}</td>
                          <td>
                            <Badge variant={extBadgeVariant(it.ext)} className="uppercase">
                              {it.ext}
                            </Badge>
                          </td>
                          <td className="whitespace-nowrap">{prettySize(it.size)}</td>
                          <td className="whitespace-nowrap">{prettyTime(it.mtime)}</td>
                          <td>
                            <a
                              href={`/artifacts/${encodeURI(it.path)}`}
                              target="_blank"
                              rel="noreferrer"
                              className="text-xs underline text-primary hover:opacity-80"
                            >
                              /artifacts/{it.path}
                            </a>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Charts */}
        <TabsContent value="charts" className="mt-4 space-y-4">
          <div className="grid gap-4 lg:grid-cols-2">
            <Card className="h-[360px]">
              <CardHeader>
                <CardTitle>GLL Histogram</CardTitle>
                <CardDescription>Shows distribution if present in summary.</CardDescription>
              </CardHeader>
              <CardContent className="h-[280px]">
                {Array.isArray(quick.gllHist) && quick.gllHist.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={quick.gllHist}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="bin" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="count" fill="hsl(var(--primary))" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="text-sm text-muted-foreground">No GLL histogram available.</div>
                )}
              </CardContent>
            </Card>

            <Card className="h-[360px]">
              <CardHeader>
                <CardTitle>Per-Planet GLL (Top 20)</CardTitle>
                <CardDescription>Higher bars indicate worse (higher) GLL.</CardDescription>
              </CardHeader>
              <CardContent className="h-[280px]">
                {quick.planetRows.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={quick.planetRows.slice(0, 20).map((r) => ({ id: r.id, gll: r.gll }))}
                      margin={{ left: 8, right: 8 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="id" hide />
                      <YAxis />
                      <Tooltip />
                      <Line
                        type="monotone"
                        dataKey="gll"
                        stroke="hsl(var(--primary))"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="text-sm text-muted-foreground">No per-planet metrics found.</div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>

      <Separator className="my-6" />

      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          API:{" "}
          <a className="underline" href="/api/diagnostics/health" target="_blank" rel="noreferrer">
            /api/diagnostics/health
          </a>{" "}
          ·{" "}
          <a className="underline" href="/api/diagnostics/summary" target="_blank" rel="noreferrer">
            /api/diagnostics/summary
          </a>{" "}
          ·{" "}
          <a className="underline" href="/api/authz/health" target="_blank" rel="noreferrer">
            /api/authz/health
          </a>
        </div>
        <div className="text-xs text-muted-foreground">CLI-first · Reproducible by construction</div>
      </div>
    </div>
  );
}
