// src/gui/app/reports.tsx
// ============================================================================
// 📄 SpectraMind V50 — Reports Browser (React + Tailwind + shadcn/ui)
// ----------------------------------------------------------------------------
// Purpose
//   A thin, read-only page to discover and preview CLI-produced HTML reports
//   (e.g., UMAP/t-SNE dashboards, diagnostic_report_v*.html) served from
//   `/artifacts`. It talks to the server-only API:
//      • GET /api/diagnostics/list?exts=html,md   (for discovery)
//      • Static files under /artifacts/*           (for rendering)
//
// Design
//   • No analytics here; the CLI is the source of truth.
//   • IFrame preview only for same-origin HTML under /artifacts.
//   • Safe guards for missing API endpoints or empty states.
//
// Notes
//   • Requires Tailwind + shadcn/ui in your project.
//   • If your server uses a different path for the list endpoint, update below.
// ============================================================================

import React from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

// -------------------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------------------

type ArtifactItem = {
  path: string; // relative under /artifacts
  size: number; // bytes
  mtime: number; // epoch seconds
  ext: string; // lowercase extension (e.g. "html", "md", "json")
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

function prettyTime(sec: number): string {
  const d = new Date((sec ?? 0) * 1000);
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : "-";
}

function isReportCandidate(path: string): boolean {
  const p = path.toLowerCase();
  // Heuristics for typical V50 report names
  return (
    p.endsWith(".html") &&
    (p.includes("umap") ||
      p.includes("tsne") ||
      p.includes("report") ||
      p.includes("dashboard") ||
      p.includes("symbolic") ||
      p.includes("overview"))
  );
}

function classifyReport(path: string): string {
  const p = path.toLowerCase();
  if (p.includes("umap")) return "UMAP";
  if (p.includes("tsne")) return "t-SNE";
  if (p.includes("symbolic")) return "Symbolic";
  if (p.includes("dashboard")) return "Dashboard";
  if (p.includes("report")) return "Report";
  return "Other";
}

function extBadgeVariant(ext: string): "default" | "secondary" | "destructive" | "outline" {
  const e = ext.toLowerCase();
  if (e === "html") return "default";
  if (e === "md") return "secondary";
  return "outline";
}

// -------------------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------------------

export default function ReportsPage(): JSX.Element {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const [items, setItems] = React.useState<ArtifactItem[]>([]);
  const [total, setTotal] = React.useState(0);

  const [query, setQuery] = React.useState("");
  const [typeFilter, setTypeFilter] = React.useState<string>("all"); // all | umap | tsne | symbolic | dashboard | report | other
  const [newestFirst, setNewestFirst] = React.useState(true);

  const [activeTab, setActiveTab] = React.useState("reports");
  const [activePath, setActivePath] = React.useState<string | null>(null);

  const [mdList, setMdList] = React.useState<ArtifactItem[]>([]);
  const [mdPreview, setMdPreview] = React.useState<string | null>(null);
  const [mdContent, setMdContent] = React.useState<string>("");

  // Fetch list of artifacts (HTML & MD)
  const fetchList = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Request both HTML and MD in a single call if supported by server
      const res = await fetch("/api/diagnostics/list?exts=html,md&limit=1000");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = (await res.json()) as ArtifactListResponse;

      // Separate html and md; pre-filter report candidates
      const html = (data.items ?? []).filter((it) => it.ext.toLowerCase() === "html");
      const md = (data.items ?? []).filter((it) => it.ext.toLowerCase() === "md");

      // Only display HTML that looks like a report by default; still searchable via filter
      const reports = html.filter((it) => isReportCandidate(it.path));

      setItems(reports);
      setMdList(md);
      setTotal(data.total ?? reports.length);
      // Autoselect the newest report
      const newest = [...reports].sort((a, b) => b.mtime - a.mtime)[0];
      setActivePath((prev) => prev ?? newest?.path ?? null);
    } catch (err: any) {
      setError(err?.message ?? "failed to load artifacts");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    fetchList();
  }, [fetchList]);

  // Derived filtered list
  const filtered = React.useMemo(() => {
    let list = [...items];

    // Type filter by name pattern
    if (typeFilter !== "all") {
      list = list.filter((it) => classifyReport(it.path).toLowerCase() === typeFilter);
    }

    // Search
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter((it) => it.path.toLowerCase().includes(q));
    }

    if (newestFirst) list.sort((a, b) => b.mtime - a.mtime);
    return list;
  }, [items, typeFilter, query, newestFirst]);

  // Markdown preview loader
  const loadMarkdown = React.useCallback(async (relPath: string) => {
    setMdPreview(relPath);
    setMdContent("Loading…");
    try {
      const res = await fetch(`/artifacts/${encodeURI(relPath)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      setMdContent(text);
    } catch (err: any) {
      setMdContent(`Error: ${err?.message ?? "failed to load markdown"}`);
    }
  }, []);

  // -----------------------------------------------------------------------------------
  // UI
  // -----------------------------------------------------------------------------------

  return (
    <div className="w-full min-h-screen px-4 py-6 lg:px-8 bg-background">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Reports</h1>
          <p className="text-sm text-muted-foreground">
            Read-only browser for CLI-produced reports (HTML/MD) under <code>/artifacts</code>.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" onClick={() => fetchList()}>
            Refresh
          </Button>
          <Button asChild>
            <a href="/api/diagnostics/health" target="_blank" rel="noreferrer">
              API Health
            </a>
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Discovered Reports</CardDescription>
            <CardTitle className="text-lg">{filtered.length}</CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            Total (server): {total}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Newest-first</CardDescription>
            <CardTitle className="text-lg">{newestFirst ? "Yes" : "No"}</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex h-10 items-center gap-3 rounded-md border px-3">
              <Switch checked={newestFirst} onCheckedChange={setNewestFirst} />
              <span className="text-xs text-muted-foreground">
                Toggle ordering
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Active Report</CardDescription>
            <CardTitle className="text-lg">
              {activePath ? activePath.split("/").slice(-1)[0] : "—"}
            </CardTitle>
          </CardHeader>
          <CardContent className="text-xs text-muted-foreground">
            {activePath ? (
              <div className="space-x-2">
                <a
                  href={`/artifacts/${encodeURI(activePath)}`}
                  target="_blank"
                  rel="noreferrer"
                  className="underline text-primary"
                >
                  Open in new tab
                </a>
                <Button
                  variant="outline"
                  size="xs"
                  onClick={() => {
                    navigator.clipboard?.writeText(
                      `${location.origin}/artifacts/${activePath}`
                    );
                  }}
                  className="ml-1"
                >
                  Copy Link
                </Button>
              </div>
            ) : (
              "No selection"
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardDescription>Quick Links</CardDescription>
            <CardTitle className="text-lg">Artifacts</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {["diagnostic_report_v1.html", "umap.html", "tsne.html"].map((name) => (
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
          <TabsTrigger value="reports">HTML Reports</TabsTrigger>
          <TabsTrigger value="notes">Markdown Notes</TabsTrigger>
        </TabsList>

        {/* Reports Tab */}
        <TabsContent value="reports" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Report Catalog</CardTitle>
              <CardDescription>Filter by type or search by path.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <div className="space-y-1">
                  <Label htmlFor="search">Search</Label>
                  <Input
                    id="search"
                    placeholder="e.g. umap, report_v2, planet_0042"
                    value={query}
                    onChange={(e) => setQuery(e.currentTarget.value)}
                  />
                </div>
                <div className="space-y-1">
                  <Label>Type</Label>
                  <Select value={typeFilter} onValueChange={setTypeFilter}>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="All" />
                    </SelectTrigger>
                    <SelectContent>
                      {["all", "umap", "tsne", "symbolic", "dashboard", "report", "other"].map((t) => (
                        <SelectItem key={t} value={t}>
                          {t.toUpperCase()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="invisible">Action</Label>
                  <div className="flex h-10 items-center">
                    <Button variant="secondary" onClick={() => fetchList()}>
                      Apply
                    </Button>
                  </div>
                </div>
              </div>

              {error ? (
                <div className="text-sm text-destructive">Error: {error}</div>
              ) : loading ? (
                <div className="text-sm text-muted-foreground">Loading…</div>
              ) : filtered.length === 0 ? (
                <div className="text-sm text-muted-foreground">No reports found.</div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-muted-foreground">
                      <tr className="[&>th]:px-2 [&>th]:py-1 text-left border-b">
                        <th>Report</th>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Modified</th>
                        <th>Open</th>
                        <th>Preview</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filtered.map((it) => {
                        const kind = classifyReport(it.path);
                        return (
                          <tr
                            key={it.path}
                            className={cn(
                              "[&>td]:px-2 [&>td]:py-1 border-b hover:bg-muted/40",
                              activePath === it.path && "bg-muted/60"
                            )}
                          >
                            <td className="max-w-[560px] truncate">{it.path}</td>
                            <td>
                              <Badge variant={extBadgeVariant(it.ext)}>{kind}</Badge>
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
                            <td>
                              <Button
                                size="xs"
                                variant={activePath === it.path ? "default" : "outline"}
                                onClick={() => setActivePath(it.path)}
                              >
                                {activePath === it.path ? "Selected" : "Preview"}
                              </Button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="h-[75vh]">
            <CardHeader>
              <CardTitle>HTML Preview</CardTitle>
              <CardDescription>
                Embedded iframe of the selected report (same-origin only).
              </CardDescription>
            </CardHeader>
            <CardContent className="h-[calc(75vh-5rem)]">
              {activePath ? (
                <iframe
                  title={activePath}
                  src={`/artifacts/${encodeURI(activePath)}`}
                  className="h-full w-full rounded border"
                />
              ) : (
                <div className="text-sm text-muted-foreground">
                  Select a report from the catalog to preview.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Notes Tab (Markdown viewer) */}
        <TabsContent value="notes" className="mt-4 space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Markdown Notes</CardTitle>
              <CardDescription>Rendered directly as raw text (client-side).</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              {mdList.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  No <code>.md</code> notes discovered.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-muted-foreground">
                      <tr className="[&>th]:px-2 [&>th]:py-1 text-left border-b">
                        <th>Note</th>
                        <th>Size</th>
                        <th>Modified</th>
                        <th>Open</th>
                        <th>Preview</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mdList
                        .sort((a, b) => b.mtime - a.mtime)
                        .map((it) => (
                          <tr key={it.path} className="[&>td]:px-2 [&>td]:py-1 border-b hover:bg-muted/40">
                            <td className="max-w-[560px] truncate">{it.path}</td>
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
                            <td>
                              <Button
                                size="xs"
                                variant={mdPreview === it.path ? "default" : "outline"}
                                onClick={() => loadMarkdown(it.path)}
                              >
                                {mdPreview === it.path ? "Selected" : "Preview"}
                              </Button>
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="h-[60vh]">
            <CardHeader>
              <CardTitle>Markdown Preview</CardTitle>
              <CardDescription>Raw text (no client-side Markdown rendering).</CardDescription>
            </CardHeader>
            <CardContent className="h-[calc(60vh-5rem)]">
              {mdPreview ? (
                <pre className="h-full w-full overflow-auto rounded bg-muted p-3 text-xs">
                  {mdContent}
                </pre>
              ) : (
                <div className="text-sm text-muted-foreground">
                  Select a <code>.md</code> note to preview.
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Separator className="my-6" />

      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          API:{" "}
          <a className="underline" href="/api/diagnostics/list" target="_blank" rel="noreferrer">
            /api/diagnostics/list
          </a>{" "}
          ·{" "}
          <a className="underline" href="/api/diagnostics/health" target="_blank" rel="noreferrer">
            /api/diagnostics/health
          </a>
        </div>
        <div className="text-xs text-muted-foreground">CLI-first · Reproducible by construction</div>
      </div>
    </div>
  );
}
