// src/gui/app/reports.tsx
// ============================================================================
// 📄 SpectraMind V50 — Reports Browser (React + Tailwind + shadcn/ui)
// ----------------------------------------------------------------------------
// Purpose
//   A thin, read-only page to discover and preview CLI-produced HTML reports
//   (e.g., UMAP/t-SNE dashboards, diagnostic\_report\_v\*.html) served from
//   `/artifacts`. It talks to server-only endpoints (same-origin by default):
//      • GET  /api/diagnostics/list?exts=html,md\&limit=1000  (preferred)
//      • GET  /api/artifacts/list?exts=html,md\&limit=1000    (fallback)
//      • POST /api/diagnostics/run                           (optional trigger)
//
// Design
//   • No analytics here; the CLI is the source of truth.
//   • IFrame preview only for same-origin HTML under /artifacts.
//   • Safe guards, robust fetch with aborts, empty states, retry/trigger run.
//
// Notes
//   • Requires Tailwind + shadcn/ui in your project.
//   • You can override bases via Vite envs:
//       VITE\_API\_BASE         (default: "")
//       VITE\_ARTIFACTS\_BASE   (default: "/artifacts")
// ============================================================================

import React from "react";
import {
Card,
CardContent,
CardDescription,
CardHeader,
CardTitle,
} from "@/components/ui/card";
import {
Tabs,
TabsContent,
TabsList,
TabsTrigger,
} from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
Select,
SelectContent,
SelectItem,
SelectTrigger,
SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

// -------------------------------------------------------------------------------------
// Constants & Utilities
// -------------------------------------------------------------------------------------

const API\_BASE = import.meta.env.VITE\_API\_BASE?.toString?.() ?? "";
const ARTIFACTS\_BASE =
import.meta.env.VITE\_ARTIFACTS\_BASE?.toString?.() ?? "/artifacts";

type ArtifactItem = {
// Relative POSIX-style path under ARTIFACTS\_BASE (no leading slash)
path: string;
// Size in bytes
size: number;
// mtime in epoch seconds
mtime: number;
// Lowercase extension (e.g. "html", "md", "json")
ext: string;
};

type ArtifactListResponse = {
dir?: string;
count?: number;
total?: number;
items?: ArtifactItem\[];
};

type SortKey = "mtime" | "size" | "name";

// Narrow type guard to validate a single item
function isArtifactItem(x: unknown): x is ArtifactItem {
if (!x || typeof x !== "object") return false;
const o = x as Record\<string, unknown>;
return (
typeof o.path === "string" &&
typeof o.size === "number" &&
typeof o.mtime === "number" &&
typeof o.ext === "string"
);
}

// Narrow type guard to validate list response
function isArtifactListResponse(x: unknown): x is ArtifactListResponse {
if (!x || typeof x !== "object") return false;
const o = x as Record\<string, unknown>;
if (!("items" in o)) return false;
const items = o.items as unknown;
if (!Array.isArray(items)) return false;
return items.every(isArtifactItem);
}

function prettySize(bytes: number): string {
if (!Number.isFinite(bytes) || bytes < 0) return "-";
if (bytes < 1024) return `${bytes} B`;
const units = \["KB", "MB", "GB", "TB"];
let v = bytes / 1024;
let i = 0;
while (v >= 1024 && i < units.length - 1) {
v /= 1024;
i++;
}
const digits = v >= 100 ? 0 : v >= 10 ? 1 : 2;
return `${v.toFixed(digits)} ${units[i]}`;
}

function prettyTime(sec: number): string {
if (!Number.isFinite(sec)) return "-";
const d = new Date(sec \* 1000);
return Number.isFinite(d.getTime()) ? d.toLocaleString() : "-";
}

// Heuristic: which HTML files are "reports" we want to surface first
function isReportCandidate(path: string): boolean {
const p = path.toLowerCase();
return (
p.endsWith(".html") &&
(p.includes("umap") ||
p.includes("tsne") ||
p.includes("report") ||
p.includes("dashboard") ||
p.includes("symbolic") ||
p.includes("overview") ||
p.includes("diagnostic"))
);
}

function classifyReport(path: string): string {
const p = path.toLowerCase();
if (p.includes("umap")) return "UMAP";
if (p.includes("tsne")) return "t-SNE";
if (p.includes("symbolic")) return "Symbolic";
if (p.includes("dashboard")) return "Dashboard";
if (p.includes("report") || p.includes("diagnostic")) return "Report";
return "Other";
}

function extBadgeVariant(
ext: string
): "default" | "secondary" | "destructive" | "outline" {
const e = ext.toLowerCase();
if (e === "html") return "default";
if (e === "md") return "secondary";
return "outline";
}

// Build URL helpers (resilient to missing/extra slashes)
function urlJoin(base: string, path: string): string {
const b = base.endsWith("/") ? base.slice(0, -1) : base;
const p = path.startsWith("/") ? path.slice(1) : path;
return `${b}/${p}`;
}

// API fetch wrapper with AbortController, JSON parsing, and error mapping
async function apiFetchJSON<T>(
pathWithQuery: string,
signal?: AbortSignal
): Promise<T> {
const res = await fetch(urlJoin(API\_BASE, pathWithQuery), { signal });
if (!res.ok) {
const msg = await safeText(res);
throw new Error(`HTTP ${res.status} ${res.statusText} — ${msg || "error"}`);
}
const ct = res.headers.get("content-type") || "";
if (!ct.includes("application/json")) {
throw new Error(`Unexpected content-type: ${ct}`);
}
return (await res.json()) as T;
}

async function safeText(res: Response): Promise<string> {
try {
return await res.text();
} catch {
return "";
}
}

// -------------------------------------------------------------------------------------
// Component
// -------------------------------------------------------------------------------------

export default function ReportsPage(): JSX.Element {
// Query/state
const \[loading, setLoading] = React.useState(false);
const \[error, setError] = React.useState\<string | null>(null);

const \[items, setItems] = React.useState\<ArtifactItem\[]>(\[]);
const \[total, setTotal] = React.useState(0);

const \[query, setQuery] = React.useState("");
const \[typeFilter, setTypeFilter] = React.useState<string>("all"); // all | umap | tsne | symbolic | dashboard | report | other
const \[sortKey, setSortKey] = React.useState<SortKey>("mtime");
const \[newestFirst, setNewestFirst] = React.useState(true);

const \[activeTab, setActiveTab] = React.useState("reports");
const \[activePath, setActivePath] = React.useState\<string | null>(null);

const \[mdList, setMdList] = React.useState\<ArtifactItem\[]>(\[]);
const \[mdPreview, setMdPreview] = React.useState\<string | null>(null);
const \[mdContent, setMdContent] = React.useState<string>("");

const abortRef = React.useRef\<AbortController | null>(null);

// Fetch list of artifacts (HTML & MD) with fallback endpoint and abort on unmount
const fetchList = React.useCallback(async () => {
abortRef.current?.abort();
const ac = new AbortController();
abortRef.current = ac;

```
setLoading(true);
setError(null);

// Try preferred endpoint, then fallback if needed
const endpoints = [
  "api/diagnostics/list?exts=html,md&limit=1000",
  "api/artifacts/list?exts=html,md&limit=1000",
];

let data: ArtifactListResponse | null = null;
let lastErr: unknown = null;

for (const ep of endpoints) {
  try {
    const json = await apiFetchJSON<unknown>(ep, ac.signal);
    if (isArtifactListResponse(json)) {
      data = json;
      break;
    } else {
      lastErr = new Error("Malformed response from server");
    }
  } catch (e) {
    lastErr = e;
  }
  if (ac.signal.aborted) return;
}

if (!data) {
  if (!ac.signal.aborted) {
    setError(
      (lastErr as Error)?.message ??
        "Failed to load artifacts (no valid endpoint)"
    );
    setLoading(false);
  }
  return;
}

// Separate html and md; pre-filter report candidates
const all = Array.isArray(data.items) ? data.items : [];
const html = all.filter((it) => it.ext.toLowerCase() === "html");
const md = all.filter((it) => it.ext.toLowerCase() === "md");

// Only display HTML that looks like a report by default; still searchable via filter
const reports = html.filter((it) => isReportCandidate(it.path));

// Update state
if (!ac.signal.aborted) {
  setItems(reports);
  setMdList(md);
  setTotal(typeof data.total === "number" ? data.total : reports.length);

  // Autoselect the newest (by mtime) if none selected
  if (!activePath && reports.length > 0) {
    const newest = [...reports].sort((a, b) => b.mtime - a.mtime)[0];
    setActivePath(newest?.path ?? null);
  }

  setLoading(false);
}
```

}, \[activePath]);

React.useEffect(() => {
fetchList();
return () => abortRef.current?.abort();
}, \[fetchList]);

// Derived filtered/sorted list
const filtered = React.useMemo(() => {
let list = \[...items];

```
// Type filter by name pattern
if (typeFilter !== "all") {
  list = list.filter(
    (it) => classifyReport(it.path).toLowerCase() === typeFilter
  );
}

// Search
if (query.trim()) {
  const q = query.toLowerCase();
  list = list.filter((it) => it.path.toLowerCase().includes(q));
}

// Sorting
list.sort((a, b) => {
  switch (sortKey) {
    case "mtime":
      return (a.mtime - b.mtime) * (newestFirst ? -1 : 1);
    case "size":
      return (a.size - b.size) * (newestFirst ? -1 : 1);
    case "name":
      return a.path.localeCompare(b.path) * (newestFirst ? -1 : 1);
    default:
      return 0;
  }
});

return list;
```

}, \[items, typeFilter, query, sortKey, newestFirst]);

// Markdown preview loader (raw text only)
const loadMarkdown = React.useCallback(async (relPath: string) => {
setMdPreview(relPath);
setMdContent("Loading…");
try {
const res = await fetch(
urlJoin(ARTIFACTS\_BASE, encodeRelativePath(relPath))
);
if (!res.ok) throw new Error(`HTTP ${res.status}`);
const text = await res.text();
setMdContent(text);
} catch (err: any) {
setMdContent(`Error: ${err?.message ?? "failed to load markdown"}`);
}
}, \[]);

// Trigger CLI run (optional): POST /api/diagnostics/run
const triggerRun = React.useCallback(async () => {
setLoading(true);
setError(null);
try {
const res = await fetch(urlJoin(API\_BASE, "api/diagnostics/run"), {
method: "POST",
headers: { "content-type": "application/json" },
body: JSON.stringify({ action: "diagnose\_dashboard" }),
});
if (!res.ok) throw new Error(`HTTP ${res.status}`);
// After a brief delay, refresh listing (server may still be writing)
setTimeout(() => fetchList(), 1200);
} catch (e: any) {
setError(e?.message ?? "Failed to request diagnostics run");
} finally {
setLoading(false);
}
}, \[fetchList]);

// Copy link helper with graceful fallback
const copyLink = React.useCallback(async (href: string) => {
try {
await navigator.clipboard.writeText(href);
} catch {
// eslint-disable-next-line no-alert
alert(href);
}
}, \[]);

// Encode a relative path for safe use after ARTIFACTS\_BASE
function encodeRelativePath(rel: string): string {
// Keep slashes but encode spaces and special characters in each segment
return rel
.split("/")
.map((seg) => encodeURIComponent(seg))
.join("/");
}

// -----------------------------------------------------------------------------------
// UI
// -----------------------------------------------------------------------------------

return ( <div className="w-full min-h-screen px-4 py-6 lg:px-8 bg-background"> <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-center md:justify-between"> <div> <h1 className="text-2xl font-semibold tracking-tight">Reports</h1> <p className="text-sm text-muted-foreground">
Read-only browser for CLI-produced reports (HTML/MD) under{" "} <code>{ARTIFACTS\_BASE}</code>. </p> </div> <div className="flex gap-2">
\<Button variant="secondary" onClick={() => fetchList()} disabled={loading}>
{loading ? "Refreshing…" : "Refresh"} </Button> <Button asChild variant="outline">
\<a href={urlJoin(API\_BASE, "api/diagnostics/health")} target="\_blank" rel="noreferrer">
API Health </a> </Button> <Button onClick={triggerRun} variant="default">
Trigger Diagnostics Run </Button> </div> </div>

```
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
        <CardDescription>Ordering</CardDescription>
        <CardTitle className="text-lg">
          {sortKey.toUpperCase()} · {newestFirst ? "DESC" : "ASC"}
        </CardTitle>
      </CardHeader>
      <CardContent className="flex items-center gap-3">
        <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
          <SelectTrigger className="w-36">
            <SelectValue placeholder="Sort key" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="mtime">Modified</SelectItem>
            <SelectItem value="size">Size</SelectItem>
            <SelectItem value="name">Name</SelectItem>
          </SelectContent>
        </Select>
        <div className="flex h-10 items-center gap-3 rounded-md border px-3">
          <Switch checked={newestFirst} onCheckedChange={setNewestFirst} />
          <span className="text-xs text-muted-foreground">Reverse</span>
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
      <CardContent className="text-xs text-muted-foreground space-x-2">
        {activePath ? (
          <>
            <a
              href={urlJoin(ARTIFACTS_BASE, encodeRelativePath(activePath))}
              target="_blank"
              rel="noreferrer"
              className="underline text-primary"
            >
              Open in new tab
            </a>
            <Button
              variant="outline"
              size="xs"
              onClick={() =>
                copyLink(
                  urlJoin(
                    window.location.origin,
                    urlJoin(ARTIFACTS_BASE, encodeRelativePath(activePath))
                  )
                )
              }
              className="ml-1"
            >
              Copy Link
            </Button>
          </>
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
            href={urlJoin(ARTIFACTS_BASE, name)}
            className="text-xs underline text-primary hover:opacity-80"
            target="_blank"
            rel="noreferrer"
          >
            {urlJoin(ARTIFACTS_BASE, name)}
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
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-5">
            <div className="space-y-1 lg:col-span-2">
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
                  {["all", "umap", "tsne", "symbolic", "dashboard", "report", "other"].map(
                    (t) => (
                      <SelectItem key={t} value={t}>
                        {t.toUpperCase()}
                      </SelectItem>
                    )
                  )}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label>Sort</Label>
              <Select value={sortKey} onValueChange={(v) => setSortKey(v as SortKey)}>
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Sort key" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="mtime">Modified</SelectItem>
                  <SelectItem value="size">Size</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label className="invisible">Apply</Label>
              <div className="flex h-10 items-center gap-3">
                <div className="flex h-10 items-center gap-3 rounded-md border px-3">
                  <Switch checked={newestFirst} onCheckedChange={setNewestFirst} />
                  <span className="text-xs text-muted-foreground">Reverse</span>
                </div>
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
            <div className="text-sm text-muted-foreground space-y-2">
              <p>No reports found.</p>
              <div className="flex gap-2">
                <Button onClick={triggerRun} size="sm">
                  Run Diagnostics Now
                </Button>
                <Button variant="outline" size="sm" onClick={() => fetchList()}>
                  Retry Listing
                </Button>
              </div>
            </div>
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
                    const href = urlJoin(
                      ARTIFACTS_BASE,
                      encodeRelativePath(it.path)
                    );
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
                            href={href}
                            target="_blank"
                            rel="noreferrer"
                            className="text-xs underline text-primary hover:opacity-80"
                          >
                            {href}
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
              // Note: Avoid dangerous HTML injection. This is a same-origin iframe.
              title={activePath}
              src={urlJoin(ARTIFACTS_BASE, encodeRelativePath(activePath))}
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
                  {[...mdList]
                    .sort((a, b) => b.mtime - a.mtime)
                    .map((it) => {
                      const href = urlJoin(
                        ARTIFACTS_BASE,
                        encodeRelativePath(it.path)
                      );
                      return (
                        <tr
                          key={it.path}
                          className="[&>td]:px-2 [&>td]:py-1 border-b hover:bg-muted/40"
                        >
                          <td className="max-w-[560px] truncate">{it.path}</td>
                          <td className="whitespace-nowrap">{prettySize(it.size)}</td>
                          <td className="whitespace-nowrap">
                            {prettyTime(it.mtime)}
                          </td>
                          <td>
                            <a
                              href={href}
                              target="_blank"
                              rel="noreferrer"
                              className="text-xs underline text-primary hover:opacity-80"
                            >
                              {href}
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
                      );
                    })}
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

  <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
    <div className="text-xs text-muted-foreground">
      API:{" "}
      <a
        className="underline"
        href={urlJoin(API_BASE, "api/diagnostics/list")}
        target="_blank"
        rel="noreferrer"
      >
        {urlJoin(API_BASE, "api/diagnostics/list")}
      </a>{" "}
      ·{" "}
      <a
        className="underline"
        href={urlJoin(API_BASE, "api/diagnostics/health")}
        target="_blank"
        rel="noreferrer"
      >
        {urlJoin(API_BASE, "api/diagnostics/health")}
      </a>
    </div>
    <div className="text-xs text-muted-foreground">
      CLI-first · Reproducible by construction
    </div>
  </div>
</div>
```

);
}
