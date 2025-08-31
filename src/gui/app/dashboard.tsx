// src/gui/app/dashboard.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — Dashboard Page
// ----------------------------------------------------------------------------
// Purpose:
//   • Provide a single-screen overview of SpectraMind V50 diagnostics.
//   • Thin wrapper over CLI artifacts (JSON, PNG, HTML).
//   • Render plots (UMAP, t-SNE, GLL heatmaps, SHAP overlays, calibration).
//   • All data is pulled via FastAPI backend (src/server/api/*).
//
// Principles:
//   • CLI-first: Dashboard only *renders* what `spectramind diagnose` produced.
//   • Reproducible: No hidden state; all actions map to logged CLI commands.
//   • Modern UX: React + Tailwind + shadcn/ui; responsive, accessible, minimal.
// ============================================================================

import React, { useEffect, useState } from "react";
import { Card, CardHeader, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Loader2, RefreshCw } from "lucide-react";

// Utility to fetch backend JSON
async function fetchJSON(path: string) {
  const res = await fetch(`/api/${path}`);
  if (!res.ok) throw new Error(`Failed to fetch ${path}`);
  return res.json();
}

// ----------------------------------------------------------------------------
// Dashboard Component
// ----------------------------------------------------------------------------

const DashboardPage: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      const data = await fetchJSON("diagnostics/summary"); // served by FastAPI
      setSummary(data);
      setError(null);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">SpectraMind V50 — Dashboard</h1>
        <Button onClick={loadData} disabled={loading}>
          {loading ? (
            <Loader2 className="w-4 h-4 animate-spin mr-2" />
          ) : (
            <RefreshCw className="w-4 h-4 mr-2" />
          )}
          Refresh
        </Button>
      </div>

      {error && (
        <div className="text-red-600">
          ❌ Error loading diagnostics: {error}
        </div>
      )}

      {loading && !error && (
        <div className="flex justify-center py-10">
          <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
        </div>
      )}

      {summary && (
        <Tabs defaultValue="overview" className="w-full">
          <TabsList>
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="umap">UMAP</TabsTrigger>
            <TabsTrigger value="tsne">t-SNE</TabsTrigger>
            <TabsTrigger value="gll">GLL Heatmap</TabsTrigger>
            <TabsTrigger value="shap">SHAP × Symbolic</TabsTrigger>
            <TabsTrigger value="calibration">Calibration</TabsTrigger>
          </TabsList>

          {/* Overview */}
          <TabsContent value="overview">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">Run Summary</h2>
              </CardHeader>
              <CardContent>
                <pre className="bg-muted p-2 rounded text-sm overflow-x-auto">
                  {JSON.stringify(summary.meta, null, 2)}
                </pre>
              </CardContent>
            </Card>
          </TabsContent>

          {/* UMAP */}
          <TabsContent value="umap">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">UMAP Embedding</h2>
              </CardHeader>
              <CardContent>
                <iframe
                  src="/artifacts/umap.html"
                  className="w-full h-[500px] border rounded"
                  title="UMAP Plot"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* t-SNE */}
          <TabsContent value="tsne">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">t-SNE Embedding</h2>
              </CardHeader>
              <CardContent>
                <iframe
                  src="/artifacts/tsne.html"
                  className="w-full h-[500px] border rounded"
                  title="t-SNE Plot"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* GLL */}
          <TabsContent value="gll">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">GLL Heatmap</h2>
              </CardHeader>
              <CardContent>
                <img
                  src="/artifacts/gll_heatmap.png"
                  alt="GLL Heatmap"
                  className="rounded shadow-md"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* SHAP */}
          <TabsContent value="shap">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">SHAP × Symbolic Overlay</h2>
              </CardHeader>
              <CardContent>
                <img
                  src="/artifacts/shap_overlay.png"
                  alt="SHAP Overlay"
                  className="rounded shadow-md"
                />
              </CardContent>
            </Card>
          </TabsContent>

          {/* Calibration */}
          <TabsContent value="calibration">
            <Card className="mt-4">
              <CardHeader>
                <h2 className="text-xl font-semibold">Calibration Diagnostics</h2>
              </CardHeader>
              <CardContent>
                <img
                  src="/artifacts/calibration_plot.png"
                  alt="Calibration Plot"
                  className="rounded shadow-md"
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      )}
    </div>
  );
};

export default DashboardPage;

// ============================================================================
// Notes:
// • All plots (umap.html, tsne.html, shap_overlay.png, etc.) are CLI artifacts
//   produced by `spectramind diagnose dashboard`:contentReference[oaicite:4]{index=4}.
// • Backend FastAPI (`src/server/api/diagnostics.py`) exposes /api/diagnostics/summary
//   which aggregates diagnostic_summary.json:contentReference[oaicite:5]{index=5}.
// • This dashboard does not compute anything: it only displays results already
//   created by the CLI to maintain full reproducibility:contentReference[oaicite:6]{index=6}.
// ============================================================================
