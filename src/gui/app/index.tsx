// src/gui/app/index.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — GUI App Entrypoint
// ----------------------------------------------------------------------------
// Purpose
//   • Bootstrap the optional GUI that *wraps* the CLI-first pipeline.
//   • Wire routes, global layout, theming, toasts, and accessibility helpers.
//   • Keep the browser dumb: zero analytics; all heavy lifting is via server API
//     that orchestrates `spectramind ...` (Typer) and renders prebuilt artifacts.
//
// Design Upgrades
//   • Route-based code-splitting via React.lazy (fast initial paint).
//   • ErrorBoundary + Suspense fallback with accessible loader.
//   • Scroll-to-top on navigation + route-change announcer (aria-live).
//   • Configurable router base path (supports sub-path hosting).
//   • Strict root element guard + clean render bootstrap.
// ----------------------------------------------------------------------------
// Requirements
//   • React Router v6+
//   • Tailwind + shadcn/ui
//   • <div id="root"> in index.html
//   • "@/components/theme-provider" and "@/components/ui/toaster" present.
//   • "./layout" global shell and feature pages under src/gui/features/\*
// ============================================================================

import \* as React from "react";
import ReactDOM from "react-dom/client";
import {
BrowserRouter as Router,
Routes,
Route,
Navigate,
useLocation,
} from "react-router-dom";

// Theme + notifications
import { ThemeProvider } from "@/components/theme-provider";
import { Toaster } from "@/components/ui/toaster";

// Global layout (sidebar + topbar + content)
import Layout from "./layout";

// ----------------------------------------------------------------------------
// Router Base (supports sub-path hosting behind reverse proxies/CDNs)
// ----------------------------------------------------------------------------
// Prefer Vite's BASE\_URL; allow override via VITE\_BASE\_PATH (e.g., "/spectramind")
const BASENAME =
(import.meta as any).env?.VITE\_BASE\_PATH ??
(import.meta as any).env?.BASE\_URL ??
"/";

// ----------------------------------------------------------------------------
// Lazy-loaded Feature Routes (route-level code splitting)
// ----------------------------------------------------------------------------
// Keep entry bundle lightweight; each feature page is loaded on demand.
// Paths assume project structure: src/gui/features/<feature>/<Feature>Page.tsx

const DashboardPage = React.lazy(() => import("../features/dashboard/DashboardPage"));
const DiagnosticsPage = React.lazy(() => import("../features/diagnostics/DiagnosticsPage"));
const UMAPPage = React.lazy(() => import("../features/umap/UMAPPage"));
const TSNEPage = React.lazy(() => import("../features/tsne/TSNEPage"));
const SHAPPage = React.lazy(() => import("../features/shap/SHAPPage"));
const SymbolicPage = React.lazy(() => import("../features/symbolic/SymbolicPage"));
const FFTPage = React.lazy(() => import("../features/fft/FFTPage"));
const CalibrationPage = React.lazy(() => import("../features/calibration/CalibrationPage"));
const ReportsPage = React.lazy(() => import("./reports")); // from src/gui/app/reports.tsx
const NotFoundPage = React.lazy(() => import("../features/misc/NotFoundPage"));

// ----------------------------------------------------------------------------
// Accessible Loader (Suspense fallback)
// ----------------------------------------------------------------------------

const AppLoader: React.FC<{ label?: string }> = ({ label = "Loading…" }) => {
return ( <div
   role="status"
   aria-live="polite"
   className="flex h-[50vh] w-full items-center justify-center text-sm text-muted-foreground"
 > <span className="mr-2 inline-block h-3 w-3 animate-pulse rounded-full bg-primary" aria-hidden />
{label} </div>
);
};

// ----------------------------------------------------------------------------
// Error Boundary (catches render errors in lazy routes/components)
// ----------------------------------------------------------------------------

class ErrorBoundary extends React.Component<
React.PropsWithChildren,
{ error: Error | null }

> {
> constructor(props: React.PropsWithChildren) {
> super(props);
> this.state = { error: null };
> }
> static getDerivedStateFromError(error: Error) {
> return { error };
> }
> componentDidCatch(error: Error) {
> // Note: we keep it simple and visible; CLI logs still live in v50\_debug\_log.md.
> // You could also forward to server via /api/diagnostics/log if desired.
> // eslint-disable-next-line no-console
> console.error("UI ErrorBoundary caught:", error);
> }
> render() {
> if (this.state.error) {
> return ( <div className="mx-auto max-w-2xl p-6"> <h2 className="mb-2 text-lg font-semibold">Something went wrong.</h2> <p className="mb-4 text-sm text-muted-foreground">
> The UI failed to render this view. Try refreshing the page. If the
> issue persists, check the CLI logs and server console. </p> <pre className="overflow-auto rounded border bg-muted p-3 text-xs">
> {String(this.state.error?.stack || this.state.error?.message || this.state.error)} </pre> </div>
> );
> }
> return this.props.children as React.ReactElement;
> }
> }

// ----------------------------------------------------------------------------
// Navigation A11y: Scroll Restoration + Route Change Announcer
// ----------------------------------------------------------------------------

/\*\*

* ScrollToTopOnNavigate
* Ensures each route transition resets window scroll to top.
  \*/
  const ScrollToTopOnNavigate: React.FC = () => {
  const { pathname } = useLocation();
  React.useEffect(() => {
  try {
  window\.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  } catch {
  window\.scrollTo(0, 0);
  }
  }, \[pathname]);
  return null;
  };

/\*\*

* RouteAnnouncer
* Announces page changes via aria-live region for screen readers.
  \*/
  const RouteAnnouncer: React.FC = () => {
  const location = useLocation();
  const \[message, setMessage] = React.useState("Page loaded");
  React.useEffect(() => {
  // Derive a friendly label from the pathname; could also inspect route meta.
  const label = location.pathname === "/" ? "Dashboard" : location.pathname.replace("/", "");
  setMessage(`Navigated to ${label}`);
  }, \[location]);
  return (

   <div
     aria-live="polite"
     aria-atomic="true"
     className="sr-only"
   >
     {message}
   </div>

);
};

// ----------------------------------------------------------------------------
// App Component (Theme + Router + Layout + Routes)
// ----------------------------------------------------------------------------

const App: React.FC = () => {
return ( <ThemeProvider defaultTheme="light" storageKey="spectramind-theme"> <Router basename={BASENAME}> <ScrollToTopOnNavigate /> <RouteAnnouncer /> <Layout> <ErrorBoundary>
\<React.Suspense fallback={<AppLoader />}> <Routes>
{/\* Default landing route \*/}
\<Route path="/" element={<Navigate to="/dashboard" replace />} />

```
            {/* Core pages (all render CLI-produced artifacts via server) */}
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/diagnostics" element={<DiagnosticsPage />} />
            <Route path="/umap" element={<UMAPPage />} />
            <Route path="/tsne" element={<TSNEPage />} />
            <Route path="/shap" element={<SHAPPage />} />
            <Route path="/symbolic" element={<SymbolicPage />} />
            <Route path="/fft" element={<FFTPage />} />
            <Route path="/calibration" element={<CalibrationPage />} />

            {/* Reports browser (HTML/MD under /artifacts) */}
            <Route path="/reports" element={<ReportsPage />} />

            {/* Catch-all */}
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </React.Suspense>
      </ErrorBoundary>
    </Layout>

    {/* Global toasts (command statuses, API errors, etc.) */}
    <Toaster />
  </Router>
</ThemeProvider>
```

);
};

// ----------------------------------------------------------------------------
// Bootstrap React App (strict root guard + clean render)
// ----------------------------------------------------------------------------

const container = document.getElementById("root") as HTMLElement | null;

if (!container) {
// Throwing ensures CI/builds fail loudly if index.html is misconfigured.
throw new Error("❌ Root element #root not found. Check index.html scaffold.");
}

const root = ReactDOM.createRoot(container);

// StrictMode helps catch subtle issues in dev; harmless in production builds.
root.render(
\<React.StrictMode> <App />
\</React.StrictMode>
);

// ============================================================================
// Implementation Notes
// ----------------------------------------------------------------------------
// • BASENAME lets you host the GUI at a sub-path (e.g., [https://host/app/preview](https://host/app/preview)).
//   Set VITE\_BASE\_PATH="/app/preview" or rely on Vite's BASE\_URL.
// • Each lazy page should remain *thin* and only read artifacts/JSON that the
//   CLI produced, typically via GET /api/\* and /artifacts/\* (static).
// • No analytics or model computations run in-browser. Keep it that way.
// • To add a new page:
//      1) Create src/gui/features/<name>/<Name>Page.tsx
//      2) Add React.lazy import above
//      3) Register a \<Route path="/name" element={<NamePage />} />
//      4) Add it to the sidebar list in layout.tsx
// • If you need hash-based routing (older static hosts), swap BrowserRouter
//   with HashRouter and remove BASENAME.
// ============================================================================
