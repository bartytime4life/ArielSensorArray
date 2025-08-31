// src/gui/app/index.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — GUI App Entrypoint
// This file bootstraps the React application for the optional GUI layer.
// It wires together the routes, global layout, state store, and theming.
// ----------------------------------------------------------------------------
// Principles followed here:
//   • CLI-first: GUI always wraps CLI commands (`spectramind ...`) via server API.
//   • Declarative UI: React + Tailwind + shadcn/ui for responsive, modern UX.
//   • Accessibility: ARIA roles, keyboard navigation, and semantic HTML.
//   • State: Provided by Zustand/Redux store under src/gui/store.
//   • Logging: All actions route through CLI bridge and are logged in v50_debug_log.md.
// ============================================================================

import React from "react";
import ReactDOM from "react-dom/client";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";

import { ThemeProvider } from "@/components/theme-provider"; // shadcn/ui wrapper
import { Toaster } from "@/components/ui/toaster"; // notifications
import Layout from "./layout"; // global layout wrapper
import DashboardPage from "../features/dashboard/DashboardPage";
import DiagnosticsPage from "../features/diagnostics/DiagnosticsPage";
import UMAPPage from "../features/umap/UMAPPage";
import TSNEPage from "../features/tsne/TSNEPage";
import SHAPPage from "../features/shap/SHAPPage";
import SymbolicPage from "../features/symbolic/SymbolicPage";
import FFTPage from "../features/fft/FFTPage";
import CalibrationPage from "../features/calibration/CalibrationPage";
import NotFoundPage from "../features/misc/NotFoundPage";

// ----------------------------------------------------------------------------
// App Component
// ----------------------------------------------------------------------------

const App: React.FC = () => {
  return (
    <ThemeProvider defaultTheme="light" storageKey="spectramind-theme">
      <Router>
        <Layout>
          <Routes>
            {/* Default landing route */}
            <Route path="/" element={<Navigate to="/dashboard" replace />} />

            {/* Core pages */}
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/diagnostics" element={<DiagnosticsPage />} />
            <Route path="/umap" element={<UMAPPage />} />
            <Route path="/tsne" element={<TSNEPage />} />
            <Route path="/shap" element={<SHAPPage />} />
            <Route path="/symbolic" element={<SymbolicPage />} />
            <Route path="/fft" element={<FFTPage />} />
            <Route path="/calibration" element={<CalibrationPage />} />

            {/* Catch-all */}
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </Layout>
        <Toaster />
      </Router>
    </ThemeProvider>
  );
};

// ----------------------------------------------------------------------------
// Bootstrap React App
// ----------------------------------------------------------------------------

const container = document.getElementById("root") as HTMLElement;
if (!container) {
  throw new Error("❌ Root element #root not found. Check index.html scaffold.");
}

ReactDOM.createRoot(container).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// ============================================================================
// Notes:
// • Layout.tsx defines the navigation shell (sidebar + topbar + content region).
// • Each feature page (UMAP, t-SNE, SHAP, etc.) wraps CLI calls via server/api.
// • Toaster provides global notifications for CLI execution status/results.
// • ThemeProvider ensures consistent dark/light theme toggling.
// ============================================================================

