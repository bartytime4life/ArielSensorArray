// ============================================================================
// 📦 Barrel File for GUI Components — SpectraMind V50
// ----------------------------------------------------------------------------
// Purpose
//   • Central export hub for all reusable GUI components.
//   • Simplifies imports elsewhere in the GUI. Example:
//
//       import { Button, Card, Table } from "@/gui/components";
//
//   • Ensures that new components are discoverable and consistently exported.
//
// Notes
//   • Each component must be default-exported in its own file.
//   • All components here are designed for CLI-first, GUI-optional integration,
//     so they should remain thin, testable shells with no business logic.
//   • Keep ordering alphabetical for readability and easier maintenance.
// ----------------------------------------------------------------------------

// Core UI primitives
export { default as Button } from "./Button";
export { default as Card } from "./Card";
export { default as Chart } from "./Chart";
export { default as Input } from "./Input";
export { default as Loader } from "./Loader";
export { default as Modal } from "./Modal";
export { default as Panel } from "./Panel";
export { default as Select } from "./Select";
export { default as Table } from "./Table";
export { default as Tabs } from "./Tabs";
export { default as Tooltip } from "./Tooltip";

// ---------------------------------------------------------------------------
// ⚠️ Integration Contract
// ---------------------------------------------------------------------------
// • Every component exported here must:
//     1. Have an accompanying test file in the same directory
//        (e.g. Button.test.tsx for Button.tsx).
//     2. Follow accessibility and keyboard navigation standards.
//     3. Be styled using Tailwind + shadcn/ui primitives.
//     4. Support reproducibility (no hidden state outside props).
//     5. Export deterministic renders (no random IDs without seeding).
//
// • When adding new components:
//     - Place file under /src/gui/components
//     - Add default export
//     - Add matching test
//     - Update this index.ts to re-export
//
// • This ensures consistency with the CLI-first, NASA-grade reproducibility
//   standards of SpectraMind V50:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
//
// ---------------------------------------------------------------------------
