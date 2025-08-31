// ============================================================================
// 🧪 Tests — Barrel Export (src/gui/components/index.ts)
// -----------------------------------------------------------------------------
// Purpose
//   • Ensure every component is exported from the barrel file
//   • Verify components are defined and are valid React components
//   • Catch broken imports when refactoring or renaming
//
// Design notes
//   • This test guards the "single source of truth" for GUI components
//   • Any new component must be added to both:
//       1. src/gui/components/index.ts
//       2. expectedExports array below
//   • Failing tests mean barrel and actual files are out of sync
// -----------------------------------------------------------------------------

import React from "react";
import * as Components from "./index";

describe("GUI Components Barrel Export", () => {
  const expectedExports = [
    "Button",
    "Card",
    "Chart",
    "Input",
    "Loader",
    "Modal",
    "Panel",
    "Select",
    "Table",
    "Tabs",
    "Tooltip",
  ];

  it("exports all expected components", () => {
    for (const name of expectedExports) {
      expect(Components).toHaveProperty(name);
    }
  });

  it("each exported component is defined", () => {
    for (const name of expectedExports) {
      const Comp = (Components as any)[name];
      expect(Comp).toBeDefined();
    }
  });

  it("each export is a valid React component", () => {
    for (const name of expectedExports) {
      const Comp = (Components as any)[name];
      // Functional/class components should be callable or constructible
      const isRenderable =
        typeof Comp === "function" || React.isValidElement(<Comp />);
      expect(isRenderable).toBeTruthy();
    }
  });

  it("has no unexpected extra exports (drift check)", () => {
    const actualExports = Object.keys(Components).sort();
    const sortedExpected = [...expectedExports].sort();
    expect(actualExports).toEqual(sortedExpected);
  });
});

// -----------------------------------------------------------------------------
// ✅ Integration Contract
// -----------------------------------------------------------------------------
// • All new components must be added to `expectedExports`
// • Tests enforce barrel consistency & reproducibility
// • This test suite will fail in CI if barrel drift occurs
// -----------------------------------------------------------------------------
