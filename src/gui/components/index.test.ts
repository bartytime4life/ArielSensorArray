// src/gui/components/index.test.ts
// ============================================================================
// 🧪 Tests — Barrel Export (src/gui/components/index.ts)
// -----------------------------------------------------------------------------
// What we test:
//   • Ensure each component is exported and defined
//   • Verify named exports match the actual component files
//   • Helps catch broken imports when refactoring
// ============================================================================

import * as Components from "./index";

describe("GUI Components Barrel Export", () => {
  const expectedExports = [
    "Card",
    "Chart",
    "Table",
    "Panel",
    "Button",
    "Input",
    "Select",
    "Tabs",
    "Modal",
    "Loader",
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
});
