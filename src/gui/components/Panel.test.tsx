// src/gui/components/Panel.test.tsx
// =============================================================================
// ✅ Tests for src/gui/components/Panel.tsx (SpectraMind V50 GUI)
// -----------------------------------------------------------------------------
// Covers: header chrome (title/description/icon/actions), collapsible behavior
// (uncontrolled & controlled), action click does not toggle collapse, scrollable
// body with maxHeight, padded vs. unpadded body, footer rendering, variants and
// highlight classes, a11y attributes on region/header.
//
// Test stack assumptions:
//   • Vitest (or Jest) + @testing-library/react
//   • tsconfig paths support for "@/..." (adjust mocks if needed)
//
// Run (Vitest):
//   npx vitest run src/gui/components/Panel.test.tsx
// =============================================================================

import * as React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Panel } from "./Panel";

// -----------------------------------------------------------------------------
// Mocks
// -----------------------------------------------------------------------------

// Strip framer-motion animations to keep DOM simple & deterministic
vi.mock("framer-motion", () => {
  const NoopDiv: React.FC<any> = ({ children, ...rest }) => <div {...rest}>{children}</div>;
  return {
    motion: { section: NoopDiv, div: NoopDiv },
    AnimatePresence: ({ children }: any) => <>{children}</>,
  };
});

// Merge classnames
vi.mock("@/lib/utils", () => {
  return { cn: (...args: any[]) => args.filter(Boolean).join(" ") };
});

// Clean up between tests
afterEach(() => cleanup());

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

function getRegion(): HTMLElement {
  // Body container has role="region"
  const region = screen.getByRole("region");
  return region as HTMLElement;
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

describe("Panel component", () => {
  it("renders header chrome (title, description, icon, actions) and body content", () => {
    render(
      <Panel
        title="Diagnostics"
        description="Top rules and overlays"
        icon={<span data-testid="icon">★</span>}
        actions={<button data-testid="action">Refresh</button>}
      >
        <div data-testid="content">Body</div>
      </Panel>
    );

    // Header pieces
    expect(screen.getByText("Diagnostics")).toBeInTheDocument();
    expect(screen.getByText("Top rules and overlays")).toBeInTheDocument();
    expect(screen.getByTestId("icon")).toBeInTheDocument();
    expect(screen.getByTestId("action")).toBeInTheDocument();

    // Body region and content
    const region = getRegion();
    expect(region).toBeInTheDocument();
    expect(screen.getByTestId("content")).toBeInTheDocument();
  });

  it("uncontrolled collapsible: chevron toggles collapsed state and data attributes", () => {
    render(
      <Panel title="Collapsible" collapsible defaultCollapsed={false}>
        <div>Body</div>
      </Panel>
    );

    const container = screen.getByRole("region").parentElement!.parentElement as HTMLElement;
    // data attributes live on the motion.section (container)
    const section = container.closest('[data-collapsible]') as HTMLElement;
    expect(section).toBeTruthy();
    // Initially expanded
    expect(section.getAttribute("data-collapsed")).toBe("false");

    // Click the chevron button
    const chevron = screen.getByLabelText("Collapse panel");
    fireEvent.click(chevron);

    // Now collapsed (body region should no longer be present)
    const sectionAfter = container.closest('[data-collapsible]') as HTMLElement;
    expect(sectionAfter.getAttribute("data-collapsed")).toBe("true");
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("controlled collapsible: clicking chevron calls onCollapsedChange but does not auto-toggle without prop change", () => {
    const onCollapsedChange = vi.fn();
    const { rerender } = render(
      <Panel
        title="Controlled"
        collapsible
        collapsed={false}
        onCollapsedChange={onCollapsedChange}
      >
        <div>Body</div>
      </Panel>
    );

    const section = screen.getByRole("region").parentElement!.parentElement!.closest(
      '[data-collapsible]'
    ) as HTMLElement;
    expect(section?.getAttribute("data-collapsed")).toBe("false");

    const chevron = screen.getByLabelText("Collapse panel");
    fireEvent.click(chevron);
    expect(onCollapsedChange).toHaveBeenCalledWith(true);

    // Without updating prop, still expanded
    const sectionStill = screen.getByRole("region").parentElement!.parentElement!.closest(
      '[data-collapsible]'
    ) as HTMLElement;
    expect(sectionStill?.getAttribute("data-collapsed")).toBe("false");

    // Now simulate parent controlling prop to collapsed
    rerender(
      <Panel
        title="Controlled"
        collapsible
        collapsed={true}
        onCollapsedChange={onCollapsedChange}
      >
        <div>Body</div>
      </Panel>
    );
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("clicking actions does not toggle collapse when collapsible", () => {
    render(
      <Panel
        title="Collapsible"
        collapsible
        defaultCollapsed={false}
        actions={<button data-testid="action">Do</button>}
      >
        <div>Body</div>
      </Panel>
    );

    const sectionBefore = screen.getByRole("region").parentElement!.parentElement!.closest(
      '[data-collapsible]'
    ) as HTMLElement;
    expect(sectionBefore?.getAttribute("data-collapsed")).toBe("false");

    // Click inside actions
    fireEvent.click(screen.getByTestId("action"));

    // Should remain expanded
    const sectionAfter = screen.getByRole("region").parentElement!.parentElement!.closest(
      '[data-collapsible]'
    ) as HTMLElement;
    expect(sectionAfter?.getAttribute("data-collapsed")).toBe("false");
  });

  it("scrollBody applies maxHeight style on the region container", () => {
    render(
      <Panel title="Scrollable" scrollBody bodyMaxHeight={400}>
        <div style={{ height: 1000 }}>Tall Content</div>
      </Panel>
    );
    const region = getRegion();
    // The motion.div with role=region receives style when scrollBody is true
    const style = (region as HTMLElement).getAttribute("style") || "";
    expect(style.replace(/\s/g, "")).toContain("max-height:400px");
  });

  it("padded by default, and uses compact padding when padded=false", () => {
    // Default padded
    const { rerender } = render(
      <Panel title="Pad-default">
        <div data-testid="inner">Body</div>
      </Panel>
    );
    const region = getRegion();
    // First child of region is the padding wrapper
    const wrapper = region.firstElementChild as HTMLElement;
    expect(wrapper.className).toMatch(/pb-3|pt-2/); // default padded adds pb-3 pt-2

    // Now unpadded
    rerender(
      <Panel title="No-pad" padded={false}>
        <div data-testid="inner">Body</div>
      </Panel>
    );
    const region2 = getRegion();
    const wrapper2 = region2.firstElementChild as HTMLElement;
    expect(wrapper2.className).toMatch(/py-2/);
  });

  it("renders footer when provided", () => {
    render(
      <Panel title="With footer" footer={<div data-testid="foot">Footer Text</div>}>
        <div>Body</div>
      </Panel>
    );
    expect(screen.getByTestId("foot")).toBeInTheDocument();
  });

  it("applies variant and highlight classes on container", () => {
    const { container, rerender } = render(
      <Panel title="v" variant="default" highlight>
        <div />
      </Panel>
    );

    const section = container.querySelector('[data-collapsible], section')?.closest("section, div") as HTMLElement;
    // highlight adds border-blue-500
    expect(section.className).toMatch(/border-blue-500/);

    // ghost variant removes border/shadow
    rerender(
      <Panel title="v" variant="ghost">
        <div />
      </Panel>
    );
    const ghost = container.querySelector('[data-collapsible], section')?.closest("section, div") as HTMLElement;
    expect(ghost.className).toMatch(/shadow-none/);
  });

  it("sets a11y attributes: aria-labelledby on section and region linkage", () => {
    render(
      <Panel title="a11y" collapsible>
        <div>Body</div>
      </Panel>
    );

    // Section has aria-labelledby referencing the header id
    const sections = screen.getAllByRole("region");
    // The first region exists, and its parent section should have aria-labelledby
    const region = sections[0];
    const section = region.parentElement?.parentElement as HTMLElement;
    const labelledby = section.getAttribute("aria-labelledby");
    expect(labelledby).toBeTruthy();

    // The region should reference that header via aria-labelledby (implicit via parent id + role=region)
    // We ensure header element exists with that id
    const headerEl = document.getElementById(labelledby!);
    expect(headerEl).toBeTruthy();
  });
});
