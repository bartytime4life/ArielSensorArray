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

function getSectionFromRegion(): HTMLElement {
  // region (motion.div) -> section (container)
  return getRegion().parentElement!.parentElement as HTMLElement;
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

    // container section has data attributes for collapsible & collapsed state
    const section = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(section).toBeTruthy();
    expect(section.getAttribute("data-collapsed")).toBe("false");

    // Click the chevron button
    const chevron = screen.getByLabelText("Collapse panel");
    fireEvent.click(chevron);

    // Now collapsed (body region should no longer be present)
    const sectionAfter = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(sectionAfter.getAttribute("data-collapsed")).toBe("true");
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("controlled collapsible: clicking chevron calls onCollapsedChange/onToggle but does not auto-toggle without prop change", () => {
    const onCollapsedChange = vi.fn();
    const onToggle = vi.fn();
    const { rerender } = render(
      <Panel
        title="Controlled"
        collapsible
        collapsed={false}
        onCollapsedChange={onCollapsedChange}
        onToggle={onToggle}
      >
        <div>Body</div>
      </Panel>
    );

    const section = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(section?.getAttribute("data-collapsed")).toBe("false");

    const chevron = screen.getByLabelText("Collapse panel");
    fireEvent.click(chevron);
    expect(onCollapsedChange).toHaveBeenCalledWith(true);
    expect(onToggle).toHaveBeenCalledWith(true);

    // Without updating prop, still expanded
    const sectionStill = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(sectionStill?.getAttribute("data-collapsed")).toBe("false");

    // Now simulate parent controlling prop to collapsed
    rerender(
      <Panel
        title="Controlled"
        collapsible
        collapsed={true}
        onCollapsedChange={onCollapsedChange}
        onToggle={onToggle}
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

    const before = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(before?.getAttribute("data-collapsed")).toBe("false");

    // Click inside actions
    fireEvent.click(screen.getByTestId("action"));

    // Should remain expanded
    const after = getSectionFromRegion().closest("[data-collapsible]") as HTMLElement;
    expect(after?.getAttribute("data-collapsed")).toBe("false");
  });

  it("scrollBody applies maxHeight style on the region container", () => {
    render(
      <Panel title="Scrollable" scrollBody bodyMaxHeight={400}>
        <div style={{ height: 1000 }}>Tall Content</div>
      </Panel>
    );
    const region = getRegion();
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
    let region = getRegion();
    // First child of region is the padding wrapper
    let wrapper = region.firstElementChild as HTMLElement;
    expect(wrapper.className).toMatch(/pb-3/);
    expect(wrapper.className).toMatch(/pt-2/);

    // Now unpadded
    rerender(
      <Panel title="No-pad" padded={false}>
        <div data-testid="inner">Body</div>
      </Panel>
    );
    region = getRegion();
    wrapper = region.firstElementChild as HTMLElement;
    expect(wrapper.className).toMatch(/py-2/);
  });

  it("renders footer when provided", () => {
    render(
      <Panel title="With footer" footer={<div data-testid="foot">Footer Text</div>}>
        <div>Body</div>
      </Panel>
    );
    expect(screen.getByTestId("foot")).toBeInTheDocument();
  });

  it("applies variant and highlight classes (tone-aware)", () => {
    const { container, rerender } = render(
      <Panel title="v" variant="default" highlight tone="info">
        <div />
      </Panel>
    );

    // highlight + tone=info adds border-blue-500
    const section = container.querySelector("[data-collapsible], section")?.closest("section, div") as HTMLElement;
    expect(section.className).toMatch(/border-blue-500/);

    // ghost variant removes border/shadow
    rerender(
      <Panel title="v" variant="ghost">
        <div />
      </Panel>
    );
    const ghost = container.querySelector("[data-collapsible], section")?.closest("section, div") as HTMLElement;
    expect(ghost.className).toMatch(/shadow-none/);
  });

  it("sets a11y attributes: section has aria-labelledby pointing at header; region linked via aria-labelledby", () => {
    render(
      <Panel title="a11y" collapsible>
        <div>Body</div>
      </Panel>
    );

    // Section has aria-labelledby referencing the header id
    const section = getSectionFromRegion();
    const labelledby = section.getAttribute("aria-labelledby");
    expect(labelledby).toBeTruthy();

    // The element with that id must exist (header)
    const headerEl = document.getElementById(labelledby!);
    expect(headerEl).toBeTruthy();

    // Region references header via aria-labelledby implicitly by id linkage and role=region
    const region = getRegion();
    expect(region.getAttribute("aria-labelledby")).toBeTruthy();
    // Should point to same header id
    expect(region.getAttribute("aria-labelledby")).toBe(labelledby);
  });

  it("sticky header/footer and dividers render appropriate classes", () => {
    render(
      <Panel
        title="Sticky"
        stickyHeader
        stickyFooter
        dividers
        footer={<div>F</div>}
      >
        <div>Body</div>
      </Panel>
    );

    // Header should exist (sticky class applied)
    const header = screen.getByTestId("PanelHeader");
    expect(header.className).toMatch(/sticky/);
    // Footer sticky
    const footer = screen.getByTestId("PanelFooter");
    expect(footer.className).toMatch(/sticky/);
    // Dividers apply border on header/footer container (border-b / border-t)
    expect(header.className).toMatch(/border-b|border/);
    expect(footer.className).toMatch(/border-t|border/);
  });
});
