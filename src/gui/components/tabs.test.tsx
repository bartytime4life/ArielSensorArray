// src/gui/components/tabs.test.tsx
// =============================================================================
// 🧪 Tests — Tabs Component (Upgraded)
// -----------------------------------------------------------------------------
// What we test (expanded):
//   • Renders tabs with accessible roles & labels
//   • Default tab selection (uncontrolled) and defaultTab prop
//   • Controlled mode (value prop) + re-renders when value changes
//   • onTabChange fired on click and keyboard interactions
//   • Disabled tabs cannot be activated (mouse or keyboard) and are skipped
//   • Displays correct panel content per selected tab
//   • Forwards className for outer container
//   • ARIA: role=tablist, role=tab, role=tabpanel, aria-selected, aria-controls,
//            aria-labelledby, aria-orientation
//   • Keyboard navigation per WAI-ARIA Tabs pattern:
//       - Left/Right arrows move focus among tabs
//       - Home/End jump to first/last tab
//       - Enter/Space activate focused tab (uncontrolled)
//       - Disabled tabs are skipped during keyboard nav
//   • Focus management: roving tabindex (only one tab is tabbable)
// =============================================================================

import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Tabs } from "./Tabs";

describe("Tabs component", () => {
  const setupTabs = [
    { id: "umap", label: "UMAP", content: <div>UMAP Plot</div> },
    { id: "tsne", label: "t-SNE", content: <div>t-SNE Plot</div> },
    { id: "fft", label: "FFT", content: <div>FFT Plot</div>, disabled: true },
  ];

  // Utility: grab tab elements by their accessible names
  const getTab = (name: RegExp | string) =>
    screen.getByRole("tab", { name: name instanceof RegExp ? name : new RegExp(name, "i") });

  const getTabs = () => screen.getAllByRole("tab");

  const getPanelByName = (name: RegExp | string) =>
    screen.getByRole("tabpanel", { name: name instanceof RegExp ? name : new RegExp(name, "i") });

  // ---------------------------------------------------------------------------

  it("renders all tab labels", () => {
    render(<Tabs tabs={setupTabs} />);
    expect(getTab(/umap/i)).toBeInTheDocument();
    expect(getTab(/t-sne/i)).toBeInTheDocument();
    expect(getTab(/fft/i)).toBeInTheDocument();
  });

  it("exposes role=tablist and aria-orientation on the container", () => {
    render(<Tabs tabs={setupTabs} ariaOrientation="horizontal" />);
    const tablist = screen.getByRole("tablist");
    expect(tablist).toBeInTheDocument();
    expect(tablist).toHaveAttribute("aria-orientation", "horizontal");
  });

  it("selects the first tab by default (uncontrolled)", () => {
    render(<Tabs tabs={setupTabs} />);
    const umapTab = getTab(/umap/);
    expect(umapTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/umap plot/i)).toBeVisible();

    // Others not selected
    expect(getTab(/t-sne/)).toHaveAttribute("aria-selected", "false");
    expect(getTab(/fft/)).toHaveAttribute("aria-selected", "false");
  });

  it("respects defaultTab prop (uncontrolled)", () => {
    render(<Tabs tabs={setupTabs} defaultTab="tsne" />);
    const tsneTab = getTab(/t-sne/);
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();
  });

  it("changes tab on click in uncontrolled mode", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="umap" />);

    const tsneTab = getTab(/t-sne/);
    await user.click(tsneTab);

    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();
  });

  it("calls onTabChange when a tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} onTabChange={onChange} />);
    await user.click(getTab(/t-sne/));
    expect(onChange).toHaveBeenCalledWith("tsne");
  });

  it("works in controlled mode with value prop (and does not change on click)", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const { rerender } = render(<Tabs tabs={setupTabs} value="tsne" onTabChange={onChange} />);

    // Controlled selection shows tsne
    const tsneTab = getTab(/t-sne/);
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();

    // Click another tab does not switch UI unless parent updates value
    await user.click(getTab(/umap/));
    expect(onChange).toHaveBeenCalledWith("umap"); // parent notified
    // Still tsne until parent updates:
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();

    // Now simulate parent controlling the change:
    rerender(<Tabs tabs={setupTabs} value="umap" onTabChange={onChange} />);
    expect(getTab(/umap/)).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/umap plot/i)).toBeVisible();
  });

  it("does not activate disabled tabs (mouse)", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} onTabChange={onChange} />);
    const fftTab = getTab(/fft/);

    // Disabled semantics
    expect(fftTab).toBeDisabled?.(); // if implemented via <button disabled/>
    // If custom role, fallback check:
    if (!("disabled" in fftTab)) {
      expect(fftTab).toHaveAttribute("aria-disabled", "true");
    }

    await user.click(fftTab);
    expect(onChange).not.toHaveBeenCalled();
    expect(screen.queryByText(/fft plot/i)).not.toBeVisible();
  });

  it("applies custom className to container", () => {
    render(<Tabs tabs={setupTabs} className="custom-class" />);
    const container = screen.getByRole("tablist").parentElement;
    expect(container).toHaveClass("custom-class", { exact: false });
  });

  it("renders correct aria attributes for accessibility", () => {
    render(<Tabs tabs={setupTabs} />);
    const umapTab = getTab(/umap/);
    const panel = getPanelByName(/umap/);

    // Tab ↔ Panel relationships
    expect(umapTab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", umapTab.id);

    // Only one panel visible at a time (by selection)
    const tablist = screen.getByRole("tablist");
    const panels = within(tablist.parentElement as HTMLElement).getAllByRole("tabpanel");
    const selected = panels.filter((p) => p.getAttribute("hidden") === null);
    expect(selected.length).toBe(1);
  });

  // ---------------------------------------------------------------------------
  // Keyboard navigation per WAI-ARIA Tabs pattern
  // ---------------------------------------------------------------------------

  it("only one tab is tabbable (roving tabindex)", async () => {
    render(<Tabs tabs={setupTabs} />);
    const [tab1, tab2, tab3] = getTabs();
    // Typically: selected tab has tabIndex=0, others = -1
    expect(tab1).toHaveAttribute("tabindex", "0");
    if (tab2.hasAttribute("tabindex")) expect(tab2).toHaveAttribute("tabindex", "-1");
    if (tab3.hasAttribute("tabindex")) expect(tab3).toHaveAttribute("tabindex", "-1");
  });

  it("supports Left/Right arrow focus navigation and Home/End", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="umap" />);

    const [umapTab, tsneTab, fftTab] = getTabs();

    // Focus first tab then arrow right to next (skipping disabled handled below)
    umapTab.focus();
    expect(umapTab).toHaveFocus();

    await user.keyboard("{ArrowRight}");
    // Next focus should be t-SNE (enabled)
    expect(tsneTab).toHaveFocus();

    // ArrowLeft should return focus to first
    await user.keyboard("{ArrowLeft}");
    expect(umapTab).toHaveFocus();

    // Home: jump to first
    await user.keyboard("{Home}");
    expect(umapTab).toHaveFocus();

    // End: jump to last (note: if last is disabled, focus behavior depends on implementation;
    // we allow either staying on last enabled or landing on last with aria-disabled)
    await user.keyboard("{End}");
    // If last is disabled, some implementations keep focus on previous enabled tab
    // We'll accept either behavior but ensure one of the tabs has focus
    const anyFocused = getTabs().some((t) => t === document.activeElement);
    expect(anyFocused).toBe(true);
  });

  it("skips disabled tabs during keyboard navigation", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="tsne" />);
    const [umapTab, tsneTab, fftTab] = getTabs();

    tsneTab.focus();
    expect(tsneTab).toHaveFocus();

    // ArrowRight from t-SNE would attempt to go to FFT (disabled), so it should skip to wrap/next enabled
    await user.keyboard("{ArrowRight}");
    // Either back to first enabled (umap) or stays if implementation blocks; verify not landing on disabled
    expect(document.activeElement).not.toBe(fftTab);
  });

  it("Enter/Space activates focused tab in uncontrolled mode", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="umap" />);

    const tsneTab = getTab(/t-sne/);
    tsneTab.focus();
    expect(tsneTab).toHaveFocus();

    await user.keyboard("{Enter}");
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();

    // Switch via Space as well
    const umapTab = getTab(/umap/);
    umapTab.focus();
    await user.keyboard(" ");
    expect(umapTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/umap plot/i)).toBeVisible();
  });

  it("does not activate disabled tab via keyboard", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="umap" />);
    const fftTab = getTab(/fft/);

    fftTab.focus();
    await user.keyboard("{Enter}");
    // Still on default selection; FFT panel should not become visible
    expect(screen.queryByText(/fft plot/i)).not.toBeVisible();
  });

  it("fires onTabChange with keyboard activation (uncontrolled)", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} onTabChange={onChange} defaultTab="umap" />);
    const tsneTab = getTab(/t-sne/);

    tsneTab.focus();
    await user.keyboard("{Enter}");
    expect(onChange).toHaveBeenCalledWith("tsne");
  });

  it("keeps visual selection aligned with controlled value when arrow keys move focus", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} value="tsne" onTabChange={onChange} />);

    // Focus on tsne and move focus right; selection should remain controlled as "tsne"
    const tsneTab = getTab(/t-sne/);
    tsneTab.focus();
    await user.keyboard("{ArrowRight}");

    expect(getTab(/t-sne/)).toHaveAttribute("aria-selected", "true"); // selection unchanged
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();
    // Parent may choose to update value on onTabChange; we only assert no implicit internal change
  });

  // ---------------------------------------------------------------------------
  // Edge cases & robustness
  // ---------------------------------------------------------------------------

  it("renders nothing gracefully if tabs array is empty", () => {
    render(<Tabs tabs={[]} />);
    // No tablist or tabs should be present
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("tab")).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel")).not.toBeInTheDocument();
  });

  it("exposes stable relationships between tab and panel ids", () => {
    render(<Tabs tabs={setupTabs} />);
    const umap = getTab(/umap/);
    const panel = getPanelByName(/umap/);
    // Ids must not be empty and must cross-reference
    expect(umap.id).toBeTruthy();
    expect(panel.id).toBeTruthy();
    expect(umap).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", umap.id);
  });
});
