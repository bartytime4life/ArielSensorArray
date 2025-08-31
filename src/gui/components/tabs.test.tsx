// src/gui/components/tabs.test.tsx
// =============================================================================
// 🧪 Tests — Tabs Component
// -----------------------------------------------------------------------------
// What we test:
//   • Renders tabs with labels
//   • Default tab selection (uncontrolled)
//   • Controlled tab selection with value prop
//   • Calls onTabChange when tab clicked
//   • Displays correct content per tab
//   • Handles disabled tabs (cannot be activated)
//   • Forwards className for container
//   • Accessibility: role, aria-selected, aria-controls
// =============================================================================

import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Tabs } from "./Tabs";

describe("Tabs component", () => {
  const setupTabs = [
    { id: "umap", label: "UMAP", content: <div>UMAP Plot</div> },
    { id: "tsne", label: "t-SNE", content: <div>t-SNE Plot</div> },
    { id: "fft", label: "FFT", content: <div>FFT Plot</div>, disabled: true },
  ];

  it("renders all tab labels", () => {
    render(<Tabs tabs={setupTabs} />);
    expect(screen.getByRole("tab", { name: /umap/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /t-sne/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /fft/i })).toBeInTheDocument();
  });

  it("selects the first tab by default (uncontrolled)", () => {
    render(<Tabs tabs={setupTabs} />);
    const umapTab = screen.getByRole("tab", { name: /umap/i });
    expect(umapTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/umap plot/i)).toBeVisible();
  });

  it("respects defaultTab prop (uncontrolled)", () => {
    render(<Tabs tabs={setupTabs} defaultTab="tsne" />);
    const tsneTab = screen.getByRole("tab", { name: /t-sne/i });
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();
  });

  it("changes tab on click in uncontrolled mode", async () => {
    const user = userEvent.setup();
    render(<Tabs tabs={setupTabs} defaultTab="umap" />);

    const tsneTab = screen.getByRole("tab", { name: /t-sne/i });
    await user.click(tsneTab);

    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();
  });

  it("calls onTabChange when a tab is clicked", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} onTabChange={onChange} />);
    const tsneTab = screen.getByRole("tab", { name: /t-sne/i });
    await user.click(tsneTab);
    expect(onChange).toHaveBeenCalledWith("tsne");
  });

  it("works in controlled mode with value prop", () => {
    render(<Tabs tabs={setupTabs} value="tsne" />);
    const tsneTab = screen.getByRole("tab", { name: /t-sne/i });
    expect(tsneTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText(/t-sne plot/i)).toBeVisible();

    // The uncontrolled state shouldn't change value if controlled
    const umapTab = screen.getByRole("tab", { name: /umap/i });
    expect(umapTab).toHaveAttribute("aria-selected", "false");
  });

  it("does not activate disabled tabs", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Tabs tabs={setupTabs} onTabChange={onChange} />);
    const fftTab = screen.getByRole("tab", { name: /fft/i });
    expect(fftTab).toBeDisabled();
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
    const umapTab = screen.getByRole("tab", { name: /umap/i });
    const umapPanel = screen.getByRole("tabpanel", { name: /umap/i });
    expect(umapTab).toHaveAttribute("aria-controls", umapPanel.id);
    expect(umapPanel).toHaveAttribute("aria-labelledby", umapTab.id);
  });
});
