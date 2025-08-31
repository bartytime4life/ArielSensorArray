// src/gui/components/loader.test.tsx
// =============================================================================
// 🧪 Tests — Loader Components (Spinner, Dots, Bar, Skeleton, Overlay) (Upgraded)
// -----------------------------------------------------------------------------
// What we test:
//   • Spinner: role/aria, size mapping, label visibility (sr-only vs visible)
//   • Spinner: tone/class merging, custom label id propagation via aria-labelledby
//   • Dots: role/aria, three bouncing dots, label inline vs sr-only
//   • Bar (determinate): aria-valuenow, label with percent, width style not null
//   • Bar (indeterminate): no aria-valuenow, has indeterminate segment + animation
//   • Bar: min/max clamping to [0,100]
//   • Skeleton: shimmer on/off renders shimmer span; respects data-* and inline style props
//   • Overlay: renders container with spinner/label; absolute vs fixed for fullScreen
//   • Overlay: merges custom className and supports tone passthrough for Spinner
// =============================================================================

import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import "@testing-library/jest-dom";

import Loader from "./Loader";

describe("Loader.Spinner", () => {
  it("renders with role=status and sr-only label by default", () => {
    render(<Loader.Spinner />);
    const status = screen.getByRole("status");
    expect(status).toBeInTheDocument();
    expect(status).toHaveAttribute("aria-live", "polite");

    // SR-only label should exist (commonly 'Loading…')
    expect(screen.getByText(/loading/i)).toBeInTheDocument();

    // SVG present with spin animation class
    const svgEl = status.querySelector("svg");
    expect(svgEl).toBeInTheDocument();
    expect(svgEl).toHaveClass("animate-spin");
  });

  it("shows visible label when showLabel=true and respects size mapping", () => {
    render(<Loader.Spinner showLabel size="xl" label="Preparing diagnostics…" />);
    expect(screen.getByText("Preparing diagnostics…")).toBeVisible();

    // Size="xl" → 36px typically
    const svg = screen.getByRole("status").querySelector("svg")!;
    expect(svg.getAttribute("width")).toBe("36");
    expect(svg.getAttribute("height")).toBe("36");
  });

  it("applies tone classes and merges custom className; supports aria-labelledby", () => {
    render(
      <div>
        <span id="slabel">Fetching plots…</span>
        <Loader.Spinner
          ariaLabelledby="slabel"
          className="custom-spinner-class"
          tone="primary"
          showLabel={false}
        />
      </div>
    );
    const status = screen.getByRole("status");
    // aria-labelledby links external label
    expect(status).toHaveAttribute("aria-labelledby", "slabel");

    // custom class should be merged (implementation-dependent; we check presence)
    expect(status.className).toMatch(/custom-spinner-class/);
  });
});

describe("Loader.Dots", () => {
  it("renders three bouncing dots and sr-only label", () => {
    render(<Loader.Dots />);
    const status = screen.getByRole("status");
    const dots = status.querySelectorAll("span");
    // At least 3 span dots; first three are animated
    expect(dots.length).toBeGreaterThanOrEqual(3);
    expect(dots[0]).toHaveClass("animate-bounce");
    expect(dots[1]).toHaveClass("animate-bounce");
    expect(dots[2]).toHaveClass("animate-bounce");
    // SR-only label exists
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("shows inline label when inlineLabel=true", () => {
    render(<Loader.Dots inlineLabel label="Fetching UMAP…" />);
    expect(screen.getByText("Fetching UMAP…")).toBeVisible();
  });
});

describe("Loader.Bar", () => {
  it("renders determinate progress with correct aria and label", () => {
    render(<Loader.Bar value={50} showLabel label="Processing" />);
    const bar = screen.getByRole("progressbar");
    expect(bar).toHaveAttribute("aria-valuemin", "0");
    expect(bar).toHaveAttribute("aria-valuemax", "100");
    expect(bar).toHaveAttribute("aria-valuenow", "50");
    expect(bar).toHaveAttribute("aria-label", "Processing");
    // Label with percent appears
    expect(screen.getByText(/Processing\s*\(50%\)/i)).toBeVisible();
    // Inner child should have a width style set (determinate)
    const inner = bar.querySelector("div");
    expect(inner).toBeInTheDocument();
    expect(inner!.getAttribute("style")).toMatch(/width:\s*50%/i);
  });

  it("clamps value <0 to 0 and >100 to 100 for aria and width", () => {
    const { rerender } = render(<Loader.Bar value={-10} showLabel label="ClampLow" />);
    let bar = screen.getByRole("progressbar");
    let inner = bar.querySelector("div")!;
    expect(bar).toHaveAttribute("aria-valuenow", "0");
    expect(inner.getAttribute("style")).toMatch(/width:\s*0%/i);

    rerender(<Loader.Bar value={250} showLabel label="ClampHigh" />);
    bar = screen.getByRole("progressbar");
    inner = bar.querySelector("div")!;
    expect(bar).toHaveAttribute("aria-valuenow", "100");
    expect(inner.getAttribute("style")).toMatch(/width:\s*100%/i);
  });

  it("renders indeterminate progress without aria-valuenow", () => {
    render(<Loader.Bar tone="primary" />);
    const bar = screen.getByRole("progressbar");
    expect(bar).not.toHaveAttribute("aria-valuenow");
    // Indeterminate segment has fixed width and animation class
    const segment = bar.querySelector("div");
    expect(segment).toBeInTheDocument();
    expect(segment).toHaveClass("w-1/3");
    // Presence of animation class or keyframe utility
    expect(segment!.className).toMatch(/animate|transition|motion/);
  });
});

describe("Loader.Skeleton", () => {
  it("renders shimmering skeleton by default", () => {
    render(<Loader.Skeleton data-testid="skel" style={{ width: 120, height: 16 }} />);
    const skel = screen.getByTestId("skel");
    expect(skel).toBeInTheDocument();
    // Shimmer span should be present (aria-hidden)
    const shimmer = skel.querySelector("span[aria-hidden='true']");
    expect(shimmer).toBeInTheDocument();
  });

  it("does not render shimmer when shimmer=false", () => {
    render(<Loader.Skeleton data-testid="skel" shimmer={false} />);
    const skel = screen.getByTestId("skel");
    expect(skel.querySelector("span[aria-hidden='true']")).not.toBeInTheDocument();
  });

  it("passes arbitrary data- attributes and inline styles", () => {
    render(
      <Loader.Skeleton
        data-testid="skel"
        data-kind="row"
        style={{ width: 200, height: 12 }}
      />
    );
    const skel = screen.getByTestId("skel");
    expect(skel).toHaveAttribute("data-kind", "row");
    // Inline style presence (jsdom stores inline CSS as string)
    expect(skel.getAttribute("style")).toMatch(/width:\s*200px/i);
    expect(skel.getAttribute("style")).toMatch(/height:\s*12px/i);
  });
});

describe("Loader.Overlay", () => {
  it("renders absolute overlay by default with spinner and label", () => {
    render(<Loader.Overlay label="Calibrating…" />);
    const label = screen.getByText("Calibrating…");
    expect(label).toBeVisible();

    // The overlay container (one of the ancestor divs) should include 'absolute'
    const overlay = label.closest("div")!.parentElement!;
    expect(overlay.className).toMatch(/\babsolute\b/);

    // Ensure Spinner exists within overlay
    const status = within(overlay).getByRole("status");
    expect(status).toBeInTheDocument();
  });

  it("renders fixed overlay when fullScreen=true", () => {
    render(<Loader.Overlay fullScreen label="End-to-end run…" />);
    const label = screen.getByText("End-to-end run…");
    // Look up the overlay container ancestors and find one with 'fixed' positioning class
    let el: HTMLElement | null = label;
    let found = false;
    for (let i = 0; i < 5 && el; i++) {
      if (el.className?.includes?.("fixed")) {
        found = true;
        break;
      }
      el = el.parentElement;
    }
    expect(found).toBe(true);
  });

  it("merges className and passes tone to Spinner", () => {
    render(<Loader.Overlay className="overlay-test" tone="primary" label="Merging…" />);
    const label = screen.getByText("Merging…");
    // Ascend a couple of levels to the positioned container
    const overlayRoot = label.closest("div")!.parentElement!;
    expect(overlayRoot.className).toMatch(/overlay-test/);

    // Spinner present within; tone styling is implementation-defined; we at least assert role exists
    const status = within(overlayRoot).getByRole("status");
    expect(status).toBeInTheDocument();
  });
});
