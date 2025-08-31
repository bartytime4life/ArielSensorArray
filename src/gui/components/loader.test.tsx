// src/gui/components/loader.test.tsx
// =============================================================================
// 🧪 Tests — Loader Components (Spinner, Dots, Bar, Skeleton, Overlay)
// -----------------------------------------------------------------------------
// What we test:
//   • Spinner: role/aria, size, label visibility (sr-only vs visible)
//   • Dots: role/aria, three bouncing dots, label inline vs sr-only
//   • Bar (determinate): aria-valuenow, label with percent, width style not null
//   • Bar (indeterminate): no aria-valuenow, has indeterminate segment
//   • Skeleton: shimmer on/off renders shimmer span
//   • Overlay: renders container with spinner label; absolute vs fixed for fullScreen
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
    // The label should exist but be visually hidden unless showLabel
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
    // SVG present with animate-spin class
    const svg = within(status).getByRole("img", { hidden: true });
    // jsdom doesn't compute roles for <svg> automatically via getByRole unless aria-hidden=false,
    // so instead assert the SVG element is present via querySelector fallback:
    const svgEl = status.querySelector("svg");
    expect(svgEl).toBeInTheDocument();
    expect(svgEl).toHaveClass("animate-spin");
  });

  it("shows visible label when showLabel=true and respects size=xl", () => {
    render(<Loader.Spinner showLabel size="xl" label="Preparing diagnostics…" />);
    expect(screen.getByText("Preparing diagnostics…")).toBeVisible();
    const svg = screen.getByRole("status").querySelector("svg")!;
    expect(svg.getAttribute("width")).toBe("36");
    expect(svg.getAttribute("height")).toBe("36");
  });
});

describe("Loader.Dots", () => {
  it("renders three bouncing dots and sr-only label", () => {
    render(<Loader.Dots />);
    const status = screen.getByRole("status");
    const dots = status.querySelectorAll("span");
    // There are at least 3 span dots; first three are the animated ones
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
    // width is set inline as a percentage, e.g. "50%"
    expect(inner!.getAttribute("style")).toMatch(/width:\s*50%/i);
  });

  it("renders indeterminate progress without aria-valuenow", () => {
    render(<Loader.Bar tone="primary" />);
    const bar = screen.getByRole("progressbar");
    expect(bar).not.toHaveAttribute("aria-valuenow");
    // Indeterminate segment uses fixed width (w-1/3) with animation class
    const segment = bar.querySelector("div");
    expect(segment).toBeInTheDocument();
    expect(segment).toHaveClass("w-1/3");
  });
});

describe("Loader.Skeleton", () => {
  it("renders shimmering skeleton by default", () => {
    render(<Loader.Skeleton data-testid="skel" style={{ width: 120, height: 16 }} />);
    const skel = screen.getByTestId("skel");
    expect(skel).toBeInTheDocument();
    // Shimmer span should be present
    const shimmer = skel.querySelector("span[aria-hidden='true']");
    expect(shimmer).toBeInTheDocument();
  });

  it("does not render shimmer when shimmer=false", () => {
    render(<Loader.Skeleton data-testid="skel" shimmer={false} />);
    const skel = screen.getByTestId("skel");
    expect(skel.querySelector("span[aria-hidden='true']")).not.toBeInTheDocument();
  });
});

describe("Loader.Overlay", () => {
  it("renders absolute overlay by default with spinner and label", () => {
    render(<Loader.Overlay label="Calibrating…" />);
    const overlay = screen.getByText("Calibrating…").closest("div")!.parentElement!;
    // The overlay root should have position class 'absolute' by default
    expect(overlay).toHaveClass("absolute");
    // Spinner label visible
    expect(screen.getByText("Calibrating…")).toBeVisible();
  });

  it("renders fixed overlay when fullScreen=true", () => {
    render(<Loader.Overlay fullScreen label="End-to-end run…" />);
    // Find the top-level overlay: has fixed class
    // The overlay is the first div matching both inset-0 & z-40 (as composed), but just assert 'fixed'
    const overlays = screen.getAllByText("End-to-end run…").map((n) => n.closest("div")!.parentElement!);
    const found = overlays.find((el) => el.className.includes("fixed"));
    expect(found).toBeTruthy();
  });
});
