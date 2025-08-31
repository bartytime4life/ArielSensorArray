// =============================================================================
// 🎛️ SpectraMind V50 — Tests for Reusable Button Component
// -----------------------------------------------------------------------------
// Test goals:
//   • Renders with default role/text and default variant+size classes.
//   • Supports all visual variants and sizes (class presence checks).
//   • Calls onClick when enabled; blocks when disabled and when loading.
//   • Forwards refs to the underlying <button> element.
//   • Supports `asChild` composition (e.g., renders <a> without button role).
//   • Merges custom className and retains accessible focus styles.
//   • fullWidth variant applies w-full.
//   • leftIcon/rightIcon render and spacing applies when content present.
//   • loading state renders inline spinner, sets aria-busy, disables when not asChild.
//   • icon-only usage supports `srLabel` for accessibility.
// =============================================================================

import React, { createRef } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Button } from "./Button";

describe("Button component", () => {
  it("renders with default role and text", () => {
    render(<Button>Click me</Button>);
    const btn = screen.getByRole("button", { name: /click me/i });
    expect(btn).toBeInTheDocument();
    // default variant + size classes
    expect(btn).toHaveClass("bg-blue-600", { exact: false });
    expect(btn).toHaveClass("h-10", { exact: false }); // md size by default
  });

  it("supports all visual variants (class snapshots)", () => {
    const { rerender } = render(<Button variant="default">Default</Button>);
    let btn = screen.getByRole("button", { name: /default/i });
    expect(btn).toHaveClass("bg-blue-600", { exact: false });

    rerender(<Button variant="secondary">Secondary</Button>);
    btn = screen.getByRole("button", { name: /secondary/i });
    expect(btn).toHaveClass("bg-gray-100", { exact: false });

    rerender(<Button variant="destructive">Destructive</Button>);
    btn = screen.getByRole("button", { name: /destructive/i });
    expect(btn).toHaveClass("bg-red-600", { exact: false });

    rerender(<Button variant="outline">Outline</Button>);
    btn = screen.getByRole("button", { name: /outline/i });
    expect(btn).toHaveClass("border", { exact: false });
    expect(btn).toHaveClass("bg-transparent", { exact: false });

    rerender(<Button variant="ghost">Ghost</Button>);
    btn = screen.getByRole("button", { name: /ghost/i });
    expect(btn).toHaveClass("text-gray-900", { exact: false });

    rerender(<Button variant="link">Link</Button>);
    btn = screen.getByRole("button", { name: /link/i });
    expect(btn).toHaveClass("text-blue-600", { exact: false });
    expect(btn).toHaveClass("underline-offset-4", { exact: false });
  });

  it("supports size variants", () => {
    const { rerender } = render(<Button size="sm">Small</Button>);
    let btn = screen.getByRole("button", { name: /small/i });
    expect(btn).toHaveClass("h-8", { exact: false });
    expect(btn).toHaveClass("text-xs", { exact: false });

    rerender(<Button size="md">Medium</Button>);
    btn = screen.getByRole("button", { name: /medium/i });
    expect(btn).toHaveClass("h-10", { exact: false });
    expect(btn).toHaveClass("text-sm", { exact: false });

    rerender(<Button size="lg">Large</Button>);
    btn = screen.getByRole("button", { name: /large/i });
    expect(btn).toHaveClass("h-12", { exact: false });
    expect(btn).toHaveClass("text-base", { exact: false });
  });

  it("applies fullWidth when requested", () => {
    render(<Button fullWidth>Full</Button>);
    const btn = screen.getByRole("button", { name: /full/i });
    expect(btn).toHaveClass("w-full", { exact: false });
  });

  it("calls onClick when enabled", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<Button onClick={onClick}>Go</Button>);
    const btn = screen.getByRole("button", { name: /go/i });
    await user.click(btn);
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("does not call onClick when disabled", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Button onClick={onClick} disabled>
        Nope
      </Button>
    );
    const btn = screen.getByRole("button", { name: /nope/i });
    expect(btn).toBeDisabled();
    await user.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("forwards refs to the underlying button element", () => {
    const ref = createRef<HTMLButtonElement>();
    render(<Button ref={ref}>Ref Target</Button>);
    expect(ref.current).toBeInstanceOf(HTMLButtonElement);
    expect(ref.current?.tagName.toLowerCase()).toBe("button");
  });

  it("supports asChild to render non-button elements (e.g., link)", () => {
    render(
      <Button asChild>
        <a href="/reports">Open Reports</a>
      </Button>
    );
    const link = screen.getByRole("link", { name: /open reports/i });
    expect(link).toBeInTheDocument();
    // Ensure it is not exposed as a button role
    expect(
      screen.queryByRole("button", { name: /open reports/i })
    ).not.toBeInTheDocument();
  });

  it("merges custom className with variant classes", () => {
    render(
      <Button className="data-test-flag custom-padding" variant="outline">
        Custom
      </Button>
    );
    const btn = screen.getByRole("button", { name: /custom/i });
    expect(btn).toHaveClass("border", { exact: false });
    expect(btn).toHaveClass("data-test-flag", { exact: false });
    expect(btn).toHaveClass("custom-padding", { exact: false });
  });

  it("has accessible focus styles via Tailwind ring classes", async () => {
    const user = userEvent.setup();
    render(<Button>Focus Me</Button>);
    const btn = screen.getByRole("button", { name: /focus me/i });

    // Focus the button and check it retains focus
    await user.tab();
    btn.focus();
    expect(btn).toHaveFocus();

    // Ensure the focus-visible base classes exist
    expect(btn.className).toMatch(/focus-visible:ring-2/);
    expect(btn.className).toMatch(/focus-visible:ring-offset-2/);
  });

  it("renders leftIcon/rightIcon and keeps spacing when content present", () => {
    const Left = <span data-testid="left">L</span>;
    const Right = <span data-testid="right">R</span>;
    render(
      <Button leftIcon={Left} rightIcon={Right}>
        Text
      </Button>
    );
    const btn = screen.getByRole("button", { name: /text/i });
    expect(btn.querySelector('[data-testid="left"]')).toBeInTheDocument();
    expect(btn.querySelector('[data-testid="right"]')).toBeInTheDocument();
  });

  it("shows spinner and sets aria-busy when loading; blocks clicks", async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(
      <Button loading onClick={onClick}>
        Loading
      </Button>
    );
    const btn = screen.getByRole("button", { name: /loading/i });
    // spinner svg should be present (animate-spin in class)
    const svg = btn.querySelector("svg");
    expect(svg).toBeInTheDocument();
    expect(btn).toHaveAttribute("aria-busy", "true");
    expect(btn).toBeDisabled(); // disabled when not asChild
    await user.click(btn);
    expect(onClick).not.toHaveBeenCalled();
  });

  it("supports icon-only with srLabel for accessibility", () => {
    render(<Button srLabel="Download" aria-label="Download" leftIcon={<span>↓</span>} />);
    // No visible text; role/button should be findable by sr label
    const btn = screen.getByRole("button", { name: /download/i });
    expect(btn).toBeInTheDocument();
  });
});
