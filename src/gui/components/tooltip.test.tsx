// src/gui/components/tooltip.test.tsx
// =============================================================================
// 🧪 Tests — Tooltip Component
// -----------------------------------------------------------------------------
// What we test:
//   • Renders nothing when closed by default
//   • Opens on hover/focus after delay and attaches aria-describedby
//   • Closes on mouse leave after small delay
//   • ESC closes (uncontrolled) and calls onOpenChange in controlled mode
//   • Hovering the panel keeps it open when disableHoverableContent=false
// =============================================================================

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import Tooltip from "./Tooltip";

// jsdom returns 0 rects by default; provide a reasonable bbox to the elements
const mockRect = (rect: Partial<DOMRect> = {}) => {
  return {
    x: 100,
    y: 100,
    top: 100,
    left: 100,
    bottom: 140,
    right: 160,
    width: 60,
    height: 40,
    toJSON: () => ({}),
    ...rect,
  } as DOMRect;
};

let getBCRSpy: vi.SpyInstance;

beforeEach(() => {
  vi.useFakeTimers();
  // Mock getBoundingClientRect for all elements
  getBCRSpy = vi
    .spyOn(Element.prototype, "getBoundingClientRect")
    .mockImplementation(function (this: Element) {
      // Trigger roughly 60x40 at 100,100; tooltip panel ~ 120x40
      if ((this as HTMLElement).getAttribute("role") === "tooltip") {
        return mockRect({ width: 120, height: 40, top: 60, left: 60 });
      }
      return mockRect();
    });
});

afterEach(() => {
  vi.runOnlyPendingTimers();
  vi.useRealTimers();
  getBCRSpy.mockRestore();
});

describe("Tooltip", () => {
  it("does not render by default (closed)", () => {
    render(
      <Tooltip content="Hello tip">
        <button>Trigger</button>
      </Tooltip>
    );
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("opens on hover after delay and applies aria-describedby", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Open diagnostics">
        <button>Open</button>
      </Tooltip>
    );

    const trigger = screen.getByRole("button", { name: /open/i });
    await user.hover(trigger);

    // Default delay is 80ms; advance timers
    vi.advanceTimersByTime(90);

    const tip = screen.getByRole("tooltip");
    expect(tip).toBeInTheDocument();
    expect(tip).toHaveTextContent("Open diagnostics");

    // aria-describedby on trigger should reference tooltip id
    const describedBy = trigger.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(tip).toHaveAttribute("id", describedBy!);
  });

  it("closes on mouse leave after a small delay", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Leaving soon">
        <button>Hover me</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /hover me/i });

    await user.hover(trigger);
    vi.advanceTimersByTime(90);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    await user.unhover(trigger);
    // hide delay is 60ms
    vi.advanceTimersByTime(61);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("ESC closes in uncontrolled mode", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Press ESC">
        <button>Focus me</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /focus me/i });
    // Open via focus
    trigger.focus();
    vi.advanceTimersByTime(90);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("controlled: ESC attempts to close via onOpenChange(false)", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onOpenChange = vi.fn();
    render(
      <Tooltip content="Details" open={true} onOpenChange={onOpenChange}>
        <button>Ctrl</button>
      </Tooltip>
    );
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    await user.keyboard("{Escape}");
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });

  it("hovering the panel keeps it open when disableHoverableContent=false", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Sticky">
        <button>Stick</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /stick/i });

    await user.hover(trigger);
    vi.advanceTimersByTime(90);
    const panel = screen.getByRole("tooltip");
    expect(panel).toBeInTheDocument();

    // Move cursor from trigger to panel: panel mouseenter should cancel hide timer
    await user.unhover(trigger);
    // Simulate hovering panel (user-event hover calls mouseEnter/over)
    await user.hover(panel);

    // Even after the 60ms hide delay, it should remain open while panel hovered
    vi.advanceTimersByTime(70);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    // Now leave the panel -> it should close after delay
    await user.unhover(panel);
    vi.advanceTimersByTime(70);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
