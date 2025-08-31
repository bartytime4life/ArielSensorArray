// src/gui/components/tooltip.test.tsx
// =============================================================================
// 🧪 Tests — Tooltip Component (Upgraded)
// -----------------------------------------------------------------------------
// What we test:
//   • Renders nothing when closed by default
//   • Opens on hover/focus after delay and attaches aria-describedby
//   • Closes on mouse leave after small delay; removes aria-describedby
//   • ESC closes (uncontrolled) and calls onOpenChange in controlled mode
//   • Hovering the panel keeps it open when disableHoverableContent=false
//   • When disableHoverableContent=true hovering the panel does NOT keep it open
//   • Controlled open: hover/focus should request open via onOpenChange(true)
//   • Controlled open: parent-driven close reflects in DOM and aria attributes
//   • Positions are stable under jsdom via mocked getBoundingClientRect
//   • Focus + blur semantics mirror hover + unhover timings
// =============================================================================

import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import Tooltip from "./Tooltip";

// Helper: jsdom returns 0 rects; provide a reasonable bbox to elements
const mockRect = (rect: Partial<DOMRect> = {}) =>
  ({
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
  } as DOMRect);

let getBCRSpy: vi.SpyInstance;

beforeEach(() => {
  vi.useFakeTimers();

  // Mock getBoundingClientRect for all elements to stabilize positioning logic
  getBCRSpy = vi
    .spyOn(Element.prototype, "getBoundingClientRect")
    .mockImplementation(function (this: Element) {
      // Give tooltip panel a different rect (e.g., 120x40 at 60,60)
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

const OPEN_DELAY = 90; // > default 80ms
const CLOSE_DELAY = 70; // > default 60ms

describe("Tooltip (uncontrolled)", () => {
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

    // Default open delay ~80ms; we go slightly over
    vi.advanceTimersByTime(OPEN_DELAY);

    const tip = screen.getByRole("tooltip");
    expect(tip).toBeInTheDocument();
    expect(tip).toHaveTextContent("Open diagnostics");

    // aria-describedby on trigger should reference tooltip id
    const describedBy = trigger.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(tip).toHaveAttribute("id", describedBy!);
  });

  it("closes on mouse leave after a small delay and removes aria-describedby", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Leaving soon">
        <button>Hover me</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /hover me/i });

    await user.hover(trigger);
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    const describedBy = trigger.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();

    await user.unhover(trigger);
    // hide delay ~60ms
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    // aria-describedby should be removed when closed
    expect(trigger).not.toHaveAttribute("aria-describedby");
  });

  it("opens on focus (keyboard) after delay and closes on ESC", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Press ESC">
        <button>Focus me</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /focus me/i });
    trigger.focus();
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("hovering the panel keeps it open when disableHoverableContent=false (default)", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Sticky">
        <button>Stick</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /stick/i });

    await user.hover(trigger);
    vi.advanceTimersByTime(OPEN_DELAY);
    const panel = screen.getByRole("tooltip");
    expect(panel).toBeInTheDocument();

    // Move cursor from trigger to panel: panel mouseenter should cancel hide timer
    await user.unhover(trigger);
    await user.hover(panel);

    // Even after hide delay, still open while panel hovered
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    // Now leave the panel -> should close after delay
    await user.unhover(panel);
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("does not keep open when disableHoverableContent=true and cursor is over panel", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Non-sticky" disableHoverableContent>
        <button>Stick?</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /stick\?/i });

    await user.hover(trigger);
    vi.advanceTimersByTime(OPEN_DELAY);
    const panel = screen.getByRole("tooltip");
    expect(panel).toBeInTheDocument();

    // Leave trigger and hover panel; since disableHoverableContent=true, panel hover should not keep it open
    await user.unhover(trigger);
    await user.hover(panel);
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("closes after blur similar to mouse leave timing", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    render(
      <Tooltip content="Blur closes">
        <button>Focus target</button>
      </Tooltip>
    );

    const trigger = screen.getByRole("button", { name: /focus target/i });
    trigger.focus();
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(screen.getByRole("tooltip")).toBeInTheDocument();

    // Blur should start close timer
    trigger.blur();
    vi.advanceTimersByTime(CLOSE_DELAY);
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});

describe("Tooltip (controlled)", () => {
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

  it("controlled: hover/focus should request open via onOpenChange(true)", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onOpenChange = vi.fn();
    render(
      <Tooltip content="Need open" open={false} onOpenChange={onOpenChange}>
        <button>Open me</button>
      </Tooltip>
    );

    const trigger = screen.getByRole("button", { name: /open me/i });
    await user.hover(trigger);
    // opening is timer-based; we simulate enough delay then expect a request
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(onOpenChange).toHaveBeenCalledWith(true);

    // Focus path should also request open
    onOpenChange.mockClear();
    trigger.focus();
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(onOpenChange).toHaveBeenCalledWith(true);
  });

  it("controlled: parent-driven open/close reflects in DOM and aria-describedby", async () => {
    const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
    const onOpenChange = vi.fn();

    const { rerender } = render(
      <Tooltip content="Parent control" open={false} onOpenChange={onOpenChange}>
        <button>Parent</button>
      </Tooltip>
    );
    const trigger = screen.getByRole("button", { name: /parent/i });

    // Initially closed
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(trigger).not.toHaveAttribute("aria-describedby");

    // Parent opens
    rerender(
      <Tooltip content="Parent control" open={true} onOpenChange={onOpenChange}>
        <button>Parent</button>
      </Tooltip>
    );
    expect(screen.getByRole("tooltip")).toBeInTheDocument();
    const tipId = trigger.getAttribute("aria-describedby");
    expect(tipId).toBeTruthy();

    // Parent closes
    rerender(
      <Tooltip content="Parent control" open={false} onOpenChange={onOpenChange}>
        <button>Parent</button>
      </Tooltip>
    );
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(trigger).not.toHaveAttribute("aria-describedby");

    // User hover requests open again
    await user.hover(trigger);
    vi.advanceTimersByTime(OPEN_DELAY);
    expect(onOpenChange).toHaveBeenCalledWith(true);
  });
});
