// src/gui/components/Input.test.tsx
// =============================================================================
// 🧪 Tests — Input Component
// -----------------------------------------------------------------------------
// What we test:
//   • Renders with label and placeholder
//   • Supports value/onChange and different input types
//   • Shows helperText and error (with ARIA compliance)
//   • Forwards refs to the underlying <input>
//   • Applies disabled state and class merging
// =============================================================================

import React, { createRef } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Input } from "./Input";

describe("Input component", () => {
  it("renders label and placeholder", () => {
    render(<Input label="Username" placeholder="Enter username" />);
    const label = screen.getByText(/username/i);
    expect(label).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/enter username/i);
    expect(input).toBeInTheDocument();
  });

  it("supports value and onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <Input
        label="Email"
        placeholder="name@example.com"
        value=""
        onChange={onChange}
        type="email"
      />
    );

    const input = screen.getByPlaceholderText(/name@example\.com/i);
    await user.type(input, "tester@example.com");

    // onChange should be called once per character
    expect(onChange).toHaveBeenCalled();
    expect(onChange.mock.calls.length).toBe("tester@example.com".length);
  });

  it("shows helper text when provided (no error)", () => {
    render(
      <Input
        id="user-input"
        label="User"
        helperText="Must be 4–20 characters"
        placeholder="user123"
      />
    );

    // Helper text present
    const helper = screen.getByText(/must be 4–20 characters/i);
    expect(helper).toBeInTheDocument();

    // aria-describedby should point to helper when no error
    const input = screen.getByPlaceholderText(/user123/i);
    expect(input).toHaveAttribute("aria-describedby", "user-input-helper");
  });

  it("shows error text and sets aria-invalid when error provided", () => {
    render(
      <Input
        label="Password"
        placeholder="••••••••"
        error="Password is too short"
      />
    );

    const input = screen.getByPlaceholderText("••••••••");
    expect(screen.getByText(/password is too short/i)).toBeInTheDocument();
    expect(input).toHaveAttribute("aria-invalid", "true");

    // When error is present, aria-describedby may be undefined if helperText is not present.
    // We only assert aria-invalid here (helperText behavior verified above).
  });

  it("forwards refs to the underlying input element", () => {
    const ref = createRef<HTMLInputElement>();
    render(<Input ref={ref} label="Ref Field" />);
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
    // sanity: ensure it can be focused via ref
    ref.current?.focus();
    expect(ref.current).toHaveFocus();
  });

  it("supports different input types (number)", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Input type="number" placeholder="0" onChange={onChange} />);

    const input = screen.getByPlaceholderText("0") as HTMLInputElement;
    await user.type(input, "123");
    expect(onChange).toHaveBeenCalled();
    expect(input).toHaveAttribute("type", "number");
  });

  it("applies disabled state", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Input label="Disabled" placeholder="nope" disabled onChange={onChange} />);

    const input = screen.getByPlaceholderText(/nope/i);
    expect(input).toBeDisabled();

    await user.type(input, "should not type");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("merges custom className", () => {
    render(
      <Input
        label="Custom"
        placeholder="custom"
        className="data-test-input-class"
      />
    );
    const input = screen.getByPlaceholderText(/custom/i);
    expect(input).toHaveClass("data-test-input-class", { exact: false });
  });
});
