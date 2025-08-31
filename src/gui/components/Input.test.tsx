// =============================================================================
// 🧪 Tests — Input Component
// -----------------------------------------------------------------------------
// What we test:
//   • Renders with label (incl. sr-only) and placeholder
//   • Supports value/onChange and different input types
//   • Shows helperText and error with ARIA compliance (aria-describedby / aria-errormessage / aria-invalid)
//   • Forwards refs to the underlying <input>
//   • Applies disabled state and class merging
//   • Size variants and slots adjust padding (leftSlot / rightSlot)
//   • Required indicator renders when required + requiredIndicator !== false
// =============================================================================

import React, { createRef } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Input } from "./Input";

describe("Input component", () => {
  it("renders label and placeholder", () => {
    render(<Input id="u" label="Username" placeholder="Enter username" />);
    const label = screen.getByText(/username/i);
    expect(label).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/enter username/i);
    expect(input).toBeInTheDocument();
    // label should be associated to input
    expect(label).toHaveAttribute("for", "u");
  });

  it("supports visually-hidden (sr-only) label", () => {
    render(<Input id="sr" label="Hidden Label" labelSrOnly placeholder="x" />);
    const label = screen.getByText(/hidden label/i);
    expect(label).toBeInTheDocument();
    expect(label.className).toMatch(/sr-only/);
    // Still associated
    const input = screen.getByPlaceholderText("x");
    expect(label).toHaveAttribute("for", "sr");
    expect(input).toHaveAttribute("id", "sr");
  });

  it("shows required indicator when required", () => {
    render(<Input id="r" label="Req" required placeholder="y" />);
    const label = screen.getByText("Req");
    // An asterisk should be rendered visually (aria-hidden)
    const star = label.querySelector("span");
    expect(star).toBeTruthy();
    expect(star).toHaveAttribute("aria-hidden", "true");
    expect(star?.textContent).toBe("*");
  });

  it("supports value and onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <Input
        id="email"
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
    // ensure type and id applied
    expect(input).toHaveAttribute("type", "email");
    expect(input).toHaveAttribute("id", "email");
  });

  it("shows helper text when provided (no error) with aria-describedby linking", () => {
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
    expect(helper).toHaveAttribute("id", "user-input-helper");

    // aria-describedby should point to helper when no error
    const input = screen.getByPlaceholderText(/user123/i);
    expect(input).toHaveAttribute("aria-describedby", "user-input-helper");
  });

  it("shows error text and sets aria-invalid, aria-errormessage when error provided", () => {
    render(
      <Input
        id="pwd"
        label="Password"
        placeholder="••••••••"
        error="Password is too short"
      />
    );

    const input = screen.getByPlaceholderText("••••••••");
    const error = screen.getByText(/password is too short/i);
    expect(error).toBeInTheDocument();
    expect(error).toHaveAttribute("id", "pwd-error");

    expect(input).toHaveAttribute("aria-invalid", "true");
    // describedby may be error id (or combined), but errormessage should reference error id
    expect(input).toHaveAttribute("aria-errormessage", "pwd-error");
  });

  it("forwards refs to the underlying input element", () => {
    const ref = createRef<HTMLInputElement>();
    render(<Input ref={ref} id="refd" label="Ref Field" />);
    expect(ref.current).toBeInstanceOf(HTMLInputElement);
    // sanity: ensure it can be focused via ref
    ref.current?.focus();
    expect(ref.current).toHaveFocus();
  });

  it("supports different input types (number)", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Input id="num" type="number" placeholder="0" onChange={onChange} />);

    const input = screen.getByPlaceholderText("0") as HTMLInputElement;
    await user.type(input, "123");
    expect(onChange).toHaveBeenCalled();
    expect(input).toHaveAttribute("type", "number");
  });

  it("applies disabled state", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<Input id="dis" label="Disabled" placeholder="nope" disabled onChange={onChange} />);

    const input = screen.getByPlaceholderText(/nope/i);
    expect(input).toBeDisabled();

    await user.type(input, "should not type");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("merges custom className", () => {
    render(
      <Input
        id="c"
        label="Custom"
        placeholder="custom"
        className="data-test-input-class"
      />
    );
    const input = screen.getByPlaceholderText(/custom/i);
    expect(input).toHaveClass("data-test-input-class", { exact: false });
  });

  it("applies size styles and slot paddings (left/right)", () => {
    // md size with leftSlot
    const { rerender } = render(
      <Input
        id="md"
        size="md"
        label="MD"
        placeholder="md"
        leftSlot={<span data-testid="ls">L</span>}
      />
    );
    let input = screen.getByPlaceholderText("md");
    expect(input.className).toMatch(/pl-9/); // md left slot adds pl-9

    // lg size with both slots
    rerender(
      <Input
        id="lg"
        size="lg"
        label="LG"
        placeholder="lg"
        leftSlot={<span data-testid="ls">L</span>}
        rightSlot={<span data-testid="rs">R</span>}
      />
    );
    input = screen.getByPlaceholderText("lg");
    expect(input.className).toMatch(/pl-10/);
    expect(input.className).toMatch(/pr-10/);
  });

  it("renders without label and still accessible via placeholder", () => {
    render(<Input id="p" placeholder="just text" />);
    const input = screen.getByPlaceholderText(/just text/i);
    expect(input).toBeInTheDocument();
  });
});
