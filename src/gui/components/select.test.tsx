// src/gui/components/select.test.tsx
// =============================================================================
// 🧪 Tests — Select Component
// -----------------------------------------------------------------------------
// What we test:
//   • Renders with label and options
//   • Supports value/onChange, controlled usage
//   • Shows helperText with aria-describedby
//   • Shows error (aria-invalid) and renders error text
//   • Forwards refs to underlying <select>
//   • Applies disabled state (blocks changes)
//   • Supports custom children when options not provided
//   • Merges custom className
// =============================================================================

import React, { createRef } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Select } from "./Select";

describe("Select component", () => {
  it("renders label and options", () => {
    render(
      <Select
        id="planet"
        label="Planet"
        options={[
          { label: "Kepler-22b", value: "kepler-22b" },
          { label: "WASP-12b", value: "wasp-12b" },
        ]}
      />
    );

    // label present
    expect(screen.getByText(/planet/i)).toBeInTheDocument();

    // options present
    const select = screen.getByLabelText(/planet/i) as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /kepler-22b/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /wasp-12b/i })).toBeInTheDocument();
  });

  it("supports value and onChange", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <Select
        id="target"
        label="Target"
        value="kepler-22b"
        onChange={onChange}
        options={[
          { label: "Kepler-22b", value: "kepler-22b" },
          { label: "WASP-12b", value: "wasp-12b" },
        ]}
      />
    );

    const select = screen.getByLabelText(/target/i) as HTMLSelectElement;
    expect(select.value).toBe("kepler-22b");
    await user.selectOptions(select, "wasp-12b");

    expect(onChange).toHaveBeenCalledTimes(1);
    // NOTE: controlled component requires parent to update value; we just assert change handler fired.
  });

  it("shows helper text and binds aria-describedby when no error", () => {
    render(
      <Select
        id="star"
        label="Star"
        helperText="Pick a host star"
        options={[
          { label: "Sun", value: "sol" },
          { label: "Vega", value: "vega" },
        ]}
      />
    );

    // helper text is shown
    const helper = screen.getByText(/pick a host star/i);
    expect(helper).toBeInTheDocument();

    const select = screen.getByLabelText(/star/i);
    expect(select).toHaveAttribute("aria-describedby", "star-helper");
  });

  it("shows error text and sets aria-invalid when error provided", () => {
    render(
      <Select
        id="instrument"
        label="Instrument"
        error="Invalid instrument selection"
        options={[
          { label: "FGS1", value: "fgs1" },
          { label: "AIRS", value: "airs" },
        ]}
      />
    );

    expect(screen.getByText(/invalid instrument selection/i)).toBeInTheDocument();
    const select = screen.getByLabelText(/instrument/i);
    expect(select).toHaveAttribute("aria-invalid", "true");
  });

  it("forwards refs to the underlying select element", () => {
    const ref = createRef<HTMLSelectElement>();
    render(
      <Select
        ref={ref}
        id="mode"
        label="Mode"
        options={[
          { label: "Fast", value: "fast" },
          { label: "Accurate", value: "accurate" },
        ]}
      />
    );
    expect(ref.current).toBeInstanceOf(HTMLSelectElement);
    ref.current?.focus();
    expect(ref.current).toHaveFocus();
  });

  it("applies disabled state and prevents changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <Select
        id="disabled-select"
        label="Disabled"
        disabled
        onChange={onChange}
        options={[
          { label: "One", value: "1" },
          { label: "Two", value: "2" },
        ]}
      />
    );

    const select = screen.getByLabelText(/disabled/i);
    expect(select).toBeDisabled();
    await user.selectOptions(select, "2");
    expect(onChange).not.toHaveBeenCalled();
  });

  it("supports custom children when options are not provided", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(
      <Select id="custom" label="Custom" onChange={onChange}>
        <option value="alpha">Alpha</option>
        <option value="beta">Beta</option>
      </Select>
    );

    const select = screen.getByLabelText(/custom/i) as HTMLSelectElement;
    expect(screen.getByRole("option", { name: /alpha/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /beta/i })).toBeInTheDocument();

    await user.selectOptions(select, "beta");
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("merges custom className", () => {
    render(
      <Select
        id="merge"
        label="Merge"
        className="data-test-select-class"
        options={[{ label: "X", value: "x" }]}
      />
    );
    const select = screen.getByLabelText(/merge/i);
    expect(select).toHaveClass("data-test-select-class", { exact: false });
  });
});
