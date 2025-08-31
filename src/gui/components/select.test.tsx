// src/gui/components/select.test.tsx
// =============================================================================
// 🧪 Tests — Select Component (Upgraded)
// -----------------------------------------------------------------------------
// What we test:
//   • Renders with label (associates via htmlFor/aria-labelledby) and options
//   • Supports value/onChange (controlled usage) and passes event value
//   • Shows helperText with aria-describedby (when no error)
//   • Shows error (aria-invalid) and binds aria-describedby to error id (over helper)
//   • Forwards refs to underlying <select>
//   • Applies disabled state (blocks changes)
//   • Supports custom children when options not provided
//   • Merges custom className
//   • Forwards standard props (name, required) and sets aria-required
//   • Optional placeholder option appears when provided and is not selectable when disabled
// =============================================================================

import React, { createRef } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom";

import { Select } from "./Select";

describe("Select component", () => {
  it("renders label and options with correct associations", () => {
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

    // Label present and associated via htmlFor -> id, or aria-labelledby fallback
    const label = screen.getByText(/planet/i);
    expect(label).toBeInTheDocument();

    const select = screen.getByLabelText(/planet/i) as HTMLSelectElement;
    expect(select).toBeInTheDocument();
    expect(select.id).toBe("planet");

    // Options present
    expect(screen.getByRole("option", { name: /kepler-22b/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /wasp-12b/i })).toBeInTheDocument();
  });

  it("supports value and onChange (controlled), including value in event target", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn((e: React.ChangeEvent<HTMLSelectElement>) => e.target.value);

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
    // Parent would update value; we at least confirm the event carries requested value
    const lastCallEvt = onChange.mock.calls[0][0] as React.ChangeEvent<HTMLSelectElement>;
    expect(lastCallEvt.target.value).toBe("wasp-12b");
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
    // By convention, helper has id `${id}-helper`
    expect(select).toHaveAttribute("aria-describedby", "star-helper");
  });

  it("shows error text and sets aria-invalid; aria-describedby points to error id (over helper)", () => {
    render(
      <Select
        id="instrument"
        label="Instrument"
        helperText="Choose instrument"
        error="Invalid instrument selection"
        options={[
          { label: "FGS1", value: "fgs1" },
          { label: "AIRS", value: "airs" },
        ]}
      />
    );

    // Error shown and aria-invalid applied
    expect(screen.getByText(/invalid instrument selection/i)).toBeInTheDocument();
    const select = screen.getByLabelText(/instrument/i);
    expect(select).toHaveAttribute("aria-invalid", "true");

    // error id should override helper id in aria-describedby
    // By convention, error has id `${id}-error`
    expect(select).toHaveAttribute("aria-describedby", "instrument-error");
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
        <option value="">—</option>
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

  it("forwards name and required props; sets aria-required", () => {
    render(
      <Select
        id="req"
        name="reqField"
        label="Required Field"
        required
        options={[
          { label: "A", value: "a" },
          { label: "B", value: "b" },
        ]}
      />
    );

    const select = screen.getByLabelText(/required field/i) as HTMLSelectElement;
    expect(select).toHaveAttribute("name", "reqField");
    expect(select).toBeRequired();
    // aria-required mirrors required for assistive tech
    expect(select).toHaveAttribute("aria-required", "true");
  });

  it("renders a placeholder option if provided; placeholder can be disabled", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(
      <Select
        id="with-placeholder"
        label="With Placeholder"
        placeholder="Select one"
        placeholderDisabled
        onChange={onChange}
        options={[
          { label: "Option 1", value: "1" },
          { label: "Option 2", value: "2" },
        ]}
      />
    );

    const select = screen.getByLabelText(/with placeholder/i) as HTMLSelectElement;
    const placeholder = screen.getByRole("option", { name: /select one/i }) as HTMLOptionElement;
    expect(placeholder).toBeInTheDocument();
    expect(placeholder.disabled).toBe(true);

    // Selecting a real option should invoke onChange
    await user.selectOptions(select, "2");
    expect(onChange).toHaveBeenCalledTimes(1);
  });

  it("keeps aria-describedby stable when helper text changes dynamically", async () => {
    const { rerender } = render(
      <Select
        id="dyn"
        label="Dynamic"
        helperText="First helper"
        options={[
          { label: "A", value: "a" },
          { label: "B", value: "b" },
        ]}
      />
    );

    const select = screen.getByLabelText(/dynamic/i);
    expect(select).toHaveAttribute("aria-describedby", "dyn-helper");

    // Update helper text — id remains stable
    rerender(
      <Select
        id="dyn"
        label="Dynamic"
        helperText="Updated helper"
        options={[
          { label: "A", value: "a" },
          { label: "B", value: "b" },
        ]}
      />
    );
    expect(select).toHaveAttribute("aria-describedby", "dyn-helper");
    expect(screen.getByText(/updated helper/i)).toBeInTheDocument();
  });
});
