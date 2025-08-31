// src/gui/components/Select.tsx
// ============================================================================
// 🔽 Select Component — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// Responsibilities
//   • Styled <select> with label, helper text, and error state
//   • Accessible (labels, aria-invalid/aria-describedby, keyboard focus styles)
//   • Tailwind + shadcn/ui-like conventions for consistency with other components
//   • Works with controlled/uncontrolled usage and forwards refs
//
// Example:
//   <Select
//     id="planet"
//     label="Planet"
//     value={planet}
//     onChange={(e) => setPlanet(e.target.value)}
//     options={[
//       { label: "Kepler-22b", value: "kepler-22b" },
//       { label: "WASP-12b", value: "wasp-12b" },
//     ]}
//     helperText="Pick a target to inspect diagnostics"
//   />
// ============================================================================

import * as React from "react";
import clsx from "clsx";

export type SelectOption = {
  label: string;
  value: string | number;
  disabled?: boolean;
};

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "children"> {
  label?: string;
  helperText?: string;
  error?: string;
  options?: SelectOption[];
  /** Optional: pass custom option nodes; ignored if `options` provided */
  children?: React.ReactNode;
}

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      id,
      label,
      helperText,
      error,
      className,
      options,
      children,
      disabled,
      ...props
    },
    ref
  ) => {
    const describedBy =
      helperText && !error ? `${id ?? props.name}-helper` : undefined;

    return (
      <div className="w-full flex flex-col gap-1">
        {label && (
          <label
            htmlFor={id}
            className="text-sm font-medium text-gray-700 dark:text-gray-300"
          >
            {label}
          </label>
        )}

        <select
          id={id}
          ref={ref}
          disabled={disabled}
          className={clsx(
            "px-3 py-2 rounded-lg border shadow-sm outline-none bg-white dark:bg-gray-800",
            "text-gray-900 dark:text-gray-100",
            "transition-colors duration-150 ease-in-out",
            "focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500",
            "disabled:cursor-not-allowed disabled:opacity-50",
            error
              ? "border-red-500 focus-visible:ring-red-500"
              : "border-gray-300 dark:border-gray-600",
            className
          )}
          aria-invalid={!!error}
          aria-describedby={describedBy}
          {...props}
        >
          {options && options.length > 0
            ? options.map((opt) => (
                <option
                  key={`${opt.value}`}
                  value={opt.value}
                  disabled={opt.disabled}
                >
                  {opt.label}
                </option>
              ))
            : children}
        </select>

        {helperText && !error && (
          <span
            id={describedBy}
            className="text-xs text-gray-500 dark:text-gray-400"
          >
            {helperText}
          </span>
        )}

        {error && (
          <span className="text-xs text-red-600 dark:text-red-400">
            {error}
          </span>
        )}
      </div>
    );
  }
);

Select.displayName = "Select";

export default Select;
