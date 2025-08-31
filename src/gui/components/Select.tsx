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
import { ChevronDown } from "lucide-react";

export type SelectOption = {
  label: string;
  value: string | number;
  disabled?: boolean;
};

type Size = "sm" | "md" | "lg";

export interface SelectProps
  extends Omit<React.SelectHTMLAttributes<HTMLSelectElement>, "children"> {
  /** Visual label text */
  label?: string;
  /** Hide label visually but keep it for screen readers */
  labelSrOnly?: boolean;
  /** Helper text shown under the field (non-error) */
  helperText?: string;
  /** Error text shown under the field (takes precedence over helper) */
  error?: string;
  /** Options list (alternatively, use `children` for custom <option> nodes) */
  options?: SelectOption[];
  /** Custom option nodes; ignored if `options` is provided */
  children?: React.ReactNode;
  /** Visual size */
  size?: Size;
  /** Show * indicator when required (purely visual; use native required too) */
  requiredIndicator?: boolean;
}

const sizeStyles: Record<Size, string> = {
  sm: "h-9 px-3 text-sm rounded-md",
  md: "h-10 px-3 text-sm rounded-lg",
  lg: "h-11 px-4 text-base rounded-lg",
};

export const Select = React.forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      id,
      name,
      label,
      labelSrOnly,
      helperText,
      error,
      className,
      options,
      children,
      disabled,
      size = "md",
      required,
      requiredIndicator,
      ...props
    },
    ref
  ) => {
    const autoId = React.useId();
    const selectId = id ?? `${name ?? "select"}-${autoId}`;

    const helperId = helperText && !error ? `${selectId}-helper` : undefined;
    const errorId = error ? `${selectId}-error` : undefined;

    const describedBy = [helperId, errorId].filter(Boolean).join(" ") || undefined;

    return (
      <div className="flex w-full flex-col gap-1">
        {label && (
          <label
            htmlFor={selectId}
            className={clsx(
              "text-sm font-medium text-gray-700 dark:text-gray-300",
              labelSrOnly && "sr-only"
            )}
          >
            {label}
            {required && requiredIndicator !== false && (
              <span aria-hidden className="ml-0.5 text-red-600 dark:text-red-400">
                *
              </span>
            )}
          </label>
        )}

        <div className="relative">
          <select
            id={selectId}
            name={name}
            ref={ref}
            disabled={disabled}
            required={required}
            className={clsx(
              "w-full appearance-none border bg-white text-gray-900 shadow-sm outline-none transition-colors duration-150 ease-in-out",
              "dark:bg-gray-800 dark:text-gray-100",
              "focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500",
              "disabled:cursor-not-allowed disabled:opacity-50",
              error
                ? "border-red-500 focus-visible:ring-red-500"
                : "border-gray-300 dark:border-gray-600",
              sizeStyles[size],
              className
            )}
            aria-invalid={!!error || undefined}
            aria-describedby={describedBy}
            aria-errormessage={error ? errorId : undefined}
            {...props}
          >
            {options && options.length > 0
              ? options.map((opt) => (
                  <option key={`${opt.value}`} value={opt.value} disabled={opt.disabled}>
                    {opt.label}
                  </option>
                ))
              : children}
          </select>

          {/* Chevron indicator */}
          <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center pr-1 text-gray-500 dark:text-gray-400">
            <ChevronDown className="h-4 w-4" aria-hidden />
          </span>
        </div>

        {helperText && !error && (
          <span
            id={helperId}
            className="text-xs text-gray-500 dark:text-gray-400"
          >
            {helperText}
          </span>
        )}

        {error && (
          <span id={errorId} className="text-xs text-red-600 dark:text-red-400">
            {error}
          </span>
        )}
      </div>
    );
  }
);

Select.displayName = "Select";

export default Select;
