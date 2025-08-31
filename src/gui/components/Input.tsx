// ============================================================================
// 📝 Input Component — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// Responsibilities
//   • Styled input field (text/password/email/number, etc.)
//   • Supports labels, helper text, error states, required indicator
//   • Tailwind + shadcn/ui-like conventions for consistency
//   • Accessible: ARIA attributes, focus-visible styles
//   • Extendable: accepts all native input props; forwards refs
//
// Usage:
//   <Input
//     id="username"
//     label="Username"
//     placeholder="Enter your username"
//     value={username}
//     onChange={(e) => setUsername(e.target.value)}
//     helperText="Must be 4–20 characters"
//     required
//   />
// ============================================================================

import * as React from "react";
import clsx from "clsx";

type Size = "sm" | "md" | "lg";

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  /** Visual label text */
  label?: string;
  /** Hide label visually but keep for screen-readers */
  labelSrOnly?: boolean;
  /** Helper text (non-error) shown under the field */
  helperText?: string;
  /** Error text shown under the field (takes precedence over helper) */
  error?: string;
  /** Visual size */
  size?: Size;
  /** Show * next to label when required (visual only; use native required too) */
  requiredIndicator?: boolean;
  /** Left adornment (icon or text) */
  leftSlot?: React.ReactNode;
  /** Right adornment (icon or text) */
  rightSlot?: React.ReactNode;
}

const sizeMap: Record<Size, { padding: string; text: string; radius: string; slotGap: string; height?: string }> = {
  sm: { padding: "px-3 py-2", text: "text-sm", radius: "rounded-md", slotGap: "gap-1.5", height: "h-9" },
  md: { padding: "px-3 py-2", text: "text-sm", radius: "rounded-lg", slotGap: "gap-2", height: "h-10" },
  lg: { padding: "px-4 py-3", text: "text-base", radius: "rounded-lg", slotGap: "gap-2.5", height: "h-11" },
};

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  (
    {
      id,
      name,
      label,
      labelSrOnly,
      helperText,
      error,
      size = "md",
      className,
      type = "text",
      required,
      requiredIndicator,
      leftSlot,
      rightSlot,
      ...props
    },
    ref
  ) => {
    const autoId = React.useId();
    const inputId = id ?? `${name ?? "input"}-${autoId}`;
    const helperId = helperText && !error ? `${inputId}-helper` : undefined;
    const errorId = error ? `${inputId}-error` : undefined;
    const describedBy = [helperId, errorId].filter(Boolean).join(" ") || undefined;

    const sz = sizeMap[size];

    return (
      <div className="w-full flex flex-col gap-1">
        {label && (
          <label
            htmlFor={inputId}
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

        <div
          className={clsx(
            "relative flex items-center",
            sz.slotGap
          )}
        >
          {leftSlot && (
            <div
              className={clsx(
                "pointer-events-none absolute left-2 flex items-center text-gray-500 dark:text-gray-400",
                size === "lg" ? "left-3" : "left-2"
              )}
              aria-hidden="true"
            >
              {leftSlot}
            </div>
          )}

          <input
            id={inputId}
            ref={ref}
            type={type}
            name={name}
            required={required}
            className={clsx(
              "w-full border bg-white text-gray-900 shadow-sm outline-none transition-colors duration-150 ease-in-out",
              "dark:bg-gray-800 dark:text-gray-100",
              "focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500",
              "disabled:cursor-not-allowed disabled:opacity-50",
              error
                ? "border-red-500 focus-visible:ring-red-500"
                : "border-gray-300 dark:border-gray-600",
              sz.padding,
              sz.text,
              sz.radius,
              sz.height,
              // add left padding when leftSlot provided
              leftSlot && (size === "lg" ? "pl-10" : "pl-9"),
              // add right padding when rightSlot provided
              rightSlot && (size === "lg" ? "pr-10" : "pr-9"),
              className
            )}
            aria-invalid={!!error || undefined}
            aria-describedby={describedBy}
            aria-errormessage={error ? errorId : undefined}
            {...props}
          />

          {rightSlot && (
            <div
              className={clsx(
                "absolute right-2 flex items-center text-gray-500 dark:text-gray-400",
                size === "lg" ? "right-3" : "right-2"
              )}
            >
              {rightSlot}
            </div>
          )}
        </div>

        {helperText && !error && (
          <span id={helperId} className="text-xs text-gray-500 dark:text-gray-400">
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

Input.displayName = "Input";

export default Input;
