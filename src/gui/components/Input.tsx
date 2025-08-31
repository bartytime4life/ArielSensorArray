// src/gui/components/Input.tsx
// ============================================================================
// 📝 Input Component — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// Responsibilities
//   • Styled input field (text/password/email/number, etc.)
//   • Supports labels, helper text, error states.
//   • Uses TailwindCSS + shadcn/ui conventions for consistency.
//   • Accessible: ARIA attributes, focus-visible styles.
//   • Extendable: can accept all native input props.
//
// Usage Example:
//   <Input
//     label="Username"
//     placeholder="Enter your username"
//     value={username}
//     onChange={(e) => setUsername(e.target.value)}
//     helperText="Must be 4–20 characters"
//   />
// ============================================================================

import * as React from "react";
import clsx from "clsx";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  helperText?: string;
  error?: string;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ label, helperText, error, className, type = "text", ...props }, ref) => {
    return (
      <div className="w-full flex flex-col gap-1">
        {label && (
          <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
            {label}
          </label>
        )}
        <input
          ref={ref}
          type={type}
          className={clsx(
            "px-3 py-2 rounded-lg border shadow-sm outline-none",
            "transition-colors duration-150 ease-in-out",
            "focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:border-blue-500",
            "disabled:cursor-not-allowed disabled:opacity-50",
            error
              ? "border-red-500 focus-visible:ring-red-500"
              : "border-gray-300 dark:border-gray-600",
            "bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100",
            className
          )}
          aria-invalid={!!error}
          aria-describedby={helperText ? `${props.id}-helper` : undefined}
          {...props}
        />
        {helperText && !error && (
          <span
            id={`${props.id}-helper`}
            className="text-xs text-gray-500 dark:text-gray-400"
          >
            {helperText}
          </span>
        )}
        {error && (
          <span className="text-xs text-red-600 dark:text-red-400">{error}</span>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

export default Input;
