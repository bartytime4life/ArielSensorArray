// =============================================================================
// 🎛️ SpectraMind V50 — Reusable Button Component
// -----------------------------------------------------------------------------
// Design goals:
//   • Consistent with shadcn/ui + Tailwind styling used in other components.
//   • Variants: default, secondary, destructive, outline, ghost, link.
//   • Sizes: sm, md, lg; optional fullWidth.
//   • Accessible: correct roles, focus rings, keyboard navigation.
//   • Flexible: forwards refs, accepts all <button> props, Slot-asChild.
//   • Quality of life: loading state, left/right icons, icon-only a11y label.
// =============================================================================

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// -----------------------------------------------------------------------------
// Class Variance Authority (CVA) for Tailwind variants
// -----------------------------------------------------------------------------
const buttonVariants = cva(
  [
    "inline-flex items-center justify-center whitespace-nowrap select-none",
    "rounded-2xl text-sm font-medium transition-colors",
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2",
    "disabled:pointer-events-none disabled:opacity-50",
    "shadow-sm",
  ].join(" "),
  {
    variants: {
      variant: {
        default:
          "bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-600",
        secondary:
          "bg-gray-100 text-gray-900 hover:bg-gray-200 focus-visible:ring-gray-400 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700",
        destructive:
          "bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-600",
        outline:
          "border border-gray-300 bg-transparent hover:bg-gray-50 focus-visible:ring-gray-400 dark:border-gray-600 dark:hover:bg-gray-800",
        ghost:
          "text-gray-900 hover:bg-gray-100 focus-visible:ring-gray-300 dark:text-gray-100 dark:hover:bg-gray-800",
        link: "text-blue-600 underline-offset-4 hover:underline focus-visible:ring-blue-600",
      },
      size: {
        sm: "h-8 px-3 text-xs gap-1.5",
        md: "h-10 px-4 text-sm gap-2",
        lg: "h-12 px-6 text-base gap-2.5",
      },
      fullWidth: {
        true: "w-full",
        false: "",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
      fullWidth: false,
    },
  }
);

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------
export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  /** Render as child (e.g., Next.js Link) using Radix Slot */
  asChild?: boolean;
  /** Display a loading spinner and disable interactions */
  loading?: boolean;
  /** Optional left/right adornments (icons, etc.) */
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  /** For icon-only buttons, provide an accessible label */
  srLabel?: string;
}

// -----------------------------------------------------------------------------
// Spinner (inline, no external deps)
// -----------------------------------------------------------------------------
const Spinner: React.FC<{ className?: string; size?: "sm" | "md" | "lg" }> = ({
  className,
  size = "md",
}) => {
  const px = size === "sm" ? 14 : size === "lg" ? 18 : 16;
  const stroke = Math.max(2, Math.round(px / 8));
  return (
    <svg
      className={cn("animate-spin", className)}
      width={px}
      height={px}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        className="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        strokeWidth={stroke}
        fill="none"
      />
      <path
        className="opacity-90"
        fill="currentColor"
        d="M12 2a10 10 0 0 1 10 10h-4a6 6 0 1 0-6-6V2z"
      />
    </svg>
  );
};

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      fullWidth,
      asChild = false,
      loading = false,
      disabled,
      leftIcon,
      rightIcon,
      srLabel,
      children,
      type = "button",
      ...props
    },
    ref
  ) => {
    const Comp: any = asChild ? Slot : "button";

    // Icon-only a11y: ensure there is a name for AT
    const ariaLabel =
      (!children || String(children).trim().length === 0) && srLabel
        ? { "aria-label": srLabel }
        : undefined;

    const computedDisabled = disabled || loading;

    // choose spinner size relative to button size
    const spinnerSize = size === "sm" ? "sm" : size === "lg" ? "lg" : "md";

    return (
      <Comp
        ref={ref}
        type={asChild ? undefined : type}
        className={cn(buttonVariants({ variant, size, fullWidth, className }))}
        aria-busy={loading || undefined}
        aria-disabled={computedDisabled || undefined}
        disabled={!asChild ? computedDisabled : undefined}
        {...ariaLabel}
        {...props}
      >
        {/* Left icon / spinner */}
        {loading ? (
          <Spinner
            size={spinnerSize as any}
            className={cn(
              variant === "link" ? "text-current" : "text-current",
              children && "mr-2"
            )}
          />
        ) : (
          leftIcon && <span className={cn(children && "mr-2")}>{leftIcon}</span>
        )}

        {/* Content (visually hidden when icon-only) */}
        {children}

        {/* Right icon */}
        {!loading && rightIcon && (
          <span className={cn(children && "ml-2")}>{rightIcon}</span>
        )}
      </Comp>
    );
  }
);

Button.displayName = "Button";

export { Button, buttonVariants };
export default Button;
