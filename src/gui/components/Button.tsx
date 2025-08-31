// src/gui/components/Button.tsx
// =============================================================================
// 🎛️ SpectraMind V50 — Reusable Button Component
// -----------------------------------------------------------------------------
// Design goals:
//   • Consistent with shadcn/ui + Tailwind styling used in other components.
//   • Variants for primary, secondary, destructive, ghost, and link buttons.
//   • Size support (sm, md, lg).
//   • Accessible: proper aria roles, focus rings, and keyboard navigation.
//   • Flexible: forwards refs, accepts all standard <button> props.
//   • Styled with rounded corners, soft shadows, hover/active states.
// =============================================================================

import * as React from "react";
import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// -----------------------------------------------------------------------------
// Class Variance Authority (CVA) for Tailwind variants
// -----------------------------------------------------------------------------
const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-2xl text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 shadow-sm",
  {
    variants: {
      variant: {
        default:
          "bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-600",
        secondary:
          "bg-gray-100 text-gray-900 hover:bg-gray-200 focus-visible:ring-gray-400",
        destructive:
          "bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-600",
        outline:
          "border border-gray-300 bg-transparent hover:bg-gray-50 focus-visible:ring-gray-400",
        ghost:
          "hover:bg-gray-100 text-gray-900 focus-visible:ring-gray-300",
        link: "text-blue-600 underline-offset-4 hover:underline focus-visible:ring-blue-600",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-10 px-4 text-sm",
        lg: "h-12 px-6 text-base",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

// -----------------------------------------------------------------------------
// Component Definition
// -----------------------------------------------------------------------------
export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean; // allows using <Link> or other element via Slot
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);

Button.displayName = "Button";

export { Button, buttonVariants };
