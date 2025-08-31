// src/gui/components/card.tsx
// =============================================================================
/* 🧩 SpectraMind V50 — Card Component
-------------------------------------------------------------------------------
• Reusable card container with header, content, and footer sections.
• Declarative + composable: React + Tailwind + shadcn/ui compatible.
• Optional title, description, icon, actions, footer.
• Variants, tone-based highlight, sticky header/footer, subtle dividers.
• Motion on mount (configurable / disable-able).
*/
// =============================================================================

import * as React from "react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import {
  Card as ShadcnCard,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";

// ----------------------------------------------------------------------------
// Types
// ----------------------------------------------------------------------------
type Variant = "default" | "bordered" | "ghost";
type Tone = "neutral" | "info" | "warning" | "danger" | "success";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  footer?: React.ReactNode;
  actions?: React.ReactNode;

  /** Visual style variant */
  variant?: Variant;

  /** Accent tone for highlight border/icon */
  tone?: Tone;

  /** Optional visual emphasis (accented border/shadow) */
  highlight?: boolean;

  /** Add thin divider lines on header/footer */
  headerDivider?: boolean;
  footerDivider?: boolean;

  /** Sticky header/footer (useful in scrollable parents) */
  stickyHeader?: boolean;
  stickyFooter?: boolean;

  /** Framer-motion mount animation control */
  motionEnabled?: boolean;
  motionDuration?: number;

  /** Data test id hook */
  "data-testid"?: string;
  headerTestId?: string;
  footerTestId?: string;
}

// ----------------------------------------------------------------------------
// Tone helpers
// ----------------------------------------------------------------------------
function toneClasses(tone: Tone, highlight: boolean) {
  switch (tone) {
    case "info":
      return {
        icon: "text-blue-600 dark:text-blue-400",
        border: highlight ? "border-blue-500" : "border-blue-500/40",
      };
    case "warning":
      return {
        icon: "text-amber-600 dark:text-amber-400",
        border: highlight ? "border-amber-500" : "border-amber-500/40",
      };
    case "danger":
      return {
        icon: "text-red-600 dark:text-red-400",
        border: highlight ? "border-red-500" : "border-red-500/40",
      };
    case "success":
      return {
        icon: "text-emerald-600 dark:text-emerald-400",
        border: highlight ? "border-emerald-500" : "border-emerald-500/40",
      };
    default:
      return {
        icon: "text-blue-600 dark:text-blue-400",
        border: highlight ? "border-foreground/50" : "border-border",
      };
  }
}

// ----------------------------------------------------------------------------
// Component
// ----------------------------------------------------------------------------
export const Card: React.FC<CardProps> = ({
  title,
  description,
  icon,
  footer,
  actions,

  variant = "default",
  tone = "neutral",
  highlight = false,

  headerDivider = false,
  footerDivider = false,

  stickyHeader = false,
  stickyFooter = false,

  motionEnabled = true,
  motionDuration = 0.25,

  className,
  children,

  "data-testid": testId,
  headerTestId,
  footerTestId,

  ...props
}) => {
  const toneC = toneClasses(tone, highlight);
  const headerId = React.useId();

  const containerClass = cn(
    "rounded-2xl bg-white shadow-sm transition dark:bg-neutral-900",
    variant !== "ghost" && "border",
    variant === "ghost" && "border-0 shadow-none",
    highlight && toneC.border,
    highlight && "shadow-md",
    className
  );

  const headerClass = cn(
    "flex items-start justify-between space-y-0 pb-2",
    stickyHeader &&
      "sticky top-0 z-10 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:bg-neutral-900/95",
    headerDivider && "border-b",
    // Inherit border token from theme
    headerDivider && "border-gray-200 dark:border-gray-800"
  );

  const footerClass = cn(
    stickyFooter &&
      "sticky bottom-0 z-10 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/60 dark:bg-neutral-900/95",
    footerDivider && "border-t",
    footerDivider && "border-gray-200 dark:border-gray-800",
    "pt-2 mt-2"
  );

  const HeaderBlock = (
    (title || description || icon || actions) && (
      <CardHeader
        id={headerId}
        className={headerClass}
        data-testid={headerTestId ?? "CardHeader"}
      >
        <div className="flex min-w-0 items-start gap-2">
          {icon && <div className={cn("mt-0.5", toneC.icon)}>{icon}</div>}
          <div className="min-w-0">
            {title && (
              <CardTitle className="truncate text-lg font-semibold">
                {title}
              </CardTitle>
            )}
            {description && (
              <CardDescription className="text-sm">{description}</CardDescription>
            )}
          </div>
        </div>
        {actions && <div className="ml-2 shrink-0">{actions}</div>}
      </CardHeader>
    )
  );

  const ContentBlock = children && (
    <CardContent className={cn((title || description || icon || actions) && "pt-0")}>
      {children}
    </CardContent>
  );

  const FooterBlock = footer && (
    <CardFooter className={footerClass} data-testid={footerTestId ?? "CardFooter"}>
      {footer}
    </CardFooter>
  );

  const Shell = (
    <ShadcnCard
      className={containerClass}
      aria-labelledby={title ? headerId : undefined}
      data-testid={testId ?? "Card"}
      {...props}
    >
      {HeaderBlock}
      {ContentBlock}
      {FooterBlock}
    </ShadcnCard>
  );

  if (!motionEnabled) return Shell;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: motionDuration }}
    >
      {Shell}
    </motion.div>
  );
};

export default Card;
