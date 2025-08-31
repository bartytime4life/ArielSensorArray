```tsx
// src/gui/components/card.tsx
// =============================================================================
// 🧩 SpectraMind V50 — Card Component
// -----------------------------------------------------------------------------
// • A reusable card container with header, content, and footer sections.
// • Used throughout the GUI for diagnostics panels, report previews, etc.
// • Declarative + composable: React + Tailwind + shadcn/ui compatible.
// • Supports optional title, description, icon, actions, and custom children.
// =============================================================================

import * as React from "react";
import { cn } from "@/lib/utils"; // utility for merging class names
import { motion } from "framer-motion";
import { Card as ShadcnCard, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from "@/components/ui/card";

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  footer?: React.ReactNode;
  actions?: React.ReactNode; // e.g. buttons, menus
  highlight?: boolean;       // optional visual emphasis
}

/**
 * 🔖 Card
 *
 * A flexible, reusable card for SpectraMind V50 dashboards and reports.
 * Follows declarative UI principles (React + Tailwind).
 *
 * Usage:
 *   <Card
 *     title="Diagnostics Summary"
 *     description="GLL, SHAP, symbolic overlays"
 *     footer={<Button>Open</Button>}
 *     highlight
 *   >
 *     <DiagnosticsChart />
 *   </Card>
 */
export const Card: React.FC<CardProps> = ({
  title,
  description,
  icon,
  footer,
  actions,
  highlight = false,
  className,
  children,
  ...props
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <ShadcnCard
        className={cn(
          "rounded-2xl border bg-white shadow-sm transition hover:shadow-md dark:bg-neutral-900",
          highlight && "border-blue-500 shadow-md",
          className
        )}
        {...props}
      >
        {(title || description || icon || actions) && (
          <CardHeader className="flex items-start justify-between space-y-0 pb-2">
            <div className="flex items-center gap-2">
              {icon && <div className="text-blue-600 dark:text-blue-400">{icon}</div>}
              <div>
                {title && <CardTitle className="text-lg font-semibold">{title}</CardTitle>}
                {description && <CardDescription>{description}</CardDescription>}
              </div>
            </div>
            {actions && <div className="ml-auto">{actions}</div>}
          </CardHeader>
        )}

        {children && <CardContent className="pt-0">{children}</CardContent>}

        {footer && (
          <CardFooter className="border-t pt-2 mt-2">
            {footer}
          </CardFooter>
        )}
      </ShadcnCard>
    </motion.div>
  );
};
```

### ✅ Features

* **Composable sections**: Header (title, description, icon, actions), Content, Footer.
* **Declarative UI**: `highlight` prop for emphasis (e.g., active diagnostics).
* **Animations**: Subtle Framer Motion fade/slide for modern UX.
* **Dark mode support**: Uses Tailwind dark variants.
* **Integration**: Compatible with `shadcn/ui` primitives for consistent design system.

---
