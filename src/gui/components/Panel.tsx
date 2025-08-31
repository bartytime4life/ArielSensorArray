// src/gui/components/Panel.tsx
// =============================================================================
// 🧩 SpectraMind V50 — Panel (section container)
// -----------------------------------------------------------------------------
// A reusable, collapsible section container for the GUI. Designed for dashboards,
// diagnostics panes, filters, and sidebars. Provides:
//   • Title, description, icon, action slot, footer
//   • Optional collapsible header with animated open/close
//   • Optional scrollable body (with maxHeight)
//   • Variants and highlight state
//   • Dark-mode friendly Tailwind tokens
//
// Usage:
//
// <Panel
//   title="Symbolic Diagnostics"
//   description="Top violations and rule overlays"
//   icon={<Beaker className="h-4 w-4" />}
//   actions={<Button size="sm">Refresh</Button>}
//   collapsible
//   defaultCollapsed={false}
//   scrollBody
//   bodyMaxHeight={400}
// >
//   <DiagnosticsTable />
// </Panel>
//
// =============================================================================

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";

export interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  footer?: React.ReactNode;

  /** Visual variant for container styling */
  variant?: "default" | "bordered" | "ghost";

  /** Emphasize border/tone (e.g., to indicate selection or warnings) */
  highlight?: boolean;

  /** Collapse / expand behavior */
  collapsible?: boolean;
  /** Uncontrolled initial collapsed state */
  defaultCollapsed?: boolean;
  /** Controlled collapsed state */
  collapsed?: boolean;
  /** Callback when collapsed changes (for controlled usage) */
  onCollapsedChange?: (collapsed: boolean) => void;

  /** Apply padding inside the body (default true) */
  padded?: boolean;

  /** Make body area scrollable with a max height */
  scrollBody?: boolean;
  /** Max pixel height for scrollable body; defaults to 320 if scrollBody is true */
  bodyMaxHeight?: number;

  /** Motion animation duration (seconds) */
  motionDuration?: number;
}

export const Panel: React.FC<PanelProps> = ({
  title,
  description,
  icon,
  actions,
  footer,
  variant = "default",
  highlight = false,

  collapsible = false,
  defaultCollapsed = false,
  collapsed: collapsedProp,
  onCollapsedChange,

  padded = true,
  scrollBody = false,
  bodyMaxHeight = 320,
  motionDuration = 0.2,

  className,
  children,
  ...rest
}) => {
  // ----- Collapsed state (controlled / uncontrolled)
  const [internalCollapsed, setInternalCollapsed] = React.useState<boolean>(defaultCollapsed);
  const isControlled = typeof collapsedProp === "boolean";
  const collapsed = isControlled ? (collapsedProp as boolean) : internalCollapsed;

  const setCollapsed = React.useCallback(
    (next: boolean) => {
      if (onCollapsedChange) onCollapsedChange(next);
      if (!isControlled) setInternalCollapsed(next);
    },
    [isControlled, onCollapsedChange]
  );

  // ----- a11y: header id, body id for aria-controls
  const headerId = React.useId();
  const bodyId = React.useId();

  // ----- Container styles
  const containerClass = cn(
    "w-full rounded-2xl bg-card shadow-sm transition-colors",
    variant === "default" && "border",
    variant === "bordered" && "border",
    variant === "ghost" && "border-0 shadow-none",
    highlight && "border-blue-500",
    className
  );

  const headerClass = cn(
    "flex w-full items-start justify-between gap-3",
    // spacing/padding gives touch targets and balance
    "px-4 pt-3"
  );

  const titleBlockClass = cn("min-w-0 flex items-start gap-2");
  const titleTextClass = cn("truncate text-base font-semibold");
  const descTextClass = cn("text-sm text-muted-foreground");

  // Toggle chevron button
  const ChevronBtn: React.FC<{ onClick: () => void }> = ({ onClick }) => (
    <button
      type="button"
      aria-label={collapsed ? "Expand panel" : "Collapse panel"}
      aria-expanded={!collapsed}
      aria-controls={bodyId}
      onClick={onClick}
      className={cn(
        "inline-flex h-7 w-7 items-center justify-center rounded-md transition",
        "hover:bg-muted focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      )}
    >
      <ChevronDown
        className={cn(
          "h-4 w-4 transition-transform duration-200",
          collapsed ? "-rotate-90" : "rotate-0"
        )}
      />
    </button>
  );

  // Body padding and optional scroll
  const bodyWrapperClass = cn(
    "px-4",
    padded ? "pb-3 pt-2" : "py-2",
    scrollBody && "overflow-auto rounded-xl",
  );

  // Footer style
  const footerClass = cn("border-t px-4 py-3 text-sm text-muted-foreground");

  // Handle header click when collapsible
  const onHeaderClick = (e: React.MouseEvent<HTMLDivElement, MouseEvent>) => {
    if (!collapsible) return;
    // don't toggle when clicking an interactive element within actions
    const target = e.target as HTMLElement;
    if (target.closest("button, a, input, label, select")) return;
    setCollapsed(!collapsed);
  };

  return (
    <motion.section
      {...rest}
      className={containerClass}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: motionDuration }}
      aria-labelledby={headerId}
      data-collapsible={collapsible ? "" : undefined}
      data-collapsed={collapsible ? String(collapsed) : undefined}
    >
      {(title || description || icon || actions || collapsible) && (
        <div id={headerId} className={headerClass} onClick={onHeaderClick}>
          <div className={titleBlockClass}>
            {icon && <div className="mt-0.5 text-blue-600 dark:text-blue-400">{icon}</div>}
            <div className="min-w-0">
              {title && <h3 className={titleTextClass}>{title}</h3>}
              {description && <p className={descTextClass}>{description}</p>}
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            {actions}
            {collapsible && <ChevronBtn onClick={() => setCollapsed(!collapsed)} />}
          </div>
        </div>
      )}

      <AnimatePresence initial={false}>
        {(!collapsible || !collapsed) && (
          <motion.div
            key="panel-body"
            id={bodyId}
            role="region"
            aria-labelledby={headerId}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: motionDuration, ease: [0.2, 0.8, 0.2, 1] }}
            className={cn(
              "overflow-hidden",
              scrollBody && "pr-1" // subtle space for scrollbar
            )}
            style={scrollBody ? { maxHeight: bodyMaxHeight } : undefined}
          >
            <div className={bodyWrapperClass}>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>

      {footer && <div className={footerClass}>{footer}</div>}
    </motion.section>
  );
};

export default Panel;
