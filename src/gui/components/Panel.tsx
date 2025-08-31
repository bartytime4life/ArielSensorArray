// =============================================================================
// 🧩 SpectraMind V50 — Panel (section container)
// -----------------------------------------------------------------------------
// Reusable, collapsible section container for dashboards, diagnostics panes,
// filters, and sidebars.
//
// Features
//   • Title, description, icon, actions, footer
//   • Optional collapsible header with animated open/close
//   • Keyboard & screen-reader friendly (ARIA, roving focus)
//   • Optional scrollable body (with maxHeight)
//   • Variants, emphasis highlight, subtle dividers, sticky header/footer
//   • Tailwind (dark-mode friendly) + small, dependency-light animations
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

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

type Variant = "default" | "bordered" | "ghost";
type Tone = "neutral" | "info" | "warning" | "danger" | "success";

export interface PanelProps extends React.HTMLAttributes<HTMLElement> {
  title?: React.ReactNode;
  description?: React.ReactNode;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  footer?: React.ReactNode;

  /** Visual container style */
  variant?: Variant;

  /** Accent tone applied to icon / border / subtle bg where applicable */
  tone?: Tone;

  /** Emphasize border/tone (e.g., to indicate selection or state) */
  highlight?: boolean;

  /** Add a thin divider between header/body and/or body/footer */
  dividers?: boolean;

  /** Sticky header/footer (useful in scrollable panels) */
  stickyHeader?: boolean;
  stickyFooter?: boolean;

  /** Collapse / expand behavior */
  collapsible?: boolean;
  /** Uncontrolled initial collapsed state */
  defaultCollapsed?: boolean;
  /** Controlled collapsed state */
  collapsed?: boolean;
  /** Callback when collapsed changes (for controlled usage) */
  onCollapsedChange?: (collapsed: boolean) => void;

  /** Called when user toggles (click/keyboard) header collapse */
  onToggle?: (nextCollapsed: boolean) => void;

  /** Apply padding inside the body (default true) */
  padded?: boolean;

  /** Make body area scrollable with a max height */
  scrollBody?: boolean;
  /** Max pixel height for scrollable body; defaults to 320 if scrollBody is true */
  bodyMaxHeight?: number;

  /** Motion animation duration (seconds) */
  motionDuration?: number;

  /** Data-testid hooks */
  "data-testid"?: string;
  headerTestId?: string;
  bodyTestId?: string;
  footerTestId?: string;

  /** Element tag; default 'section' */
  as?: keyof JSX.IntrinsicElements;
}

// -----------------------------------------------------------------------------
// Tone helpers
// -----------------------------------------------------------------------------

function toneClasses(tone: Tone, highlight: boolean) {
  // minimal, subtle emphasis; highlight increases contrast
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

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export const Panel = React.forwardRef<HTMLElement, PanelProps>((props, ref) => {
  const {
    // content
    title,
    description,
    icon,
    actions,
    footer,

    // styles
    variant = "default",
    tone = "neutral",
    highlight = false,
    dividers = false,
    stickyHeader = false,
    stickyFooter = false,

    // collapse
    collapsible = false,
    defaultCollapsed = false,
    collapsed: collapsedProp,
    onCollapsedChange,
    onToggle,

    // layout
    padded = true,
    scrollBody = false,
    bodyMaxHeight = 320,

    // motion
    motionDuration = 0.2,

    // misc
    className,
    children,
    as: As = "section",
    "data-testid": testId,
    headerTestId,
    bodyTestId,
    footerTestId,

    ...rest
  } = props;

  // ----- Collapsed state (controlled / uncontrolled)
  const [internalCollapsed, setInternalCollapsed] =
    React.useState<boolean>(defaultCollapsed);
  const isControlled = typeof collapsedProp === "boolean";
  const collapsed = isControlled ? (collapsedProp as boolean) : internalCollapsed;

  const setCollapsed = React.useCallback(
    (next: boolean) => {
      onCollapsedChange?.(next);
      onToggle?.(next);
      if (!isControlled) setInternalCollapsed(next);
    },
    [isControlled, onCollapsedChange, onToggle]
  );

  // ----- a11y: header id, body id for aria-controls
  const headerId = React.useId();
  const bodyId = React.useId();

  const toneC = toneClasses(tone, highlight);

  // ----- Container styles
  const containerClass = cn(
    "w-full rounded-2xl bg-card shadow-sm transition-colors",
    variant === "default" && "border",
    variant === "bordered" && "border",
    variant === "ghost" && "border-0 shadow-none",
    highlight && toneC.border,
    className
  );

  const headerClass = cn(
    "flex w-full items-start justify-between gap-3",
    stickyHeader && "sticky top-0 z-10 bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60",
    // spacing/padding gives touch targets and balance
    "px-4 pt-3"
  );

  const titleBlockClass = cn("min-w-0 flex items-start gap-2");
  const titleTextClass = cn("truncate text-base font-semibold");
  const descTextClass = cn("text-sm text-muted-foreground");

  const headerDivider = dividers ? "border-b" : "";
  const footerClass = cn(
    "px-4 py-3 text-sm text-muted-foreground",
    stickyFooter &&
      "sticky bottom-0 z-10 bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/60",
    dividers && "border-t"
  );

  // Body padding and optional scroll
  const bodyWrapperClass = cn(
    "px-4",
    padded ? "pb-3 pt-2" : "py-2",
    scrollBody && "overflow-auto rounded-xl"
  );

  // Keyboard toggle on header: Enter/Space
  const onHeaderKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!collapsible) return;
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setCollapsed(!collapsed);
    }
  };

  // Prevent header click toggling when clicking interactive elements
  const onHeaderClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!collapsible) return;
    const target = e.target as HTMLElement;
    if (target.closest("button, a, input, label, select, [role='menu'], [role='menuitem']")) return;
    setCollapsed(!collapsed);
  };

  return (
    <As
      {...rest}
      ref={ref}
      className={containerClass}
      aria-labelledby={headerId}
      data-testid={testId ?? "Panel"}
      data-collapsible={collapsible ? "" : undefined}
      data-collapsed={collapsible ? String(collapsed) : undefined}
    >
      {(title || description || icon || actions || collapsible) && (
        <motion.div
          id={headerId}
          className={cn(headerClass, headerDivider)}
          onClick={onHeaderClick}
          onKeyDown={onHeaderKeyDown}
          role={collapsible ? "button" : undefined}
          aria-expanded={collapsible ? !collapsed : undefined}
          aria-controls={collapsible ? bodyId : undefined}
          tabIndex={collapsible ? 0 : -1}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: motionDuration }}
          data-testid={headerTestId ?? "PanelHeader"}
        >
          <div className={titleBlockClass}>
            {icon && <div className={cn("mt-0.5", toneC.icon)}>{icon}</div>}
            <div className="min-w-0">
              {title && <h3 className={titleTextClass}>{title}</h3>}
              {description && <p className={descTextClass}>{description}</p>}
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            {actions}
            {collapsible && (
              <button
                type="button"
                aria-label={collapsed ? "Expand panel" : "Collapse panel"}
                aria-expanded={!collapsed}
                aria-controls={bodyId}
                onClick={(e) => {
                  e.stopPropagation();
                  setCollapsed(!collapsed);
                }}
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
            )}
          </div>
        </motion.div>
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
            className={cn("overflow-hidden", scrollBody && "pr-1")}
            style={scrollBody ? { maxHeight: bodyMaxHeight } : undefined}
            data-testid={bodyTestId ?? "PanelBody"}
          >
            <div className={bodyWrapperClass}>{children}</div>
          </motion.div>
        )}
      </AnimatePresence>

      {footer && (
        <div className={footerClass} data-testid={footerTestId ?? "PanelFooter"}>
          {footer}
        </div>
      )}
    </As>
  );
});

Panel.displayName = "Panel";

export default Panel;
