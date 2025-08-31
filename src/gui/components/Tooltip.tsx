// src/gui/components/Tooltip.tsx
// ============================================================================
// 💬 Tooltip — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// Features
//   • Accessible tooltip with hover/focus/keyboard (ESC to close)
//   • Controlled or uncontrolled (`open`, `defaultOpen`, `onOpenChange`)
//   • Smart positioning with viewport collision handling
//   • Portal to <body> by default; optional inline rendering
//   • Arrow, offset, and placement: top | bottom | left | right
//   • Dark-mode aware Tailwind styling
//
// Usage:
//   <Tooltip content="Open diagnostics" placement="bottom">
//     <button aria-label="Open">?</button>
//   </Tooltip>
//
//   // Controlled
//   <Tooltip content="Details" open={isOpen} onOpenChange={setOpen}>
//     <span>Hover me</span>
//   </Tooltip>
// ============================================================================

import * as React from "react";
import { createPortal } from "react-dom";
import clsx from "clsx";

type Placement = "top" | "bottom" | "left" | "right";

export interface TooltipProps {
  /** The trigger element to wrap (must be a single ReactElement) */
  children: React.ReactElement;
  /** Tooltip content (string or nodes) */
  content: React.ReactNode;
  /** Preferred placement relative to trigger */
  placement?: Placement;
  /** Pixel offset between trigger and tooltip panel */
  offset?: number;
  /** Controlled open state */
  open?: boolean;
  /** Uncontrolled initial state */
  defaultOpen?: boolean;
  /** Change handler for controlled/uncontrolled usage */
  onOpenChange?: (open: boolean) => void;
  /** Show delay in ms (for hover/focus) */
  delay?: number;
  /** If true, hovering the panel does NOT keep it open */
  disableHoverableContent?: boolean;
  /** Render via portal to document.body (default: true) */
  portal?: boolean;
  /** Show small arrow pointing to trigger (default: true) */
  arrow?: boolean;
  /** Optional id; used as tooltip id and applied to aria-describedby */
  id?: string;
  /** Extra classes for the tooltip panel */
  className?: string;
}

const DEFAULT_OFFSET = 8;
const VIEWPORT_PADDING = 8;

function useIsomorphicLayoutEffect(effect: React.EffectCallback, deps: React.DependencyList) {
  const useLayout = typeof window !== "undefined" ? React.useLayoutEffect : React.useEffect;
  useLayout(effect, deps);
}

function mergeHandlers<T extends React.SyntheticEvent>(
  a?: (e: T) => void,
  b?: (e: T) => void
) {
  return (e: T) => {
    a?.(e);
    b?.(e);
  };
}

export const Tooltip: React.FC<TooltipProps> = ({
  children,
  content,
  placement = "top",
  offset = DEFAULT_OFFSET,
  open: controlledOpen,
  defaultOpen = false,
  onOpenChange,
  delay = 80,
  disableHoverableContent = false,
  portal = true,
  arrow = true,
  id,
  className,
}) => {
  const isControlled = controlledOpen !== undefined;
  const [uncontrolledOpen, setUncontrolledOpen] = React.useState(defaultOpen);
  const open = isControlled ? !!controlledOpen : uncontrolledOpen;

  const triggerRef = React.useRef<HTMLElement | null>(null);
  const panelRef = React.useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = React.useState(false);

  // Positioning state
  const [coords, setCoords] = React.useState<{ top: number; left: number; actualPlacement: Placement }>({
    top: 0,
    left: 0,
    actualPlacement: placement,
  });

  // Track show/hide timers for delay
  const showTimer = React.useRef<number | null>(null);
  const hideTimer = React.useRef<number | null>(null);

  const tooltipId = React.useId();
  const computedId = id ?? `tooltip-${tooltipId}`;

  const setOpen = (next: boolean) => {
    if (!isControlled) setUncontrolledOpen(next);
    onOpenChange?.(next);
  };

  // Mount flag (for portal safety)
  React.useEffect(() => setMounted(true), []);

  // Clear timers on unmount
  React.useEffect(() => {
    return () => {
      if (showTimer.current) window.clearTimeout(showTimer.current);
      if (hideTimer.current) window.clearTimeout(hideTimer.current);
    };
  }, []);

  // Handlers
  const scheduleOpen = React.useCallback(() => {
    if (hideTimer.current) {
      window.clearTimeout(hideTimer.current);
      hideTimer.current = null;
    }
    if (open) return;
    showTimer.current = window.setTimeout(() => setOpen(true), delay);
  }, [delay, open]);

  const scheduleClose = React.useCallback(() => {
    if (showTimer.current) {
      window.clearTimeout(showTimer.current);
      showTimer.current = null;
    }
    if (!open) return;
    hideTimer.current = window.setTimeout(() => setOpen(false), 60);
  }, [open]);

  // If hovering content should keep the tooltip open
  const onPanelMouseEnter = React.useCallback(() => {
    if (!disableHoverableContent) {
      if (hideTimer.current) {
        window.clearTimeout(hideTimer.current);
        hideTimer.current = null;
      }
    }
  }, [disableHoverableContent]);

  const onPanelMouseLeave = React.useCallback(() => {
    if (!disableHoverableContent) scheduleClose();
  }, [disableHoverableContent, scheduleClose]);

  // Keyboard: Escape closes when tooltip is open
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("keydown", onKey, { capture: true });
    return () => document.removeEventListener("keydown", onKey, { capture: true } as any);
  }, [open]);

  // Compute position each time it opens / on scroll / resize
  const computeAndSetPosition = React.useCallback(() => {
    const triggerEl = triggerRef.current;
    const panelEl = panelRef.current;
    if (!triggerEl || !panelEl) return;

    const tr = triggerEl.getBoundingClientRect();
    const pr = panelEl.getBoundingClientRect();
    const scrollX = window.scrollX || window.pageXOffset;
    const scrollY = window.scrollY || window.pageYOffset;

    let top = 0;
    let left = 0;
    let actualPlacement: Placement = placement;

    const tryPlacement = (pl: Placement) => {
      switch (pl) {
        case "top":
          top = tr.top + scrollY - pr.height - offset;
          left = tr.left + scrollX + tr.width / 2 - pr.width / 2;
          break;
        case "bottom":
          top = tr.bottom + scrollY + offset;
          left = tr.left + scrollX + tr.width / 2 - pr.width / 2;
          break;
        case "left":
          top = tr.top + scrollY + tr.height / 2 - pr.height / 2;
          left = tr.left + scrollX - pr.width - offset;
          break;
        case "right":
          top = tr.top + scrollY + tr.height / 2 - pr.height / 2;
          left = tr.right + scrollX + offset;
          break;
      }
    };

    // initial attempt
    tryPlacement(placement);

    // viewport collision handling with padding
    const vw = document.documentElement.clientWidth;
    const vh = document.documentElement.clientHeight;

    const clamp = (val: number, min: number, max: number) => Math.max(min, Math.min(max, val));
    let needFlip = false;

    // Check vertical overflow
    const topOverflow = top - scrollY < VIEWPORT_PADDING;
    const bottomOverflow = scrollY + vh - (top + pr.height) < VIEWPORT_PADDING;

    // Check horizontal overflow
    const leftOverflow = left - scrollX < VIEWPORT_PADDING;
    const rightOverflow = scrollX + vw - (left + pr.width) < VIEWPORT_PADDING;

    // Flip logic: if overflow on the chosen side, try the opposite placement
    if (placement === "top" && topOverflow) {
      needFlip = true;
      actualPlacement = "bottom";
    } else if (placement === "bottom" && bottomOverflow) {
      needFlip = true;
      actualPlacement = "top";
    } else if (placement === "left" && leftOverflow) {
      needFlip = true;
      actualPlacement = "right";
    } else if (placement === "right" && rightOverflow) {
      needFlip = true;
      actualPlacement = "left";
    }

    if (needFlip) tryPlacement(actualPlacement);

    // Clamp within viewport horizontally/vertically
    if (actualPlacement === "top" || actualPlacement === "bottom") {
      left = clamp(left, scrollX + VIEWPORT_PADDING, scrollX + vw - pr.width - VIEWPORT_PADDING);
    } else {
      top = clamp(top, scrollY + VIEWPORT_PADDING, scrollY + vh - pr.height - VIEWPORT_PADDING);
    }

    setCoords({ top, left, actualPlacement });
  }, [offset, placement]);

  useIsomorphicLayoutEffect(() => {
    if (open) {
      // next frame to ensure panelRef has layout
      const id = window.requestAnimationFrame(computeAndSetPosition);
      return () => window.cancelAnimationFrame(id);
    }
  }, [open, computeAndSetPosition, content]);

  React.useEffect(() => {
    if (!open) return;
    const onScroll = () => computeAndSetPosition();
    const onResize = () => computeAndSetPosition();
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", onResize);
    return () => {
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", onResize);
    };
  }, [open, computeAndSetPosition]);

  // Clone child to add aria-describedby + event handlers + ref
  const child = React.Children.only(children);
  const childProps: Record<string, any> = {
    ref: (node: HTMLElement) => {
      triggerRef.current = node;
      const { ref } = child as any;
      if (typeof ref === "function") ref(node);
      else if (ref && typeof ref === "object") (ref as React.MutableRefObject<HTMLElement | null>).current = node;
    },
    onMouseEnter: mergeHandlers(child.props.onMouseEnter, scheduleOpen),
    onMouseLeave: mergeHandlers(child.props.onMouseLeave, scheduleClose),
    onFocus: mergeHandlers(child.props.onFocus, scheduleOpen),
    onBlur: mergeHandlers(child.props.onBlur, scheduleClose),
    "aria-describedby": open ? computedId : child.props["aria-describedby"],
  };

  const panel = open ? (
    <div
      ref={panelRef}
      id={computedId}
      role="tooltip"
      className={clsx(
        "pointer-events-auto z-50 max-w-xs rounded-md border",
        "border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 shadow-lg",
        "dark:border-gray-700 dark:bg-gray-900 dark:text-gray-100",
        "transition-opacity duration-100",
        className
      )}
      style={{
        position: portal ? "absolute" : "fixed", // We'll override below based on portal usage
        top: coords.top,
        left: coords.left,
      }}
      onMouseEnter={onPanelMouseEnter}
      onMouseLeave={onPanelMouseLeave}
    >
      {arrow && (
        <span
          aria-hidden="true"
          className={clsx(
            "absolute block h-2 w-2 rotate-45 border",
            "border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-900"
          )}
          style={{
            // Position arrow opposite to actual placement
            top:
              coords.actualPlacement === "bottom"
                ? -4
                : coords.actualPlacement === "top"
                ? undefined
                : "50%",
            bottom: coords.actualPlacement === "top" ? -4 : undefined,
            left:
              coords.actualPlacement === "right"
                ? -4
                : coords.actualPlacement === "left"
                ? undefined
                : "50%",
            right: coords.actualPlacement === "left" ? -4 : undefined,
            transform:
              coords.actualPlacement === "top" || coords.actualPlacement === "bottom"
                ? "translateX(-50%) rotate(45deg)"
                : "translateY(-50%) rotate(45deg)",
          }}
        />
      )}
      <div className="relative z-10">{content}</div>
    </div>
  ) : null;

  const panelNode = panel && portal && mounted ? createPortal(panel, document.body) : panel;

  return (
    <>
      {React.cloneElement(child, childProps)}
      {panelNode}
    </>
  );
};

export default Tooltip;
