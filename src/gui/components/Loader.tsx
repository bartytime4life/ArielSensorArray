// ============================================================================
// ⏳ Loader Components — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// What’s included:
//   • <Loader.Spinner>  — Accessible SVG spinner (sizes: sm|md|lg|xl)
//   • <Loader.Dots>     — Animated dot ellipsis
//   • <Loader.Bar>      — Indeterminate or determinate progress bar
//   • <Loader.Skeleton> — Shimmering skeleton placeholder block
//   • <Loader.Overlay>  — Full-screen or container overlay with spinner+label
//
// Accessibility:
//   • Spinner/Dots use role="status" + aria-live="polite" with optional label
//   • Progress bar uses role="progressbar" and proper aria-* attributes
//   • Respect reduced motion via prefers-reduced-motion
//
// Styling: Tailwind + dark-mode aware, shadcn/ui-like tokens, focus-safe
// Determinism: No random IDs; predictable animations, stable markup
// ============================================================================

import * as React from "react";
import clsx from "clsx";

// ----------------------------------------
// Shared types & utilities
// ----------------------------------------

type Size = "sm" | "md" | "lg" | "xl";
type Tone = "default" | "muted" | "primary" | "success" | "warning" | "danger";

const sizeToPx: Record<Size, number> = { sm: 16, md: 20, lg: 28, xl: 36 };

const toneToClass: Record<Tone, string> = {
  default: "text-gray-600 dark:text-gray-300",
  muted: "text-gray-400 dark:text-gray-500",
  primary: "text-blue-600 dark:text-blue-400",
  success: "text-emerald-600 dark:text-emerald-400",
  warning: "text-amber-600 dark:text-amber-400",
  danger: "text-red-600 dark:text-red-400",
};

// Reduced-motion detection (memoized per mount)
function usePrefersReducedMotion() {
  const [reduced, setReduced] = React.useState(false);
  React.useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(!!m.matches);
    onChange();
    m.addEventListener?.("change", onChange);
    return () => m.removeEventListener?.("change", onChange);
  }, []);
  return reduced;
}

// Invisible but readable text for screen readers
const SrOnly: React.FC<React.HTMLAttributes<HTMLSpanElement>> = ({
  className,
  ...props
}) => (
  <span
    className={clsx(
      "sr-only",
      // Fallback if project doesn't include Tailwind sr-only preset:
      "absolute -m-px h-px w-px overflow-hidden whitespace-nowrap border-0 p-0",
      className
    )}
    {...props}
  />
);

// ----------------------------------------
// Spinner
// ----------------------------------------

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: Size;
  tone?: Tone;
  label?: string;
  /** If true, shows a faint track ring behind the spinner arc */
  withTrack?: boolean;
  /** If true, centers label beneath spinner with small gap */
  showLabel?: boolean;
  /** Render as block (stackable) instead of inline */
  block?: boolean;
}

const Spinner: React.FC<SpinnerProps> = ({
  size = "md",
  tone = "primary",
  label = "Loading…",
  withTrack = true,
  showLabel = false,
  block = false,
  className,
  ...rest
}) => {
  const px = sizeToPx[size];
  const strokeWidth = Math.max(2, Math.round(px / 10));
  const reduced = usePrefersReducedMotion();
  return (
    <div
      role="status"
      aria-live="polite"
      className={clsx(
        block ? "flex" : "inline-flex",
        "items-center",
        showLabel && "flex-col gap-2",
        className
      )}
      {...rest}
    >
      <svg
        className={clsx(!reduced && "animate-spin", toneToClass[tone])}
        width={px}
        height={px}
        viewBox="0 0 24 24"
        aria-hidden="true"
      >
        {withTrack && (
          <circle
            className="opacity-25"
            cx="12"
            cy="12"
            r="10"
            stroke="currentColor"
            strokeWidth={strokeWidth}
            fill="none"
          />
        )}
        {/* Arc wedge */}
        <path
          className="opacity-90"
          fill="currentColor"
          d="M12 2a10 10 0 0 1 10 10h-4a6 6 0 1 0-6-6V2z"
        />
      </svg>
      {showLabel ? (
        <div className="text-xs text-gray-600 dark:text-gray-300">{label}</div>
      ) : (
        <SrOnly>{label}</SrOnly>
      )}
    </div>
  );
};

// ----------------------------------------
// Dots (ellipsis)
// ----------------------------------------

export interface DotsProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: Size;
  tone?: Tone;
  label?: string;
  /** If true, shows the label inline next to dots */
  inlineLabel?: boolean;
  /** Render as block instead of inline */
  block?: boolean;
}

const Dots: React.FC<DotsProps> = ({
  size = "md",
  tone = "default",
  label = "Loading…",
  inlineLabel = false,
  block = false,
  className,
  ...rest
}) => {
  const reduced = usePrefersReducedMotion();
  const dotSize = Math.round(sizeToPx[size] / 4);
  const style = { width: dotSize, height: dotSize, borderRadius: dotSize / 2 };
  const anim = reduced ? "" : "animate-bounce";

  return (
    <div
      role="status"
      aria-live="polite"
      className={clsx(block ? "flex" : "inline-flex", "items-center gap-1", className)}
      {...rest}
    >
      <span
        style={style}
        className={clsx(anim, "[animation-delay:-0.2s] bg-current", toneToClass[tone])}
      />
      <span
        style={style}
        className={clsx(anim, "[animation-delay:-0.1s] bg-current", toneToClass[tone])}
      />
      <span style={style} className={clsx(anim, "bg-current", toneToClass[tone])} />
      {inlineLabel ? (
        <span className="ms-2 text-xs text-gray-600 dark:text-gray-300">{label}</span>
      ) : (
        <SrOnly>{label}</SrOnly>
      )}
    </div>
  );
};

// ----------------------------------------
// Bar (progress)
// ----------------------------------------

export interface BarProps extends React.HTMLAttributes<HTMLDivElement> {
  /** If provided, renders a determinate bar 0..100; otherwise indeterminate */
  value?: number;
  min?: number;
  max?: number;
  tone?: Tone;
  label?: string;
  showLabel?: boolean;
  /** Height of the bar in Tailwind h-* classes (e.g., 'h-1.5', 'h-2') */
  heightClass?: string;
  /** Rounded corners (default true) */
  rounded?: boolean;
  /** Show value as text inside the bar when determinate */
  showValueInside?: boolean;
}

const Bar: React.FC<BarProps> = ({
  value,
  min = 0,
  max = 100,
  tone = "primary",
  label = "Loading…",
  showLabel = false,
  heightClass = "h-2",
  rounded = true,
  showValueInside = false,
  className,
  ...rest
}) => {
  const reduced = usePrefersReducedMotion();
  const isDeterminate = typeof value === "number";
  const clamp = (v: number) => Math.max(min, Math.min(max, v));
  const pct = isDeterminate ? clamp(value!) : undefined;
  const ratio = isDeterminate ? (pct! - min) / (max - min || 1) : 0;
  const widthStyle = isDeterminate ? { width: `${Math.round(ratio * 100)}%` } : undefined;

  return (
    <div className={clsx("w-full", className)} {...rest}>
      {showLabel && (
        <div className="mb-1 text-xs text-gray-600 dark:text-gray-300">
          {label} {isDeterminate ? `(${Math.round(ratio * 100)}%)` : null}
        </div>
      )}
      <div
        className={clsx(
          "w-full bg-gray-200 dark:bg-gray-800 overflow-hidden",
          heightClass,
          rounded && "rounded-full"
        )}
        role="progressbar"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={isDeterminate ? pct : undefined}
        aria-label={label}
      >
        {isDeterminate ? (
          <div
            className={clsx(
              "relative h-full transition-all duration-300 ease-out",
              rounded && "rounded-full",
              toneToClass[tone],
              "bg-current"
            )}
            style={widthStyle}
          >
            {showValueInside && (
              <span className="absolute inset-0 flex items-center justify-center text-[10px] font-medium text-gray-900 dark:text-gray-100">
                {Math.round(ratio * 100)}%
              </span>
            )}
          </div>
        ) : (
          <div
            className={clsx(
              "h-full w-1/3",
              !reduced && "animate-[progress-indeterminate_1.2s_ease-in-out_infinite]",
              "bg-current",
              toneToClass[tone],
              rounded && "rounded-full"
            )}
          />
        )}
      </div>
      {/* Keyframes (fallbacks if Tailwind config doesn't include) */}
      <style>{`
        @keyframes progress-indeterminate {
          0% { transform: translateX(-100%); }
          50% { transform: translateX(50%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
};

// ----------------------------------------
// Skeleton
// ----------------------------------------

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  rounded?: boolean | "full" | "lg" | "md" | "sm" | "none";
  shimmer?: boolean;
  /** Allow children to render inside the skeleton (useful for masking layout) */
  asChild?: boolean;
}

const roundedToClass = (rounded: SkeletonProps["rounded"]) =>
  rounded === true || rounded === "md"
    ? "rounded-md"
    : rounded === "full"
    ? "rounded-full"
    : rounded === "lg"
    ? "rounded-lg"
    : rounded === "sm"
    ? "rounded"
    : rounded === "none"
    ? "rounded-none"
    : "rounded-md";

const Skeleton: React.FC<SkeletonProps> = ({
  className,
  rounded = "md",
  shimmer = true,
  style,
  asChild = false,
  children,
  ...rest
}) => {
  const reduced = usePrefersReducedMotion();
  const shimmerCls =
    shimmer && !reduced
      ? "pointer-events-none absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/60 to-transparent dark:via-white/10 animate-[skeleton_1.2s_ease-in-out_infinite]"
      : "hidden";

  const content = (
    <div
      className={clsx(
        "relative overflow-hidden bg-gray-200 dark:bg-gray-800",
        roundedToClass(rounded),
        className
      )}
      style={style}
      {...rest}
    >
      <span aria-hidden="true" className={shimmerCls} />
      {children}
    </div>
  );

  return (
    <>
      {asChild ? children : content}
      <style>{`
        @keyframes skeleton {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </>
  );
};

// ----------------------------------------
// Overlay
// ----------------------------------------

export interface OverlayProps extends React.HTMLAttributes<HTMLDivElement> {
  label?: string;
  SpinnerProps?: Partial<SpinnerProps>;
  /** If true, covers the entire screen; else fills the parent container */
  fullScreen?: boolean;
  /** Optional: dim level for backdrop */
  backdropOpacity?: "sm" | "md" | "lg";
  /** Optional: show label below spinner */
  showLabel?: boolean;
}

const backdropMap = {
  sm: "bg-black/20",
  md: "bg-black/40",
  lg: "bg-black/60",
};

const Overlay: React.FC<OverlayProps> = ({
  label = "Loading…",
  SpinnerProps,
  fullScreen = false,
  backdropOpacity = "md",
  showLabel = true,
  className,
  ...rest
}) => {
  return (
    <div
      className={clsx(
        "inset-0 z-40 flex items-center justify-center backdrop-blur-[1px]",
        backdropMap[backdropOpacity],
        fullScreen ? "fixed" : "absolute",
        className
      )}
      role="status"
      aria-live="polite"
      {...rest}
    >
      <div className="flex flex-col items-center gap-3 rounded-2xl bg-white/80 p-4 shadow-lg dark:bg-gray-900/80">
        <Spinner
          tone="primary"
          size="lg"
          showLabel={showLabel}
          label={label}
          {...SpinnerProps}
        />
      </div>
    </div>
  );
};

// ----------------------------------------
// Namespace export
// ----------------------------------------

export const Loader = {
  Spinner,
  Dots,
  Bar,
  Skeleton,
  Overlay,
};

export default Loader;
