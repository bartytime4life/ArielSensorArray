// =============================================================================
// 📈 SpectraMind V50 — Reusable Chart Component (React + Recharts + Tailwind)
// -----------------------------------------------------------------------------
// Purpose
//   A single, declarative chart component used across the GUI for diagnostics,
//   calibration checks, GLL heatmaps (aggregated), etc. Wraps Recharts with a
//   consistent look-and-feel, optional Card header, loading/empty/error states,
//   responsive behavior, and dual Y-axes support.
//
// Design notes
//   • Declarative props: xKey, yKeys[], type = "line" | "area" | "bar" | "composed"
//   • Optional Card header: title, description, icon, actions
//   • Optional reference lines (x or y)
//   • Optional grid/legend; configurable tick/tooltip formatters
//   • Sensible defaults; non-intrusive styling with Tailwind
//   • Dark-mode friendly (inherits Tailwind dark vars)
//   • No color hardcoding required; supports per-series color override
//   • Reduced-motion aware; deterministic rendering
//
// Example
//   <Chart
//     title="GLL over Time"
//     description="Validation epoch metrics"
//     type="line"
//     data={rows}
//     xKey="step"
//     yKeys={[{ key: "gll", label: "GLL", color: "#2563eb" }]}
//     grid
//     legend
//   />
//
// Dependencies
//   • "recharts"
//   • "framer-motion"
//   • TailwindCSS
//   • Local Card component (src/gui/components/card.tsx)
// =============================================================================

import * as React from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  ResponsiveContainer,
  LineChart,
  ComposedChart,
  AreaChart,
  BarChart,
  Line,
  Area,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ReferenceLine,
  Brush,
} from "recharts";
import { Card } from "./card";

type SeriesType = "line" | "area" | "bar";
type ChartType = "line" | "area" | "bar" | "composed";

export interface ChartSeries {
  key: string;                   // data key
  label?: string;                // legend label
  color?: string;                // override color (stroke/fill)
  strokeWidth?: number;          // line/bar stroke width
  dot?: boolean;                 // show dots for line series
  yAxisId?: "left" | "right";    // attach to left/right axis
  type?: "linear" | "monotone";  // Recharts curve type (for line/area)
  seriesType?: SeriesType;       // only used for composed charts
  fillOpacity?: number;          // for area/bar
  barSize?: number;              // for bar
  stacked?: boolean;             // stackId auto-assigned when true
  gradient?: boolean;            // add gradient fill (area/bar)
}

export interface ReferenceMark {
  x?: number | string;
  y?: number | string;
  label?: string;
  color?: string;
  dash?: string;               // e.g., "3 3"
  yAxisId?: "left" | "right";
}

export interface ChartProps extends React.HTMLAttributes<HTMLDivElement> {
  // Layout / chrome
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;

  // Data & configuration
  type?: ChartType;
  data: Array<Record<string, any>>;
  xKey: string;
  yKeys: ChartSeries[];

  height?: number;             // pixel height (default 280)
  grid?: boolean;              // show cartesian grid
  legend?: boolean;            // show legend
  referenceLines?: ReferenceMark[];
  brush?: boolean;             // show Brush control (zooming window)
  syncId?: string;             // sync multiple charts

  // Axes & formatting
  leftAxisLabel?: string;
  rightAxisLabel?: string;
  xTickFormatter?: (val: any) => string;
  yTickFormatterLeft?: (val: any) => string;
  yTickFormatterRight?: (val: any) => string;
  tooltipFormatter?: (value: any, name: string, entry: any) => any;
  tooltipLabelFormatter?: (label: any) => any;
  yDomainLeft?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];
  yDomainRight?: [number | "auto" | "dataMin" | "dataMax", number | "auto" | "dataMin" | "dataMax"];

  // States
  loading?: boolean;
  emptyMessage?: string;
  error?: string | null;

  // Events
  onPointClick?: (payload: any) => void;

  // A11y / test hooks
  "aria-label"?: string;
  "data-testid"?: string;
}

/** Lightweight empty state */
const EmptyState: React.FC<{ message?: string }> = ({ message }) => (
  <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
    {message ?? "No data to display"}
  </div>
);

/** Lightweight error state */
const ErrorState: React.FC<{ message?: string }> = ({ message }) => (
  <div className="flex h-40 items-center justify-center text-sm text-red-600 dark:text-red-400">
    {message ?? "Something went wrong rendering the chart."}
  </div>
);

/** Lightweight loading skeleton */
const LoadingState: React.FC = () => (
  <div className="h-40 w-full animate-pulse rounded-xl bg-muted/50" />
);

/** Default axis value formatter (noop passthrough) */
const identityFormat = (v: any) => (v == null ? "" : String(v));

/** Build a consistent Recharts color tuple for stroke/fill */
function seriesColors(c?: string) {
  // Fallback to CSS vars so themes can govern palette
  const stroke = c ?? "hsl(var(--primary))";
  const fill = c ?? "hsl(var(--primary))";
  return { stroke, fill };
}

/** Renders a Recharts <ReferenceLine> from a ReferenceMark definition */
function ReferenceLineEl({ mark }: { mark: ReferenceMark }) {
  const color = mark.color ?? "hsl(var(--muted-foreground))";
  const strokeDasharray = mark.dash ?? "4 4";
  if (mark.y != null) {
    return (
      <ReferenceLine
        y={mark.y as any}
        yAxisId={mark.yAxisId ?? "left"}
        stroke={color}
        strokeDasharray={strokeDasharray}
        label={mark.label}
      />
    );
  }
  if (mark.x != null) {
    return (
      <ReferenceLine
        x={mark.x as any}
        stroke={color}
        strokeDasharray={strokeDasharray}
        label={mark.label}
      />
    );
  }
  return null;
}

/** Reduced-motion detection */
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

/** Series renderers for composed charts */
function renderSeriesForComposed(
  s: ChartSeries,
  idx: number,
  onPointClick?: (p: any) => void
) {
  const { stroke, fill } = seriesColors(s.color);
  const common = {
    key: `${s.key}-${idx}`,
    dataKey: s.key,
    name: s.label ?? s.key,
    yAxisId: s.yAxisId ?? "left",
  };
  const handleClick = onPointClick ? { onClick: onPointClick } : {};
  const stackId = s.stacked ? "stack-0" : undefined;

  switch (s.seriesType ?? "line") {
    case "bar":
      return (
        <Bar
          {...common}
          fill={fill}
          barSize={s.barSize ?? 12}
          fillOpacity={s.fillOpacity ?? 0.9}
          stackId={stackId}
          {...handleClick}
        />
      );
    case "area":
      return (
        <Area
          {...common}
          type={s.type ?? "monotone"}
          stroke={stroke}
          fill={fill}
          strokeWidth={s.strokeWidth ?? 2}
          fillOpacity={s.fillOpacity ?? 0.15}
          dot={s.dot ?? false}
          activeDot={{ r: 4 }}
          stackId={stackId}
          {...handleClick}
        />
      );
    case "line":
    default:
      return (
        <Line
          {...common}
          type={s.type ?? "monotone"}
          stroke={stroke}
          strokeWidth={s.strokeWidth ?? 2}
          dot={s.dot ?? false}
          activeDot={{ r: 4 }}
          {...handleClick}
        />
      );
  }
}

/** Optional gradient defs for series that request gradient fill */
function Gradients({ yKeys }: { yKeys: ChartSeries[] }) {
  return (
    <defs>
      {yKeys.map((s, i) =>
        s.gradient ? (
          <linearGradient key={`grad-${s.key}-${i}`} id={`grad-${s.key}-${i}`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={s.color ?? "hsl(var(--primary))"} stopOpacity={0.6} />
            <stop offset="100%" stopColor={s.color ?? "hsl(var(--primary))"} stopOpacity={0.05} />
          </linearGradient>
        ) : null
      )}
    </defs>
  );
}

/**
 * Chart
 * The main, exported chart component with optional Card header/chrome.
 */
export const Chart: React.FC<ChartProps> = ({
  // chrome
  title,
  description,
  icon,
  actions,
  className,

  // data/config
  type = "line",
  data,
  xKey,
  yKeys,

  height = 280,
  grid = true,
  legend = true,
  referenceLines,
  brush = false,
  syncId,

  // formatting
  leftAxisLabel,
  rightAxisLabel,
  xTickFormatter,
  yTickFormatterLeft,
  yTickFormatterRight,
  tooltipFormatter,
  tooltipLabelFormatter,
  yDomainLeft = ["auto", "auto"],
  yDomainRight = ["auto", "auto"],

  // states
  loading,
  emptyMessage,
  error,

  // events
  onPointClick,

  // a11y/test
  "aria-label": ariaLabel,
  "data-testid": testId,

  ...rest
}) => {
  const hasData = Array.isArray(data) && data.length > 0;
  const reducedMotion = usePrefersReducedMotion();

  // Choose base chart by type
  const isComposed = type === "composed";
  const isLine = type === "line";
  const isArea = type === "area";
  const isBar = type === "bar";

  const XTickFmt = xTickFormatter ?? identityFormat;
  const YTickLeftFmt = yTickFormatterLeft ?? identityFormat;
  const YTickRightFmt = yTickFormatterRight ?? identityFormat;

  // Tooltip formatters
  const tooltipFmt =
    tooltipFormatter ??
    ((value: any, name: string) => {
      if (typeof value === "number") return [value, name];
      return [String(value ?? ""), name];
    });

  // Determine whether any series uses right axis
  const useRightAxis = React.useMemo(
    () => yKeys.some((s) => (s.yAxisId ?? "left") === "right"),
    [yKeys]
  );

  // Inner chart element
  const ChartInner = React.useMemo(() => {
    if (loading) return <LoadingState />;
    if (error) return <ErrorState message={error} />;
    if (!hasData) return <EmptyState message={emptyMessage} />;

    const commonAxes = (
      <>
        <XAxis
          dataKey={xKey}
          tickFormatter={XTickFmt}
          stroke="hsl(var(--muted-foreground))"
        />
        <YAxis
          yAxisId="left"
          tickFormatter={YTickLeftFmt}
          stroke="hsl(var(--muted-foreground))"
          domain={yDomainLeft as any}
          label={
            leftAxisLabel
              ? {
                  value: leftAxisLabel,
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                }
              : undefined
          }
        />
        {useRightAxis && (
          <YAxis
            yAxisId="right"
            orientation="right"
            tickFormatter={YTickRightFmt}
            stroke="hsl(var(--muted-foreground))"
            domain={yDomainRight as any}
            label={
              rightAxisLabel
                ? {
                    value: rightAxisLabel,
                    angle: -90,
                    position: "insideRight",
                    offset: 10,
                  }
                : undefined
            }
          />
        )}
        {grid && (
          <CartesianGrid
            stroke="hsl(var(--muted))"
            strokeOpacity={0.5}
            vertical={false}
          />
        )}
        <Tooltip
          formatter={tooltipFmt}
          labelFormatter={tooltipLabelFormatter}
          contentStyle={{
            background: "hsl(var(--popover))",
            border: "1px solid hsl(var(--border))",
            borderRadius: 8,
            color: "hsl(var(--foreground))",
          }}
          labelStyle={{ color: "hsl(var(--muted-foreground))" }}
          isAnimationActive={!reducedMotion}
        />
        {legend && <Legend />}
        {referenceLines?.map((mark, i) => (
          <ReferenceLineEl key={`ref-${i}`} mark={mark} />
        ))}
      </>
    );

    // Render by chart type
    if (isComposed) {
      return (
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data} syncId={syncId}>
            <Gradients yKeys={yKeys} />
            {commonAxes}
            {yKeys.map((s, i) => renderSeriesForComposed(s, i, onPointClick))}
            {brush && <Brush dataKey={xKey} height={20} stroke="hsl(var(--muted-foreground))" />}
          </ComposedChart>
        </ResponsiveContainer>
      );
    }

    if (isLine) {
      return (
        <ResponsiveContainer width="100%" height={height}>
          <LineChart data={data} syncId={syncId}>
            {commonAxes}
            {yKeys.map((s, i) => {
              const { stroke } = seriesColors(s.color);
              return (
                <Line
                  key={`${s.key}-${i}`}
                  type={s.type ?? "monotone"}
                  dataKey={s.key}
                  name={s.label ?? s.key}
                  stroke={stroke}
                  strokeWidth={s.strokeWidth ?? 2}
                  dot={s.dot ?? false}
                  activeDot={{ r: 4 }}
                  yAxisId={s.yAxisId ?? "left"}
                  onClick={onPointClick}
                  isAnimationActive={!reducedMotion}
                />
              );
            })}
            {brush && <Brush dataKey={xKey} height={20} stroke="hsl(var(--muted-foreground))" />}
          </LineChart>
        </ResponsiveContainer>
      );
    }

    if (isArea) {
      return (
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data} syncId={syncId}>
            <Gradients yKeys={yKeys} />
            {commonAxes}
            {yKeys.map((s, i) => {
              const { stroke, fill } = seriesColors(s.color);
              const fillRef = s.gradient ? `url(#grad-${s.key}-${i})` : fill;
              return (
                <Area
                  key={`${s.key}-${i}`}
                  type={s.type ?? "monotone"}
                  dataKey={s.key}
                  name={s.label ?? s.key}
                  stroke={stroke}
                  fill={fillRef}
                  strokeWidth={s.strokeWidth ?? 2}
                  fillOpacity={s.fillOpacity ?? (s.gradient ? 1 : 0.15)}
                  dot={s.dot ?? false}
                  yAxisId={s.yAxisId ?? "left"}
                  onClick={onPointClick}
                  isAnimationActive={!reducedMotion}
                />
              );
            })}
            {brush && <Brush dataKey={xKey} height={20} stroke="hsl(var(--muted-foreground))" />}
          </AreaChart>
        </ResponsiveContainer>
      );
    }

    // default: bar
    return (
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} syncId={syncId}>
          <Gradients yKeys={yKeys} />
          {commonAxes}
          {yKeys.map((s, i) => {
            const { fill } = seriesColors(s.color);
            const fillRef = s.gradient ? `url(#grad-${s.key}-${i})` : fill;
            const stackId = s.stacked ? "stack-0" : undefined;
            return (
              <Bar
                key={`${s.key}-${i}`}
                dataKey={s.key}
                name={s.label ?? s.key}
                fill={fillRef}
                yAxisId={s.yAxisId ?? "left"}
                barSize={s.barSize ?? 12}
                fillOpacity={s.fillOpacity ?? (s.gradient ? 1 : 0.9)}
                onClick={onPointClick}
                stackId={stackId}
                isAnimationActive={!reducedMotion}
              />
            );
          })}
          {brush && <Brush dataKey={xKey} height={20} stroke="hsl(var(--muted-foreground))" />}
        </BarChart>
      </ResponsiveContainer>
    );
  }, [
    loading,
    error,
    hasData,
    emptyMessage,
    isComposed,
    isLine,
    isArea,
    height,
    grid,
    legend,
    data,
    xKey,
    yKeys,
    XTickFmt,
    YTickLeftFmt,
    YTickRightFmt,
    leftAxisLabel,
    rightAxisLabel,
    tooltipFmt,
    tooltipLabelFormatter,
    referenceLines,
    useRightAxis,
    onPointClick,
    yDomainLeft,
    yDomainRight,
    brush,
    syncId,
    reducedMotion,
  ]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={cn(className)}
      aria-label={ariaLabel ?? title ?? "Chart"}
      data-testid={testId ?? "Chart"}
      {...rest}
    >
      <Card
        title={title}
        description={description}
        icon={icon}
        actions={actions}
        className="overflow-hidden"
      >
        <div className="w-full">{ChartInner}</div>
      </Card>
    </motion.div>
  );
};

export default Chart;
