// src/gui/components/Chart.test.tsx
// =============================================================================
// ✅ Tests for src/gui/components/Chart.tsx (SpectraMind V50 GUI)
// -----------------------------------------------------------------------------
// Validates: loading/error/empty states, chart type switching (line/area/bar/
// composed), series rendering, grid/legend, reference lines, right-axis usage,
// and point click forwarding.
//
// Test stack assumptions:
//   • Vitest (or Jest) + @testing-library/react
//   • tsconfig paths supporting "@/..." (adjust imports if needed)
//   • Recharts is mocked for deterministic JSDOM behavior
//
// Run (Vitest):
//   npx vitest run src/gui/components/Chart.test.tsx
// =============================================================================

import * as React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Chart } from "./Chart";

// -----------------------------------------------------------------------------
// Mocks
// -----------------------------------------------------------------------------

// Mock framer-motion to strip animations.
vi.mock("framer-motion", () => {
  return {
    motion: {
      div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    },
  };
});

// Mock Card wrapper to avoid external dependencies.
vi.mock("./card", () => {
  const Card: React.FC<any> = ({ children, title, description }) => (
    <div data-testid="card">
      {title && <div data-testid="card-title">{title}</div>}
      {description && <div data-testid="card-desc">{description}</div>}
      {children}
    </div>
  );
  return { Card };
});

// Mock className util.
vi.mock("@/lib/utils", () => {
  return { cn: (...args: any[]) => args.filter(Boolean).join(" ") };
});

// Helper to create simple passthrough components with testids.
// We also preserve props on elements via data-* so we can assert yAxisId etc.
function makeEl(name: string, extraProps?: string[]) {
  const Comp: React.FC<any> = ({ children, onClick, ...rest }) => {
    const dataProps: Record<string, any> = { "data-testid": name };
    (extraProps ?? []).forEach((k) => {
      const v = rest[k];
      if (v !== undefined) dataProps[`data-${k.toLowerCase()}`] = String(v);
    });
    return (
      <div {...dataProps} onClick={onClick}>
        {children}
      </div>
    );
  };
  Comp.displayName = name;
  return Comp;
}

// Mock recharts components we rely on.
vi.mock("recharts", () => {
  const comps: Record<string, any> = {};
  // containers / charts
  comps.ResponsiveContainer = ({ children, width, height }: any) => (
    <div data-testid="ResponsiveContainer" data-width={width} data-height={height}>
      {children}
    </div>
  );
  comps.LineChart = makeEl("LineChart");
  comps.AreaChart = makeEl("AreaChart");
  comps.BarChart = makeEl("BarChart");
  comps.ComposedChart = makeEl("ComposedChart");

  // primitives
  comps.CartesianGrid = makeEl("CartesianGrid");
  comps.XAxis = makeEl("XAxis");
  comps.YAxis = makeEl("YAxis", ["yAxisId", "orientation", "label"]);
  comps.Tooltip = makeEl("Tooltip");
  comps.Legend = makeEl("Legend");
  comps.ReferenceLine = makeEl("ReferenceLine", ["x", "y", "yAxisId"]);

  // series
  comps.Line = makeEl("Line", ["dataKey", "yAxisId", "type"]);
  comps.Area = makeEl("Area", ["dataKey", "yAxisId", "type"]);
  comps.Bar = makeEl("Bar", ["dataKey", "yAxisId"]);

  return comps;
});

// Clean up after each test.
afterEach(() => cleanup());

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

const sampleData = [
  { step: 0, gll: 0.5, loss: 1.0, aux: 2.0 },
  { step: 1, gll: 0.6, loss: 0.9, aux: 2.1 },
  { step: 2, gll: 0.55, loss: 0.95, aux: 2.2 },
];

const lineSeries = [
  { key: "gll", label: "GLL" },
  { key: "loss", label: "Loss" },
];

const areaSeries = [
  { key: "gll", label: "GLL", seriesType: "area" as const },
];

const barSeries = [
  { key: "aux", label: "Aux", seriesType: "bar" as const },
];

const composedSeries = [
  { key: "gll", label: "GLL", seriesType: "line" as const },
  { key: "loss", label: "Loss", seriesType: "area" as const },
  { key: "aux", label: "Aux", seriesType: "bar" as const, yAxisId: "right" as const },
];

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

describe("Chart component", () => {
  it("renders loading state", () => {
    render(
      <Chart
        title="Loading Chart"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
        loading
      />
    );
    // Loading skeleton rendered (h-40 pulse)
    const skeleton = document.querySelector(".animate-pulse");
    expect(skeleton).toBeTruthy();
  });

  it("renders error state", () => {
    render(
      <Chart
        title="Error Chart"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
        error="Boom!"
      />
    );
    expect(screen.getByText("Boom!")).toBeInTheDocument();
  });

  it("renders empty state when no data", () => {
    render(
      <Chart
        title="Empty Chart"
        type="line"
        data={[]}
        xKey="step"
        yKeys={lineSeries}
        emptyMessage="Nothing here"
      />
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("renders LineChart with matching Line series when type=line", () => {
    render(
      <Chart title="Line" type="line" data={sampleData} xKey="step" yKeys={lineSeries} />
    );
    expect(screen.getByTestId("LineChart")).toBeInTheDocument();
    const lines = screen.getAllByTestId("Line");
    expect(lines.length).toBe(lineSeries.length);
    // Axes should be present
    expect(screen.getByTestId("XAxis")).toBeInTheDocument();
    expect(screen.getAllByTestId("YAxis").length).toBeGreaterThanOrEqual(1);
  });

  it("renders AreaChart with Area series when type=area", () => {
    render(
      <Chart
        title="Area"
        type="area"
        data={sampleData}
        xKey="step"
        yKeys={areaSeries as any}
      />
    );
    expect(screen.getByTestId("AreaChart")).toBeInTheDocument();
    const areas = screen.getAllByTestId("Area");
    expect(areas.length).toBe(areaSeries.length);
  });

  it("renders BarChart with Bar series when type=bar", () => {
    render(
      <Chart
        title="Bar"
        type="bar"
        data={sampleData}
        xKey="step"
        yKeys={barSeries as any}
      />
    );
    expect(screen.getByTestId("BarChart")).toBeInTheDocument();
    const bars = screen.getAllByTestId("Bar");
    expect(bars.length).toBe(barSeries.length);
  });

  it("renders ComposedChart with mixed series and right axis when type=composed", () => {
    render(
      <Chart
        title="Composed"
        type="composed"
        data={sampleData}
        xKey="step"
        yKeys={composedSeries as any}
        leftAxisLabel="Left"
        rightAxisLabel="Right"
      />
    );
    expect(screen.getByTestId("ComposedChart")).toBeInTheDocument();

    // Expect one of each series
    expect(screen.getAllByTestId("Line").length).toBe(1);
    expect(screen.getAllByTestId("Area").length).toBe(1);
    expect(screen.getAllByTestId("Bar").length).toBe(1);

    // Right axis should render due to a yAxisId="right" series
    const yAxes = screen.getAllByTestId("YAxis");
    const hasRight = yAxes.some((el) => el.getAttribute("data-orientation") === "right");
    expect(hasRight).toBe(true);
  });

  it("renders grid and legend when enabled", () => {
    render(
      <Chart
        title="Decorated"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
        grid
        legend
      />
    );
    expect(screen.getByTestId("CartesianGrid")).toBeInTheDocument();
    expect(screen.getByTestId("Legend")).toBeInTheDocument();
  });

  it("renders reference lines for x and y", () => {
    render(
      <Chart
        title="Refs"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
        referenceLines={[
          { x: 1, label: "x-ref" },
          { y: 0.55, label: "y-ref", yAxisId: "left" },
        ]}
      />
    );
    const refs = screen.getAllByTestId("ReferenceLine");
    expect(refs.length).toBe(2);
    // Ensure attributes are passed through in mock
    const yRef = refs.find((el) => el.getAttribute("data-y") === "0.55");
    expect(yRef).toBeTruthy();
  });

  it("forwards onPointClick via series onClick", () => {
    const onPointClick = vi.fn();
    render(
      <Chart
        title="Clicks"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
        onPointClick={onPointClick}
      />
    );
    const firstLine = screen.getAllByTestId("Line")[0];
    fireEvent.click(firstLine);
    expect(onPointClick).toHaveBeenCalledTimes(1);
  });

  it("renders header chrome (title/description) in Card", () => {
    render(
      <Chart
        title="Header Title"
        description="Header Description"
        type="line"
        data={sampleData}
        xKey="step"
        yKeys={lineSeries}
      />
    );
    expect(screen.getByTestId("card")).toBeInTheDocument();
    expect(screen.getByTestId("card-title")).toHaveTextContent("Header Title");
    expect(screen.getByTestId("card-desc")).toHaveTextContent("Header Description");
  });
});
