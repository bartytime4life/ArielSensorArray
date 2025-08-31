// src/gui/components/Table.test.tsx
// =============================================================================
// ✅ Tests for src/gui/components/Table.tsx (SpectraMind V50 GUI)
// -----------------------------------------------------------------------------
// Covers: loading/error/empty states, rendering, sorting, selection (row & all),
// global filter, pagination controls, row actions, and row click behavior.
//
// Test stack assumptions:
//   • Vitest (or Jest) + @testing-library/react
//   • tsconfig paths support for "@/..." (adjust mocks if needed)
//
// Run (Vitest):
//   npx vitest run src/gui/components/Table.test.tsx
// =============================================================================

import * as React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, fireEvent } from "@testing-library/react";
import { Table } from "./Table";

// -----------------------------------------------------------------------------
// Mocks
// -----------------------------------------------------------------------------

// Strip framer-motion animations
vi.mock("framer-motion", () => {
  return {
    motion: {
      div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    },
  };
});

// Merge classnames
vi.mock("@/lib/utils", () => {
  return { cn: (...args: any[]) => args.filter(Boolean).join(" ") };
});

// Mock shadcn/ui table primitives
vi.mock("@/components/ui/table", () => {
  const wrap = (tid: string) =>
    function Comp({ children, ...rest }: any) {
      return (
        <div data-testid={tid} {...rest}>
          {children}
        </div>
      );
    };
  return {
    Table: wrap("UiTable"),
    TableHeader: wrap("TableHeader"),
    TableBody: wrap("TableBody"),
    TableHead: wrap("TableHead"),
    TableRow: wrap("TableRow"),
    TableCell: wrap("TableCell"),
  };
});

// Mock Checkbox with a native input so we can toggle checked state
vi.mock("@/components/ui/checkbox", () => {
  const Checkbox: React.FC<any> = ({ checked, onCheckedChange, ...rest }) => (
    <label>
      <input
        data-testid="Checkbox"
        type="checkbox"
        checked={checked === true}
        onChange={(e) => onCheckedChange?.(e.target.checked)}
      />
      <span className="sr-only">checkbox</span>
    </label>
  );
  return { Checkbox };
});

// Mock Input
vi.mock("@/components/ui/input", () => {
  const Input: React.FC<any> = ({ value, onChange, ...rest }) => (
    <input data-testid="Input" value={value ?? ""} onChange={onChange} {...rest} />
  );
  return { Input };
});

// Mock Button
vi.mock("@/components/ui/button", () => {
  const Button: React.FC<any> = ({ children, onClick, disabled, ...rest }) => (
    <button data-testid="Button" onClick={onClick} disabled={disabled} {...rest}>
      {children}
    </button>
  );
  return { Button };
});

// Clean up between tests
afterEach(() => cleanup());

// -----------------------------------------------------------------------------
// Fixtures
// -----------------------------------------------------------------------------

type Row = { id: string; planet: string; gll: number; violations: number };

const rows: Row[] = [
  { id: "p-001", planet: "Kepler-1", gll: 0.6001, violations: 3 },
  { id: "p-002", planet: "Ariel-5", gll: 0.4321, violations: 7 },
  { id: "p-003", planet: "Kepler-2", gll: 0.7003, violations: 1 },
  { id: "p-004", planet: "TRAPPIST-1d", gll: 0.5102, violations: 2 },
];

const columns = [
  { key: "planet", header: "Planet", sortable: true as const },
  { key: "gll", header: "GLL", sortable: true as const, align: "right" as const, format: (v: number) => v.toFixed(4) },
  { key: "violations", header: "Violations", align: "right" as const },
];

// Convenience queries
function getAllTableRows() {
  // Each body row is "TableRow" with descendant "TableCell"
  return screen.getAllByTestId("TableRow").slice(1); // skip header row (first TableRow)
}

function cellText(r: HTMLElement, cellIndex: number) {
  return r.querySelectorAll('[data-testid="TableCell"]')[cellIndex]?.textContent ?? "";
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

describe("Table component", () => {
  it("renders loading state", () => {
    render(<Table<Row> title="t" data={rows} columns={columns} loading />);
    const skeleton = document.querySelector(".animate-pulse");
    expect(skeleton).toBeTruthy();
  });

  it("renders error state", () => {
    render(<Table<Row> title="t" data={rows} columns={columns} error="Boom!" />);
    expect(screen.getByText("Boom!")).toBeInTheDocument();
  });

  it("renders empty state when no data", () => {
    render(<Table<Row> title="t" data={[]} columns={columns} emptyMessage="Nothing here" />);
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
  });

  it("renders rows and formatted cells", () => {
    render(<Table<Row> title="t" data={rows} columns={columns} />);
    const bodyRows = getAllTableRows();
    expect(bodyRows.length).toBe(rows.length);
    // Check formatting on GLL (second column)
    const firstGll = cellText(bodyRows[0], 1);
    expect(firstGll).toMatch(/\d\.\d{4}/);
  });

  it("toggles sorting (asc → desc → none) on sortable header", () => {
    render(<Table<Row> title="t" data={rows} columns={columns} />);
    // Header cells are "TableHead"; first is (optional) selection, here not rendered
    const headers = screen.getAllByTestId("TableHead");
    const planetHeader = headers.find((h) => h.textContent?.includes("Planet"))!;
    // Asc
    fireEvent.click(planetHeader);
    let bodyRows = getAllTableRows();
    expect(cellText(bodyRows[0], 0)).toBe("Ariel-5");
    // Desc
    fireEvent.click(planetHeader);
    bodyRows = getAllTableRows();
    expect(cellText(bodyRows[0], 0)).toBe("TRAPPIST-1d");
    // None (original order restored starting with Kepler-1)
    fireEvent.click(planetHeader);
    bodyRows = getAllTableRows();
    expect(cellText(bodyRows[0], 0)).toBe("Kepler-1");
  });

  it("supports row selection and select-all on visible page", () => {
    const onSelectionChange = vi.fn();
    render(
      <Table<Row>
        title="t"
        data={rows}
        columns={columns}
        selectable
        defaultPageSize={2}
        onSelectionChange={onSelectionChange}
        rowId={(r) => r.id}
      />
    );
    // Two rows visible on first page
    const bodyRows = getAllTableRows();
    expect(bodyRows.length).toBe(2);

    // Check first row checkbox (inside first cell)
    const firstRowCheckbox = bodyRows[0].querySelector('input[type="checkbox"]')!;
    fireEvent.click(firstRowCheckbox);
    expect(onSelectionChange).toHaveBeenCalled();
    // Select all visible
    const headerCheckbox = screen.getAllByTestId("Checkbox")[0].querySelector('input[type="checkbox"]')!;
    fireEvent.click(headerCheckbox);
    expect(onSelectionChange).toHaveBeenCalledTimes(2);
  });

  it("filters with global filter over specified keys", () => {
    render(
      <Table<Row>
        title="t"
        data={rows}
        columns={columns}
        globalFilter={{ placeholder: "Search planet…", keys: ["planet"] }}
        defaultPageSize={10}
      />
    );
    // Initially all rows visible
    expect(getAllTableRows().length).toBe(rows.length);
    // Filter to Kepler only
    const input = screen.getByTestId("Input");
    fireEvent.change(input, { target: { value: "Kepler" } });
    const bodyRows = getAllTableRows();
    // Two Kepler matches (Kepler-1 & Kepler-2)
    expect(bodyRows.length).toBe(2);
    expect(cellText(bodyRows[0], 0)).toContain("Kepler");
  });

  it("paginates and updates range text / next/prev actions", () => {
    render(
      <Table<Row>
        title="t"
        data={rows}
        columns={columns}
        defaultPageSize={2}
        pageSizeOptions={[2, 4]}
      />
    );

    // Range text "Showing 1–2 of 4"
    expect(screen.getByText(/Showing/i).textContent).toMatch(/1–2 of 4/);

    // Next page
    const buttons = screen.getAllByTestId("Button");
    const next = buttons.find((b) => b.getAttribute("aria-label") === "Next page")!;
    fireEvent.click(next);

    // New range "Showing 3–4 of 4"
    expect(screen.getByText(/Showing/i).textContent).toMatch(/3–4 of 4/);

    // Change page size to 4
    const pageSizeSelect = screen.getByLabelText("Rows per page") as HTMLSelectElement;
    fireEvent.change(pageSizeSelect, { target: { value: "4" } });
    expect(screen.getByText(/Showing/i).textContent).toMatch(/1–4 of 4/);
  });

  it("renders row actions and does not trigger row click when interacting with actions", () => {
    const onRowClick = vi.fn();
    render(
      <Table<Row>
        title="t"
        data={rows}
        columns={columns}
        rowActions={(row) => <button data-testid={`row-action-${row.id}`}>Act</button>}
        onRowClick={onRowClick}
        rowId={(r) => r.id}
        defaultPageSize={2}
      />
    );
    const bodyRows = getAllTableRows();
    // Click action button in first row
    const actionBtn = bodyRows[0].querySelector('[data-testid^="row-action-"]') as HTMLButtonElement;
    fireEvent.click(actionBtn);
    expect(onRowClick).not.toHaveBeenCalled();

    // Click on row surface triggers onRowClick
    fireEvent.click(bodyRows[0]);
    expect(onRowClick).toHaveBeenCalledTimes(1);
  });

  it("applies alignment class on cells and custom render/format", () => {
    render(
      <Table<Row>
        title="t"
        data={rows.slice(0, 1)}
        columns={[
          { key: "planet", header: "Planet", align: "left" },
          { key: "gll", header: "GLL", align: "right", format: (v: number) => `g=${v.toFixed(2)}` },
          { key: "violations", header: "Viol", align: "center", render: (r: Row) => <span>v={r.violations}</span> },
        ]}
      />
    );
    const bodyRows = getAllTableRows();
    const cells = bodyRows[0].querySelectorAll('[data-testid="TableCell"]');

    // Left (default), Right, Center
    expect(cells[0].className).toMatch(/text-left/);
    expect(cells[1].className).toMatch(/text-right/);
    expect(cells[2].className).toMatch(/text-center/);

    expect(cells[1].textContent).toMatch(/^g=\d\.\d{2}$/);
    expect(cells[2].textContent).toBe(`v=${rows[0].violations}`);
  });
});
