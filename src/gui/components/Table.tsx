// =============================================================================
// 🧬 SpectraMind V50 — Reusable Data Table (React + Tailwind + shadcn/ui)
// -----------------------------------------------------------------------------
// Purpose
//   A fast, declarative table for diagnostics dashboards and reports. Supports:
//   • Column definitions with optional custom cell renderers
//   • Accessors and formatters for flexible value mapping
//   • Client-side sorting (per-column, ASC/DESC)
//   • Optional pagination (controlled or internal)
//   • Optional row selection via checkboxes (page-aware or full-table)
//   • Optional global text filtering over selected fields
//   • Loading / empty / error states with a11y live regions
//   • Optional row actions and row click handler
//   • Deterministic behavior in SSR/CSR
//
// Design Notes
//   • No heavy table libs to keep bundle small; shadcn/ui primitives only.
//   • Tailwind for styling. Dark-mode friendly via Tailwind tokens.
//   • Composable and safe defaults; everything optional.
//   • A11y: aria-sort on headers, aria-live for state regions, keyboard support.
//   • Reproducible: no random IDs, stable keys; optional caller-provided rowId.
//
// Example
//   <Table
//     title="Diagnostics per Planet"
//     description="GLL, RMSE, violations"
//     data={rows}
//     columns={[
//       { key: "planet_id", header: "Planet", sortable: true },
//       { key: "gll", header: "GLL", sortable: true, align: "right",
//         format: (v) => v?.toFixed?.(4) },
//       { key: "violations", header: "Violations", align: "right" },
//     ]}
//     selectable
//     onSelectionChange={(ids) => console.log(ids)}
//     globalFilter={{ placeholder: "Search planet…", keys: ["planet_id"] }}
//     pageSizeOptions={[10, 20, 50]}
//     defaultPageSize={10}
//   />
//
// =============================================================================

import * as React from "react";
import { cn } from "@/lib/utils";
import {
  Table as UiTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  ArrowUp,
  ArrowDown,
  ArrowUpDown,
  MoreVertical,
} from "lucide-react";
import { motion } from "framer-motion";

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

type Align = "left" | "center" | "right";

export interface ColumnDef<T extends Record<string, any>> {
  /** Property key for the column, used for sorting and default cell content */
  key: keyof T | string;
  /** Optional header content; defaults to `key` string */
  header?: React.ReactNode;
  /** Whether column can sort */
  sortable?: boolean;
  /** Horizontal alignment for cell content */
  align?: Align;
  /** Width constraint (px or CSS width) */
  width?: string | number;
  /** Header cell class */
  headerClassName?: string;
  /** Body cell class */
  className?: string;
  /** A11y label for cells in this column */
  ariaLabel?: string;

  /** Accessor to compute the raw value for this column from the row */
  accessor?: (row: T, rowIndex: number) => any;
  /** Formatter for a value to display */
  format?: (value: any, row: T, rowIndex: number) => React.ReactNode;
  /** Full custom cell renderer (bypasses accessor/format) */
  render?: (row: T, rowIndex: number) => React.ReactNode;
}

export interface Pagination {
  pageIndex: number;
  pageSize: number;
  total: number;
}

export interface GlobalFilter {
  value?: string;
  placeholder?: string;
  keys?: (string | keyof any)[];
}

export interface TableProps<T extends Record<string, any>>
  extends React.HTMLAttributes<HTMLDivElement> {
  // Chrome
  title?: string;
  description?: string;
  className?: string;

  // Data model
  data: T[];
  columns: ColumnDef<T>[];
  /** Generate stable IDs; default: index-based string (page-aware) */
  rowId?: (row: T) => string;

  // Sorting (controlled or internal)
  defaultSortBy?: { key: string; direction: "asc" | "desc" };
  sortBy?: { key: string; direction: "asc" | "desc" } | null;
  onSortChange?: (sort: { key: string; direction: "asc" | "desc" } | null) => void;

  // Selection
  selectable?: boolean;
  defaultSelectedIds?: string[];
  selectedIds?: string[];
  onSelectionChange?: (ids: string[]) => void;

  // Global filter (simple contains across keys)
  globalFilter?: GlobalFilter;
  onGlobalFilterChange?: (value: string) => void;

  // Pagination (controlled or internal)
  defaultPageSize?: number;
  pageSizeOptions?: number[];
  pagination?: Pagination;
  onPageChange?: (pageIndex: number) => void;
  onPageSizeChange?: (pageSize: number) => void;

  // Row interactions
  rowActions?: (row: T, rowIndex: number) => React.ReactNode;
  onRowClick?: (row: T, rowIndex: number, id: string) => void;

  // States
  loading?: boolean;
  emptyMessage?: string;
  error?: string | null;

  // Misc
  /** Customize selection scope: 'page' (default) applies select-all to visible page; 'all' applies to all filtered rows */
  selectionScope?: "page" | "all";
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/** Skeleton loading block (kept deterministic) */
const LoadingState: React.FC = () => (
  <div
    role="status"
    aria-live="polite"
    className="h-40 w-full animate-pulse rounded-xl bg-muted/50"
    data-testid="table-loading"
  />
);

/** Error region */
const ErrorState: React.FC<{ message?: string }> = ({ message }) => (
  <div
    className="flex h-40 items-center justify-center text-sm text-red-600 dark:text-red-400"
    role="alert"
    aria-live="assertive"
    data-testid="table-error"
  >
    {message ?? "Failed to load data."}
  </div>
);

/** Empty region */
const EmptyState: React.FC<{ message?: string }> = ({ message }) => (
  <div
    className="flex h-40 items-center justify-center text-sm text-muted-foreground"
    role="status"
    aria-live="polite"
    data-testid="table-empty"
  >
    {message ?? "No rows to display."}
  </div>
);

/** Safe row ID derivation with caller override; deterministic and stable */
function getRowId<T extends Record<string, any>>(
  row: T,
  zeroBasedAbsoluteIndex: number,
  rowId?: (r: T) => string
) {
  try {
    if (rowId) {
      const v = rowId(row);
      if (typeof v === "string" && v.length) return v;
    }
  } catch {
    // ignore and fallback
  }
  // deterministic fallback: prefix + index
  return `row-${zeroBasedAbsoluteIndex}`;
}

/** Basic comparator with null safety + stable tiebreaker */
function cmp(a: any, b: any) {
  if (a == null && b == null) return 0;
  if (a == null) return -1;
  if (b == null) return 1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

/** Convert alignment to Tailwind classes */
function getCellAlignClass(align?: Align) {
  switch (align) {
    case "center":
      return "text-center";
    case "right":
      return "text-right";
    default:
      return "text-left";
  }
}

/** Resolve column width style */
function widthStyleOf(width?: string | number) {
  if (width === undefined) return undefined;
  return typeof width === "number" ? { width: `${width}px` } : { width };
}

/** Read column value via accessor/key */
function getColumnValue<T extends Record<string, any>>(
  col: ColumnDef<T>,
  row: T,
  absoluteIndex: number
) {
  if (typeof col.render === "function") return undefined; // render bypasses value
  if (typeof col.accessor === "function") return col.accessor(row, absoluteIndex);
  return (row as any)[col.key as any];
}

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export function Table<T extends Record<string, any>>(props: TableProps<T>) {
  const {
    // Chrome
    title,
    description,
    className,

    // Data
    data,
    columns,
    rowId,

    // Sorting
    defaultSortBy,
    sortBy: sortByProp,
    onSortChange,

    // Selection
    selectable,
    defaultSelectedIds,
    selectedIds: selectedIdsProp,
    onSelectionChange,
    selectionScope = "page",

    // Global filter
    globalFilter,
    onGlobalFilterChange,

    // Pagination
    defaultPageSize = 20,
    pageSizeOptions = [10, 20, 50, 100],
    pagination: paginationProp,
    onPageChange,
    onPageSizeChange,

    // Row interactions
    rowActions,
    onRowClick,

    // States
    loading,
    emptyMessage,
    error,

    // div props
    ...rest
  } = props;

  // ----- Sorting (controlled / uncontrolled) -----
  const [sortByInternal, setSortByInternal] = React.useState<
    { key: string; direction: "asc" | "desc" } | null
  >(defaultSortBy ?? null);

  const sortBy = sortByProp !== undefined ? sortByProp : sortByInternal;

  const setSort = React.useCallback(
    (next: { key: string; direction: "asc" | "desc" } | null) => {
      if (onSortChange) onSortChange(next);
      else setSortByInternal(next);
    },
    [onSortChange]
  );

  // ----- Selection (controlled / uncontrolled) -----
  const [selectedIdsInternal, setSelectedIdsInternal] = React.useState<string[]>(
    defaultSelectedIds ?? []
  );
  const selectedIds =
    selectedIdsProp !== undefined ? selectedIdsProp : selectedIdsInternal;

  const setSelected = React.useCallback(
    (ids: string[]) => {
      if (onSelectionChange) onSelectionChange(ids);
      else setSelectedIdsInternal(ids);
    },
    [onSelectionChange]
  );

  // ----- Filter (controlled / uncontrolled) -----
  const [filterValueInternal, setFilterValueInternal] = React.useState(
    globalFilter?.value ?? ""
  );
  const filterValue =
    globalFilter?.value !== undefined ? globalFilter.value : filterValueInternal;

  const changeFilter = React.useCallback(
    (v: string) => {
      if (onGlobalFilterChange) onGlobalFilterChange(v);
      else setFilterValueInternal(v);
    },
    [onGlobalFilterChange]
  );

  // ----- Pagination (controlled / uncontrolled) -----
  const [pageIndexInternal, setPageIndexInternal] = React.useState(0);
  const [pageSizeInternal, setPageSizeInternal] = React.useState(defaultPageSize);

  const pageIndex = paginationProp?.pageIndex ?? pageIndexInternal;
  const pageSize = paginationProp?.pageSize ?? pageSizeInternal;
  const totalExternal = paginationProp?.total;

  const setPageIndex = React.useCallback(
    (idx: number) => {
      if (onPageChange) onPageChange(idx);
      else setPageIndexInternal(idx);
    },
    [onPageChange]
  );

  const setPageSize = React.useCallback(
    (size: number) => {
      if (onPageSizeChange) onPageSizeChange(size);
      else setPageSizeInternal(size);
      // reset to first page when page size changes
      if (onPageChange) onPageChange(0);
      else setPageIndexInternal(0);
    },
    [onPageChange, onPageSizeChange]
  );

  // ----- Derived: filter → sort → paginate -----

  const filtered = React.useMemo(() => {
    const keys = globalFilter?.keys?.map(String) ?? [];
    const q = (filterValue ?? "").trim().toLowerCase();
    if (!q || keys.length === 0) return data;
    return data.filter((row) =>
      keys.some((k) => {
        const val = (row as any)[k];
        if (val == null) return false;
        return String(val).toLowerCase().includes(q);
      })
    );
  }, [data, globalFilter?.keys, filterValue]);

  const sorted = React.useMemo(() => {
    if (!sortBy) return filtered;
    const { key, direction } = sortBy;
    const arr = [...filtered];
    // Stable sort with tiebreaker on derived id to keep determinism
    arr.sort((a, b) => {
      const ia = getRowId(a, data.indexOf(a), rowId);
      const ib = getRowId(b, data.indexOf(b), rowId);
      const va = (a as any)[key];
      const vb = (b as any)[key];
      const c = cmp(va, vb);
      if (c !== 0) return direction === "asc" ? c : -c;
      return ia.localeCompare(ib);
    });
    return arr;
  }, [filtered, sortBy, data, rowId]);

  const total = totalExternal ?? sorted.length;

  const paged = React.useMemo(() => {
    const start = pageIndex * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [sorted, pageIndex, pageSize]);

  // Keep current page in range when deps change
  React.useEffect(() => {
    const maxIndex = Math.max(0, Math.ceil(total / pageSize) - 1);
    if (pageIndex > maxIndex) setPageIndex(0);
  }, [total, pageSize, pageIndex, setPageIndex]);

  // ----- Visible IDs + selection meta -----
  const visibleIds = React.useMemo(() => {
    const base = pageIndex * pageSize;
    return paged.map((row, i) => getRowId(row, base + i, rowId));
  }, [paged, pageIndex, pageSize, rowId]);

  const allSelectableIds = React.useMemo(() => {
    if (selectionScope === "all") {
      // Apply to all *filtered* rows, in keeping with user expectations
      return filtered.map((row, i) => getRowId(row, i, rowId));
    }
    return visibleIds;
  }, [selectionScope, filtered, visibleIds, rowId]);

  const allSelected =
    allSelectableIds.length > 0 &&
    allSelectableIds.every((id) => selectedIds.includes(id));
  const someSelected = allSelectableIds.some((id) => selectedIds.includes(id));

  const toggleBulk = React.useCallback(
    (checked: boolean | "indeterminate") => {
      if (checked === true) {
        const union = Array.from(new Set([...selectedIds, ...allSelectableIds]));
        setSelected(union);
      } else {
        const remaining = selectedIds.filter((id) => !allSelectableIds.includes(id));
        setSelected(remaining);
      }
    },
    [selectedIds, setSelected, allSelectableIds]
  );

  // ----- Sorting toggler -----
  const toggleSort = React.useCallback(
    (key: string, sortable?: boolean) => {
      if (!sortable) return;
      // cycle: none → asc → desc → none
      if (!sortBy || sortBy.key !== key) return setSort({ key, direction: "asc" });
      if (sortBy.direction === "asc") return setSort({ key, direction: "desc" });
      return setSort(null);
    },
    [sortBy, setSort]
  );

  // ----- Derived state flags -----
  const hasData = data && data.length > 0;
  const showLoading = !!loading;
  const showError = !!error;
  const showEmpty = !showLoading && !showError && (!hasData || paged.length === 0);

  // ----- Keyboard helpers for row activation (Enter/Space) -----
  const onRowKeyDown = React.useCallback(
    (
      e: React.KeyboardEvent<HTMLTableRowElement>,
      row: T,
      absoluteIndex: number,
      id: string
    ) => {
      if (!onRowClick) return;
      const k = e.key;
      if (k === "Enter" || k === " ") {
        e.preventDefault();
        onRowClick(row, absoluteIndex, id);
      }
    },
    [onRowClick]
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <motion.div
      className={cn("w-full rounded-2xl border bg-card p-4 shadow-sm", className)}
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      {...rest}
    >
      {/* Header */}
      {(title || description || globalFilter) && (
        <div className="mb-3 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div className="min-w-0">
            {title && (
              <h3 className="truncate text-base font-semibold" data-testid="table-title">
                {title}
              </h3>
            )}
            {description && (
              <p className="text-sm text-muted-foreground" data-testid="table-description">
                {description}
              </p>
            )}
          </div>

          {globalFilter && (
            <div className="w-full max-w-xs">
              <Input
                data-testid="table-filter"
                aria-label={globalFilter.placeholder ?? "Filter"}
                placeholder={globalFilter.placeholder ?? "Filter…"}
                value={filterValue}
                onChange={(e) => changeFilter(e.target.value)}
              />
            </div>
          )}
        </div>
      )}

      {/* Body states */}
      {showLoading && <LoadingState />}
      {showError && <ErrorState message={error ?? undefined} />}
      {showEmpty && <EmptyState message={emptyMessage} />}

      {/* Table */}
      {!showLoading && !showError && !showEmpty && (
        <div className="w-full overflow-x-auto rounded-xl border" role="region" aria-label={title ?? "Table"}>
          <UiTable>
            <TableHeader>
              <TableRow>
                {selectable && (
                  <TableHead className="w-[44px]">
                    <Checkbox
                      aria-label={selectionScope === "all" ? "Select all filtered" : "Select all visible"}
                      checked={allSelected ? true : someSelected ? "indeterminate" : false}
                      onCheckedChange={toggleBulk}
                      data-testid="table-select-all"
                    />
                  </TableHead>
                )}

                {columns.map((col, ci) => {
                  const isSorted = sortBy?.key === String(col.key);
                  const sortDir = isSorted ? sortBy?.direction : undefined;
                  const alignClass = getCellAlignClass(col.align);
                  const widthStyle = widthStyleOf(col.width);

                  // aria-sort states: "none" | "ascending" | "descending" | "other"
                  const ariaSort: React.AriaAttributes["aria-sort"] = col.sortable
                    ? isSorted
                      ? sortDir === "asc"
                        ? "ascending"
                        : "descending"
                      : "none"
                    : undefined;

                  return (
                    <TableHead
                      key={`h-${ci}-${String(col.key)}`}
                      style={widthStyle}
                      className={cn(
                        "select-none whitespace-nowrap",
                        col.headerClassName,
                        col.sortable && "cursor-pointer",
                        alignClass
                      )}
                      onClick={() => toggleSort(String(col.key), col.sortable)}
                      aria-sort={ariaSort}
                      scope="col"
                      role="columnheader"
                      tabIndex={col.sortable ? 0 : -1}
                      onKeyDown={(e) => {
                        if (!col.sortable) return;
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          toggleSort(String(col.key), col.sortable);
                        }
                      }}
                    >
                      <div className="flex items-center gap-1">
                        <span className="truncate">{col.header ?? String(col.key)}</span>
                        {col.sortable && (
                          <>
                            {!isSorted && <ArrowUpDown className="h-3.5 w-3.5 opacity-60" aria-hidden />}
                            {isSorted && sortDir === "asc" && <ArrowUp className="h-3.5 w-3.5" aria-hidden />}
                            {isSorted && sortDir === "desc" && <ArrowDown className="h-3.5 w-3.5" aria-hidden />}
                          </>
                        )}
                      </div>
                    </TableHead>
                  );
                })}

                {rowActions && <TableHead className="w-[44px] text-right" />}
              </TableRow>
            </TableHeader>

            <TableBody>
              {paged.map((row, ri) => {
                const absoluteIndex = pageIndex * pageSize + ri;
                const id = getRowId(row, absoluteIndex, rowId);
                const selected = selectedIds.includes(id);

                return (
                  <TableRow
                    key={id}
                    className={cn(onRowClick && "cursor-pointer hover:bg-muted/40")}
                    onClick={(e) => {
                      // prevent row click when clicking interactive elements
                      const target = e.target as HTMLElement;
                      if (target.closest("button, a, input, label, [role='menu'], [role='menuitem']")) return;
                      onRowClick?.(row, absoluteIndex, id);
                    }}
                    onKeyDown={(e) => onRowKeyDown(e, row, absoluteIndex, id)}
                    tabIndex={onRowClick ? 0 : -1}
                    role="row"
                    aria-selected={selectable ? selected : undefined}
                    data-testid={`table-row-${id}`}
                  >
                    {selectable && (
                      <TableCell className="w-[44px]">
                        <Checkbox
                          aria-label={`Select row ${id}`}
                          checked={selected}
                          onCheckedChange={(checked) => {
                            if (checked) setSelected([...selectedIds, id]);
                            else setSelected(selectedIds.filter((x) => x !== id));
                          }}
                          onClick={(e) => e.stopPropagation()}
                          data-testid={`table-select-${id}`}
                        />
                      </TableCell>
                    )}

                    {columns.map((col, ci) => {
                      // Resolve raw value through accessor/key
                      const raw = getColumnValue(col, row, absoluteIndex);

                      // Resolve final display: render > format > raw
                      const content =
                        typeof col.render === "function"
                          ? col.render(row, absoluteIndex)
                          : typeof col.format === "function"
                          ? col.format(raw, row, absoluteIndex)
                          : raw;

                      const alignClass = getCellAlignClass(col.align);
                      const widthStyle = widthStyleOf(col.width);

                      return (
                        <TableCell
                          key={`c-${id}-${ci}-${String(col.key)}`}
                          style={widthStyle}
                          className={cn("whitespace-nowrap", alignClass, col.className)}
                          aria-label={col.ariaLabel}
                          role="cell"
                        >
                          {content}
                        </TableCell>
                      );
                    })}

                    {rowActions && (
                      <TableCell className="w-[44px] text-right">
                        <div className="flex items-center justify-end">
                          <Button
                            size="icon"
                            variant="ghost"
                            onClick={(e) => e.stopPropagation()}
                            aria-label={`Row actions for ${id}`}
                          >
                            <MoreVertical className="h-4 w-4" aria-hidden />
                          </Button>
                          {/* Inline actions (consumer-provided) */}
                          <div onClick={(e) => e.stopPropagation()}>{rowActions(row, absoluteIndex)}</div>
                        </div>
                      </TableCell>
                    )}
                  </TableRow>
                );
              })}
            </TableBody>
          </UiTable>
        </div>
      )}

      {/* Footer / Pagination */}
      {!showLoading && !showError && !showEmpty && (
        <div className="mt-3 flex flex-col items-center justify-between gap-2 sm:flex-row">
          <div className="text-xs text-muted-foreground" aria-live="polite">
            Showing{" "}
            <span className="font-medium">
              {Math.min(pageIndex * pageSize + 1, total)}–{Math.min((pageIndex + 1) * pageSize, total)}
            </span>{" "}
            of <span className="font-medium">{total}</span>
          </div>

          <div className="flex items-center gap-2">
            {/* Rows per page (hidden on very small screens) */}
            <div className="hidden items-center gap-2 sm:flex">
              <label htmlFor="rows-per-page" className="text-xs text-muted-foreground">
                Rows per page
              </label>
              <select
                id="rows-per-page"
                className="h-8 rounded-md border bg-background px-2 text-sm"
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value))}
                aria-label="Rows per page"
                data-testid="table-rows-per-page"
              >
                {pageSizeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            {/* Pager */}
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                aria-label="First page"
                onClick={() => setPageIndex(0)}
                disabled={pageIndex === 0}
                data-testid="table-page-first"
              >
                <ChevronsLeft className="h-4 w-4" aria-hidden />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Previous page"
                onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
                disabled={pageIndex === 0}
                data-testid="table-page-prev"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Next page"
                onClick={() =>
                  setPageIndex(Math.min(Math.ceil(total / pageSize) - 1, pageIndex + 1))
                }
                disabled={pageIndex >= Math.ceil(total / pageSize) - 1}
                data-testid="table-page-next"
              >
                <ChevronRight className="h-4 w-4" aria-hidden />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Last page"
                onClick={() => setPageIndex(Math.max(0, Math.ceil(total / pageSize) - 1))}
                disabled={pageIndex >= Math.ceil(total / pageSize) - 1}
                data-testid="table-page-last"
              >
                <ChevronsRight className="h-4 w-4" aria-hidden />
              </Button>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default Table;
