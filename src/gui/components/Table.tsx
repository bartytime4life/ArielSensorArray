// src/gui/components/Table.tsx
// =============================================================================
// 🧬 SpectraMind V50 — Reusable Data Table (React + Tailwind + shadcn/ui)
// -----------------------------------------------------------------------------
// Purpose
//   A fast, declarative table for diagnostics dashboards and reports. Supports:
//   • Column definitions with optional custom cell renderers
//   • Client-side sorting (per-column, ASC/DESC)
//   • Optional pagination (controlled or internal)
//   • Optional row selection via checkboxes
//   • Optional global text filtering over selected fields
//   • Loading / empty / error states
//   • Optional row actions and row click handler
//
// Design Notes
//   • No heavy table libs to keep bundle small; shadcn/ui primitives only.
//   • Tailwind for styling. Dark-mode friendly via Tailwind tokens.
//   • Deterministic behavior in SSR/CSR.
//   • Composable and safe defaults; everything optional.
//
// Example
//   <Table
//     title="Diagnostics per Planet"
//     description="GLL, RMSE, violations"
//     data={rows}
//     columns={[
//       { key: "planet_id", header: "Planet", sortable: true },
//       { key: "gll", header: "GLL", sortable: true, align: "right",
//         format: (v) => v.toFixed(4) },
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
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUp, ArrowDown, ArrowUpDown, MoreVertical } from "lucide-react";
import { motion } from "framer-motion";

// -----------------------------------------------------------------------------
// Types
// -----------------------------------------------------------------------------

type Align = "left" | "center" | "right";

export interface ColumnDef<T extends Record<string, any>> {
  key: keyof T | string;
  header?: React.ReactNode;
  sortable?: boolean;
  align?: Align;
  // Cell formatting or custom renderer
  format?: (value: any, row: T, rowIndex: number) => React.ReactNode;
  render?: (row: T, rowIndex: number) => React.ReactNode;
  width?: string | number;
  className?: string;
  headerClassName?: string;
  // For accessibility/testing
  ariaLabel?: string;
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

export interface TableProps<T extends Record<string, any>> extends React.HTMLAttributes<HTMLDivElement> {
  // Chrome
  title?: string;
  description?: string;
  className?: string;
  // Data model
  data: T[];
  columns: ColumnDef<T>[];
  rowId?: (row: T) => string; // default: index-based
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
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

const LoadingState: React.FC = () => (
  <div className="h-40 w-full animate-pulse rounded-xl bg-muted/50" />
);

const ErrorState: React.FC<{ message?: string }> = ({ message }) => (
  <div className="flex h-40 items-center justify-center text-sm text-red-600 dark:text-red-400">
    {message ?? "Failed to load data."}
  </div>
);

const EmptyState: React.FC<{ message?: string }> = ({ message }) => (
  <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
    {message ?? "No rows to display."}
  </div>
);

function getRowId<T extends Record<string, any>>(row: T, idx: number, rowId?: (r: T) => string) {
  try {
    if (rowId) return rowId(row);
  } catch (_) {}
  return String(idx);
}

function cmp(a: any, b: any) {
  if (a == null && b == null) return 0;
  if (a == null) return -1;
  if (b == null) return 1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b));
}

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

// -----------------------------------------------------------------------------
// Component
// -----------------------------------------------------------------------------

export function Table<T extends Record<string, any>>(props: TableProps<T>) {
  const {
    title,
    description,
    className,
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
    // Filtering
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
    ...rest
  } = props;

  // ----- Sorting (controlled / uncontrolled)
  const [sortByInternal, setSortByInternal] = React.useState<{ key: string; direction: "asc" | "desc" } | null>(
    defaultSortBy ?? null
  );
  const sortBy = sortByProp !== undefined ? sortByProp : sortByInternal;

  const setSort = React.useCallback(
    (next: { key: string; direction: "asc" | "desc" } | null) => {
      if (onSortChange) onSortChange(next);
      else setSortByInternal(next);
    },
    [onSortChange]
  );

  // ----- Selection (controlled / uncontrolled)
  const [selectedIdsInternal, setSelectedIdsInternal] = React.useState<string[]>(defaultSelectedIds ?? []);
  const selectedIds = selectedIdsProp !== undefined ? selectedIdsProp : selectedIdsInternal;
  const setSelected = React.useCallback(
    (ids: string[]) => {
      if (onSelectionChange) onSelectionChange(ids);
      else setSelectedIdsInternal(ids);
    },
    [onSelectionChange]
  );

  // ----- Filter (controlled / uncontrolled)
  const [filterValueInternal, setFilterValueInternal] = React.useState(globalFilter?.value ?? "");
  const filterValue = globalFilter?.value !== undefined ? globalFilter.value : filterValueInternal;
  const changeFilter = React.useCallback(
    (v: string) => {
      if (onGlobalFilterChange) onGlobalFilterChange(v);
      else setFilterValueInternal(v);
    },
    [onGlobalFilterChange]
  );

  // ----- Pagination (controlled / uncontrolled)
  const [pageIndexInternal, setPageIndexInternal] = React.useState(0);
  const [pageSizeInternal, setPageSizeInternal] = React.useState(defaultPageSize);

  const pageIndex =
    paginationProp?.pageIndex ?? pageIndexInternal;
  const pageSize =
    paginationProp?.pageSize ?? pageSizeInternal;
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

  // ----- Derived rows (filter + sort)
  const filtered = React.useMemo(() => {
    const keys = globalFilter?.keys?.map(String) ?? [];
    const query = (filterValue ?? "").trim().toLowerCase();
    if (!query) return data;
    if (keys.length === 0) return data;
    return data.filter((row) =>
      keys.some((k) => {
        const val = (row as any)[k];
        if (val == null) return false;
        return String(val).toLowerCase().includes(query);
      })
    );
  }, [data, globalFilter?.keys, filterValue]);

  const sorted = React.useMemo(() => {
    if (!sortBy) return filtered;
    const { key, direction } = sortBy;
    const arr = [...filtered];
    arr.sort((a, b) => {
      const va = (a as any)[key];
      const vb = (b as any)[key];
      const c = cmp(va, vb);
      return direction === "asc" ? c : -c;
    });
    return arr;
  }, [filtered, sortBy]);

  const total = totalExternal ?? sorted.length;

  // Page slicing only if not controlled via external pagination total (still show slice locally)
  const paged = React.useMemo(() => {
    const start = pageIndex * pageSize;
    return sorted.slice(start, start + pageSize);
  }, [sorted, pageIndex, pageSize]);

  // Reset page when data/filter/sort changes and current page is out-of-bounds
  React.useEffect(() => {
    const maxIndex = Math.max(0, Math.ceil(total / pageSize) - 1);
    if (pageIndex > maxIndex) setPageIndex(0);
  }, [total, pageSize, pageIndex, setPageIndex]);

  // Select-all checkbox state (based on visible page)
  const visibleIds = React.useMemo(
    () => paged.map((row, idx) => getRowId(row, idx + pageIndex * pageSize, rowId)),
    [paged, pageIndex, pageSize, rowId]
  );
  const allVisibleSelected = visibleIds.length > 0 && visibleIds.every((id) => selectedIds.includes(id));
  const someVisibleSelected = visibleIds.some((id) => selectedIds.includes(id));

  const toggleAllVisible = React.useCallback(
    (checked: boolean | "indeterminate") => {
      if (checked === true) {
        const union = Array.from(new Set([...selectedIds, ...visibleIds]));
        setSelected(union);
      } else {
        const remaining = selectedIds.filter((id) => !visibleIds.includes(id));
        setSelected(remaining);
      }
    },
    [selectedIds, setSelected, visibleIds]
  );

  // Sorting toggler
  const toggleSort = React.useCallback(
    (key: string, sortable?: boolean) => {
      if (!sortable) return;
      if (!sortBy || sortBy.key !== key) return setSort({ key, direction: "asc" });
      if (sortBy.direction === "asc") return setSort({ key, direction: "desc" });
      return setSort(null); // third click clears sorting
    },
    [sortBy, setSort]
  );

  // Render
  const hasData = data && data.length > 0;
  const showLoading = loading;
  const showError = !!error;
  const showEmpty = !showLoading && !showError && (!hasData || paged.length === 0);

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
            {title && <h3 className="truncate text-base font-semibold">{title}</h3>}
            {description && <p className="text-sm text-muted-foreground">{description}</p>}
          </div>
          {globalFilter && (
            <div className="w-full max-w-xs">
              <Input
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
        <div className="w-full overflow-x-auto rounded-xl border">
          <UiTable>
            <TableHeader>
              <TableRow>
                {selectable && (
                  <TableHead className="w-[44px]">
                    <Checkbox
                      aria-label="Select all"
                      checked={allVisibleSelected ? true : someVisibleSelected ? "indeterminate" : false}
                      onCheckedChange={toggleAllVisible}
                    />
                  </TableHead>
                )}
                {columns.map((col, ci) => {
                  const isSorted = sortBy?.key === String(col.key);
                  const sortDir = isSorted ? sortBy?.direction : undefined;
                  const alignClass = getCellAlignClass(col.align);
                  const widthStyle =
                    col.width !== undefined
                      ? typeof col.width === "number"
                        ? { width: `${col.width}px` }
                        : { width: col.width }
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
                    >
                      <div className="flex items-center gap-1">
                        <span className="truncate">{col.header ?? String(col.key)}</span>
                        {col.sortable && (
                          <>
                            {!isSorted && <ArrowUpDown className="h-3.5 w-3.5 opacity-60" />}
                            {isSorted && sortDir === "asc" && <ArrowUp className="h-3.5 w-3.5" />}
                            {isSorted && sortDir === "desc" && <ArrowDown className="h-3.5 w-3.5" />}
                          </>
                        )}
                      </div>
                    </TableHead>
                  );
                })}
                {rowActions && <TableHead className="w-[44px] text-right"></TableHead>}
              </TableRow>
            </TableHeader>

            <TableBody>
              {paged.map((row, ri) => {
                const id = getRowId(row, ri + pageIndex * pageSize, rowId);
                const selected = selectedIds.includes(id);

                return (
                  <TableRow
                    key={id}
                    className={cn(onRowClick && "cursor-pointer hover:bg-muted/40")}
                    onClick={(e) => {
                      // prevent row click when clicking interactive elements
                      const target = e.target as HTMLElement;
                      if (target.closest("button, a, input, label")) return;
                      onRowClick?.(row, ri + pageIndex * pageSize, id);
                    }}
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
                        />
                      </TableCell>
                    )}
                    {columns.map((col, ci) => {
                      const raw = (row as any)[col.key as any];
                      const content =
                        typeof col.render === "function"
                          ? col.render(row, ri + pageIndex * pageSize)
                          : typeof col.format === "function"
                          ? col.format(raw, row, ri + pageIndex * pageSize)
                          : raw;

                      const alignClass = getCellAlignClass(col.align);

                      const widthStyle =
                        col.width !== undefined
                          ? typeof col.width === "number"
                            ? { width: `${col.width}px` }
                            : { width: col.width }
                          : undefined;

                      return (
                        <TableCell
                          key={`c-${id}-${ci}-${String(col.key)}`}
                          style={widthStyle}
                          className={cn("whitespace-nowrap", alignClass, col.className)}
                          aria-label={col.ariaLabel}
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
                            <MoreVertical className="h-4 w-4" />
                          </Button>
                          {/* Action portal */}
                          <div className="sr-only" aria-hidden>
                            {/* screen-reader hidden; consumers render menus/popovers elsewhere */}
                          </div>
                          {/* Inline actions (optional) */}
                          <div onClick={(e) => e.stopPropagation()}>{rowActions(row, ri + pageIndex * pageSize)}</div>
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
          <div className="text-xs text-muted-foreground">
            Showing{" "}
            <span className="font-medium">
              {Math.min(pageIndex * pageSize + 1, total)}–{Math.min((pageIndex + 1) * pageSize, total)}
            </span>{" "}
            of <span className="font-medium">{total}</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 sm:flex">
              <span className="text-xs text-muted-foreground">Rows per page</span>
              <select
                className="h-8 rounded-md border bg-background px-2 text-sm"
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value))}
                aria-label="Rows per page"
              >
                {pageSizeOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="icon"
                aria-label="First page"
                onClick={() => setPageIndex(0)}
                disabled={pageIndex === 0}
              >
                <ChevronsLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Previous page"
                onClick={() => setPageIndex(Math.max(0, pageIndex - 1))}
                disabled={pageIndex === 0}
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Next page"
                onClick={() => setPageIndex(Math.min(Math.ceil(total / pageSize) - 1, pageIndex + 1))}
                disabled={pageIndex >= Math.ceil(total / pageSize) - 1}
              >
                <ChevronRight className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Last page"
                onClick={() => setPageIndex(Math.max(0, Math.ceil(total / pageSize) - 1))}
                disabled={pageIndex >= Math.ceil(total / pageSize) - 1}
              >
                <ChevronsRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default Table;
