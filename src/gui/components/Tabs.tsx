// src/gui/components/Tabs.tsx
// ============================================================================
// 📑 Tabs Component — SpectraMind V50 GUI (CLI-first, GUI-optional)
// ----------------------------------------------------------------------------
// Responsibilities
//   • Render a tabbed navigation interface with accessible keyboard controls
//   • Support controlled/uncontrolled usage (selected tab via props or internal)
//   • Provide Tailwind + shadcn/ui-like styling (rounded, shadows, focus rings)
//   • Work with any ReactNode as tab content
//
// A11y & Keyboard (WAI-ARIA Tabs pattern):
//   • role="tablist" with aria-orientation="horizontal"
//   • role="tab" per tab; role="tabpanel" per panel
//   • roving tabindex: only one tab is tabbable (tabIndex=0), others -1
//   • ArrowLeft/ArrowRight move focus among tabs (skip disabled)
//   • Home/End jump focus to first/last enabled tab
//   • Enter/Space activate focused tab (fires onTabChange; in uncontrolled, selects)
//
// Example:
//   <Tabs
//     tabs={[
//       { id: "umap", label: "UMAP", content: <UmapPlot /> },
//       { id: "tsne", label: "t-SNE", content: <TsnePlot /> },
//     ]}
//     defaultTab="umap"
//     onTabChange={(id) => console.log("Switched to:", id)}
//   />
// ============================================================================

import * as React from "react";
import clsx from "clsx";

export interface TabItem {
  id: string;
  label: string;
  content: React.ReactNode;
  disabled?: boolean;
}

export interface TabsProps {
  tabs: TabItem[];
  /** Default selected tab id (uncontrolled) */
  defaultTab?: string;
  /** Controlled selected tab id */
  value?: string;
  /** Callback when tab changes */
  onTabChange?: (id: string) => void;
  className?: string;
  /** Optional aria-orientation (default: 'horizontal') */
  ariaOrientation?: "horizontal" | "vertical";
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  defaultTab,
  value,
  onTabChange,
  className,
  ariaOrientation = "horizontal",
}) => {
  const isControlled = value !== undefined;

  // Resolve initial tab (first enabled if unspecified)
  const firstEnabledId = React.useMemo(() => {
    const t = tabs.find((x) => !x.disabled);
    return t?.id;
  }, [tabs]);

  const initialTab = defaultTab ?? firstEnabledId ?? tabs[0]?.id;

  // Selected tab state (for uncontrolled)
  const [internalTab, setInternalTab] = React.useState<string | undefined>(initialTab);
  const activeTab = isControlled ? value : internalTab;

  // Refs to tab buttons for focus management
  const tabRefs = React.useRef<(HTMLButtonElement | null)[]>([]);

  // Focused index (roving tabindex)
  const [focusedIndex, setFocusedIndex] = React.useState<number>(() => {
    const i = tabs.findIndex((t) => t.id === activeTab && !t.disabled);
    if (i >= 0) return i;
    const j = tabs.findIndex((t) => !t.disabled);
    return j >= 0 ? j : 0;
  });

  // Sync focusedIndex when activeTab changes (e.g., controlled mode updates)
  React.useEffect(() => {
    if (!activeTab) return;
    const i = tabs.findIndex((t) => t.id === activeTab && !t.disabled);
    if (i >= 0) setFocusedIndex(i);
  }, [activeTab, tabs]);

  const enabledIndexes = React.useMemo(
    () => tabs.map((t, i) => (!t.disabled ? i : -1)).filter((i) => i >= 0),
    [tabs]
  );

  const getNextEnabled = (start: number, dir: 1 | -1) => {
    if (enabledIndexes.length === 0) return start;
    let idx = start;
    for (let step = 0; step < tabs.length; step++) {
      idx = (idx + dir + tabs.length) % tabs.length;
      if (!tabs[idx]?.disabled) return idx;
    }
    return start;
  };

  const getEdgeEnabled = (edge: "first" | "last") => {
    if (enabledIndexes.length === 0) return focusedIndex;
    return edge === "first" ? enabledIndexes[0] : enabledIndexes[enabledIndexes.length - 1];
  };

  const focusTab = (index: number) => {
    setFocusedIndex(index);
    const el = tabRefs.current[index];
    el?.focus();
  };

  const handleActivate = (index: number) => {
    const tab = tabs[index];
    if (!tab || tab.disabled) return;
    if (!isControlled) setInternalTab(tab.id);
    onTabChange?.(tab.id);
  };

  const handleClick = (index: number) => {
    handleActivate(index);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (tabs.length === 0) return;
    const key = e.key;
    let nextIndex = focusedIndex;

    if (key === "ArrowRight" && ariaOrientation === "horizontal") {
      e.preventDefault();
      nextIndex = getNextEnabled(focusedIndex, 1);
      focusTab(nextIndex);
      return;
    }
    if (key === "ArrowLeft" && ariaOrientation === "horizontal") {
      e.preventDefault();
      nextIndex = getNextEnabled(focusedIndex, -1);
      focusTab(nextIndex);
      return;
    }
    if (key === "ArrowDown" && ariaOrientation === "vertical") {
      e.preventDefault();
      nextIndex = getNextEnabled(focusedIndex, 1);
      focusTab(nextIndex);
      return;
    }
    if (key === "ArrowUp" && ariaOrientation === "vertical") {
      e.preventDefault();
      nextIndex = getNextEnabled(focusedIndex, -1);
      focusTab(nextIndex);
      return;
    }
    if (key === "Home") {
      e.preventDefault();
      focusTab(getEdgeEnabled("first"));
      return;
    }
    if (key === "End") {
      e.preventDefault();
      focusTab(getEdgeEnabled("last"));
      return;
    }
    if (key === "Enter" || key === " ") {
      e.preventDefault();
      handleActivate(focusedIndex);
      return;
    }
  };

  return (
    <div className={clsx("w-full", className)}>
      {/* Tab list */}
      <div
        role="tablist"
        aria-label="Tabs"
        aria-orientation={ariaOrientation}
        className="flex gap-2 border-b border-gray-200 dark:border-gray-700"
        onKeyDown={onKeyDown}
      >
        {tabs.map((tab, i) => {
          const selected = activeTab === tab.id;
          const isDisabled = !!tab.disabled;
          // Roving tabindex: only the focused tab is tabbable
          const tabIndex = i === focusedIndex ? 0 : -1;

          return (
            <button
              key={tab.id}
              ref={(el) => (tabRefs.current[i] = el)}
              role="tab"
              aria-selected={selected}
              aria-controls={`tab-panel-${tab.id}`}
              id={`tab-${tab.id}`}
              tabIndex={tabIndex}
              disabled={isDisabled}
              onClick={() => handleClick(i)}
              className={clsx(
                "px-4 py-2 text-sm font-medium rounded-t-lg focus:outline-none focus-visible:ring-2",
                selected
                  ? "bg-white dark:bg-gray-900 border border-b-0 border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-400"
                  : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white",
                isDisabled &&
                  "cursor-not-allowed opacity-50 hover:text-gray-400 dark:hover:text-gray-500"
              )}
            >
              {tab.label}
            </button>
          );
        })}
      </div>

      {/* Tab panels */}
      <div className="p-4 border border-t-0 border-gray-300 dark:border-gray-600 rounded-b-lg bg-white dark:bg-gray-900">
        {tabs.map((tab) => (
          <div
            key={tab.id}
            role="tabpanel"
            id={`tab-panel-${tab.id}`}
            aria-labelledby={`tab-${tab.id}`}
            hidden={activeTab !== tab.id}
            className="focus:outline-none"
          >
            {tab.content}
          </div>
        ))}
      </div>
    </div>
  );
};

export default Tabs;
