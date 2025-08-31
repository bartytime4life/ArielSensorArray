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
}

export const Tabs: React.FC<TabsProps> = ({
  tabs,
  defaultTab,
  value,
  onTabChange,
  className,
}) => {
  const isControlled = value !== undefined;
  const [internalTab, setInternalTab] = React.useState<string>(
    defaultTab ?? tabs[0]?.id
  );

  const activeTab = isControlled ? value : internalTab;

  const handleChange = (id: string) => {
    if (!isControlled) setInternalTab(id);
    onTabChange?.(id);
  };

  return (
    <div className={clsx("w-full", className)}>
      {/* Tab list */}
      <div
        role="tablist"
        aria-label="Tabs"
        className="flex gap-2 border-b border-gray-200 dark:border-gray-700"
      >
        {tabs.map((tab) => (
          <button
            key={tab.id}
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`tab-panel-${tab.id}`}
            id={`tab-${tab.id}`}
            disabled={tab.disabled}
            onClick={() => handleChange(tab.id)}
            className={clsx(
              "px-4 py-2 text-sm font-medium rounded-t-lg focus:outline-none focus-visible:ring-2",
              activeTab === tab.id
                ? "bg-white dark:bg-gray-900 border border-b-0 border-gray-300 dark:border-gray-600 text-blue-600 dark:text-blue-400"
                : "text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white",
              tab.disabled &&
                "cursor-not-allowed opacity-50 hover:text-gray-400 dark:hover:text-gray-500"
            )}
          >
            {tab.label}
          </button>
        ))}
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
