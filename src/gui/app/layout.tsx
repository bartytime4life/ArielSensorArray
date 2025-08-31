// src/gui/app/layout.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — Global App Layout (Sidebar + Topbar + Content)
// ----------------------------------------------------------------------------
// Purpose
//   • Provide a persistent navigation shell around all feature routes.
//   • Keep UX lightweight, accessible, and responsive (Tailwind + shadcn/ui).
//   • Mirror CLI-first diagnostics: Dashboard, Diagnostics, UMAP, t-SNE, SHAP,
//     Symbolic, FFT, Calibration — each view renders CLI-produced artifacts.
//   • Offer a command palette (⌘/Ctrl + K) to jump between sections.
//   • Persist theme preference (light/dark/system) with system-change tracking.
//   • Be framework-agnostic about analytics: browser does zero analytics.
// ----------------------------------------------------------------------------
// Notes
//   • Expects shadcn/ui primitives under "@/components/ui/\*" and Tailwind set up.
//   • Uses react-router v6+ (NavLink/Link).
//   • Uses lucide-react icons.
//   • Theming works by toggling the `dark` class on <html> and setting color-scheme.
//   • Safe for client-only apps; if SSR is used, guards avoid hydration issues.
// ============================================================================

import \* as React from "react";
import { NavLink, Link, useLocation } from "react-router-dom";

import {
Tooltip,
TooltipContent,
TooltipProvider,
TooltipTrigger,
} from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { Button } from "@/components/ui/button";
import {
Sheet,
SheetContent,
SheetTrigger,
SheetHeader,
SheetTitle,
SheetDescription,
} from "@/components/ui/sheet";
import {
Command,
CommandEmpty,
CommandGroup,
CommandInput,
CommandItem,
CommandList,
CommandSeparator,
} from "@/components/ui/command";
import { Input } from "@/components/ui/input"; // used for quick filter in palette footer (optional)

import {
LayoutDashboard,
ActivitySquare,
Network,
ScatterChart,
Atom,
Sigma,
Waves,
Wrench,
Menu,
Github,
ExternalLink,
Sun,
Moon,
Monitor,
ChevronLeft,
ChevronRight,
Search,
} from "lucide-react";

// ----------------------------------------------------------------------------
// Constants & Types
// ----------------------------------------------------------------------------

const APP\_TITLE = "SpectraMind V50";
const THEME\_STORAGE\_KEY = "spectramind-theme";
const SIDEBAR\_STATE\_KEY = "spectramind-sidebar"; // "expanded" | "collapsed"

type ThemeMode = "light" | "dark" | "system";

type NavItem = {
label: string;
to: string;
icon: React.ElementType;
desc: string;
};

const NAV\_ITEMS: NavItem\[] = \[
{ label: "Dashboard",    to: "/dashboard",    icon: LayoutDashboard, desc: "Overview, recent runs, quick links" },
{ label: "Diagnostics",  to: "/diagnostics",  icon: ActivitySquare,  desc: "GLL summaries, calibration checks, reports" },
{ label: "UMAP",         to: "/umap",         icon: Network,         desc: "Latent UMAP explorer with overlays & links" },
{ label: "t-SNE",        to: "/tsne",         icon: ScatterChart,    desc: "Interactive t-SNE projection diagnostics" },
{ label: "SHAP",         to: "/shap",         icon: Atom,            desc: "Explainability: SHAP overlays & barplots" },
{ label: "Symbolic",     to: "/symbolic",     icon: Sigma,           desc: "Neuro-symbolic rules, violations, influence" },
{ label: "FFT",          to: "/fft",          icon: Waves,           desc: "FFT & autocorr analysis with molecular regions" },
{ label: "Calibration",  to: "/calibration",  icon: Wrench,          desc: "Run/view calibration pipeline & artifacts" },
];

// ----------------------------------------------------------------------------
// Theming
// ----------------------------------------------------------------------------

/\*\*

* Safely read from localStorage in a way that won't explode under SSR.
  \*/
  function lsGet(key: string): string | null {
  try {
  return typeof window !== "undefined" ? window\.localStorage.getItem(key) : null;
  } catch {
  return null;
  }
  }

/\*\*

* Safely write to localStorage in a way that won't explode under SSR/private modes.
  \*/
  function lsSet(key: string, value: string) {
  try {
  if (typeof window !== "undefined") window\.localStorage.setItem(key, value);
  } catch {
  // ignore
  }
  }

/\*\*

* Detect if system prefers dark. Guard for SSR.
  \*/
  function getSystemPrefersDark(): boolean {
  if (typeof window === "undefined" || !window\.matchMedia) return false;
  try {
  return window\.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
  return false;
  }
  }

/\*\*

* Apply theme to document root.
  \*/
  function applyTheme(mode: ThemeMode) {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  if (mode === "system") {
  const dark = getSystemPrefersDark();
  root.classList.toggle("dark", dark);
  root.style.colorScheme = dark ? "dark" : "light";
  } else {
  const dark = mode === "dark";
  root.classList.toggle("dark", dark);
  root.style.colorScheme = dark ? "dark" : "light";
  }
  lsSet(THEME\_STORAGE\_KEY, mode);
  }

/\*\*

* Subscribe to system theme changes if mode === "system".
* Returns an unsubscribe function.
  \*/
  function subscribeSystemThemeChange(onChange: () => void): () => void {
  if (typeof window === "undefined" || !window\.matchMedia) return () => {};
  const mq = window\.matchMedia("(prefers-color-scheme: dark)");
  const handler = () => onChange();
  try {
  mq.addEventListener("change", handler);
  return () => mq.removeEventListener("change", handler);
  } catch {
  // Safari <14
  // @ts-ignore
  mq.addListener(handler);
  // @ts-ignore
  return () => mq.removeListener(handler);
  }
  }

/\*\*

* Theme toggle control.
  \*/
  const ThemeToggle: React.FC = () => {
  const \[mode, setMode] = React.useState<ThemeMode>(() => {
  const stored = (lsGet(THEME\_STORAGE\_KEY) as ThemeMode) || "light";
  return stored;
  });

// Apply initially and whenever mode changes
React.useEffect(() => {
applyTheme(mode);
}, \[mode]);

// If system changes while in "system", reflect it immediately
React.useEffect(() => {
if (mode !== "system") return;
const unsub = subscribeSystemThemeChange(() => applyTheme("system"));
return unsub;
}, \[mode]);

return ( <div className="inline-flex items-center gap-1" role="group" aria-label="Theme toggle"> <TooltipProvider> <Tooltip> <TooltipTrigger asChild>
\<Button
variant={mode === "light" ? "default" : "ghost"}
size="icon"
aria-pressed={mode === "light"}
aria-label="Light theme"
onClick={() => setMode("light")}
\> <Sun className="h-4 w-4" /> </Button> </TooltipTrigger> <TooltipContent>Light</TooltipContent> </Tooltip>

```
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={mode === "dark" ? "default" : "ghost"}
          size="icon"
          aria-pressed={mode === "dark"}
          aria-label="Dark theme"
          onClick={() => setMode("dark")}
        >
          <Moon className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>Dark</TooltipContent>
    </Tooltip>

    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant={mode === "system" ? "default" : "ghost"}
          size="icon"
          aria-pressed={mode === "system"}
          aria-label="System theme"
          onClick={() => setMode("system")}
        >
          <Monitor className="h-4 w-4" />
        </Button>
      </TooltipTrigger>
      <TooltipContent>System</TooltipContent>
    </Tooltip>
  </TooltipProvider>
</div>
```

);
};

// ----------------------------------------------------------------------------
// Sidebar Link
// ----------------------------------------------------------------------------

const SidebarLink: React.FC<{
to: string;
icon: React.ElementType;
label: string;
onClick?: () => void;
collapsed?: boolean;
}> = ({ to, icon: Icon, label, onClick, collapsed }) => {
return (
\<NavLink
to={to}
onClick={onClick}
className={({ isActive }) =>
\[
"group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition",
isActive
? "bg-primary/10 text-primary"
: "text-muted-foreground hover\:bg-accent hover\:text-accent-foreground",
].join(" ")
}
aria-label={label}
\> <Icon className="h-4 w-4 shrink-0" />
\<span className={collapsed ? "sr-only" : ""}>{label}</span> </NavLink>
);
};

// ----------------------------------------------------------------------------
// Command Palette (⌘/Ctrl + K)
// ----------------------------------------------------------------------------

const CommandPalette: React.FC = () => {
const \[open, setOpen] = React.useState(false);
const \[filter, setFilter] = React.useState("");
const location = useLocation();

React.useEffect(() => {
const onKey = (e: KeyboardEvent) => {
const isMac = typeof navigator !== "undefined" && navigator.platform?.includes("Mac");
const mod = isMac ? e.metaKey : e.ctrlKey;
if (mod && e.key.toLowerCase() === "k") {
e.preventDefault();
setOpen((v) => !v);
}
};
window\.addEventListener("keydown", onKey);
return () => window\.removeEventListener("keydown", onKey);
}, \[]);

const list = React.useMemo(
() =>
NAV\_ITEMS.filter(
(n) =>
!filter.trim() ||
n.label.toLowerCase().includes(filter.toLowerCase()) ||
n.desc.toLowerCase().includes(filter.toLowerCase()) ||
n.to.toLowerCase().includes(filter.toLowerCase())
),
\[filter]
);

return (
<>
\<Button
variant="outline"
className="hidden md\:inline-flex gap-2 text-sm"
onClick={() => setOpen(true)}
aria-label="Open command palette"
\> <Search className="h-4 w-4" /> <span className="text-muted-foreground">Search or jump…</span> <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground"> <span className="text-xs">⌘</span>K </kbd> </Button>

```
  <div
    className={open ? "fixed inset-0 z-[60] flex items-start justify-center bg-black/20 p-4" : "hidden"}
    aria-hidden={!open}
    onClick={() => setOpen(false)}
  >
    <div
      className="w-full max-w-xl rounded-lg border bg-background shadow-lg"
      onClick={(e) => e.stopPropagation()}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <Command>
        <CommandInput
          placeholder="Type a page, e.g., 'shap' or 'umap'…"
          autoFocus
          onKeyDown={(e) => e.key === "Escape" && setOpen(false)}
          value={filter}
          onValueChange={setFilter}
        />
        <CommandList>
          <CommandEmpty>No results.</CommandEmpty>

          <CommandGroup heading="Navigate">
            {list.map((item) => (
              <Link key={item.to} to={item.to} onClick={() => setOpen(false)} className="block">
                <CommandItem className="cursor-pointer">
                  <item.icon className="mr-2 h-4 w-4" />
                  <span>{item.label}</span>
                </CommandItem>
              </Link>
            ))}
          </CommandGroup>

          <CommandSeparator />

          <CommandGroup heading="Info">
            <CommandItem disabled>
              <span>Current: {location.pathname}</span>
            </CommandItem>
          </CommandGroup>
        </CommandList>

        {/* Optional footer quick filter (same state as input above) */}
        <div className="flex items-center gap-2 border-t p-2">
          <Input
            value={filter}
            onChange={(e) => setFilter(e.currentTarget.value)}
            placeholder="Filter…"
            className="h-8"
            aria-label="Filter pages"
          />
          <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
            Close
          </Button>
        </div>
      </Command>
    </div>
  </div>
</>
```

);
};

// ----------------------------------------------------------------------------
// Mobile Sidebar (Sheet)
// ----------------------------------------------------------------------------

const MobileNav: React.FC<{ items: NavItem\[] }> = ({ items }) => {
const \[open, setOpen] = React.useState(false);
return ( <Sheet open={open} onOpenChange={setOpen}> <SheetTrigger asChild> <Button variant="ghost" size="icon" aria-label="Open navigation" className="md:hidden"> <Menu className="h-5 w-5" /> </Button> </SheetTrigger> <SheetContent side="left" className="p-0"> <SheetHeader className="px-6 pt-6 pb-2 text-left"> <SheetTitle className="text-base font-semibold">{APP\_TITLE}</SheetTitle> <SheetDescription>Navigate</SheetDescription> </SheetHeader> <div className="px-3 py-2">
{items.map((n) => (
\<SidebarLink
key={n.to}
to={n.to}
icon={n.icon}
label={n.label}
onClick={() => setOpen(false)}
/>
))} </div> <Separator className="my-2" /> <div className="px-3 pb-6"> <ThemeToggle /> </div> </SheetContent> </Sheet>
);
};

// ----------------------------------------------------------------------------
// Sidebar (Desktop, Collapsible)
// ----------------------------------------------------------------------------

const Sidebar: React.FC<{ items: NavItem\[] }> = ({ items }) => {
// Collapsed sidebar saves horizontal space on large dashboards.
const \[collapsed, setCollapsed] = React.useState<boolean>(() => {
const stored = lsGet(SIDEBAR\_STATE\_KEY);
return stored === "collapsed" ? true : false;
});

React.useEffect(() => {
lsSet(SIDEBAR\_STATE\_KEY, collapsed ? "collapsed" : "expanded");
}, \[collapsed]);

return (
\<aside
className={\[
"hidden md\:flex md\:flex-col md\:border-r md\:bg-background transition-\[width] duration-200",
collapsed ? "md\:w-16" : "md\:w-60",
].join(" ")}
aria-label="Primary"
\> <div className="flex h-14 items-center justify-between gap-2 px-2"> <Link to="/" className="inline-flex items-center gap-2 px-2"> <div className="h-3 w-3 rounded-sm bg-primary" aria-hidden />
\<span className={\["text-sm font-semibold tracking-wide", collapsed ? "sr-only" : ""].join(" ")}>
{APP\_TITLE} </span> </Link>
\<Button
variant="ghost"
size="icon"
aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
onClick={() => setCollapsed((v) => !v)}
\>
{collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />} </Button> </div> <Separator /> <nav className="flex-1 overflow-auto px-2 py-2"> <TooltipProvider>
{items.map((n) => ( <Tooltip key={n.to} delayDuration={300}> <TooltipTrigger asChild> <div> <SidebarLink to={n.to} icon={n.icon} label={n.label} collapsed={collapsed} /> </div> </TooltipTrigger> <TooltipContent side="right">{n.desc}</TooltipContent> </Tooltip>
))} </TooltipProvider> </nav> <Separator /> <div className="flex items-center justify-between p-2"> <ThemeToggle />
\<div className={\["flex items-center gap-2", collapsed ? "sr-only" : ""].join(" ")}> <a
         href="https://github.com/"
         target="_blank"
         rel="noreferrer"
         aria-label="Open GitHub"
         className="text-muted-foreground hover:text-foreground inline-flex"
         title="GitHub"
       > <Github className="h-4 w-4" /> </a> <a
         href="about:blank"
         target="_blank"
         rel="noreferrer"
         aria-label="Open Docs"
         className="text-muted-foreground hover:text-foreground inline-flex"
         title="Docs"
       > <ExternalLink className="h-4 w-4" /> </a> </div> </div> </aside>
);
};

// ----------------------------------------------------------------------------
// Skip Link (Accessibility)
// ----------------------------------------------------------------------------

const SkipToContent: React.FC = () => {
return ( <a
   href="#main-content"
   className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground"
 >
Skip to content </a>
);
};

// ----------------------------------------------------------------------------
// Topbar
// ----------------------------------------------------------------------------

const Topbar: React.FC<{ items: NavItem\[] }> = ({ items }) => {
return ( <header className="sticky top-0 z-40 flex h-14 items-center gap-2 border-b bg-background px-3"> <MobileNav items={items} /> <div className="flex items-center gap-2"> <Link to="/" className="md:hidden inline-flex items-center gap-2"> <div className="h-3 w-3 rounded-sm bg-primary" aria-hidden /> <span className="text-sm font-semibold tracking-wide">{APP\_TITLE}</span> </Link> </div> <div className="ml-auto flex items-center gap-2"> <CommandPalette /> </div> </header>
);
};

// ----------------------------------------------------------------------------
// Layout Shell
// ----------------------------------------------------------------------------

const Layout: React.FC\<React.PropsWithChildren> = ({ children }) => {
return ( <div className="flex min-h-dvh w-full">
{/\* Accessibility: allow keyboard users to jump straight to main content \*/} <SkipToContent />

```
  <Sidebar items={NAV_ITEMS} />

  <div className="flex w-0 flex-1 flex-col">
    <Topbar items={NAV_ITEMS} />
    <main id="main-content" className="flex-1">
      <section className="mx-auto max-w-[1400px] p-4 md:p-6">
        {children}
      </section>
    </main>
    {/* Optional footer area for global notices or build metadata */}
    <footer className="border-t px-4 py-2 text-xs text-muted-foreground">
      CLI-first · Reproducible by construction · GUI is a thin veneer over the pipeline
    </footer>
  </div>
</div>
```

);
};

export default Layout;

// ============================================================================
// Implementation Notes (for reviewers)
// ----------------------------------------------------------------------------
// • Sidebar collapse state is persisted to localStorage for a consistent UX.
// • Theme mode persists and updates live if "system" is selected and the user
//   flips OS appearance while the app is open.
// • Command palette uses NAV\_ITEMS as a single source of truth, just like the
//   sidebar. Add or rename routes in one place.
// • Keyboard: ⌘/Ctrl + K opens palette; "Escape" closes it.
// • Accessibility: Skip link, proper aria-labels, and role hints are included.
// • Security: All links with target=\_blank include rel="noreferrer".
// • Styling: Tailwind utility classes and shadcn/ui components for simplicity.
// ============================================================================
