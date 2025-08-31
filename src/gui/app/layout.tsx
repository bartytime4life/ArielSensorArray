// src/gui/app/layout.tsx
// ============================================================================
// 🎛️ SpectraMind V50 — Global App Layout (Sidebar + Topbar + Content)
// ----------------------------------------------------------------------------
// • Provides a persistent navigation shell around all feature routes.
// • Uses Tailwind + shadcn/ui components for accessible, modern UX.
// • Sidebar links mirror core CLI-first diagnostics: Dashboard, Diagnostics,
//   UMAP, t-SNE, SHAP, Symbolic, FFT, Calibration.
// • Topbar includes app title, theme toggle, and a mobile menu.
// • All navigation ultimately maps to CLI-backed server endpoints.
// ============================================================================

import * as React from "react";
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
} from "lucide-react";

// ----------------------------------------------------------------------------
// Constants
// ----------------------------------------------------------------------------

const APP_TITLE = "SpectraMind V50";
const THEME_STORAGE_KEY = "spectramind-theme";

// Keep routes defined in one place (source of truth for sidebar + command palette).
const NAV_ITEMS: Array<{
  label: string;
  to: string;
  icon: React.ElementType;
  desc: string;
}> = [
  { label: "Dashboard", to: "/dashboard", icon: LayoutDashboard, desc: "Overview, recent runs, quick links" },
  { label: "Diagnostics", to: "/diagnostics", icon: ActivitySquare, desc: "GLL summaries, calibration checks, reports" },
  { label: "UMAP", to: "/umap", icon: Network, desc: "Latent UMAP explorer with overlays & links" },
  { label: "t-SNE", to: "/tsne", icon: ScatterChart, desc: "Interactive t-SNE projection diagnostics" },
  { label: "SHAP", to: "/shap", icon: Atom, desc: "Explainability: SHAP overlays & barplots" },
  { label: "Symbolic", to: "/symbolic", icon: Sigma, desc: "Neuro-symbolic rules, violations, influence" },
  { label: "FFT", to: "/fft", icon: Waves, desc: "FFT & autocorr analysis with molecular regions" },
  { label: "Calibration", to: "/calibration", icon: Wrench, desc: "Run/view calibration pipeline & artifacts" },
];

// ----------------------------------------------------------------------------
// Theme Toggle (respects ThemeProvider storageKey)
// ----------------------------------------------------------------------------

type ThemeMode = "light" | "dark" | "system";

function getSystemPrefersDark(): boolean {
  return typeof window !== "undefined" && window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyTheme(mode: ThemeMode) {
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
  localStorage.setItem(THEME_STORAGE_KEY, mode);
}

const ThemeToggle: React.FC = () => {
  const [mode, setMode] = React.useState<ThemeMode>(() => {
    const stored = (localStorage.getItem(THEME_STORAGE_KEY) as ThemeMode) || "light";
    return stored;
  });

  React.useEffect(() => {
    applyTheme(mode);
  }, [mode]);

  return (
    <div className="inline-flex items-center gap-1" role="group" aria-label="Theme toggle">
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={mode === "light" ? "default" : "ghost"}
              size="icon"
              aria-pressed={mode === "light"}
              aria-label="Light theme"
              onClick={() => setMode("light")}
            >
              <Sun className="h-4 w-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Light</TooltipContent>
        </Tooltip>

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
}> = ({ to, icon: Icon, label, onClick }) => {
  const location = useLocation();
  const active = location.pathname === to;
  return (
    <NavLink
      to={to}
      onClick={onClick}
      className={({ isActive }) =>
        [
          "group flex items-center gap-2 rounded-md px-3 py-2 text-sm transition",
          (isActive || active)
            ? "bg-primary/10 text-primary"
            : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
        ].join(" ")
      }
      aria-current={active ? "page" : undefined}
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span>{label}</span>
    </NavLink>
  );
};

// ----------------------------------------------------------------------------
// Command Palette (Ctrl/Cmd + K)
// ----------------------------------------------------------------------------

const CommandPalette: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  const location = useLocation();

  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = navigator.platform.includes("Mac") ? e.metaKey : e.ctrlKey;
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setOpen((v) => !v);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <Button
        variant="outline"
        className="hidden md:inline-flex gap-2 text-sm"
        onClick={() => setOpen(true)}
        aria-label="Open command palette"
      >
        <span className="text-muted-foreground">Search or jump…</span>
        <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border bg-muted px-1.5 font-mono text-[10px] text-muted-foreground">
          <span className="text-xs">⌘</span>K
        </kbd>
      </Button>

      <div className={open ? "fixed inset-0 z-[60] flex items-start justify-center bg-black/20 p-4" : "hidden"} aria-hidden={!open}>
        <div className="w-full max-w-xl rounded-lg border bg-background shadow-lg">
          <Command>
            <CommandInput placeholder="Type a page, e.g., 'shap' or 'umap'…" autoFocus onKeyDown={(e) => e.key === "Escape" && setOpen(false)} />
            <CommandList>
              <CommandEmpty>No results.</CommandEmpty>

              <CommandGroup heading="Navigate">
                {NAV_ITEMS.map((item) => (
                  <Link key={item.to} to={item.to} onClick={() => setOpen(false)} className="block">
                    <CommandItem className="cursor-pointer">
                      <item.icon className="mr-2 h-4 w-4" />
                      <span>{item.label}</span>
                    </CommandItem>
                  </Link>
                ))}
              </CommandGroup>

              <CommandSeparator />

              <CommandGroup heading="Shortcuts">
                <CommandItem disabled>
                  <span>Current: {location.pathname}</span>
                </CommandItem>
              </CommandGroup>
            </CommandList>
          </Command>
        </div>
      </div>
    </>
  );
};

// ----------------------------------------------------------------------------
// Mobile Sidebar
// ----------------------------------------------------------------------------

const MobileNav: React.FC = () => {
  const [open, setOpen] = React.useState(false);
  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="icon" aria-label="Open navigation" className="md:hidden">
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="left" className="p-0">
        <SheetHeader className="px-6 pt-6 pb-2 text-left">
          <SheetTitle className="text-base font-semibold">{APP_TITLE}</SheetTitle>
          <SheetDescription>Navigate</SheetDescription>
        </SheetHeader>
        <div className="px-3 py-2">
          {NAV_ITEMS.map((n) => (
            <SidebarLink key={n.to} to={n.to} icon={n.icon} label={n.label} onClick={() => setOpen(false)} />
          ))}
        </div>
        <Separator className="my-2" />
        <div className="px-3 pb-6">
          <ThemeToggle />
        </div>
      </SheetContent>
    </Sheet>
  );
};

// ----------------------------------------------------------------------------
// Sidebar (Desktop)
// ----------------------------------------------------------------------------

const Sidebar: React.FC = () => {
  return (
    <aside
      className="hidden md:flex md:w-60 md:flex-col md:border-r md:bg-background"
      aria-label="Primary"
    >
      <div className="flex h-14 items-center gap-2 px-4">
        <Link to="/" className="inline-flex items-center gap-2">
          <div className="h-3 w-3 rounded-sm bg-primary" aria-hidden />
          <span className="text-sm font-semibold tracking-wide">{APP_TITLE}</span>
        </Link>
      </div>
      <Separator />
      <nav className="flex-1 overflow-auto px-3 py-2">
        <TooltipProvider>
          {NAV_ITEMS.map((n) => (
            <Tooltip key={n.to} delayDuration={300}>
              <TooltipTrigger asChild>
                <div>
                  <SidebarLink to={n.to} icon={n.icon} label={n.label} />
                </div>
              </TooltipTrigger>
              <TooltipContent side="right">{n.desc}</TooltipContent>
            </Tooltip>
          ))}
        </TooltipProvider>
      </nav>
      <Separator />
      <div className="flex items-center justify-between p-3">
        <ThemeToggle />
        <div className="flex items-center gap-2">
          <a
            href="https://github.com/"
            target="_blank"
            rel="noreferrer"
            aria-label="Open GitHub"
            className="text-muted-foreground hover:text-foreground inline-flex"
            title="GitHub"
          >
            <Github className="h-4 w-4" />
          </a>
          <a
            href="about:blank"
            target="_blank"
            rel="noreferrer"
            aria-label="Open Docs"
            className="text-muted-foreground hover:text-foreground inline-flex"
            title="Docs"
          >
            <ExternalLink className="h-4 w-4" />
          </a>
        </div>
      </div>
    </aside>
  );
};

// ----------------------------------------------------------------------------
// Topbar
// ----------------------------------------------------------------------------

const Topbar: React.FC = () => {
  return (
    <header className="sticky top-0 z-40 flex h-14 items-center gap-2 border-b bg-background px-3">
      <MobileNav />
      <div className="flex items-center gap-2">
        <Link to="/" className="md:hidden inline-flex items-center gap-2">
          <div className="h-3 w-3 rounded-sm bg-primary" aria-hidden />
          <span className="text-sm font-semibold tracking-wide">{APP_TITLE}</span>
        </Link>
      </div>
      <div className="ml-auto flex items-center gap-2">
        <CommandPalette />
      </div>
    </header>
  );
};

// ----------------------------------------------------------------------------
// Layout
// ----------------------------------------------------------------------------

const Layout: React.FC<React.PropsWithChildren> = ({ children }) => {
  return (
    <div className="flex min-h-dvh w-full">
      <Sidebar />
      <div className="flex w-0 flex-1 flex-col">
        <Topbar />
        <main className="flex-1">
          <section className="mx-auto max-w-[1400px] p-4 md:p-6">
            {children}
          </section>
        </main>
      </div>
    </div>
  );
};

export default Layout;

// ============================================================================
// Notes:
// • This file is self-contained for easy drop-in. It expects shadcn/ui primitives
//   to be available under "@/components/ui/*" and Tailwind configured.
// • Icons from lucide-react are used for a clean, lightweight icon set.
// • Command palette is intentionally simple; you can extend to run CLI actions.
// • The theme toggle syncs with ThemeProvider via localStorage key.
// ============================================================================
