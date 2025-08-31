// src/gui/components/card.test.tsx
// =============================================================================
// ✅ Tests for src/gui/components/card.tsx (SpectraMind V50 GUI) — Upgraded
// -----------------------------------------------------------------------------
// These tests validate rendering, composability, a11y semantics, and styling
// behavior for the Card component, including title/description/icon/actions/
// footer, highlight emphasis, and child content rendering. Mocks are provided
// for framer-motion and shadcn/ui Card primitives to keep tests deterministic.
//
// Test Stack Assumptions:
//   • Vitest (or Jest) + @testing-library/react
//   • tsconfig paths support for "@/..." (or adjust imports accordingly)
//
// Run (Vitest):
//   npx vitest run src/gui/components/card.test.tsx
// =============================================================================

import * as React from "react";
import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, cleanup, within } from "@testing-library/react";
import "@testing-library/jest-dom";
import { Card } from "./card";

// -----------------------------------------------------------------------------
// Mocks
// -----------------------------------------------------------------------------

// Mock framer-motion to strip animations for unit tests.
vi.mock("framer-motion", () => {
  return {
    motion: {
      // Render a plain div that passes through props/children
      div: ({ children, ...rest }: any) => <div {...rest}>{children}</div>,
    },
  };
});

// Mock shadcn/ui card primitives used by Card.
// If your project provides real implementations at "@/components/ui/card",
// you can remove this mock. We mock to keep tests self-contained and portable.
vi.mock("@/components/ui/card", () => {
  const Base: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-root" {...rest}>
      {children}
    </div>
  );
  const Header: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-header" {...rest}>
      {children}
    </div>
  );
  const Title: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-title" {...rest}>
      {children}
    </div>
  );
  const Description: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-description" {...rest}>
      {children}
    </div>
  );
  const Content: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-content" {...rest}>
      {children}
    </div>
  );
  const Footer: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ children, ...rest }) => (
    <div data-testid="card-footer" {...rest}>
      {children}
    </div>
  );

  return {
    Card: Base,
    CardHeader: Header,
    CardTitle: Title,
    CardDescription: Description,
    CardContent: Content,
    CardFooter: Footer,
  };
});

// Clean up after each test to avoid bleed.
afterEach(() => cleanup());

// Simple helper icon component for tests
const TestIcon: React.FC = () => <span aria-label="test-icon">★</span>;

// -----------------------------------------------------------------------------
// Test Cases
// -----------------------------------------------------------------------------

describe("Card component", () => {
  it("renders title, description, and child content", () => {
    render(
      <Card title="Diagnostics Summary" description="GLL, SHAP, symbolic overlays">
        <div>Child Content</div>
      </Card>
    );

    // Title and description
    expect(screen.getByText("Diagnostics Summary")).toBeInTheDocument();
    expect(screen.getByText("GLL, SHAP, symbolic overlays")).toBeInTheDocument();

    // Content
    expect(screen.getByText("Child Content")).toBeInTheDocument();

    // Header + Content testids (from mocked ui/card)
    expect(screen.getByTestId("card-header")).toBeInTheDocument();
    expect(screen.getByTestId("card-content")).toBeInTheDocument();
  });

  it("renders description without title and still shows header", () => {
    render(
      <Card description="Only description present">
        <div>Body</div>
      </Card>
    );
    expect(screen.getByText("Only description present")).toBeInTheDocument();
    expect(screen.getByTestId("card-header")).toBeInTheDocument();
  });

  it("renders an icon when provided", () => {
    render(
      <Card title="With Icon" icon={<TestIcon />}>
        <div>Body</div>
      </Card>
    );

    expect(screen.getByLabelText("test-icon")).toBeInTheDocument();
    expect(screen.getByText("With Icon")).toBeInTheDocument();
  });

  it("renders header if only icon is provided (no title/description)", () => {
    render(
      <Card icon={<TestIcon />}>
        <div>Body</div>
      </Card>
    );
    // Header exists to host the icon
    expect(screen.getByTestId("card-header")).toBeInTheDocument();
    expect(screen.getByLabelText("test-icon")).toBeInTheDocument();
  });

  it("renders actions area when provided", () => {
    render(
      <Card
        title="With Actions"
        actions={
          <button type="button" aria-label="card-action">
            Action
          </button>
        }
      >
        <div>Body</div>
      </Card>
    );

    const header = screen.getByTestId("card-header");
    expect(screen.getByLabelText("card-action")).toBeInTheDocument();
    // Ensure the action button is inside header region
    expect(within(header).getByLabelText("card-action")).toBeInTheDocument();
  });

  it("renders footer when provided", () => {
    render(
      <Card title="Footer Test" footer={<div>Footer Content</div>}>
        <div>Body</div>
      </Card>
    );

    // Footer testid (from mocked ui/card)
    expect(screen.getByTestId("card-footer")).toBeInTheDocument();
    expect(screen.getByText("Footer Content")).toBeInTheDocument();
  });

  it("applies highlight styling when highlight=true", () => {
    const { container } = render(
      <Card title="Highlighted" highlight>
        <div>Body</div>
      </Card>
    );

    // We expect the highlighted class to be present on the outermost card container.
    // The Card component adds "border-blue-500" when highlight=true.
    const root = container.firstChild as HTMLElement;
    expect(root).toBeTruthy();
    expect(root.className).toContain("border-blue-500");
  });

  it("omits header section entirely if no header props are provided", () => {
    render(
      <Card>
        <div>Only Content</div>
      </Card>
    );

    // With no title/description/icon/actions, header should not exist.
    // Our mocked Header adds data-testid, so query should fail.
    const header = screen.queryByTestId("card-header");
    expect(header).toBeNull();

    // Content still renders
    expect(screen.getByText("Only Content")).toBeInTheDocument();
  });

  it("merges custom className into the card container", () => {
    const { container } = render(
      <Card className="test-class">
        <div>Body</div>
      </Card>
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("test-class");
  });

  it("supports a complex header with icon + title + description + actions", () => {
    render(
      <Card
        icon={<TestIcon />}
        title="Complex"
        description="Header combo"
        actions={<button aria-label="more">More</button>}
      >
        <div>Content</div>
      </Card>
    );

    const header = screen.getByTestId("card-header");
    expect(within(header).getByLabelText("test-icon")).toBeInTheDocument();
    expect(within(header).getByText("Complex")).toBeInTheDocument();
    expect(within(header).getByText("Header combo")).toBeInTheDocument();
    expect(within(header).getByLabelText("more")).toBeInTheDocument();
  });
});
