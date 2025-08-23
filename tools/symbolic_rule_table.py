# symbolic_rule_table.py
# SPDX-License-Identifier: Apache-2.0
"""
Symbolic Rule Table — SpectraMind V50
-------------------------------------

A compact neuro-symbolic “rule table” engine for SpectraMind V50 that:

* Loads rules from YAML/JSON or registers them programmatically.
* Evaluates boolean/threshold expressions over per-target feature records
  (e.g., spectrum arrays, uncertainties, and derived metadata).
* Emits a structured result table, plus CSV/Markdown exports.
* Provides a small Typer sub-CLI suitable for mounting under the unified CLI.
* Integrates nicely with Hydra-driven configs (optional) and Rich logging.

Typical Uses
------------
1) Physics-informed sanity checks (e.g., non-negativity, uncertainty > 0).
2) Band logic (e.g., H2O band depth > continuum threshold).
3) Diagnostics gating (block submission if any ERROR-severity rule fires).
4) Lightweight symbolic priors to aid neuro-symbolic interpretation.

Design Notes
------------
- Evaluation is record-wise: you pass an iterable of dict-like records.
  Each record is expected to contain at least:
    * "spectrum": np.ndarray shape (N,)
    * "uncertainty": np.ndarray shape (N,)
  and any scalar features you wish to reference (e.g., "white_light_depth").
- Expressions are evaluated with a restricted eval() environment. We expose:
    math, numpy (as np), and helper functions: between, mean, std, band,
    band_mean, band_std, slope.
- Band definitions are optional; if provided (as {band_name: [start, end]}),
  helpers can compute on spectral slices by index or wavelength.
- This keeps dependencies slim: PyYAML (or stdlib json), numpy, pandas, rich, typer.

Security
--------
Rule expressions are evaluated using Python's eval() with restricted globals.
This is intended for *trusted* project configs. Do not load untrusted rules.

Author: SpectraMind V50 team
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

try:
    import yaml  # PyYAML
except Exception:  # pragma: no cover
    yaml = None

try:
    import typer  # Typer for CLI glue
except Exception:  # pragma: no cover
    typer = None

console = Console()


# ---------------------------
# Data structures & results
# ---------------------------

Severity = str  # "info" | "warn" | "error" | "block"


@dataclass
class Rule:
    """A symbolic rule definition."""
    id: str
    name: str
    description: str = ""
    severity: Severity = "warn"  # info|warn|error|block
    condition: str = ""          # Python expression evaluated over a record
    when: Optional[str] = None   # Optional gate expression (record-scoped)
    tags: List[str] = field(default_factory=list)
    applies_to: Optional[str] = None  # Optional hint, e.g., "spectrum" / "uncertainty"
    score_delta: float = 0.0     # Optional numeric contribution if fired
    metadata: Dict[str, Any] = field(default_factory=dict)

    def short(self) -> str:
        return f"{self.id}: {self.name} ({self.severity})"


@dataclass
class RuleResult:
    """Evaluation outcome for a single rule on a single record."""
    rule_id: str
    name: str
    severity: Severity
    fired: bool
    score_delta: float
    tags: List[str]
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------
# Helper functions (safe env)
# ---------------------------

def between(x: Union[float, np.ndarray], lo: float, hi: float) -> Union[bool, np.ndarray]:
    """Return True where lo <= x <= hi (vectorized)."""
    return (x >= lo) & (x <= hi)


def band(arr: np.ndarray, start: int, end: int) -> np.ndarray:
    """Return slice of an array by index range [start:end] (end exclusive)."""
    start = max(0, int(start))
    end = int(end)
    return arr[start:end]


def band_mean(arr: np.ndarray, start: int, end: int) -> float:
    """Mean over an index slice; returns float('nan') if empty."""
    sl = band(arr, start, end)
    return float(np.nanmean(sl)) if sl.size else float("nan")


def band_std(arr: np.ndarray, start: int, end: int) -> float:
    """Std over an index slice; returns float('nan') if empty."""
    sl = band(arr, start, end)
    return float(np.nanstd(sl)) if sl.size else float("nan")


def mean(x: Union[np.ndarray, Sequence[float], float]) -> float:
    return float(np.nanmean(x))


def std(x: Union[np.ndarray, Sequence[float], float]) -> float:
    return float(np.nanstd(x))


def slope(y: np.ndarray, x: Optional[np.ndarray] = None) -> float:
    """Return least-squares slope of y (vs x or index)."""
    if x is None:
        x = np.arange(len(y))
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    xmean = x.mean()
    ymean = y.mean()
    denom = np.sum((x - xmean) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((x - xmean) * (y - ymean)) / denom)


def _default_eval_env() -> Dict[str, Any]:
    """Restricted globals for eval()."""
    return {
        "__builtins__": None,  # disable builtins
        "math": math,
        "np": np,
        # helpers
        "between": between,
        "mean": mean,
        "std": std,
        "band": band,
        "band_mean": band_mean,
        "band_std": band_std,
        "slope": slope,
    }


# --------------------------------
# Rule table core implementation
# --------------------------------

class SymbolicRuleTable:
    """
    Manages symbolic rules and evaluates them against feature records.

    Each record is a Mapping[str, Any]. Conventionally:
      - record["spectrum"] -> np.ndarray (length N)
      - record["uncertainty"] -> np.ndarray (length N)
      - record may include scalars, arrays, or derived features.
    """

    def __init__(self,
                 rules: Optional[List[Rule]] = None,
                 bands: Optional[Dict[str, Tuple[int, int]]] = None):
        self._rules: List[Rule] = rules or []
        self._bands: Dict[str, Tuple[int, int]] = bands or {}

    # -- Registration & I/O --

    def register(self, rule: Rule) -> None:
        if any(r.id == rule.id for r in self._rules):
            raise ValueError(f"Duplicate rule id '{rule.id}'")
        self._rules.append(rule)

    def extend(self, rules: Iterable[Rule]) -> None:
        for r in rules:
            self.register(r)

    @classmethod
    def from_file(cls, path: Union[str, Path], bands: Optional[Dict[str, Tuple[int, int]]] = None) -> "SymbolicRuleTable":
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML required to load YAML rule files.")
            data = yaml.safe_load(path.read_text())
        elif path.suffix.lower() == ".json":
            data = json.loads(path.read_text())
        else:
            raise ValueError(f"Unsupported rule file type: {path.suffix}")

        rules = []
        for item in data.get("rules", []):
            rules.append(Rule(**item))
        file_bands = data.get("bands", {})
        merged_bands = {**(bands or {}), **file_bands}
        return cls(rules=rules, bands=merged_bands)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rules": [asdict(r) for r in self._rules],
            "bands": self._bands
        }

    # -- Evaluation --

    def _band_lookup(self, key: str) -> Tuple[int, int]:
        if key not in self._bands:
            raise KeyError(f"Unknown band '{key}'. Define it in 'bands'.")
        return self._bands[key]

    def _inject_band_helpers(self, local_env: Dict[str, Any]) -> None:
        """
        Injects dynamic helpers that let rule expressions use named bands:

            band_mean_named('h2o')  -> band_mean(record['spectrum'], *bands['h2o'])
            band_std_named('cont')   -> band_std(record['spectrum'], *bands['cont'])
        """
        def band_mean_named(name: str, target: str = "spectrum") -> float:
            start, end = self._band_lookup(name)
            arr = local_env.get(target, None)
            if arr is None:
                return float("nan")
            return band_mean(np.asarray(arr), start, end)

        def band_std_named(name: str, target: str = "spectrum") -> float:
            start, end = self._band_lookup(name)
            arr = local_env.get(target, None)
            if arr is None:
                return float("nan")
            return band_std(np.asarray(arr), start, end)

        def band_slice_named(name: str, target: str = "spectrum") -> np.ndarray:
            start, end = self._band_lookup(name)
            arr = local_env.get(target, None)
            if arr is None:
                return np.array([])
            return band(np.asarray(arr), start, end)

        local_env["band_mean_named"] = band_mean_named
        local_env["band_std_named"] = band_std_named
        local_env["band_slice_named"] = band_slice_named

    def _eval_expr(self, expr: str, record: Mapping[str, Any]) -> Any:
        if not expr:
            return True
        g = _default_eval_env()
        l = dict(record)
        # band helpers depend on record & bands
        self._inject_band_helpers(l)
        try:
            return eval(expr, g, l)
        except Exception as ex:
            raise RuntimeError(f"Error evaluating expression '{expr}': {ex}")

    def evaluate_record(self, record: Mapping[str, Any]) -> List[RuleResult]:
        """Evaluate all rules on a single record."""
        results: List[RuleResult] = []
        for rule in self._rules:
            # Optional gate
            gated_in = True
            if rule.when:
                gated_in = bool(self._eval_expr(rule.when, record))
            fired = False
            msg = ""
            if gated_in:
                cond = bool(self._eval_expr(rule.condition, record))
                fired = bool(cond)
                if fired:
                    msg = f"Rule fired: {rule.short()}"
            rr = RuleResult(
                rule_id=rule.id,
                name=rule.name,
                severity=rule.severity,
                fired=fired,
                score_delta=(rule.score_delta if fired else 0.0),
                tags=list(rule.tags),
                message=msg,
                details={"applies_to": rule.applies_to} if rule.applies_to else {},
            )
            results.append(rr)
        return results

    def evaluate(self, records: Iterable[Mapping[str, Any]], with_index: bool = True) -> pd.DataFrame:
        """
        Evaluate rules on an iterable of records.
        Returns a tidy pandas DataFrame with one row per (index, rule).
        """
        rows = []
        for idx, rec in enumerate(records):
            rec_results = self.evaluate_record(rec)
            for rr in rec_results:
                rows.append({
                    "index": idx if with_index else None,
                    "rule_id": rr.rule_id,
                    "name": rr.name,
                    "severity": rr.severity,
                    "fired": rr.fired,
                    "score_delta": rr.score_delta,
                    "tags": ",".join(rr.tags) if rr.tags else "",
                    "message": rr.message,
                    **rr.details,
                })
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["index", "rule_id", "name", "severity", "fired",
                                       "score_delta", "tags", "message"])
        return df

    # -- Aggregation & export --

    @staticmethod
    def aggregate(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate rule firings per record (index).
        Provides per-severity counts and total score_delta.
        """
        if df.empty:
            return pd.DataFrame(columns=["index", "info", "warn", "error", "block", "total_score_delta"])
        pivot = (
            df.assign(cnt=1)
              .pivot_table(index="index", columns="severity", values="cnt", aggfunc="sum", fill_value=0)
              .reset_index()
        )
        for sev in ("info", "warn", "error", "block"):
            if sev not in pivot.columns:
                pivot[sev] = 0
        scores = df.groupby("index")["score_delta"].sum().reset_index(name="total_score_delta")
        out = pivot.merge(scores, on="index", how="left").fillna({"total_score_delta": 0.0})
        return out[["index", "info", "warn", "error", "block", "total_score_delta"]]

    @staticmethod
    def to_markdown(df: pd.DataFrame, title: str = "Symbolic Rule Results") -> str:
        """Render a DataFrame as Markdown table with a header."""
        if df.empty:
            return f"# {title}\n\n_No results._\n"
        buf = [f"# {title}\n"]
        buf.append(df.to_markdown(index=False))
        return "\n".join(buf)

    @staticmethod
    def save_csv(df: pd.DataFrame, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def save_markdown(df: pd.DataFrame, path: Union[str, Path], title: str = "Symbolic Rule Results") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(SymbolicRuleTable.to_markdown(df, title=title), encoding="utf-8")

    # -- Introspection --

    def show(self) -> None:
        """Pretty-print current rules and bands."""
        table = Table(title="Symbolic Rule Table", box=box.SIMPLE_HEAVY)
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="bold")
        table.add_column("Severity", style="magenta")
        table.add_column("Expr", overflow="fold")
        for r in self._rules:
            table.add_row(r.id, r.name, r.severity, r.condition)
        console.print(table)
        if self._bands:
            bt = Table(title="Bands", box=box.SIMPLE)
            bt.add_column("Name", style="cyan")
            bt.add_column("Start")
            bt.add_column("End")
            for k, (s, e) in self._bands.items():
                bt.add_row(k, str(s), str(e))
            console.print(bt)


# --------------------------------
# Default rules (sane, physicsy)
# --------------------------------

def default_rules() -> List[Rule]:
    """
    A small starter set of physics‑informed sanity rules.
    These are intentionally conservative; tune per project.
    """
    return [
        Rule(
            id="spec_nonneg",
            name="Spectrum non-negative",
            description="All spectral bins should be >= 0 (after baseline correction).",
            severity="error",
            condition="np.all(spectrum >= 0)",
            tags=["sanity", "physics"],
        ),
        Rule(
            id="unc_pos",
            name="Uncertainty positive",
            description="Uncertainty values must be strictly positive.",
            severity="error",
            condition="np.all(uncertainty > 0)",
            tags=["sanity"],
        ),
#if needed: upper bound on uncertainty relative to signal to noise
        Rule(
            id="snr_floor",
            name="SNR floor",
            description="Mean(S/N) should exceed a minimal floor.",
            severity="warn",
            condition="mean(spectrum / uncertainty) >= 1.5",
            tags=["quality"],
            score_delta=+0.5,
        ),
        Rule(
            id="smoothness",
            name="Local smoothness",
            description="Adjacent-bin variation not excessively large.",
            severity="warn",
            # crude smoothness: std of first differences less than 5x global std
            condition="std(np.diff(spectrum)) <= 5 * std(spectrum)",
            tags=["shape"],
        ),
        Rule(
            id="band_water_depth",
            name="H2O band has depth",
            description="Mean depth in 'h2o' band should be below 'cont' band by threshold.",
            severity="info",
            # Requires bands: h2o and cont defined (by index)
            condition="(band_mean_named('cont') - band_mean_named('h2o')) >= 0.00005",
            when="'h2o' in bands and 'cont' in bands if 'bands' in locals() else True",
            tags=["chemistry", "hints"],
            score_delta=+0.2,
        ),
        Rule(
            id="no_nan",
            name="No NaNs",
            description="No NaN in spectrum or uncertainty.",
            severity="error",
            condition="not (np.any(np.isnan(spectrum)) or np.any(np.isnan(uncertainty)))",
            tags=["sanity"],
        ),
    ]


# ---------------------------
# Minimal Typer sub-CLI
# ---------------------------

def _install_typer() -> "typer.Typer":
    app = typer.Typer(add_completion=False, no_args_is_help=True, help="Symbolic Rule Table CLI")

    @app.command("show")
    def cli_show(
        rules_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Rules YAML/JSON"),
    ):
        """Pretty-print rules and bands from a file."""
        srt = SymbolicRuleTable.from_file(rules_path)
        srt.show()

    @app.command("eval")
    def cli_eval(
        rules_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Rules file"),
        records_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="JSONL or Parquet of records"),
        out_csv: Optional[Path] = typer.Option(None, "--out-csv", help="Save detailed results as CSV"),
        out_md: Optional[Path] = typer.Option(None, "--out-md", help="Save aggregated markdown"),
        title: str = typer.Option("Symbolic Rule Results", "--title", help="Markdown title"),
    ):
        """Evaluate rules over records and export results."""
        srt = SymbolicRuleTable.from_file(rules_path)
        # Load records
        if records_path.suffix.lower() in {".jsonl", ".json"}:
            # Expect JSON lines: one object per line
            recs = []
            for line in records_path.read_text().splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                # Convert lists to np.arrays for spectrum/uncertainty if present
                for k in ("spectrum", "uncertainty"):
                    if k in obj and isinstance(obj[k], list):
                        obj[k] = np.asarray(obj[k], dtype=float)
                recs.append(obj)
        elif records_path.suffix.lower() in {".parquet"}:
            df = pd.read_parquet(records_path)
            # Convert 'spectrum'/'uncertainty' object columns back to arrays if needed
            recs = []
            for _, row in df.iterrows():
                rec = dict(row)
                for k in ("spectrum", "uncertainty"):
                    if k in rec and not isinstance(rec[k], np.ndarray):
                        rec[k] = np.asarray(rec[k])
                recs.append(rec)
        else:
            raise typer.BadParameter("records_path must be .jsonl(.json) or .parquet")

        console.rule("[bold]Evaluating symbolic rules")
        df = srt.evaluate(recs)
        agg = SymbolicRuleTable.aggregate(df)
        console.print(Panel.fit(f"Evaluated {len(recs)} records; {df['fired'].sum() if not df.empty else 0} rule hits."))

        # Show quick summaries
        srt.show()
        console.rule("[bold]Summary")
        console.print(agg if not agg.empty else "No results.")
        if out_csv:
            SymbolicRuleTable.save_csv(df, out_csv)
            console.print(f"[green]Saved detailed CSV:[/green] {out_csv}")
        if out_md:
            SymbolicRuleTable.save_markdown(agg, out_md, title=title)
            console.print(f"[green]Saved markdown:[/green] {out_md}")

    @app.command("scaffold")
    def cli_scaffold(
        out_path: Path = typer.Argument(..., help="Where to write a starter YAML file"),
        with_defaults: bool = typer.Option(True, "--with-defaults/--no-defaults", help="Include default rules"),
    ):
        """Write a starter YAML with bands and optional default rules."""
        if yaml is None:
            raise typer.Exit(code=1)
        data = {
            "bands": {
                # Example band indices (adjust per dataset’s spectral index range)
                "cont": [20, 50],
                "h2o": [110, 140],
            },
            "rules": [asdict(r) for r in (default_rules() if with_defaults else [])],
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        console.print(f"[green]Scaffold written:[/green] {out_path}")

    return app


# ---------------------------
# Public helpers to mount
# ---------------------------

def get_typer_app() -> "typer.Typer":
    """
    Return a Typer app for mounting into the unified CLI, e.g.:

        from symbolic_rule_table import get_typer_app
        app = typer.Typer()
        app.add_typer(get_typer_app(), name="rules")

    Then: `spectramind rules eval ...`
    """
    if typer is None:
        raise RuntimeError("Typer not installed")
    return _install_typer()


# ---------------------------
# Example standalone usage
# ---------------------------

if __name__ == "__main__":
    if typer is None:
        console.print("[red]Typer is not installed. Install with `pip install typer[all]`.[/red]")
        sys.exit(1)
    _install_typer()()
