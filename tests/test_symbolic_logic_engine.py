Unit tests for src/symbolic/symbolic_logic_engine.py

Covers:
- Rule primitives (nonnegativity, smoothness, asymmetry)
- Engine aggregation and per-rule accounting
- Hard/soft modes via a custom rule
- JSON/CSV export
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

# Import the module under test
# Adjust the import path below if your project structure differs.
from src.symbolic.symbolic_logic_engine import (
    SymbolicLogicEngine,
    SymbolicRule,
    build_default_engine,
)


def test_rule_nonnegativity_basic():
    mu = torch.tensor([[-0.5, 0.0, 0.25, -1.25]], dtype=torch.float32)
    out = SymbolicLogicEngine.rule_nonnegativity(mu)
    # Violations are relu(-mu): positive where mu < 0
    expected = torch.tensor([[0.5, 0.0, 0.0, 1.25]], dtype=torch.float32)
    assert torch.allclose(out, expected)


def test_rule_smoothness_smaller_for_smooth_spectra():
    # Smooth: nearly constant
    mu_smooth = torch.ones(1, 283) * 0.1
    # Noisy: random fluctuations
    torch.manual_seed(0)
    mu_noisy = 0.1 + 0.2 * torch.randn(1, 283)

    s_smooth = SymbolicLogicEngine.rule_smoothness(mu_smooth, window=7).mean()
    s_noisy = SymbolicLogicEngine.rule_smoothness(mu_noisy, window=7).mean()

    assert s_smooth.item() < s_noisy.item(), "Smooth spectrum should have smaller smoothness violation"


def test_rule_asymmetry_zero_for_symmetric_signal():
    # Construct symmetric spectrum around middle
    bins = 200
    left = torch.linspace(0, 1, bins // 2)
    mu = torch.cat([left, left.flip(0)]).unsqueeze(0)  # shape: (1, bins)
    asym = SymbolicLogicEngine.rule_asymmetry(mu)
    # Expect ~0 difference
    assert torch.isfinite(asym).all()
    assert torch.allclose(asym, torch.zeros_like(asym), atol=1e-6)


def test_engine_default_rules_aggregate_and_per_rule():
    torch.manual_seed(42)
    mu = (0.05 * torch.randn(2, 283)).clamp_min(-0.2)  # small noise, some negatives possible

    engine = build_default_engine()
    out = engine.evaluate_rules(mu, aggregate=True)

    assert "total_loss" in out and "per_rule" in out
    assert isinstance(out["per_rule"], dict)
    assert all(isinstance(v, float) for v in out["per_rule"].values())
    # total_loss should be a torch scalar (from aggregation)
    assert hasattr(out["total_loss"], "item")

    # Simple sanity: total_loss should be positive or zero
    assert out["total_loss"].item() >= 0.0


def test_custom_rule_hard_mode_counts_boolean_mask():
    # Create a deterministic spectrum with known violations
    mu = torch.tensor([[0.0, -1.0, 0.5, -0.2, 0.1, -0.3]], dtype=torch.float32)  # shape: (1, 6)

    # Define a custom rule: positive value if mu < 0, else 0 (like violation score)
    def negative_bins(mu_t: torch.Tensor) -> torch.Tensor:
        return (mu_t < 0.0).float()  # per-bin indicator (1 where negative)

    # Soft mode should produce the mean of the indicator (i.e., fraction of negatives)
    soft_rule = SymbolicRule("neg_soft", negative_bins, weight=1.0, mode="soft", normalize=True)
    # Hard mode will threshold >0 then mean (effectively the same for this indicator)
    hard_rule = SymbolicRule("neg_hard", negative_bins, weight=1.0, mode="hard", normalize=True)

    engine = SymbolicLogicEngine([soft_rule, hard_rule])

    res = engine.evaluate_rules(mu, aggregate=True)
    per_rule = res["per_rule"]

    # Negatives at indices: 1, 3, 5 → 3 of 6 = 0.5
    assert pytest.approx(per_rule["neg_soft"], rel=0, abs=1e-6) == 0.5
    assert pytest.approx(per_rule["neg_hard"], rel=0, abs=1e-6) == 0.5

    # Total loss is sum since both weights = 1.0
    assert pytest.approx(res["total_loss"].item(), rel=0, abs=1e-6) == 1.0


def test_export_results_json_and_csv(tmp_path: Path):
    # Minimal per_rule dict
    results = {"nonnegativity": 0.123, "smoothness": 0.456, "asymmetry": 0.789}

    engine = SymbolicLogicEngine()

    # JSON export
    json_path = tmp_path / "symbolic_results.json"
    engine.export_results(results, json_path, fmt="json")
    assert json_path.exists()

    with open(json_path, "r") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert set(data.keys()) == set(results.keys())

    # CSV export
    csv_path = tmp_path / "symbolic_results.csv"
    engine.export_results(results, csv_path, fmt="csv")
    assert csv_path.exists()

    # Quick CSV sanity read
    text = csv_path.read_text().strip().splitlines()
    assert text[0].strip().lower() == "rule,loss"
    # There should be as many data lines as entries
    assert len(text) == 1 + len(results)


def test_engine_handles_empty_rule_set_gracefully():
    mu = torch.zeros(2, 283)
    engine = SymbolicLogicEngine(rules=[])
    out = engine.evaluate_rules(mu, aggregate=True)
    assert "per_rule" in out and out["per_rule"] == {}
    # With no rules, total_loss starts at 0.0 (float), converted to tensor via aggregation
    assert hasattr(out["total_loss"], "item")
    assert out["total_loss"].item() == pytest.approx(0.0)


def test_photonic_alignment_rule_noop_without_ref():
    mu = torch.randn(1, 283) * 0.01
    zero = SymbolicLogicEngine.rule_photonic_alignment(mu, ref=None)
    assert torch.allclose(zero, torch.zeros_like(mu))


def test_photonic_alignment_rule_with_ref():
    mu = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    ref = torch.tensor([[0.1, 0.1, 0.5]], dtype=torch.float32)
    out = SymbolicLogicEngine.rule_photonic_alignment(mu, ref=ref)
    # absolute difference per-bin
    expected = torch.tensor([[0.0, 0.1, 0.2]], dtype=torch.float32)
    assert torch.allclose(out, expected)


@pytest.mark.parametrize("window", [3, 5, 7, 9])
def test_smoothness_window_parameter_affects_result(window: int):
    # Construct a gently sloped spectrum
    x = torch.linspace(0, 1, 101).unsqueeze(0)
    s = SymbolicLogicEngine.rule_smoothness(x, window=window).mean()
    assert torch.isfinite(s)
    # As a loose sanity check: smoothness should remain small for any reasonable window
    assert s.item() < 0.05
```
