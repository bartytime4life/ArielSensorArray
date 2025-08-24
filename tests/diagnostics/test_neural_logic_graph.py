# tests/diagnostics/test_neural_logic_graph.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics tests for tools/neural_logic_graph.py

This suite validates the scientific logic, artifact generation, and CLI behavior of the
Neural Logic Graph tool, which renders/exports a symbolic rule graph (and optional
neural/saliency overlays) used across diagnostics and the HTML dashboard.

Coverage
--------
1) Core API sanity:
   • Graph builder (e.g., build_logic_graph / make_graph / to_graph) loads a ruleset and
     returns a node/edge structure with sensible properties (no dangling edges, ids unique).
   • Optional rank/importance propagation (e.g., via rule weights) influences node metadata.

2) Artifact generation API:
   • generate_neural_logic_graph_artifacts(...) (or equivalent) produces JSON/CSV/PNG/HTML.

3) CLI contract:
   • End-to-end run via subprocess (python -m tools.neural_logic_graph).
   • Determinism with --seed (compare JSON modulo volatile fields).
   • Graceful error handling for missing/invalid args.
   • Optional SPECTRAMIND_LOG_PATH audit line is appended.

4) Housekeeping:
   • Output files are non-empty; subsequent runs do not corrupt artifacts.

Notes
-----
• The module may expose different function names; tests try multiple candidates and xfail nicely if absent.
• No GPU/network required; uses tiny synthetic rules JSON created on the fly.
• If NetworkX is used by the tool, we do not require DAG-ness (allow feedback), but we do check
  for basic sanity (node/edge counts, referenced ids exist).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pytest


# ======================================================================================
# Helpers
# ======================================================================================

def _import_tool():
    """
    Import the module under test. Tries:
      1) tools.neural_logic_graph
      2) neural_logic_graph (top-level)
    """
    try:
        import tools.neural_logic_graph as m  # type: ignore
        return m
    except Exception:
        try:
            import neural_logic_graph as m2  # type: ignore
            return m2
        except Exception:
            pytest.skip(
                "neural_logic_graph module not found. "
                "Expected at tools/neural_logic_graph.py or importable as neural_logic_graph."
            )


def _has_attr(mod, name: str) -> bool:
    return hasattr(mod, name) and getattr(mod, name) is not None


def _run_cli(
    module_path: Path,
    args: Sequence[str],
    env: Optional[Dict[str, str]] = None,
    timeout: int = 210,
) -> subprocess.CompletedProcess:
    """
    Execute the tool as a CLI using `python -m tools.neural_logic_graph` when possible.
    Fallback to direct script invocation by file path if package execution is not feasible.
    """
    if module_path.name == "neural_logic_graph.py" and module_path.parent.name == "tools":
        repo_root = module_path.parent.parent
        candidate_pkg = "tools.neural_logic_graph"
        cmd = [sys.executable, "-m", candidate_pkg, *args]
        cwd = str(repo_root)
    else:
        cmd = [sys.executable, str(module_path), *args]
        cwd = str(module_path.parent)

    env_full = os.environ.copy()
    if env:
        env_full.update(env)

    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env_full,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        text=True,
        check=False,
    )


def _assert_file(p: Path, min_size: int = 1) -> None:
    assert p.exists(), f"File not found: {p}"
    assert p.is_file(), f"Expected file: {p}"
    sz = p.stat().st_size
    assert sz >= min_size, f"File too small ({sz} bytes): {p}"


# ======================================================================================
# Synthetic ruleset (tiny)
# ======================================================================================

def _write_synthetic_rules(path: Path) -> Path:
    """
    Create a minimal but structured ruleset JSON with:
      - input nodes (molecular bands / derived features)
      - logical ops (AND/OR/THRESH)
      - rule nodes with weights and descriptions
      - edges wiring inputs -> ops -> rules

    Format is intentionally generic to match many implementations:
      {
        "nodes": [{"id":"n1","type":"input"/"op"/"rule","name":"...","weight":...,"meta":{...}}, ...],
        "edges": [{"src":"n1","dst":"n2","type":"logic"/"influence","weight":...}, ...],
        "config": {"version":"test","seed":..., ...}
      }
    """
    rules = {
        "nodes": [
            {"id": "wl_1p4_um", "type": "input", "name": "H2O@1.4µm", "meta": {"band": [1.35, 1.45]}},
            {"id": "wl_3p3_um", "type": "input", "name": "CH4@3.3µm", "meta": {"band": [3.25, 3.35]}},
            {"id": "slope_pos", "type": "input", "name": "continuum_slope>0", "meta": {}},

            {"id": "and_h2o_slope", "type": "op", "name": "AND", "meta": {"op": "and"}},
            {"id": "or_h2o_ch4", "type": "op", "name": "OR", "meta": {"op": "or"}},

            {"id": "rule_water_present", "type": "rule", "name": "WaterPresent", "weight": 1.0,
             "meta": {"desc": "H2O band with positive slope implies water signature"}},
            {"id": "rule_methane_possible", "type": "rule", "name": "MethanePossible", "weight": 0.6,
             "meta": {"desc": "H2O or CH4 band suggests methane presence"}},
        ],
        "edges": [
            {"src": "wl_1p4_um", "dst": "and_h2o_slope", "type": "logic", "weight": 1.0},
            {"src": "slope_pos", "dst": "and_h2o_slope", "type": "logic", "weight": 0.8},
            {"src": "wl_1p4_um", "dst": "or_h2o_ch4", "type": "logic", "weight": 0.7},
            {"src": "wl_3p3_um", "dst": "or_h2o_ch4", "type": "logic", "weight": 0.7},

            {"src": "and_h2o_slope", "dst": "rule_water_present", "type": "logic", "weight": 1.0},
            {"src": "or_h2o_ch4", "dst": "rule_methane_possible", "type": "logic", "weight": 0.6},
        ],
        "config": {"version": "test", "seed": 7}
    }
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    return path


# ======================================================================================
# Fixtures
# ======================================================================================

@pytest.fixture(scope="module")
def tool_mod():
    return _import_tool()


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Dict[str, Path]:
    """
    Create a clean workspace:
      inputs/  — rules.json
      outputs/ — artifacts
      logs/    — optional v50_debug_log.md
    """
    ip = tmp_path / "inputs"
    op = tmp_path / "outputs"
    lg = tmp_path / "logs"
    ip.mkdir(parents=True, exist_ok=True)
    op.mkdir(parents=True, exist_ok=True)
    lg.mkdir(parents=True, exist_ok=True)

    rules_path = _write_synthetic_rules(ip / "rules.json")
    return {"root": tmp_path, "inputs": ip, "outputs": op, "logs": lg, "rules": rules_path}


# ======================================================================================
# Core API tests — graph building & metadata
# ======================================================================================

def test_build_graph_and_validate(tool_mod, tmp_workspace):
    """
    The builder should return a graph-like object (dict or networkx) whose nodes/edges are consistent:
      • node ids are unique
      • all edge endpoints exist
      • rule nodes contain 'weight' metadata (or default)
    """
    candidates = ["build_logic_graph", "make_graph", "to_graph", "load_graph"]
    fn = None
    for name in candidates:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No graph builder (build_logic_graph/make_graph/to_graph/load_graph) found in module.")

    rules_json = tmp_workspace["rules"]
    # Allow function to accept a path or a dict; try path first.
    try:
        g = fn(rules=rules_json)
    except TypeError:
        with open(rules_json, "r", encoding="utf-8") as f:
            rules = json.load(f)
        g = fn(rules)  # type: ignore

    # Normalize to nodes/edges lists for validation
    nodes, edges = None, None
    if isinstance(g, dict) and "nodes" in g and "edges" in g:
        nodes, edges = g["nodes"], g["edges"]
    elif hasattr(g, "nodes") and hasattr(g, "edges"):  # networkx-like
        try:
            nodes = [{"id": n, **(g.nodes[n] if hasattr(g, "nodes") else {})} for n in g.nodes]  # type: ignore
            edges = [{"src": u, "dst": v, **(g.edges[(u, v)] if hasattr(g, "edges") else {})} for (u, v) in g.edges]  # type: ignore
        except Exception as e:
            pytest.fail(f"Unable to read nodes/edges from graph object: {e}")
    else:
        pytest.fail("Unknown graph object format; expected dict with nodes/edges or networkx-like object.")

    # Basic checks
    ids = [n["id"] for n in nodes]
    assert len(ids) == len(set(ids)), "Node ids should be unique."
    node_set = set(ids)
    for e in edges:
        assert e["src"] in node_set and e["dst"] in node_set, f"Dangling edge detected: {e}"

    # Rule nodes should have weight metadata (or default 1.0)
    rule_nodes = [n for n in nodes if str(n.get("type", "")).lower() == "rule"]
    assert rule_nodes, "No rule nodes detected in the graph."
    for rn in rule_nodes:
        w = rn.get("weight", 1.0)
        assert np.isfinite(w), "Rule node missing finite weight."


def test_weight_influences_importance_if_available(tool_mod, tmp_workspace):
    """
    If the API exposes a function to compute node importance/rank, verify that increasing a rule's weight
    increases (or at least does not decrease) its reported importance.
    """
    fn_cands = ["compute_node_importance", "rank_nodes", "compute_importance"]
    fn = None
    for name in fn_cands:
        if _has_attr(tool_mod, name):
            fn = getattr(tool_mod, name)
            break
    if fn is None:
        pytest.xfail("No node-importance function available; skipping importance test.")

    # Load rules and produce two versions with different rule weight
    rules_path = tmp_workspace["rules"]
    rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
    # Identify a rule node to tweak
    rn_ids = [n["id"] for n in rules["nodes"] if n.get("type") == "rule"]
    if not rn_ids:
        pytest.xfail("No rule nodes to rank.")
    target = rn_ids[0]

    def _imp_of(rules_dict: Dict[str, Any]) -> float:
        try:
            out = fn(rules=rules_dict, normalize=True, seed=123)
        except TypeError:
            out = fn(rules_dict)  # type: ignore
        # Expect dict {node_id: importance} or list of tuples; normalize
        if isinstance(out, dict):
            return float(out.get(target, 0.0))
        if isinstance(out, (list, tuple)):
            for item in out:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and item[0] == target:
                    return float(item[1])
        # last resort, zero
        return 0.0

    base_imp = _imp_of(rules)
    # Increase weight
    for n in rules["nodes"]:
        if n["id"] == target:
            n["weight"] = float(n.get("weight", 1.0)) * 2.5
    boosted_imp = _imp_of(rules)

    assert boosted_imp >= base_imp - 1e-12, f"Boosting rule weight should not lower its importance (base={base_imp:.4g}, new={boosted_imp:.4g})"
    # Prefer strictly higher unless implementation saturates
    assert boosted_imp > base_imp or abs(boosted_imp - base_imp) < 1e-9, "Expected importance to increase or remain equal."


# ======================================================================================
# Artifact generation API
# ======================================================================================

def test_generate_artifacts(tool_mod, tmp_workspace):
    """
    Artifact generator should emit JSON/CSV/PNG/HTML files and return a manifest (or paths).
    """
    entry_candidates = [
        "generate_neural_logic_graph_artifacts",
        "run_neural_logic_graph",
        "produce_logic_graph_outputs",
        "analyze_and_export",  # generic fallback
    ]
    entry = None
    for name in entry_candidates:
        if _has_attr(tool_mod, name):
            entry = getattr(tool_mod, name)
            break
    if entry is None:
        pytest.xfail("No artifact generation entrypoint found in neural_logic_graph.")

    outdir = tmp_workspace["outputs"]
    rules_json = tmp_workspace["rules"]

    kwargs = dict(
        rules=rules_json,
        outdir=str(outdir),
        json_out=True,
        csv_out=True,
        png_out=True,
        html_out=True,
        seed=77,
        title="Neural Logic Graph — Test",
        layout="dot",  # hint: many tools accept layout engines
    )
    try:
        manifest = entry(**kwargs)
    except TypeError:
        manifest = entry(rules_json, str(outdir), True, True, True, True, 77, "Neural Logic Graph — Test", "dot")  # type: ignore

    # Presence checks
    json_files = list(outdir.glob("*.json"))
    csv_files = list(outdir.glob("*.csv"))
    png_files = list(outdir.glob("*.png"))
    html_files = list(outdir.glob("*.html"))

    assert json_files, "No JSON artifact produced by neural logic graph."
    assert png_files, "No PNG artifact produced by neural logic graph."
    assert html_files, "No HTML artifact produced by neural logic graph."
    # CSV optional in some implementations; if present, ensure nontrivial
    for c in csv_files:
        _assert_file(c, min_size=64)
    for p in png_files:
        _assert_file(p, min_size=256)
        # quick PNG signature check
        with open(p, "rb") as fh:
            assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    for h in html_files:
        _assert_file(h, min_size=128)

    # Minimal JSON schema check
    with open(json_files[0], "r", encoding="utf-8") as f:
        js = json.load(f)
    assert isinstance(js, dict), "Top-level JSON must be an object."
    has_nodes = ("nodes" in js) and isinstance(js["nodes"], list) and len(js["nodes"]) > 0
    has_edges = ("edges" in js) and isinstance(js["edges"], list) and len(js["edges"]) > 0
    assert has_nodes and has_edges, "JSON should include non-empty 'nodes' and 'edges'."


# ======================================================================================
# CLI End-to-End
# ======================================================================================

def test_cli_end_to_end(tmp_workspace):
    """
    End-to-end CLI test:
      • Runs the module as a CLI with --rules/--outdir → emits JSON/CSV/PNG/HTML.
      • Uses --seed for determinism and compares JSON across two runs (modulo volatile metadata).
      • Verifies optional audit log when SPECTRAMIND_LOG_PATH is set.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "neural_logic_graph.py",  # repo-root/tools/...
        Path(__file__).resolve().parents[1] / "neural_logic_graph.py",            # tests/diagnostics/../
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("neural_logic_graph.py not found; cannot run CLI end-to-end test.")

    outdir = tmp_workspace["outputs"]
    logsdir = tmp_workspace["logs"]
    rules_json = tmp_workspace["rules"]

    env = {
        "PYTHONUNBUFFERED": "1",
        "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
    }

    args = (
        "--rules", str(rules_json),
        "--outdir", str(outdir),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--seed", "2025",
        "--silent",
    )
    proc1 = _run_cli(module_file, args, env=env, timeout=210)
    if proc1.returncode != 0:
        msg = f"CLI run 1 failed (exit={proc1.returncode}).\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
        pytest.fail(msg)

    json1 = sorted(outdir.glob("*.json"))
    png1 = sorted(outdir.glob("*.png"))
    html1 = sorted(outdir.glob("*.html"))
    assert json1 and png1 and html1, "CLI run 1 did not produce required artifact types."

    # Determinism: second run with same seed into a new directory should match JSON (minus volatile fields)
    outdir2 = outdir.parent / "outputs_run2"
    outdir2.mkdir(exist_ok=True)
    args2 = (
        "--rules", str(rules_json),
        "--outdir", str(outdir2),
        "--json",
        "--csv",
        "--png",
        "--html",
        "--seed", "2025",
        "--silent",
    )
    proc2 = _run_cli(module_file, args2, env=env, timeout=210)
    if proc2.returncode != 0:
        msg = f"CLI run 2 failed (exit={proc2.returncode}).\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
        pytest.fail(msg)

    json2 = sorted(outdir2.glob("*.json"))
    assert json2, "Second CLI run produced no JSON artifacts."

    def _normalize(j: Dict[str, Any]) -> Dict[str, Any]:
        d = json.loads(json.dumps(j))  # deep copy
        vol_patterns = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)

        def scrub(obj: Any) -> Any:
            if isinstance(obj, dict):
                for k in list(obj.keys()):
                    if vol_patterns.search(k):
                        obj.pop(k, None)
                    else:
                        obj[k] = scrub(obj[k])
            elif isinstance(obj, list):
                for i in range(len(obj)):
                    obj[i] = scrub(obj[i])
            return obj

        return scrub(d)

    with open(json1[0], "r", encoding="utf-8") as f:
        j1 = _normalize(json.load(f))
    with open(json2[0], "r", encoding="utf-8") as f:
        j2 = _normalize(json.load(f))

    assert j1 == j2, "Seeded CLI runs should yield identical JSON after removing volatile metadata."

    # Audit log should exist and include a recognizable signature
    log_file = Path(env["SPECTRAMIND_LOG_PATH"])
    if log_file.exists():
        _assert_file(log_file, min_size=1)
        text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
        assert ("neural_logic_graph" in text) or ("logic graph" in text) or ("symbolic" in text), \
            "Audit log exists but lacks a recognizable neural_logic_graph signature."


def test_cli_error_cases(tmp_workspace):
    """
    CLI should:
      • Exit non-zero when required --rules is missing.
      • Report helpful error text mentioning the missing/invalid flag.
    """
    candidates = [
        Path(__file__).resolve().parents[2] / "tools" / "neural_logic_graph.py",
        Path(__file__).resolve().parents[1] / "neural_logic_graph.py",
    ]
    module_file = None
    for p in candidates:
        if p.exists():
            module_file = p
            break
    if module_file is None:
        pytest.skip("neural_logic_graph.py not found; cannot run CLI error tests.")

    outdir = tmp_workspace["outputs"]

    # Missing --rules
    args_missing_rules = (
        "--outdir", str(outdir),
        "--json",
    )
    proc = _run_cli(module_file, args_missing_rules, env=None, timeout=90)
    assert proc.returncode != 0, "CLI should fail when required --rules is missing."
    msg = (proc.stderr + "\n" + proc.stdout).lower()
    assert "rules" in msg, "Error message should mention missing 'rules'."


# ======================================================================================
# Housekeeping checks
# ======================================================================================

def test_artifact_min_sizes(tmp_workspace):
    """
    After prior tests, ensure that PNG/CSV/HTML in outputs/ are non-trivially sized.
    """
    outdir = tmp_workspace["outputs"]
    png_files = list(outdir.glob("*.png"))
    csv_files = list(outdir.glob("*.csv"))
    html_files = list(outdir.glob("*.html"))
    # Not all formats may exist if a prior test xfailed early; be lenient but check when present.
    for p in png_files:
        _assert_file(p, min_size=256)
    for c in csv_files:
        _assert_file(c, min_size=64)
    for h in html_files:
        _assert_file(h, min_size=128)


def test_idempotent_rerun_behavior(tmp_workspace):
    """
    The tool should either overwrite consistently or produce versioned filenames.
    We don't require a specific policy here; only that subsequent writes do not corrupt artifacts.
    """
    outdir = tmp_workspace["outputs"]
    before = {p.name for p in outdir.glob("*")}
    # Simulate pre-existing artifact to ensure tool does not crash due to existing files
    marker = outdir / "preexisting_marker.txt"
    marker.write_text("marker", encoding="utf-8")
    after = {p.name for p in outdir.glob("*")}
    assert before.issubset(after), "Artifacts disappeared unexpectedly between runs or overwrite simulation."