import json
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0e_ohj_audit.analyze_scientific_audit import (
    choose_route, gate4_decomposition)


REPO_ROOT = Path(__file__).resolve().parents[3]


def _source_config():
    return json.loads((
        REPO_ROOT / "configs/experiment3/phase0/"
        "ohj_cable_phase0d_r13_lslot.json").read_text(encoding="utf-8"))


def test_gate4_audit_separates_absolute_and_repeat_relative_terms():
    config = _source_config()
    pair = {
        "future_peak_visible_rmse_m": 0.004053722490296285,
        "repeat_peak_visible_rmse_m": 0.0006030962301782327,
    }
    result = gate4_decomposition(pair, config)
    assert np.isclose(result["absolute_floor_m"], 0.005)
    assert np.isclose(
        result["repeat_relative_floor_m"],
        3.0 * 0.0006030962301782327)
    assert result["absolute_pass"] is False
    assert result["repeat_relative_pass"] is True
    assert result["formal_pass"] is False


def test_control_relevant_plus_absolute_only_failure_routes_to_definition_review():
    gate4 = {
        "repeat_relative_pass": True,
        "absolute_pass": False,
        "formal_pass": False,
    }
    control = {"control_relevant": True}
    assert choose_route(gate4, control) == (
        "GATE4_ABSOLUTE_FLOOR_SCIENTIFIC_REVIEW_REQUIRED")


def test_missing_control_structure_routes_to_benchmark_review():
    gate4 = {
        "repeat_relative_pass": True,
        "absolute_pass": False,
        "formal_pass": False,
    }
    control = {"control_relevant": False}
    assert choose_route(gate4, control) == (
        "OHJ_CONTROL_STRUCTURE_INSUFFICIENT")


def test_audit_config_is_diagnostic_only_and_training_disabled():
    audit = json.loads((
        REPO_ROOT / "configs/experiment3/phase0e/"
        "ohj_r13_scientific_audit.json").read_text(encoding="utf-8"))
    assert audit["diagnostic_only"] is True
    assert audit["gate4_definition_audit"]["no_threshold_change"] is True
    assert audit["training"]["enabled"] is False
