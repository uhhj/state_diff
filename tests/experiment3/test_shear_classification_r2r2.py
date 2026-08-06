import copy
from pathlib import Path

import pytest

from scripts.experiment3.phase0_soft_blockpush_r2r2.analyze_coupon import (
    classify_r2r2)
from scripts.experiment3.phase0_soft_blockpush_r2r2.common import load_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2r2.json"


def _metrics():
    return {"mode": "shear", "engineering_gate": {"finite": True},
            "common_gate": {"no_action": True, "anchor_drift": True,
                            "force_cap": True, "recovery": True,
                            "edge_ratios": True},
            "peak_primary_displacement_m": .008,
            "peak_rigid_aligned_rmse_m": .003,
            "peak_lateral_leakage_m": 0.0}


@pytest.mark.parametrize("field,value", [
    ("peak_primary_displacement_m", .033),
    ("peak_primary_displacement_m", .013),
    ("peak_rigid_aligned_rmse_m", .009)])
def test_large_finite_response_is_too_soft_before_edge_overflow(field, value):
    config = load_config(str(CONFIG)); metrics = _metrics()
    metrics[field] = value; metrics["common_gate"]["edge_ratios"] = False
    assert classify_r2r2(metrics, config) == "PHASE0B_R2_SHEAR_TOO_SOFT"


@pytest.mark.parametrize("field,value", [
    ("peak_primary_displacement_m", .001),
    ("peak_rigid_aligned_rmse_m", .0005)])
def test_small_response_is_too_stiff(field, value):
    config = load_config(str(CONFIG)); metrics = _metrics(); metrics[field] = value
    assert classify_r2r2(metrics, config) == "PHASE0B_R2_SHEAR_TOO_STIFF"


def test_remaining_stability_engineering_and_complete_order():
    config = load_config(str(CONFIG)); metrics = _metrics()
    metrics["common_gate"]["edge_ratios"] = False
    assert classify_r2r2(metrics, config).endswith("UNSTABLE")
    metrics = _metrics(); metrics["common_gate"]["force_cap"] = False
    assert classify_r2r2(metrics, config).endswith("UNSTABLE")
    metrics = _metrics(); metrics["engineering_gate"]["finite"] = False
    assert classify_r2r2(metrics, config).endswith("ENGINEERING_BLOCKED")
    assert classify_r2r2(_metrics(), config).endswith("COMPLETE")
