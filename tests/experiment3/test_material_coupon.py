from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush_r1.analyze_material_coupon import (
    classify_coupon)
from scripts.experiment3.phase0_soft_blockpush_r1.common import load_config
from state_diff.env.block_pushing.material_coupon import (
    MaterialCoupon, coupon_ramp_scale)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r1.json"
PROFILE = ROOT / "configs/experiment3/material_profiles/soft_block_kv_c0.json"


def test_coupon_faces_ramp_and_verdicts():
    config = load_config(str(CONFIG))
    assert coupon_ramp_scale(1, 120) > 0
    assert np.isclose(coupon_ramp_scale(120, 120), 1)
    stable = {"recovery_ratio": .2, "force_cap_fraction": 0,
              "edge_ratio_min": .9, "edge_ratio_max": 1.1,
              "no_action_visible_peak_m": 0}
    assert classify_coupon(True, .002, .004, stable, config) == "PHASE0B_R1_COUPON_COMPLETE"
    assert classify_coupon(True, .0001, .0002, stable, config) == "PHASE0B_R1_COUPON_TOO_RIGID"
    unstable = dict(stable, recovery_ratio=.9)
    assert classify_coupon(True, .002, .004, unstable, config) == "PHASE0B_R1_COUPON_UNSTABLE"


def test_real_short_coupon_smoke_and_force_schedule():
    config = load_config(str(CONFIG))
    for key in ("no_action_steps", "load_ramp_steps", "load_hold_steps", "recovery_steps"):
        config["coupon"][key] = 2
    coupon = MaterialCoupon(config, str(PROFILE))
    try:
        trace = coupon.run()
        assert len(coupon.anchor_indices) == 12 and len(coupon.load_indices) == 12
        assert [row["phase"] for row in trace] == [
            "no_action", "no_action", "load_ramp", "load_ramp",
            "load_hold", "load_hold", "recovery", "recovery"]
        assert trace[2]["external_load_scale"] > 0
        assert trace[3]["external_load_scale"] == 1
        assert np.isclose(np.linalg.norm(trace[4]["external_load_force_n"]), .12)
        assert np.array_equal(trace[-1]["external_load_force_n"], [0, 0, 0])
        assert all(np.all(np.isfinite(row["node_positions"])) for row in trace)
    finally:
        coupon.close()
