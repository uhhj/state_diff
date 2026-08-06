import copy
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_material_coupon import (
    classify_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_config, select_next_profile)
from state_diff.env.block_pushing.material_coupon_r2 import MaterialCouponR2


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2.json"
PROFILE = ROOT / "configs/experiment3/material_profiles_r2/soft_block_kv_r2_a6_z025.json"


def test_coupon_verdicts_and_bounded_profile_selection():
    config = load_config(str(CONFIG))
    assert classify_coupon("axial", True, True, .004, .002, .001, config).endswith("COMPLETE")
    assert classify_coupon("axial", True, True, .001, .002, .001, config).endswith("TOO_STIFF")
    assert classify_coupon("axial", True, True, .02, .002, .001, config).endswith("TOO_SOFT")
    assert classify_coupon("shear", True, False, .004, .002, .001, config).endswith("UNSTABLE")
    assert classify_coupon("shear", False, True, .004, .002, .001, config).endswith("ENGINEERING_BLOCKED")
    assert select_next_profile("kv_r2_a6_z025", "PHASE0B_R2_AXIAL_TOO_SOFT", 1) == "kv_r2_a8_z025"
    assert select_next_profile("kv_r2_a6_z025", "PHASE0B_R2_AXIAL_TOO_STIFF", 1) == "kv_r2_a4_z025"
    assert select_next_profile("kv_r2_a6_z025", "PHASE0B_R2_AXIAL_TOO_SOFT", 2) is None


def test_short_axial_and_shear_coupon_fixture_and_force_schedule():
    for mode, direction in (("axial", [-1, 0, 0]), ("shear", [0, 1, 0])):
        config = copy.deepcopy(load_config(str(CONFIG)))
        for key in ("no_action_outer_steps", "load_ramp_outer_steps",
                    "load_hold_outer_steps", "recovery_outer_steps"):
            config["coupon"][key] = 2
        coupon = MaterialCouponR2(config, str(PROFILE), 2, mode)
        try:
            trace = coupon.run()
            assert len(coupon.anchor_indices) == len(coupon.load_indices) == 12
            assert coupon.client.getNumConstraints() == 0
            assert coupon.client.getNumBodies() == 72
            assert np.allclose(coupon.direction, direction)
            assert [row["phase"] for row in trace] == [
                "no_action", "no_action", "load_ramp", "load_ramp",
                "load_hold", "load_hold", "recovery", "recovery"]
            assert np.isclose(np.linalg.norm(trace[3]["external_load_force_n"]), .12)
            assert trace[3]["spring_force_evaluation_count"] == 512 * 2
            assert all(coupon.client.getDynamicsInfo(
                coupon.block.body_ids[int(i)], -1)[0] == 0 for i in coupon.anchor_indices)
            assert all(np.isfinite(row["primary_face_displacement_m"]) for row in trace)
        finally:
            coupon.close()
