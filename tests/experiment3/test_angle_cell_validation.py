import json

from state_diff.env.block_pushing.angle_cell_validation import (
    run_angle_cell_validation)
from state_diff.env.block_pushing.angle_elastic_soft_block import (
    load_angle_material_profile)


def test_angle_cell_returns_required_finite_trajectory():
    config = json.load(open("configs/experiment3/soft_blockpush_phase0b_r3.json"))
    config["angle_validation"]["outer_steps"] = 3
    _, angle, _ = load_angle_material_profile(
        "configs/experiment3/material_profiles_r3/soft_block_kv_angle_r3_a070_z025.json")
    result = run_angle_cell_validation(config, angle, 8)
    assert len(result["outer_step"]) == 3
    assert result["arm_lengths"].shape == (3, 2)
    assert result["cap_count"].sum() == 0
