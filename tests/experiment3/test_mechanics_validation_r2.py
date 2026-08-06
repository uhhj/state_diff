import copy
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_mechanics_validation import (
    select_smallest_converged)
from scripts.experiment3.phase0_soft_blockpush_r2.common import (
    load_config, load_profile, soft_block_config)
from state_diff.env.block_pushing.mechanics_validation import (
    run_cube_validation, run_two_node_validation)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiment3/soft_blockpush_phase0b_r2.json"
PROFILE = ROOT / "configs/experiment3/material_profiles_r2/soft_block_kv_r2_a8_z025.json"


def _summary(value=1.0, stable=True):
    return {"stable": stable, "peak_extension_m": value,
            "final_energy_ratio": value, "first_zero_crossing_time_s": value,
            "peak_primary_face_displacement_m": value,
            "peak_rigid_aligned_rmse_m": value, "recovery_ratio": value}


def test_convergence_selects_smallest_and_never_freezes_32_alone():
    config = load_config(str(CONFIG)); config = copy.deepcopy(config)
    config["mechanics_validation"]["convergence_relative_tolerance"] = .05
    two = {4: _summary(1), 8: _summary(1.01), 16: _summary(1.01), 32: _summary(1.01)}
    cube = copy.deepcopy(two)
    selected, comparisons = select_smallest_converged(two, cube, config)
    assert selected == 4 and comparisons["4_vs_8"]["passed"]
    two[4]["stable"] = two[8]["stable"] = two[16]["stable"] = False
    cube[4]["stable"] = cube[8]["stable"] = cube[16]["stable"] = False
    selected, _ = select_smallest_converged(two, cube, config)
    assert selected is None


def test_short_real_two_node_and_cube_are_finite_and_anchored():
    config = load_config(str(CONFIG)); config = copy.deepcopy(config)
    settings = config["mechanics_validation"]
    settings.update(two_node_duration_outer_steps=8, cube_no_action_outer_steps=2,
                    cube_ramp_outer_steps=2, cube_hold_outer_steps=2,
                    cube_recovery_outer_steps=2)
    material, _ = load_profile(str(PROFILE))
    two = run_two_node_validation(config, material, 4)
    cube = run_cube_validation(config, material, 4, soft_block_config(config, (2, 2, 2)))
    assert all(np.all(np.isfinite(value)) for value in two.values())
    assert np.sum(two["capped_force_count"]) == 0
    assert cube["phase"].tolist() == [
        "no_action", "no_action", "load_ramp", "load_ramp",
        "load_hold", "load_hold", "recovery", "recovery"]
    assert np.all(cube["force_evaluation_count"] > 0)
    assert np.max(cube["net_internal_force_residual"].astype(float)) <= 1e-10
