from pathlib import Path

import pytest

from state_diff.env.block_pushing.angle_elastic_soft_block import (
    load_angle_material_profile)


@pytest.mark.parametrize("name,stiffness", [
    ("a040", .0004), ("a055", .00055), ("a070", .0007)])
def test_strict_r3_profiles(name, stiffness):
    path = Path("configs/experiment3/material_profiles_r3") / (
        "soft_block_kv_angle_r3_{}_z025.json".format(name))
    edge, angle, payload = load_angle_material_profile(str(path))
    assert edge.structural_stiffness_n_per_m == 10.0
    assert angle.stiffness_n_m == stiffness
    assert payload["angle"]["expected_constraint_count"] == 121
