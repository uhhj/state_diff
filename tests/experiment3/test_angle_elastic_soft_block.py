import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.angle_elastic_soft_block import (
    AngleElasticSoftBlock, load_angle_material_profile)
from state_diff.env.block_pushing.soft_block_lattice import SoftBlockConfig


def test_composite_block_counts_and_zero_rest_force():
    material, angle, _ = load_angle_material_profile(
        "configs/experiment3/material_profiles_r3/soft_block_kv_angle_r3_a055_z025.json")
    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        block = AngleElasticSoftBlock(
            client, SoftBlockConfig(linear_damping=0), material, angle, .3, (0, 0))
        assert len(block.angle_triplets) == 121
        assert block.angle_plane_counts == {"xy": 45, "xz": 40, "yz": 36}
        assert block.internal_evaluations_per_microstep == 633
        stats = block.apply_internal_forces()
        assert stats.force_evaluation_count == 633
        assert stats.energy_by_kind_j["angle"] < 1e-25
        assert stats.net_internal_force_residual_n < 1e-12
    finally:
        client.disconnect()
