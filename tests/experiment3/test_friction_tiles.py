import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.friction_tiles import (
    FrictionFloorConfig, FrictionTileFloor, tile_bounds)


def _area(bounds):
    return (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])


def _overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[2], b[2]))


def test_tile_partition_has_exact_area_and_no_positive_overlap():
    config = FrictionFloorConfig()
    bounds = tile_bounds(config)
    expected = (config.x_bounds[1] - config.x_bounds[0]) * (
        config.y_bounds[1] - config.y_bounds[0])
    assert np.isclose(sum(_area(value) for value in bounds.values()), expected)
    names = list(bounds)
    assert all(_overlap(bounds[names[i]], bounds[names[j]]) == 0
               for i in range(len(names)) for j in range(i + 1, len(names)))


def test_condition_switch_changes_patch_dynamics_only():
    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        floor = FrictionTileFloor(client, FrictionFloorConfig())
        ids = dict(floor.body_ids)
        poses = floor.poses()
        before = {name: client.getDynamicsInfo(body, -1)
                  for name, body in ids.items()}
        assert client.getVisualShapeData(floor.patch_body_id) == ()
        assert all(np.isclose(client.getAABB(body)[1][2], 0.0)
                   for body in ids.values())
        floor.set_condition("right_local_high")
        after = {name: client.getDynamicsInfo(body, -1)
                 for name, body in ids.items()}
        assert ids == floor.body_ids
        assert all(np.array_equal(poses[name], floor.poses()[name]) for name in ids)
        assert before["patch"][1] != after["patch"][1]
        assert all(before[name] == after[name] for name in ids if name != "patch")
    finally:
        client.disconnect()
