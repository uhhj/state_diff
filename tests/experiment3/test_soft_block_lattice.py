import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.soft_block_lattice import (
    SoftBlockConfig, SoftBlockLattice, build_edge_metadata,
    initial_node_positions, node_index)


def test_lattice_topology_and_geometry_are_fixed():
    config = SoftBlockConfig()
    edges = build_edge_metadata(config)
    assert config.num_nodes == 72
    assert node_index(0, 0, 0, config) == 0
    assert node_index(5, 3, 2, config) == 71
    assert {key: len(value) for key, value in edges.items()} == {
        "structural": 162, "shear": 242, "bending": 108}
    all_pairs = [(edge["a"], edge["b"]) for values in edges.values() for edge in values]
    assert len(all_pairs) == len(set(all_pairs))
    structural = np.asarray([edge["rest_length"] for edge in edges["structural"]])
    assert np.allclose(structural, 0.0105)
    first = initial_node_positions(config, (.4, -.15))
    second = initial_node_positions(config, (.4, -.15))
    assert first.shape == (72, 3)
    assert np.array_equal(first, second)


def test_lattice_short_bullet_settle_is_finite():
    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        client.setGravity(0, 0, -9.8)
        floor_shape = client.createCollisionShape(client.GEOM_PLANE)
        client.createMultiBody(0, floor_shape)
        lattice = SoftBlockLattice(client, SoftBlockConfig(), (.4, -.15))
        assert np.array_equal(lattice.bottom_indices, np.arange(24))
        assert np.array_equal(lattice.top_indices, np.arange(48, 72))
        for _ in range(5):
            client.stepSimulation()
        assert lattice.positions().shape == (72, 3)
        assert np.all(np.isfinite(lattice.positions()))
        assert np.all(np.isfinite(lattice.structural_edge_ratios()))
    finally:
        client.disconnect()
