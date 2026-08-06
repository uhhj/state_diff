import numpy as np
import pybullet
import pybullet_utils.bullet_client as bullet_client

from state_diff.env.block_pushing.kelvin_voigt_soft_block import (
    KelvinVoigtMaterial, KelvinVoigtSoftBlock, evaluate_edge_family,
    kelvin_voigt_edge_force)
from state_diff.env.block_pushing.soft_block_lattice import SoftBlockConfig


def test_vectorized_matches_scalar_for_rest_stretch_compression_and_damping():
    positions = np.asarray([[0, 0, 0], [1.2, 0, 0], [2.0, 0, 0]], float)
    velocities = np.asarray([[0, 0, 0], [.2, 0, 0], [-.1, 0, 0]], float)
    pairs = np.asarray([[0, 1], [1, 2]], int); rest = np.asarray([1., 1.])
    result = evaluate_edge_family(positions, velocities, pairs, rest, 2., .5, .25)
    scalar = [kelvin_voigt_edge_force(
        positions[a], velocities[a], positions[b], velocities[b], r, 2., .5, .25)
              for (a, b), r in zip(pairs, rest)]
    assert np.allclose(result.edge_forces_on_a, [row[0] for row in scalar])
    assert np.array_equal(result.capped, [row[1] for row in scalar])
    assert np.allclose(result.energy_j, [row[2] for row in scalar])
    assert np.max(result.uncapped_force_norm_n) > np.max(result.applied_force_norm_n)
    assert np.max(result.applied_force_norm_n) <= .25 + 1e-15

    rest_state = evaluate_edge_family(
        np.asarray([[0, 0, 0], [1, 0, 0]], float), np.zeros((2, 3)),
        np.asarray([[0, 1]]), np.asarray([1.]), 2., .5, 1.)
    assert np.array_equal(rest_state.edge_forces_on_a, np.zeros((1, 3)))


def test_vectorized_order_balanced_aggregation_and_static_default():
    pairs = np.asarray([[0, 2], [0, 1]], int)
    result = evaluate_edge_family(
        np.asarray([[0, 0, 0], [1.1, 0, 0], [0, 1.2, 0]], float),
        np.zeros((3, 3)), pairs, np.ones(2), 1., .1, 10.)
    aggregate = np.zeros((3, 3))
    np.add.at(aggregate, pairs[:, 0], result.edge_forces_on_a)
    np.add.at(aggregate, pairs[:, 1], -result.edge_forces_on_a)
    assert np.linalg.norm(np.sum(aggregate, axis=0)) <= 1e-15
    assert result.edge_lengths_m.tolist() == [1.2, 1.1]

    client = bullet_client.BulletClient(pybullet.DIRECT)
    try:
        block = KelvinVoigtSoftBlock(
            client, SoftBlockConfig(nx=2, ny=2, nz=2),
            KelvinVoigtMaterial(8, .02, 3, .01, .8, .004), .3, (0, 0))
        assert block.static_indices.size == 0
        masses = [client.getDynamicsInfo(body, -1)[0] for body in block.body_ids]
        assert all(mass > 0 for mass in masses)
        stats = block.apply_internal_forces()
        assert stats.force_evaluation_count == sum(len(v) for v in block.edges.values())
        assert stats.net_internal_force_residual_n <= 1e-15
    finally:
        client.disconnect()
