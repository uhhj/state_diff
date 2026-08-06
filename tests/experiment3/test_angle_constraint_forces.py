import numpy as np

from state_diff.env.block_pushing.angle_elastic_soft_block import (
    evaluate_angle_constraints)


def evaluate(positions, velocities=None, damping=0.0):
    if velocities is None:
        velocities = np.zeros_like(positions)
    return evaluate_angle_constraints(
        positions, velocities, np.asarray([[0, 1, 2]]), np.asarray([0.0]),
        0.0007, damping, 10.0)


def test_force_balance_rest_and_orthogonal_extension():
    rest = evaluate(np.asarray([[0., 0, 0], [1., 0, 0], [0., 2., 0]]))
    total = rest.force_center_n + rest.force_first_n + rest.force_second_n
    assert np.allclose(total, 0, atol=1e-15)
    assert np.allclose(rest.force_center_n, 0, atol=1e-15)


def test_shear_restoring_force_and_damping_oppose_cosine_rate():
    positions = np.asarray([[0., 0, 0], [1., 0, 0], [.2, 1., 0]])
    elastic = evaluate(positions)
    assert elastic.cosine_error[0] > 0
    moving = evaluate(positions, np.asarray([[0., 0, 0], [0., 0, 0], [1., 0, 0]]), 1e-3)
    assert moving.cosine_rate[0] > 0
    assert np.linalg.norm(moving.force_first_n) > np.linalg.norm(elastic.force_first_n)


def test_rigid_transform_invariance():
    positions = np.asarray([[0., 0, 0], [1., 0, 0], [.2, 1., 0]])
    base = evaluate(positions)
    angle = .7
    rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    transformed = evaluate(positions @ rotation.T + [3., -2., .4])
    assert np.allclose(transformed.cosine_error, base.cosine_error)
    assert np.allclose(transformed.force_first_n, base.force_first_n @ rotation.T)


def test_force_matches_negative_finite_difference_energy_gradient():
    positions = np.asarray([[0., 0, 0], [1., 0, 0], [.2, 1., 0]])
    analytic = evaluate(positions).force_second_n[0, 0]
    epsilon = 1e-6
    plus, minus = positions.copy(), positions.copy()
    plus[2, 0] += epsilon; minus[2, 0] -= epsilon
    numerical = -(evaluate(plus).energy_j[0] - evaluate(minus).energy_j[0]) / (2 * epsilon)
    assert np.isclose(analytic, numerical, rtol=1e-6, atol=1e-10)
