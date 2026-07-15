from __future__ import annotations

import numpy as np
import pytest

from ccda_phase3.phase314b_r256_stageb_cable_diffusion import (
    BEADS,
    CABLE_DIM,
    CONDITION_DIM,
    FUTURE_STEPS,
    DiagnosticSpec,
    branch_metrics,
    cable_points,
    compare_worker_results,
    constant_velocity_baseline,
    deterministic_group_split,
    fit_geometry_contract,
    fit_standardizer,
    last_state_baseline,
    linear_beta_schedule,
    physical_validity,
    q_sample_numpy,
    scheduler_arrays,
    segment_lengths,
    self_intersection_count_single,
)


def synthetic_condition(rows: int) -> np.ndarray:
    condition = np.zeros((rows, CONDITION_DIM), dtype=np.float32)
    state_history = condition[:, : 3 * 67].reshape(rows, 3, 67)
    for row in range(rows):
        base = np.linspace(0.0, 1.0, CABLE_DIM, dtype=np.float32)
        state_history[row, 0, :CABLE_DIM] = base
        state_history[row, 1, :CABLE_DIM] = base + 0.1
        state_history[row, 2, :CABLE_DIM] = base + 0.2
    return condition


def straight_cable(offset: float = 0.0) -> np.ndarray:
    x = np.linspace(0.0, 1.0, BEADS, dtype=np.float32)
    y = np.full(BEADS, offset, dtype=np.float32)
    return np.stack([x, y], axis=1).reshape(CABLE_DIM)


def test_linear_beta_schedule_is_valid():
    beta = linear_beta_schedule(
        100,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
    )
    assert beta.shape == (100,)
    assert np.all(beta > 0.0)
    assert np.all(beta < 1.0)
    assert np.all(np.diff(beta) > 0.0)


def test_q_sample_zero_noise_matches_scaled_clean():
    spec = DiagnosticSpec(train_steps=1)
    scheduler = scheduler_arrays(spec)
    clean = np.ones((2, 4, 48), dtype=np.float32)
    timestep = np.asarray([0, 25], dtype=np.int64)
    result = q_sample_numpy(
        clean,
        np.zeros_like(clean),
        timestep,
        scheduler["alpha_bar"],
    )
    expected = np.broadcast_to(
        np.sqrt(scheduler["alpha_bar"][timestep])[:, None, None],
        clean.shape,
    )
    np.testing.assert_allclose(result, expected, atol=1.0e-7)


def test_group_split_keeps_groups_together():
    groups = np.asarray(
        ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]
    )
    train, probe, mapping = deterministic_group_split(
        groups,
        folds=4,
        probe_fold=0,
    )
    assert len(mapping) == 4
    for group in mapping:
        assert not (
            np.any(train[groups == group])
            and np.any(probe[groups == group])
        )


def test_baselines_use_state_history_only():
    condition = synthetic_condition(2)
    last = last_state_baseline(condition)
    velocity = constant_velocity_baseline(condition)
    assert last.shape == (2, 4, 48)
    assert velocity.shape == (2, 4, 48)
    np.testing.assert_allclose(
        velocity[:, 0] - last[:, 0],
        np.full((2, 48), 0.1, dtype=np.float32),
        atol=1.0e-6,
    )


def test_cable_points_and_segment_lengths():
    cable = straight_cable()
    points = cable_points(cable)
    lengths = segment_lengths(cable)
    assert points.shape == (BEADS, 2)
    assert lengths.shape == (BEADS - 1,)
    assert np.all(lengths > 0.0)


def test_straight_cable_has_no_self_intersection():
    assert self_intersection_count_single(straight_cable()) == 0


def test_geometry_contract_accepts_training_like_candidates():
    target = np.empty((10, 4, 48), dtype=np.float32)
    for row in range(10):
        for horizon in range(4):
            target[row, horizon] = straight_cable(
                offset=0.01 * row + 0.001 * horizon
            )
    contract = fit_geometry_contract(target)
    candidates = np.repeat(target[:2, None], 3, axis=1)
    result = physical_validity(candidates, contract)
    assert result["candidate_rate"] == 1.0


def test_standardizer_round_trip():
    rng = np.random.RandomState(0)
    value = rng.randn(20, 4, 48).astype(np.float32)
    standardizer = fit_standardizer(value)
    reconstructed = standardizer.denormalize(
        standardizer.normalize(value)
    )
    np.testing.assert_allclose(reconstructed, value, atol=1.0e-6)


def test_branch_metrics_detects_own_branch_support():
    target = np.zeros((2, 4, 48), dtype=np.float32)
    target[1] = 2.0
    candidates = np.zeros((2, 2, 4, 48), dtype=np.float32)
    candidates[0, 0] = 0.0
    candidates[0, 1] = 2.0
    candidates[1, 0] = 2.0
    candidates[1, 1] = 0.0
    result = branch_metrics(
        pair_key=["p", "p"],
        condition_name=["free", "hidden_slack_breakaway_pin_v2"],
        target=target,
        candidates=candidates,
    )
    assert result["row_own_branch_support_rate"] == 1.0


def test_compare_worker_results_uses_identity_projection():
    base = {
        "root_cause": "x",
        "required_next_path": "y",
        "identity": {"a": "1"},
        "training": {
            "loss_first": 1.0,
            "loss_final": 0.5,
            "loss_tail_mean": 0.5,
            "gradient_tail_mean": 1.0,
        },
        "evaluation": {"diagnostic_pass": False},
        "split": {"probe_rows": 2},
        "standardizers": {"x": "1"},
        "geometry_contract": {"x": "1"},
    }
    assert compare_worker_results(base, dict(base))["exact"]


def test_spec_rejects_wrong_reverse_steps():
    with pytest.raises(ValueError):
        DiagnosticSpec(reverse_steps=50).validate()
