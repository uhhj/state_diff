from __future__ import annotations

import inspect

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stageg_joint_direction_feasibility as stageg


def test_stageg_candidate_population_is_frozen() -> None:
    assert stageg.BASE_DIRECTION_IDS == (
        "anchor_point_rr32_all",
        "anchor_point_rr32_feasible",
        "full_point_rr32_all",
        "full_point_rr32_feasible",
        "full_point_rr64_feasible",
        "segment_target_rr32_all",
        "segment_target_rr32_feasible",
        "segment_target_rr64_feasible",
        "full_point_rff256_feasible",
    )
    assert len(stageg.JOINT_CANDIDATE_DEFINITIONS) == 9


def test_scale_bank_is_predeclared_and_positive() -> None:
    spec = stageg.JointDirectionFeasibilitySpec()
    spec.validate()
    assert spec.scale_multipliers == (0.25, 0.5, 1.0, 2.0)


def test_invalid_scale_bank_is_rejected() -> None:
    spec = stageg.JointDirectionFeasibilitySpec(
        scale_multipliers=(0.5, 0.25, 1.0, 2.0)
    )
    with pytest.raises(ValueError, match="scale bank"):
        spec.validate()


def test_row_target_distance_reduction_has_expected_sign() -> None:
    control = np.zeros((2, 1, 2), dtype=np.float64)
    target = np.ones_like(control)
    candidate = np.asarray([[[0.5, 0.5]], [[-1.0, -1.0]]], dtype=np.float64)
    result = stageg.row_target_distance_reduction(
        control=control,
        candidate=candidate,
        target=target,
        epsilon=1.0e-12,
    )
    assert result[0] > 0.0
    assert result[1] < 0.0


def test_scalar_ridge_fits_linear_signal() -> None:
    x = np.arange(20, dtype=np.float64)[:, None]
    y = 1.5 + 2.0 * x[:, 0]
    model = stageg._fit_scalar_ridge(x, y, alpha=1.0e-8, epsilon=1.0e-12)
    predicted = stageg._predict_scalar_ridge(model, x)
    assert np.max(np.abs(predicted - y)) < 1.0e-5


def test_logistic_constant_population_is_explicit() -> None:
    x = np.zeros((12, 3), dtype=np.float64)
    y = np.ones(12, dtype=np.float64)
    model = stageg._fit_logistic_irls(
        x,
        y,
        l2=1.0,
        max_iterations=48,
        tolerance=1.0e-10,
        epsilon=1.0e-12,
    )
    assert model["mode"] == "constant"
    predicted = stageg._predict_logistic(model, x)
    assert np.all(predicted > 0.999)


def test_logistic_separates_simple_population() -> None:
    x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float64)
    y = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    model = stageg._fit_logistic_irls(
        x,
        y,
        l2=0.1,
        max_iterations=48,
        tolerance=1.0e-10,
        epsilon=1.0e-12,
    )
    probability = stageg._predict_logistic(model, x)
    assert probability[0] < probability[-1]
    assert probability[1] < 0.5 < probability[2]


def test_binary_metrics_compare_prevalence_baseline() -> None:
    probability = np.asarray([0.1, 0.2, 0.8, 0.9])
    target = np.asarray([False, False, True, True])
    metrics = stageg.binary_metrics(
        probability=probability,
        target=target,
        cutoff=0.5,
    )
    assert metrics["brier_improvement"] > 0.0
    assert metrics["balanced_accuracy"] == 1.0


def _fake_bank() -> dict:
    rows = 3
    bank = 4
    direction = np.zeros((rows, bank, 1, 2), dtype=np.float64)
    for index in range(bank):
        direction[:, index, 0, 0] = float(index + 1)
    return {
        "directions": direction,
        "descriptors": np.zeros((rows, bank, 2), dtype=np.float64),
        "observable_feasible": np.asarray(
            [
                [True, True, True, True],
                [False, True, True, False],
                [False, False, False, False],
            ],
            dtype=np.bool_,
        ),
        "candidates": np.zeros((rows, bank, 1, 2), dtype=np.float64),
    }


def test_score_bank_uses_probability_cutoff_and_observable_mask(monkeypatch) -> None:
    bank = _fake_bank()
    monkeypatch.setattr(
        stageg,
        "_predict_scalar_ridge",
        lambda _model, _x: np.asarray(
            [0.0, 1.0, 2.0, 3.0] * 3, dtype=np.float64
        ),
    )
    monkeypatch.setattr(
        stageg,
        "_predict_logistic",
        lambda _model, _x: np.asarray(
            [0.1, 0.6, 0.7, 0.8, 0.9, 0.2, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
            dtype=np.float64,
        ),
    )
    scoring = stageg.score_integrated_bank(
        ranker={"utility_model": {}, "feasibility_model": {}},
        bank=bank,
        spec=stageg.JointDirectionFeasibilitySpec(),
    )
    assert scoring["selected_index"].tolist() == [3, 2, 0]
    assert scoring["fallback"].tolist() == [False, False, True]


def test_gather_selected_direction_is_rowwise() -> None:
    bank = _fake_bank()
    selected = stageg.gather_selected_direction(
        bank=bank,
        selected_index=np.asarray([3, 1, 0]),
    )
    assert selected[:, 0, 0].tolist() == [4.0, 2.0, 1.0]


def test_ranker_metrics_reports_fixed_scale_margin() -> None:
    utility = np.asarray(
        [[0.0, 0.1, 0.2, 0.3], [0.0, 0.1, 0.2, 0.3]], dtype=np.float64
    )
    labels = {
        "utility": utility,
        "beneficial": utility > 0.0,
    }
    scoring = {
        "predicted_utility": utility.copy(),
        "predicted_probability": np.where(utility > 0.0, 0.9, 0.1),
        "selected_index": np.asarray([3, 3]),
        "fallback": np.asarray([False, False]),
    }
    bank = {
        "observable_feasible": np.ones_like(utility, dtype=np.bool_),
    }
    metrics = stageg.ranker_metrics(
        scoring=scoring,
        labels=labels,
        bank=bank,
        spec=stageg.JointDirectionFeasibilitySpec(
            fixed_scale_reduction_margin_min=1.0e-6
        ),
    )
    assert metrics["selected_mean_reduction"] == pytest.approx(0.3)
    assert metrics["best_fixed_mean_reduction"] == pytest.approx(0.3)
    assert metrics["ranker_margin_over_best_fixed"] == pytest.approx(0.0)


def test_fit_ranker_permutation_changes_training_identity() -> None:
    descriptors = np.arange(48, dtype=np.float64).reshape(6, 4, 2)
    utility = np.linspace(-0.5, 0.5, 24).reshape(6, 4)
    labels = {"utility": utility, "beneficial": utility > 0.0}
    spec = stageg.JointDirectionFeasibilitySpec()
    plain = stageg.fit_feasibility_ranker(
        descriptors=descriptors,
        labels=labels,
        spec=spec,
    )
    permuted = stageg.fit_feasibility_ranker(
        descriptors=descriptors,
        labels=labels,
        spec=spec,
        permutation_seed=spec.ranker_seed,
    )
    assert plain["identity"]["permuted_labels"] is False
    assert permuted["identity"]["permuted_labels"] is True
    assert (
        plain["identity"]["training_utility_sha256"]
        != permuted["identity"]["training_utility_sha256"]
    )


def test_ranker_descriptor_api_has_no_target_argument() -> None:
    signature = inspect.signature(stageg.candidate_descriptors)
    assert "target" not in signature.parameters
    source = inspect.getsource(stageg.candidate_descriptors)
    assert "holdout_target" not in source
    assert "frozen_probe" not in source


def test_holdout_target_is_opened_after_locked_selection() -> None:
    source = inspect.getsource(stageg.locked_holdout_evaluation)
    selection_offset = source.index("_fit_full_joint_and_select")
    target_offset = source.index('context["holdout_target"]')
    assert selection_offset < target_offset
    assert '"holdout_target_used_for_ranker_selection": False' in source


def test_stageg_does_not_modify_stagee_or_stagef_contract() -> None:
    source = inspect.getsource(stageg.run_calibration)
    assert "stagef.calibrate_float32_constraint_z_policy" in source
    assert "EXPECTED_SELECTED_ULP_FACTOR" in source
    assert "stagee_integrator_modified\": False" in source
    assert "stagef_scientific_source_modified\": False" in source
