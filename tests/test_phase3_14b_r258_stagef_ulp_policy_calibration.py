from __future__ import annotations

import inspect

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef


def controls() -> dict:
    return {
        10: np.zeros((5, 4, 48), dtype=np.float32),
        25: np.zeros((5, 4, 48), dtype=np.float32),
        50: np.zeros((5, 4, 48), dtype=np.float32),
    }


def fake_result(*, factor: float, admitted: bool) -> dict:
    bound = factor * 0.004
    return {
        "rounding_model_covers_length": True,
        "formula_covers_observed": True,
        "maximum_observed_resolvable_z_difference": 0.003,
        "derived_bound": bound,
        "applied_bound": bound,
        "maximum_allowed_bound": 0.01,
        "admission_pass": admitted,
        "structural_zero_count": 2,
        "resolvable_segment_count": 458,
        "formula_diagnostic": {
            "factor": factor,
            "policy": {
                "holdout_used": False,
                "frozen_probe_used": False,
            },
        },
    }


def test_policy_selects_largest_predeclared_factor_admitted_on_all_timesteps(
    monkeypatch,
):
    calls = []

    def evaluator(*, control, offset_xy, context, spec, enforce_admission=True):
        del control, offset_xy, context
        calls.append((spec.translation_float32_z_ulp_factor, enforce_admission))
        factor = float(spec.translation_float32_z_ulp_factor)
        return fake_result(factor=factor, admitted=factor <= 2.0)

    monkeypatch.setattr(stagef, "_float32_constraint_z_bound", evaluator)
    result = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep=controls(),
        context={"objective_train_only": True},
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["selected_factor"] == 2.0
    assert result["eligible_factors"] == [1.0, 2.0]
    assert result["selection_witness"]["selected_is_largest_eligible"] is True
    assert len(calls) == 4 * 3
    assert all(enforce is False for _, enforce in calls)


def test_policy_requires_every_frozen_timestep(monkeypatch):
    def evaluator(*, control, offset_xy, context, spec, enforce_admission=True):
        del offset_xy, context, enforce_admission
        factor = float(spec.translation_float32_z_ulp_factor)
        # The t=50 synthetic control is marked by its first scalar.
        timestep_50 = bool(np.asarray(control).reshape(-1)[0] == 50.0)
        return fake_result(
            factor=factor,
            admitted=(factor <= 2.0 and not timestep_50),
        )

    values = controls()
    values[50][0, 0, 0] = 50.0
    monkeypatch.setattr(stagef, "_float32_constraint_z_bound", evaluator)
    with pytest.raises(stagef.ConstraintZULPAdmissionError) as captured:
        stagef.calibrate_float32_constraint_z_policy(
            controls_by_timestep=values,
            context={},
            spec=stagef.ConstraintAwareSpec(),
        )
    assert captured.value.required_next_path == (
        "REDESIGN_CONSTRAINT_Z_TRANSLATION_NUMERICS_ON_OBJECTIVE_TRAIN_ONLY"
    )
    assert captured.value.diagnostic["eligible_factors"] == []


def test_policy_rejects_changed_timestep_population():
    with pytest.raises(stagef.ConstraintAwareSurrogateError):
        stagef.calibrate_float32_constraint_z_policy(
            controls_by_timestep={10: controls()[10], 25: controls()[25]},
            context={},
            spec=stagef.ConstraintAwareSpec(),
        )


def test_policy_candidate_population_and_selection_rule_are_frozen():
    stagef.ConstraintAwareSpec().validate()
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(
            translation_float32_z_ulp_factor_candidates=(1.0, 8.0),
        ).validate()
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(
            translation_float32_z_policy_selection_rule="smallest",
        ).validate()


def test_active_factor_must_come_from_predeclared_population():
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(
            translation_float32_z_ulp_factor=1.5,
        ).validate()


def test_policy_does_not_change_frozen_maximum_or_infer_tolerance(monkeypatch):
    def evaluator(*, control, offset_xy, context, spec, enforce_admission=True):
        del control, offset_xy, context, enforce_admission
        return fake_result(
            factor=float(spec.translation_float32_z_ulp_factor),
            admitted=float(spec.translation_float32_z_ulp_factor) == 1.0,
        )

    monkeypatch.setattr(stagef, "_float32_constraint_z_bound", evaluator)
    result = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep=controls(),
        context={},
        spec=stagef.ConstraintAwareSpec(),
    )
    policy = result["policy"]
    assert policy["exact_gate_relaxed"] is False
    assert policy["maximum_allowed_bound_changed"] is False
    assert policy["automatic_tolerance_inferred"] is False
    assert result["predeclared_policy"]["maximum_allowed_bound"] == 0.01


def test_policy_population_is_objective_train_only(monkeypatch):
    def evaluator(*, control, offset_xy, context, spec, enforce_admission=True):
        del control, offset_xy, context, enforce_admission
        return fake_result(
            factor=float(spec.translation_float32_z_ulp_factor),
            admitted=float(spec.translation_float32_z_ulp_factor) == 1.0,
        )

    monkeypatch.setattr(stagef, "_float32_constraint_z_bound", evaluator)
    result = stagef.calibrate_float32_constraint_z_policy(
        controls_by_timestep=controls(),
        context={"selection_holdout": object(), "frozen_probe": object()},
        spec=stagef.ConstraintAwareSpec(),
    )
    population = result["population"]
    assert population["name"] == "objective_train_only"
    assert population["selection_holdout_accessed"] is False
    assert population["frozen_probe_accessed"] is False
    assert population["condition_label_used"] is False
    assert population["candidate_model_result_used"] is False


def test_run_calibration_locks_selected_factor_before_candidate_fitting():
    source = inspect.getsource(stagef.run_calibration)
    policy_index = source.index("calibrate_float32_constraint_z_policy")
    replace_index = source.index("translation_float32_z_ulp_factor=selected_ulp_factor")
    oracle_index = source.index("oracle_targets = {}")
    candidate_index = source.index("candidate_record(")
    assert policy_index < replace_index < oracle_index < candidate_index
    assert "selection_holdout" not in source[policy_index:replace_index]
    assert "frozen_probe" not in source[policy_index:replace_index]
