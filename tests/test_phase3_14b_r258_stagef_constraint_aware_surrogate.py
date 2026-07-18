from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef


def make_condition(rows: int = 48) -> np.ndarray:
    condition = np.zeros((rows, 243), dtype=np.float32)
    history = condition[:, : 3 * 67].reshape(rows, 3, 67)
    x = np.linspace(-0.23, 0.23, 24, dtype=np.float32)
    for row in range(rows):
        for step in range(3):
            points = np.stack(
                [
                    x + np.float32(0.001 * row),
                    np.float32(0.01 * step)
                    + np.float32(0.005)
                    * np.sin(np.linspace(0.0, np.pi, 24, dtype=np.float32)),
                ],
                axis=1,
            )
            history[row, step, :48] = points.reshape(-1)
            robot = history[row, step, 48:]
            robot[:6] = np.linspace(-0.2, 0.2, 6, dtype=np.float32)
            robot[6:12] = np.linspace(0.1, -0.1, 6, dtype=np.float32)
            robot[12:15] = np.asarray(
                [points[:, 0].mean() + 0.03, points[:, 1].mean() - 0.02, 0.15],
                dtype=np.float32,
            )
            robot[15:19] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    condition[:, 201:] = np.linspace(-0.5, 0.5, 42, dtype=np.float32)[None]
    return condition


def make_control(rows: int = 48) -> np.ndarray:
    output = np.zeros((rows, 4, 48), dtype=np.float32)
    x = np.linspace(-0.22, 0.22, 24, dtype=np.float32)
    for row in range(rows):
        for horizon in range(4):
            points = np.stack(
                [
                    x + np.float32(0.001 * row + 0.004 * horizon),
                    np.float32(0.01 * horizon)
                    + np.float32(0.004)
                    * np.sin(np.linspace(0.0, np.pi, 24, dtype=np.float32)),
                ],
                axis=1,
            )
            output[row, horizon] = points.reshape(-1)
    return output


def make_names(rows: int = 48) -> np.ndarray:
    values = np.asarray(stagef.CONDITION_VALUES)
    return values[np.arange(rows) % len(values)]


class FakeReference:
    def __init__(self) -> None:
        self.center_log = np.full((4, 23), np.log(0.02), dtype=np.float64)
        self.scale_log = np.full((4, 23), 0.25, dtype=np.float64)

    def validate(self) -> None:
        assert self.center_log.shape == (4, 23)
        assert self.scale_log.shape == (4, 23)


def make_context() -> dict:
    return {
        "stage_d_contract": SimpleNamespace(reference=FakeReference()),
    }


def make_oracle(control: np.ndarray, amount: float = 0.01) -> dict:
    delta = np.full_like(control, amount, dtype=np.float64)
    candidate = (control.astype(np.float64) + delta).astype(np.float32)
    feasible = np.ones(control.shape[0], dtype=np.bool_)
    return {
        "candidate": candidate,
        "projected_delta": delta,
        "feasible_mask": feasible,
        "candidate_sha256": stagef.sha256_array(candidate),
        "projected_delta_sha256": stagef.sha256_array(delta),
        "feasible_mask_sha256": stagef.sha256_array(feasible),
    }


def fake_metrics(mean: float = 0.6, nonnegative: float = 0.9) -> dict:
    return {
        "cosine": {"mean": mean},
        "nonnegative_rate": nonnegative,
        "by_condition": {
            name: {"cosine": {"mean": mean - 0.02}}
            for name in stagef.CONDITION_VALUES
        },
        "by_fold": {
            str(index): {"cosine": {"mean": mean - 0.04}}
            for index in range(6)
        },
    }


def fake_timestep_record(scientific: bool = True, direction: bool = True) -> dict:
    return {
        "projected_direction_gates": {"all": direction},
        "projected_direction_metrics": {"cosine": {"mean": 0.5}},
        "scientific_pass": scientific,
        "evaluation": {
            "target_distance_reduction_fraction": {"mean": 0.2},
            "normalized_mse_ratio": 0.5,
            "historical_physical_row_any_rate": 0.95,
        },
    }


def fake_candidate(
    candidate_id: str = "candidate",
    *,
    role: str = "selectable",
    eligible: bool = True,
    scientific: bool = True,
    direction: bool = True,
) -> dict:
    definition = copy.deepcopy(stagef.CANDIDATE_DEFINITIONS[3])
    return {
        "candidate_id": candidate_id,
        "definition": {**stagef.asdict(definition), "candidate_id": candidate_id},
        "role": role,
        "eligible": eligible,
        "timestep_records": {
            str(t): fake_timestep_record(scientific, direction)
            for t in stagef.TIMESTEPS
        },
    }


def test_phase_constant():
    assert stagef.PHASE == "Phase3.14b-r2.5.8 Stage F"


def test_base_commit_constant():
    assert stagef.BASE_EVIDENCE_COMMIT == "6758ea7ad800667a436b0243d3b1f6c63256d854"


def test_submodule_constant():
    assert stagef.EXPECTED_SUBMODULE_COMMIT.startswith("633a887")


def test_candidate_count():
    assert len(stagef.CANDIDATE_DEFINITIONS) == 13


def test_candidate_ids_unique():
    ids = [item.candidate_id for item in stagef.CANDIDATE_DEFINITIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("definition", stagef.CANDIDATE_DEFINITIONS)
def test_candidate_definitions_validate(definition):
    definition.validate()


def test_spec_validates():
    stagef.ConstraintAwareSpec().validate()


def test_spec_rejects_fold_change():
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(grouped_cv_folds=5).validate()


def test_spec_rejects_negative_threshold():
    with pytest.raises(ValueError):
        stagef.ConstraintAwareSpec(projected_direction_mean_min=-1.0).validate()


def test_definition_by_id():
    assert stagef.definition_by_id("full_point_rr64_feasible").output_rank == 64


def test_definition_by_id_rejects_unknown():
    with pytest.raises(KeyError):
        stagef.definition_by_id("missing")


def test_fixed_integrator_definition():
    value = stagef.fixed_integrator_definition()
    assert value.bound_mode == "robust_intersection"
    assert value.maximum_scale == 2.0


def test_projection_only_definition():
    value = stagef.projection_only_definition()
    assert value.direction_source_id == "zero"
    assert value.role == "diagnostic_control"


def test_stable_json_deterministic():
    assert stagef.stable_json_bytes({"b": 2, "a": 1}) == stagef.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        stagef.stable_json_bytes({"x": float("nan")})


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    stagef.atomic_write_once(path, b"{}\n")
    with pytest.raises(FileExistsError):
        stagef.atomic_write_once(path, b"{}\n")


def test_sha256_array_shape_sensitive():
    value = np.arange(6, dtype=np.float32)
    assert stagef.sha256_array(value) != stagef.sha256_array(value.reshape(2, 3))


def test_condition_parts():
    cable, robot, actions = stagef._condition_parts(make_condition(4))
    assert cable.shape == (4, 3, 24, 2)
    assert robot.shape == (4, 3, 19)
    assert actions.shape == (4, 42)


def test_control_points():
    assert stagef._control_points(make_control(4)).shape == (4, 4, 24, 2)


@pytest.mark.parametrize(
    "mode,dimension",
    [
        ("none", 0),
        ("condition_only", 2),
        ("anchor", 211),
        ("full_centered_constraint", 527),
        ("full_segment_constraint", 525),
    ],
)
def test_feature_dimensions(mode, dimension):
    result = stagef.build_constraint_features(
        condition=make_condition(12),
        control=make_control(12),
        condition_name=make_names(12),
        feature_mode=mode,
        context=make_context(),
    )
    assert result.shape == (12, dimension)


@pytest.mark.parametrize(
    "mode",
    ("anchor", "full_centered_constraint", "full_segment_constraint"),
)
def test_selectable_features_ignore_condition_labels(mode):
    condition = make_condition(12)
    control = make_control(12)
    names = make_names(12)
    first = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=names,
        feature_mode=mode,
        context=make_context(),
    )
    second = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=names[::-1],
        feature_mode=mode,
        context=make_context(),
    )
    assert np.array_equal(first, second)


def test_constraint_z_features_finite():
    z = stagef.constraint_z_features(make_control(8), make_context())
    assert z.shape == (8, 4, 23)
    assert np.all(np.isfinite(z))


def test_translation_audit_passes():
    audit = stagef.translation_invariance_audit(
        condition=make_condition(12),
        control=make_control(12),
        condition_name=make_names(12),
        context=make_context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert audit["all_selectable_modes_invariant"]


def test_point_target_roundtrip():
    control = make_control(8)
    candidate = control + np.float32(0.02)
    encoded = stagef.encode_projected_target(
        control=control,
        oracle_candidate=candidate,
        target_mode="projected_point",
    )
    decoded = stagef.decode_projected_prediction(
        control=control,
        encoded=encoded,
        target_mode="projected_point",
    )
    assert np.allclose(decoded, candidate - control)


def test_segment_target_roundtrip():
    control = make_control(8)
    candidate = control.copy().reshape(8, 4, 24, 2)
    candidate += np.asarray([0.01, -0.02], dtype=np.float32)[None, None, None]
    candidate[:, :, :, 1] += np.linspace(0.0, 0.01, 24, dtype=np.float32)[None, None]
    candidate = candidate.reshape(8, 4, 48)
    encoded = stagef.encode_projected_target(
        control=control,
        oracle_candidate=candidate,
        target_mode="projected_segment",
    )
    decoded = stagef.decode_projected_prediction(
        control=control,
        encoded=encoded,
        target_mode="projected_segment",
    )
    assert np.allclose(decoded, candidate - control, atol=1.0e-7)


def test_encode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        stagef.encode_projected_target(
            control=make_control(2),
            oracle_candidate=make_control(2),
            target_mode="bad",
        )


def test_decode_rejects_shape():
    with pytest.raises(ValueError):
        stagef.decode_projected_prediction(
            control=make_control(2),
            encoded=np.zeros((2, 3)),
            target_mode="projected_point",
        )


def test_candidate_seed_deterministic():
    assert stagef._candidate_seed("x", 3) == stagef._candidate_seed("x", 3)
    assert stagef._candidate_seed("x", 3) != stagef._candidate_seed("y", 3)


def test_ridge_solve():
    x = np.eye(4, dtype=np.float64)
    y = np.arange(8, dtype=np.float64).reshape(4, 2)
    coefficient = stagef._ridge_solve(x, y, alpha=1.0, jitter=1.0e-10)
    assert coefficient.shape == (4, 2)
    assert np.all(np.isfinite(coefficient))


@pytest.mark.parametrize(
    "candidate_id",
    (
        "projected_global_mean",
        "projected_condition_mean",
        "anchor_point_rr32_all",
        "full_point_rr64_feasible",
        "full_point_rff256_feasible",
    ),
)
def test_fit_predict_models(candidate_id):
    definition = stagef.definition_by_id(candidate_id)
    rows = 48
    feature_dim = 0 if definition.feature_mode == "none" else 12
    if definition.feature_mode == "condition_only":
        features = np.eye(2, dtype=np.float64)[np.arange(rows) % 2]
    else:
        random = np.random.RandomState(7)
        features = random.normal(size=(rows, feature_dim)).astype(np.float64)
    target = np.random.RandomState(8).normal(size=(rows, 192)).astype(np.float64)
    model = stagef.fit_constraint_model(
        definition=definition,
        features=features,
        encoded_target=target,
        condition_name=make_names(rows),
        fit_mask=np.ones(rows, dtype=np.bool_),
        spec=stagef.ConstraintAwareSpec(),
    )
    prediction = stagef.predict_constraint_model(
        definition=definition,
        model=model,
        features=features[:5],
        condition_name=make_names(rows)[:5],
    )
    assert prediction.shape == (5, 192)
    assert np.all(np.isfinite(prediction))


def test_model_rejects_small_fit_population():
    definition = stagef.definition_by_id("anchor_point_rr32_all")
    with pytest.raises(stagef.ConstraintAwareSurrogateError):
        stagef.fit_constraint_model(
            definition=definition,
            features=np.zeros((40, 5)),
            encoded_target=np.zeros((40, 192)),
            condition_name=make_names(40),
            fit_mask=np.arange(40) < 10,
            spec=stagef.ConstraintAwareSpec(),
        )


def test_rff_model_deterministic():
    definition = stagef.definition_by_id("full_point_rff256_feasible")
    features = np.random.RandomState(4).normal(size=(48, 9))
    target = np.random.RandomState(5).normal(size=(48, 192))
    kwargs = dict(
        definition=definition,
        features=features,
        encoded_target=target,
        condition_name=make_names(48),
        fit_mask=np.ones(48, dtype=np.bool_),
        spec=stagef.ConstraintAwareSpec(),
    )
    first = stagef.fit_constraint_model(**kwargs)
    second = stagef.fit_constraint_model(**kwargs)
    assert first["identity"]["model_sha256"] == second["identity"]["model_sha256"]


def test_fit_oof_projected_point():
    rows = 48
    control = make_control(rows)
    oracle = make_oracle(control)
    features = np.random.RandomState(2).normal(size=(rows, 15))
    result = stagef.fit_oof_candidate(
        definition=stagef.definition_by_id("anchor_point_rr32_all"),
        features=features,
        control=control,
        raw_target=control + 0.03,
        oracle_target=oracle,
        condition_name=make_names(rows),
        fold_assignment=np.arange(rows) % 6,
        spec=stagef.ConstraintAwareSpec(),
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
    )
    assert result["prediction"].shape == control.shape
    assert len(result["fold_records"]) == 6
    assert all(not item["test_target_used_for_fit"] for item in result["fold_records"])


def test_fit_oof_segment_target():
    rows = 48
    control = make_control(rows)
    oracle = make_oracle(control)
    result = stagef.fit_oof_candidate(
        definition=stagef.definition_by_id("segment_target_rr32_all"),
        features=np.random.RandomState(2).normal(size=(rows, 15)),
        control=control,
        raw_target=control + 0.03,
        oracle_target=oracle,
        condition_name=make_names(rows),
        fold_assignment=np.arange(rows) % 6,
        spec=stagef.ConstraintAwareSpec(),
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
    )
    assert result["prediction"].shape == control.shape


def test_fit_oof_permutation_flag():
    rows = 48
    control = make_control(rows)
    result = stagef.fit_oof_candidate(
        definition=stagef.definition_by_id("anchor_point_rr32_all"),
        features=np.random.RandomState(2).normal(size=(rows, 15)),
        control=control,
        raw_target=control + 0.03,
        oracle_target=make_oracle(control),
        condition_name=make_names(rows),
        fold_assignment=np.arange(rows) % 6,
        spec=stagef.ConstraintAwareSpec(),
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
        permutation_seed=17,
    )
    assert result["permuted_training_targets"]


def test_fit_oof_oracle_marks_target_use():
    rows = 48
    control = make_control(rows)
    result = stagef.fit_oof_candidate(
        definition=stagef.definition_by_id("projected_oracle_control"),
        features=np.zeros((rows, 0)),
        control=control,
        raw_target=control + 0.03,
        oracle_target=make_oracle(control),
        condition_name=make_names(rows),
        fold_assignment=np.arange(rows) % 6,
        spec=stagef.ConstraintAwareSpec(),
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
    )
    assert result["oracle_test_target_used_for_prediction"]


def test_projected_direction_gate_passes():
    result = stagef.projected_direction_gate(
        metrics=fake_metrics(),
        global_mean=0.1,
        condition_mean=0.2,
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["all"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", 0.1),
        ("nonnegative", 0.2),
    ],
)
def test_projected_direction_gate_fails(field, value):
    mean = value if field == "mean" else 0.6
    nonnegative = value if field == "nonnegative" else 0.9
    result = stagef.projected_direction_gate(
        metrics=fake_metrics(mean, nonnegative),
        global_mean=0.1,
        condition_mean=0.2,
        spec=stagef.ConstraintAwareSpec(),
    )
    assert not result["all"]


def test_holdout_direction_gate():
    assert stagef.holdout_projected_direction_gate(
        fake_metrics(), spec=stagef.ConstraintAwareSpec()
    )["all"]


def test_selection_none():
    assert stagef.select_candidate([fake_candidate(eligible=False)]) is None


def test_selection_deterministic():
    first = fake_candidate("a")
    second = fake_candidate("b")
    assert stagef.select_candidate([second, first])["candidate_id"] == "a"


def test_selection_key_stable():
    record = fake_candidate()
    assert stagef._selection_key(record) == stagef._selection_key(copy.deepcopy(record))


def oracle_targets(valid: bool = True) -> dict:
    return {
        timestep: {"evaluation": {"scientific_pass": valid}}
        for timestep in stagef.TIMESTEPS
    }


def test_classify_projected_oracle_failure():
    result = stagef.classify(
        records=[fake_candidate(eligible=False)],
        oracle_targets=oracle_targets(False),
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "projected_oracle_target"


def test_classify_no_direction_signal():
    result = stagef.classify(
        records=[fake_candidate(eligible=False, scientific=False, direction=False)],
        oracle_targets=oracle_targets(),
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "constraint_aware_direction_prediction"


def test_classify_not_integrable():
    result = stagef.classify(
        records=[fake_candidate(eligible=False, scientific=False, direction=True)],
        oracle_targets=oracle_targets(),
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "post_prediction_integrability"


def test_classify_permutation_failure():
    selected = fake_candidate()
    result = stagef.classify(
        records=[selected],
        oracle_targets=oracle_targets(),
        selected=selected,
        permutation={"all_pass": False},
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "permutation_shortcut"


def test_classify_projection_shortcut():
    selected = fake_candidate()
    result = stagef.classify(
        records=[selected],
        oracle_targets=oracle_targets(),
        selected=selected,
        permutation={"all_pass": True},
        projection_only={"pass": False},
        holdout=None,
    )
    assert result["primary_failure_locus"] == "projection_only_shortcut"


def test_classify_holdout_failure():
    selected = fake_candidate()
    result = stagef.classify(
        records=[selected],
        oracle_targets=oracle_targets(),
        selected=selected,
        permutation={"all_pass": True},
        projection_only={"pass": True},
        holdout={"scientific_pass": False},
    )
    assert result["primary_failure_locus"] == "holdout_generalization"


def test_classify_success():
    selected = fake_candidate()
    result = stagef.classify(
        records=[selected],
        oracle_targets=oracle_targets(),
        selected=selected,
        permutation={"all_pass": True},
        projection_only={"pass": True},
        holdout={"scientific_pass": True},
    )
    assert result["primary_failure_locus"] == "constraint_aware_surrogate_validated"


def test_compare_workers_exact():
    result = stagef.compare_worker_results({"a": 1}, {"a": 1})
    assert result["exact"]
    assert result["left_sha256"] == result["right_sha256"]


def test_compare_workers_detects_difference():
    assert not stagef.compare_worker_results({"a": 1}, {"a": 2})["exact"]


def test_status_paths_sorted(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "Experiment1"], cwd=tmp_path, check=True)
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert stagef.status_paths(tmp_path) == ("a.txt", "b.txt")


def test_selectable_model_ignores_condition_names():
    definition = stagef.definition_by_id("anchor_point_rr32_all")
    features = np.random.RandomState(31).normal(size=(48, 12))
    target = np.random.RandomState(32).normal(size=(48, 192))
    kwargs = dict(
        definition=definition,
        features=features,
        encoded_target=target,
        fit_mask=np.ones(48, dtype=np.bool_),
        spec=stagef.ConstraintAwareSpec(),
    )
    first = stagef.fit_constraint_model(condition_name=make_names(48), **kwargs)
    second = stagef.fit_constraint_model(condition_name=make_names(48)[::-1], **kwargs)
    assert first["identity"]["model_sha256"] == second["identity"]["model_sha256"]


def test_segment_decode_preserves_predicted_centroid_delta():
    control = make_control(3)
    encoded = np.zeros((3, 192), dtype=np.float64)
    centroid_delta = np.asarray(
        [[[0.01, -0.02], [0.02, -0.01], [0.03, 0.0], [0.04, 0.01]]],
        dtype=np.float64,
    )
    encoded[:, :8] = np.repeat(centroid_delta, 3, axis=0).reshape(3, 8)
    direction = stagef.decode_projected_prediction(
        control=control,
        encoded=encoded,
        target_mode="projected_segment",
    )
    control_points = control.reshape(3, 4, 24, 2)
    candidate_points = (control + direction).reshape(3, 4, 24, 2)
    observed = np.mean(candidate_points, axis=2) - np.mean(control_points, axis=2)
    assert np.allclose(observed, np.repeat(centroid_delta, 3, axis=0))


def test_generate_projected_oracle_target(monkeypatch):
    control = make_control(8)
    target = control + np.float32(0.02)

    def fake_integrate(**kwargs):
        candidate = target.copy()
        return {
            "candidate": candidate,
            "selected_scale": np.ones(8, dtype=np.float64),
            "observable_feasible_rate": 1.0,
            "candidate_sha256": stagef.sha256_array(candidate),
        }

    def fake_evaluate(**kwargs):
        return {"scientific_pass": True}

    monkeypatch.setattr(stagef.stagee258, "integrate_rowwise", fake_integrate)
    monkeypatch.setattr(stagef.stagee258, "evaluate_integrated_candidate", fake_evaluate)
    result = stagef.generate_projected_oracle_target(
        control=control,
        target=target,
        groups=np.asarray(["g{}".format(i) for i in range(8)]),
        condition_name=make_names(8),
        timestep=10,
        context={},
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
        integrator_spec=stagef.stagee258.ConstrainedIntegratorSpec(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["feasible_rate"] == 1.0
    assert result["target_used_for_supervision"]
    assert np.allclose(result["projected_delta"], 0.02)


def test_generate_projected_oracle_rejects_low_feasible(monkeypatch):
    control = make_control(8)

    def fake_integrate(**kwargs):
        return {
            "candidate": control.copy(),
            "selected_scale": np.zeros(8, dtype=np.float64),
            "observable_feasible_rate": 0.0,
        }

    monkeypatch.setattr(stagef.stagee258, "integrate_rowwise", fake_integrate)
    monkeypatch.setattr(
        stagef.stagee258,
        "evaluate_integrated_candidate",
        lambda **kwargs: {"scientific_pass": True},
    )
    with pytest.raises(stagef.ConstraintAwareSurrogateError):
        stagef.generate_projected_oracle_target(
            control=control,
            target=control + 0.02,
            groups=np.asarray(["g{}".format(i) for i in range(8)]),
            condition_name=make_names(8),
            timestep=10,
            context={},
            direction_spec=stagef.staged258.DirectionSurrogateSpec(),
            integrator_spec=stagef.stagee258.ConstrainedIntegratorSpec(),
            spec=stagef.ConstraintAwareSpec(),
        )


def test_projected_direction_gate_rejects_global_shortcut():
    result = stagef.projected_direction_gate(
        metrics=fake_metrics(mean=0.4, nonnegative=0.9),
        global_mean=0.35,
        condition_mean=0.2,
        spec=stagef.ConstraintAwareSpec(),
    )
    assert not result["global_margin"]
    assert not result["all"]


def test_feasible_fit_population_is_fold_local():
    rows = 48
    control = make_control(rows)
    oracle = make_oracle(control)
    oracle["feasible_mask"][::8] = False
    result = stagef.fit_oof_candidate(
        definition=stagef.definition_by_id("anchor_point_rr32_feasible"),
        features=np.random.RandomState(41).normal(size=(rows, 15)),
        control=control,
        raw_target=control + 0.03,
        oracle_target=oracle,
        condition_name=make_names(rows),
        fold_assignment=np.arange(rows) % 6,
        spec=stagef.ConstraintAwareSpec(),
        direction_spec=stagef.staged258.DirectionSurrogateSpec(),
    )
    assert all(item["fit_rows"] <= item["train_rows"] for item in result["fold_records"])
    assert any(item["fit_rows"] < item["train_rows"] for item in result["fold_records"])


def test_full_feature_modes_are_distinct():
    condition = make_condition(10)
    control = make_control(10)
    context = make_context()
    centered = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=make_names(10),
        feature_mode="full_centered_constraint",
        context=context,
    )
    segment = stagef.build_constraint_features(
        condition=condition,
        control=control,
        condition_name=make_names(10),
        feature_mode="full_segment_constraint",
        context=context,
    )
    assert centered.shape[1] != segment.shape[1]
    assert stagef.sha256_array(centered) != stagef.sha256_array(segment)


def test_projected_direction_gate_rejects_condition_shortcut():
    result = stagef.projected_direction_gate(
        metrics=fake_metrics(mean=0.40, nonnegative=0.90),
        global_mean=0.20,
        condition_mean=0.37,
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["global_margin"]
    assert not result["condition_margin"]
    assert not result["all"]


def test_fixed_integrator_is_derived_from_stagee_oracle():
    fixed = stagef.fixed_integrator_definition()
    oracle = stagef.oracle_integrator_definition()
    assert fixed.integration_mode == oracle.integration_mode
    assert fixed.bound_mode == oracle.bound_mode
    assert fixed.lower_z == oracle.lower_z
    assert fixed.upper_z == oracle.upper_z
    assert fixed.retention_min == oracle.retention_min
    assert fixed.maximum_scale == oracle.maximum_scale
