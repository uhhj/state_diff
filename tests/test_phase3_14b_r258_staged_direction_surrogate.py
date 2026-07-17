from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_staged_direction_surrogate as staged


def make_condition(rows=12):
    rng = np.random.RandomState(7)
    value = rng.normal(
        size=(rows, staged.stageb.CONDITION_DIM)
    ).astype(np.float32)
    history = value[
        :,
        : staged.stageb.HISTORY_STEPS
        * staged.stageb.STATE_DIM,
    ].reshape(
        rows,
        staged.stageb.HISTORY_STEPS,
        staged.stageb.STATE_DIM,
    )
    for row in range(rows):
        for step in range(
            staged.stageb.HISTORY_STEPS
        ):
            points = np.zeros(
                (staged.stageb.BEADS, 2),
                dtype=np.float32,
            )
            points[:, 0] = (
                np.linspace(
                    0.0,
                    1.0,
                    staged.stageb.BEADS,
                )
                + row * 0.01
                + step * 0.02
            )
            points[:, 1] = (
                0.1 * np.sin(
                    np.linspace(
                        0.0,
                        np.pi,
                        staged.stageb.BEADS,
                    )
                )
                + row * 0.005
            )
            history[
                row,
                step,
                : staged.stageb.CABLE_DIM,
            ] = points.reshape(-1)
    return value


def make_control(rows=12):
    control = np.zeros(
        (
            rows,
            staged.stageb.FUTURE_STEPS,
            staged.stageb.CABLE_DIM,
        ),
        dtype=np.float32,
    )
    for row in range(rows):
        for step in range(
            staged.stageb.FUTURE_STEPS
        ):
            points = np.zeros(
                (staged.stageb.BEADS, 2),
                dtype=np.float32,
            )
            points[:, 0] = (
                np.linspace(
                    0.0,
                    1.0,
                    staged.stageb.BEADS,
                )
                + row * 0.01
                + step * 0.03
            )
            points[:, 1] = (
                0.2 * np.sin(
                    np.linspace(
                        0.0,
                        np.pi,
                        staged.stageb.BEADS,
                    )
                )
                + row * 0.004
            )
            control[row, step] = (
                points.reshape(-1)
            )
    return control


def make_names(rows=12):
    return np.asarray(
        [
            staged.CONDITION_VALUES[
                index % 2
            ]
            for index in range(rows)
        ]
    )


def make_groups(rows=12):
    return np.asarray(
        [
            "g{:02d}".format(index // 2)
            for index in range(rows)
        ]
    )


def make_target_standardizer():
    return staged.stageb.ArrayStandardizer(
        mean=np.zeros(
            (
                staged.stageb.FUTURE_STEPS,
                staged.stageb.CABLE_DIM,
            ),
            dtype=np.float32,
        ),
        scale=np.ones(
            (
                staged.stageb.FUTURE_STEPS,
                staged.stageb.CABLE_DIM,
            ),
            dtype=np.float32,
        ),
        active=np.ones(
            (
                staged.stageb.FUTURE_STEPS,
                staged.stageb.CABLE_DIM,
            ),
            dtype=np.bool_,
        ),
    )


def simple_definition(
    *,
    candidate_id="centered_ridge_a1",
    feature_mode="centered_anchor",
    model_mode="ridge",
    alpha=1.0,
    rank=0,
    role="selectable",
):
    return staged.SurrogateDefinition(
        candidate_id,
        feature_mode,
        model_mode,
        alpha,
        rank,
        role,
    )


def direction_metrics(mean=0.5):
    by_condition = {
        name: {
            "rows": 4,
            "cosine": {
                "mean": mean,
            },
            "nonnegative_rate": 0.9,
        }
        for name in staged.CONDITION_VALUES
    }
    by_fold = {
        str(index): {
            "rows": 2,
            "cosine": {
                "mean": mean,
            },
            "nonnegative_rate": 0.9,
        }
        for index in range(6)
    }
    return {
        "cosine": {
            "mean": mean,
        },
        "nonnegative_rate": 0.9,
        "by_condition": by_condition,
        "by_fold": by_fold,
    }


def fake_evaluation(
    *,
    upper=1.0,
    nmse=0.8,
    lower=1.0,
    physical=1.0,
    collapse=0.0,
    stretch=0.0,
    movement=0.6,
    reduction=0.2,
):
    return {
        "upper_row_pass_rate": upper,
        "normalized_mse_ratio": nmse,
        "lower_row_pass_rate": lower,
        "historical_physical_row_any_rate":
            physical,
        "segment_length": {
            "collapse_fraction": collapse,
            "stretch_fraction": stretch,
        },
        "target_direction_cosine": {
            "mean": movement,
        },
        "target_distance_reduction_fraction": {
            "mean": reduction,
        },
    }


def fake_candidate_record(
    *,
    candidate_id="centered_rridge_k16_a10",
    role="selectable",
    eligible=True,
    direction_all=True,
    state_all=True,
):
    definition = {
        "candidate_id": candidate_id,
        "feature_mode": (
            "absolute_anchor"
            if role == "diagnostic_control"
            else "centered_anchor"
        ),
        "model_mode":
            "reduced_rank_ridge",
        "ridge_alpha": 10.0,
        "output_rank": 16,
        "role": role,
    }
    records = {}
    for timestep in (10, 25, 50):
        witness = (
            {
                "scale": 1.0,
                "evaluation":
                    fake_evaluation(),
            }
            if state_all
            else None
        )
        records[str(timestep)] = {
            "direction_metrics":
                direction_metrics(
                    0.5
                    if direction_all
                    else 0.0
                ),
            "direction_gates": {
                "all": direction_all,
            },
            "scale_sweep": {
                "scientific_reachable":
                    state_all,
                "scientific_witness":
                    witness,
            },
        }
    return {
        "candidate_id": candidate_id,
        "definition": definition,
        "timestep_records": records,
        "role": role,
        "eligible": eligible,
    }


def test_phase_constant():
    assert staged.PHASE == (
        "Phase3.14b-r2.5.8 Stage D"
    )


def test_base_commit_constant():
    assert staged.BASE_EVIDENCE_COMMIT == (
        "6866507c42b9bc9d2d271becd1a9423f61710405"
    )


def test_base_worker_constant():
    assert staged.EXPECTED_BASE_WORKER_SHA256 == (
        "8af4981760c2959667379221215c067fc"
        "d838b3e73349ab2af5517e197806913"
    )


def test_base_contract_constant():
    assert staged.EXPECTED_BASE_CONTRACT_SHA256 == (
        "355e3a12e13533b4a6f283344cfda1be"
        "20c4d6a4bef2fdd28f2ba03b3397b696"
    )


def test_base_selection_constant():
    assert staged.EXPECTED_BASE_SELECTION_SHA256 == (
        "613dd19b4611fddd24e5a5a71c29054b"
        "8a668a868d40198caccb42c82f4b3f50"
    )


def test_candidate_count():
    assert len(
        staged.SURROGATE_DEFINITIONS
    ) == 9


def test_candidate_order():
    assert tuple(
        value.candidate_id
        for value
        in staged.SURROGATE_DEFINITIONS
    ) == (
        "global_mean",
        "condition_mean",
        "absolute_rridge_k32_a10",
        "centered_ridge_a1",
        "centered_ridge_a10",
        "centered_rridge_k16_a10",
        "centered_rridge_k32_a10",
        "segment_rridge_k16_a10",
        "segment_rridge_k32_a10",
    )


@pytest.mark.parametrize(
    "definition",
    staged.SURROGATE_DEFINITIONS,
)
def test_all_definitions_validate(
    definition,
):
    definition.validate()


def test_selectable_rejects_absolute():
    value = simple_definition(
        feature_mode="absolute_anchor",
    )
    with pytest.raises(ValueError):
        value.validate()


def test_diagnostic_requires_absolute():
    value = simple_definition(
        feature_mode="centered_anchor",
        model_mode="reduced_rank_ridge",
        alpha=10.0,
        rank=32,
        role="diagnostic_control",
    )
    with pytest.raises(ValueError):
        value.validate()


def test_reduced_rank_rejects_rank():
    value = simple_definition(
        model_mode="reduced_rank_ridge",
        alpha=10.0,
        rank=8,
    )
    with pytest.raises(ValueError):
        value.validate()


def test_spec_validates():
    staged.DirectionSurrogateSpec().validate()


def test_spec_rejects_fold_count():
    with pytest.raises(ValueError):
        staged.DirectionSurrogateSpec(
            grouped_cv_folds=5
        ).validate()


def test_spec_rejects_scale_grid():
    with pytest.raises(ValueError):
        staged.DirectionSurrogateSpec(
            scale_grid=(1.0, 0.5)
        ).validate()


def test_spec_rejects_noise_offset():
    with pytest.raises(ValueError):
        staged.DirectionSurrogateSpec(
            objective_train_noise_seed_offset=1
        ).validate()


def test_stable_json_deterministic():
    assert staged.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == staged.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    staged.atomic_write_once(
        path,
        b"{}\n",
    )
    with pytest.raises(FileExistsError):
        staged.atomic_write_once(
            path,
            b"{}\n",
        )


def test_group_folds_keep_groups():
    groups = make_groups(24)
    assignment, mapping = (
        staged.deterministic_group_folds(
            groups,
            folds=6,
        )
    )
    assert len(mapping) == 12
    for group in set(groups.tolist()):
        assert len(
            set(
                assignment[
                    groups == group
                ].tolist()
            )
        ) == 1


def test_group_folds_reject_small_population():
    with pytest.raises(
        staged.DirectionSurrogateError
    ):
        staged.deterministic_group_folds(
            np.asarray(["a", "b"]),
            folds=6,
        )


def test_condition_one_hot():
    value = staged.condition_one_hot(
        make_names(4)
    )
    assert value.shape == (4, 2)
    assert np.all(
        np.sum(value, axis=1) == 1
    )


def test_condition_one_hot_rejects_unknown():
    with pytest.raises(
        staged.DirectionSurrogateError
    ):
        staged.condition_one_hot(
            ["unknown"]
        )


@pytest.mark.parametrize(
    "mode",
    (
        "centered_anchor",
        "segment_anchor",
        "absolute_anchor",
    ),
)
def test_feature_shape(mode):
    result = (
        staged.build_surrogate_features(
            condition=make_condition(),
            control=make_control(),
            condition_name=make_names(),
            feature_mode=mode,
        )
    )
    assert result.shape == (12, 211)


@pytest.mark.parametrize(
    "mode",
    (
        "centered_anchor",
        "segment_anchor",
        "absolute_anchor",
    ),
)
def test_deployable_features_ignore_condition_labels(
    mode,
):
    condition = make_condition()
    control = make_control()
    names = make_names()
    reversed_names = names[::-1].copy()
    first = staged.build_surrogate_features(
        condition=condition,
        control=control,
        condition_name=names,
        feature_mode=mode,
    )
    second = staged.build_surrogate_features(
        condition=condition,
        control=control,
        condition_name=reversed_names,
        feature_mode=mode,
    )
    assert np.array_equal(
        first,
        second,
    )


def test_none_feature_shape():
    result = staged.build_surrogate_features(
        condition=make_condition(),
        control=make_control(),
        condition_name=make_names(),
        feature_mode="none",
    )
    assert result.shape == (12, 0)


def test_condition_feature_shape():
    result = staged.build_surrogate_features(
        condition=make_condition(),
        control=make_control(),
        condition_name=make_names(),
        feature_mode="condition_only",
    )
    assert result.shape == (12, 2)


def test_centered_feature_translation_invariant():
    condition = make_condition()
    control = make_control()
    translated_condition, translated_control = (
        staged.translate_cable_inputs(
            condition=condition,
            control=control,
            offset_xy=(0.3, -0.5),
        )
    )
    original = staged.build_surrogate_features(
        condition=condition,
        control=control,
        condition_name=make_names(),
        feature_mode="centered_anchor",
    )
    translated = staged.build_surrogate_features(
        condition=translated_condition,
        control=translated_control,
        condition_name=make_names(),
        feature_mode="centered_anchor",
    )
    assert np.max(
        np.abs(original - translated)
    ) < 1.0e-6


def test_segment_feature_translation_invariant():
    condition = make_condition()
    control = make_control()
    translated_condition, translated_control = (
        staged.translate_cable_inputs(
            condition=condition,
            control=control,
            offset_xy=(0.3, -0.5),
        )
    )
    original = staged.build_surrogate_features(
        condition=condition,
        control=control,
        condition_name=make_names(),
        feature_mode="segment_anchor",
    )
    translated = staged.build_surrogate_features(
        condition=translated_condition,
        control=translated_control,
        condition_name=make_names(),
        feature_mode="segment_anchor",
    )
    assert np.max(
        np.abs(original - translated)
    ) < 1.0e-6


def test_absolute_feature_translation_sensitive():
    condition = make_condition()
    control = make_control()
    translated_condition, translated_control = (
        staged.translate_cable_inputs(
            condition=condition,
            control=control,
            offset_xy=(0.3, -0.5),
        )
    )
    original = staged.build_surrogate_features(
        condition=condition,
        control=control,
        condition_name=make_names(),
        feature_mode="absolute_anchor",
    )
    translated = staged.build_surrogate_features(
        condition=translated_condition,
        control=translated_control,
        condition_name=make_names(),
        feature_mode="absolute_anchor",
    )
    assert np.max(
        np.abs(original - translated)
    ) > 0.1


def test_translation_audit():
    result = (
        staged.translation_invariance_audit(
            condition=make_condition(),
            control=make_control(),
            condition_name=make_names(),
            spec=
                staged.DirectionSurrogateSpec(),
        )
    )
    assert result[
        "selectable_modes_invariant"
    ]
    assert result[
        "absolute_control_sensitive"
    ]


def test_feature_standardizer():
    x = np.asarray(
        [[1.0, 2.0], [3.0, 2.0]],
        dtype=np.float64,
    )
    result = (
        staged.fit_feature_standardizer(
            x,
            epsilon=1.0e-12,
        )
    )
    assert result["active"].tolist() == [
        True,
        False,
    ]
    transformed = (
        staged.apply_feature_standardizer(
            x,
            result,
        )
    )
    assert np.allclose(
        transformed[:, 1],
        0.0,
    )


def test_ridge_solve_exact_mapping():
    x = np.asarray(
        [[1.0], [2.0], [3.0]],
        dtype=np.float64,
    )
    y = 2.0 * x
    coefficient = staged._ridge_solve(
        x,
        y,
        alpha=1.0e-8,
        jitter=1.0e-12,
    )
    assert coefficient[0, 0] == pytest.approx(
        2.0,
        rel=1.0e-6,
    )


def test_global_mean_model():
    definition = staged.SURROGATE_DEFINITIONS[0]
    y = np.arange(
        12 * 4,
        dtype=np.float64,
    ).reshape(12, 2, 2)
    model = staged.fit_surrogate_model(
        definition=definition,
        features=np.zeros((12, 0)),
        target_residual=y,
        condition_name=make_names(),
        spec=staged.DirectionSurrogateSpec(),
    )
    prediction = (
        staged.predict_surrogate_model(
            definition=definition,
            model=model,
            features=np.zeros((3, 0)),
            condition_name=make_names(3),
            target_shape=(2, 2),
        )
    )
    assert np.allclose(
        prediction[0],
        np.mean(y, axis=0),
    )


def test_condition_mean_model():
    definition = staged.SURROGATE_DEFINITIONS[1]
    names = make_names()
    y = np.zeros((12, 2, 2))
    y[names == staged.CONDITION_VALUES[1]] = 2.0
    model = staged.fit_surrogate_model(
        definition=definition,
        features=staged.condition_one_hot(names),
        target_residual=y,
        condition_name=names,
        spec=staged.DirectionSurrogateSpec(),
    )
    prediction = (
        staged.predict_surrogate_model(
            definition=definition,
            model=model,
            features=staged.condition_one_hot(names),
            condition_name=names,
            target_shape=(2, 2),
        )
    )
    assert np.allclose(
        prediction[names == staged.CONDITION_VALUES[0]],
        0.0,
    )
    assert np.allclose(
        prediction[names == staged.CONDITION_VALUES[1]],
        2.0,
    )


def test_ridge_model_predicts_linear_target():
    rng = np.random.RandomState(11)
    x = rng.normal(size=(40, 6))
    weight = rng.normal(size=(6, 8))
    y = (x @ weight).reshape(40, 2, 4)
    definition = simple_definition()
    model = staged.fit_surrogate_model(
        definition=definition,
        features=x,
        target_residual=y,
        condition_name=make_names(40),
        spec=staged.DirectionSurrogateSpec(),
    )
    prediction = (
        staged.predict_surrogate_model(
            definition=definition,
            model=model,
            features=x,
            condition_name=make_names(40),
            target_shape=(2, 4),
        )
    )
    assert np.mean(
        (prediction - y) ** 2
    ) < 0.02


def test_reduced_rank_model_shape():
    rng = np.random.RandomState(13)
    x = rng.normal(size=(50, 10))
    y = rng.normal(size=(50, 4, 48))
    definition = simple_definition(
        candidate_id="rr",
        model_mode="reduced_rank_ridge",
        alpha=10.0,
        rank=16,
    )
    model = staged.fit_surrogate_model(
        definition=definition,
        features=x,
        target_residual=y,
        condition_name=make_names(50),
        spec=staged.DirectionSurrogateSpec(),
    )
    prediction = (
        staged.predict_surrogate_model(
            definition=definition,
            model=model,
            features=x,
            condition_name=make_names(50),
            target_shape=(4, 48),
        )
    )
    assert prediction.shape == y.shape
    assert model[
        "identity"
    ][
        "basis_orthogonality_error"
    ] < 1.0e-10


def test_active_residual_uses_all_raw_dimensions():
    control = np.zeros((2, 4, 48), np.float32)
    target = np.ones((2, 4, 48), np.float32)
    standardizer = make_target_standardizer()
    standardizer.active[0, 0] = False
    residual = staged.active_residual(
        control=control,
        target=target,
        target_standardizer=standardizer,
    )
    assert residual[0, 0, 0] == 1.0
    assert residual[0, 0, 1] == 1.0


def test_direction_metrics_parallel():
    target = np.ones((4, 2, 2))
    metrics = staged.row_direction_metrics(
        predicted_residual=target,
        target_residual=target,
        condition_name=make_names(4),
        fold_assignment=np.asarray(
            [0, 1, 2, 3]
        ),
        epsilon=1.0e-12,
    )
    assert metrics[
        "cosine"
    ]["mean"] == pytest.approx(1.0)


def test_direction_metrics_opposite():
    target = np.ones((4, 2, 2))
    metrics = staged.row_direction_metrics(
        predicted_residual=-target,
        target_residual=target,
        condition_name=make_names(4),
        fold_assignment=np.asarray(
            [0, 1, 2, 3]
        ),
        epsilon=1.0e-12,
    )
    assert metrics[
        "cosine"
    ]["mean"] == pytest.approx(-1.0)


def test_direction_gate_pass():
    gates = staged.direction_gate(
        metrics=direction_metrics(0.5),
        global_baseline_mean=0.1,
        condition_baseline_mean=0.2,
        spec=staged.DirectionSurrogateSpec(),
    )
    assert gates["all"]


def test_direction_gate_margin_fail():
    gates = staged.direction_gate(
        metrics=direction_metrics(0.3),
        global_baseline_mean=0.25,
        condition_baseline_mean=0.28,
        spec=staged.DirectionSurrogateSpec(),
    )
    assert not gates["all"]


def test_holdout_direction_gate_pass():
    gates = staged.holdout_direction_gate(
        metrics=direction_metrics(0.3),
        spec=staged.DirectionSurrogateSpec(),
    )
    assert gates["all"]


def test_state_candidate_scale():
    control = np.zeros((2, 4, 48), np.float32)
    residual = np.ones((2, 4, 48), np.float64)
    candidate = staged.state_candidate_from_residual(
        control=control,
        predicted_residual=residual,
        scale=0.5,
        target_standardizer=
            make_target_standardizer(),
    )
    assert np.allclose(candidate, 0.5)


def test_state_gates_pass():
    gates = staged.state_scientific_gates(
        timestep=10,
        evaluation=fake_evaluation(),
        spec=staged.DirectionSurrogateSpec(),
    )
    assert gates["all"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("upper", 0.5),
        ("nmse", 1.0),
        ("lower", 0.8),
        ("physical", 0.8),
        ("collapse", 0.1),
        ("stretch", 0.1),
        ("movement", 0.1),
        ("reduction", 0.01),
    ],
)
def test_state_gates_fail(field, value):
    gates = staged.state_scientific_gates(
        timestep=10,
        evaluation=fake_evaluation(
            **{field: value}
        ),
        spec=staged.DirectionSurrogateSpec(),
    )
    assert not gates["all"]


def test_oof_predictions_complete():
    rng = np.random.RandomState(17)
    rows = 24
    x = rng.normal(size=(rows, 5))
    y = rng.normal(size=(rows, 2, 4))
    groups = make_groups(rows)
    folds, _ = staged.deterministic_group_folds(
        groups,
        folds=6,
    )
    definition = simple_definition()
    result = staged.fit_oof_predictions(
        definition=definition,
        features=x,
        target_residual=y,
        condition_name=make_names(rows),
        fold_assignment=folds,
        spec=staged.DirectionSurrogateSpec(),
    )
    assert result[
        "prediction"
    ].shape == y.shape
    assert len(
        result["fold_records"]
    ) == 6


def test_oof_permutation_records_sha():
    rng = np.random.RandomState(19)
    rows = 24
    x = rng.normal(size=(rows, 5))
    y = rng.normal(size=(rows, 2, 4))
    folds, _ = staged.deterministic_group_folds(
        make_groups(rows),
        folds=6,
    )
    result = staged.fit_oof_predictions(
        definition=simple_definition(),
        features=x,
        target_residual=y,
        condition_name=make_names(rows),
        fold_assignment=folds,
        spec=staged.DirectionSurrogateSpec(),
        permutation_seed=123,
    )
    assert result[
        "permuted_training_targets"
    ]
    assert all(
        record[
            "permutation_sha256"
        ] is not None
        for record in result[
            "fold_records"
        ]
    )


def test_collect_baseline_means():
    records = []
    for candidate_id in (
        "global_mean",
        "condition_mean",
    ):
        records.append(
            {
                "candidate_id":
                    candidate_id,
                "timestep_records": {
                    str(timestep): {
                        "direction_metrics": {
                            "cosine": {
                                "mean":
                                    float(timestep)
                            }
                        }
                    }
                    for timestep
                    in (10, 25, 50)
                },
            }
        )
    result = staged.collect_baseline_means(
        records
    )
    assert result[
        "global_mean"
    ]["10"] == 10.0


def test_select_candidate_none():
    assert staged.select_oof_candidate(
        [
            fake_candidate_record(
                eligible=False
            )
        ]
    ) is None


def test_select_candidate():
    selected = staged.select_oof_candidate(
        [
            fake_candidate_record()
        ]
    )
    assert selected[
        "candidate_id"
    ] == "centered_rridge_k16_a10"
    assert selected[
        "locked_scales"
    ] == {
        "10": 1.0,
        "25": 1.0,
        "50": 1.0,
    }


def test_select_excludes_diagnostic():
    selected = staged.select_oof_candidate(
        [
            fake_candidate_record(
                candidate_id="absolute",
                role="diagnostic_control",
                eligible=True,
            )
        ]
    )
    assert selected is None


def test_classify_no_signal():
    selectable = fake_candidate_record(
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[
            selectable,
            diagnostic,
        ],
        selected=None,
        permutation=None,
        holdout=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "no_grouped_cv_direction_signal"


def test_classify_absolute_dependence():
    selectable = fake_candidate_record(
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=True,
        state_all=True,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[
            selectable,
            diagnostic,
        ],
        selected=None,
        permutation=None,
        holdout=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "absolute_position_dependence"


def test_classify_direction_state_candidate_mismatch():
    directional = fake_candidate_record(
        candidate_id="directional",
        eligible=False,
        direction_all=True,
        state_all=False,
    )
    safe = fake_candidate_record(
        candidate_id="safe",
        eligible=False,
        direction_all=False,
        state_all=True,
    )
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[
            directional,
            safe,
            diagnostic,
        ],
        selected=None,
        permutation=None,
        holdout=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "direction_state_candidate_mismatch"


def test_classify_direction_without_state():
    selectable = fake_candidate_record(
        eligible=False,
        direction_all=True,
        state_all=False,
    )
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[
            selectable,
            diagnostic,
        ],
        selected=None,
        permutation=None,
        holdout=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "direction_without_state_reachability"


def selected_stub():
    return {
        "candidate_id": "selected",
        "definition": {
            "candidate_id": "selected",
            "feature_mode":
                "centered_anchor",
            "model_mode":
                "reduced_rank_ridge",
            "ridge_alpha": 10.0,
            "output_rank": 16,
            "role": "selectable",
        },
        "locked_scales": {
            "10": 1.0,
            "25": 1.0,
            "50": 1.0,
        },
    }


def test_classify_permutation_leakage():
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[diagnostic],
        selected=selected_stub(),
        permutation={"all_pass": False},
        holdout=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "permutation_leakage"


def test_classify_holdout_direction_failure():
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[diagnostic],
        selected=selected_stub(),
        permutation={"all_pass": True},
        holdout={
            "direction_all_timesteps":
                False,
            "state_all_timesteps":
                False,
            "scientific_pass":
                False,
        },
    )
    assert result[
        "primary_failure_locus"
    ] == "holdout_direction_generalization"


def test_classify_integrator_needed():
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[diagnostic],
        selected=selected_stub(),
        permutation={"all_pass": True},
        holdout={
            "direction_all_timesteps":
                True,
            "state_all_timesteps":
                False,
            "scientific_pass":
                False,
        },
    )
    assert result[
        "primary_failure_locus"
    ] == "direction_without_holdout_state_reachability"


def test_classify_validated():
    diagnostic = fake_candidate_record(
        candidate_id="absolute",
        role="diagnostic_control",
        eligible=False,
        direction_all=False,
        state_all=False,
    )
    result = staged.classify_direction_surrogate(
        candidate_records=[diagnostic],
        selected=selected_stub(),
        permutation={"all_pass": True},
        holdout={
            "direction_all_timesteps":
                True,
            "state_all_timesteps":
                True,
            "scientific_pass":
                True,
        },
    )
    assert result[
        "primary_failure_locus"
    ] == "direction_surrogate_validated"


def fake_result():
    return {
        "root_cause": "root",
        "required_next_path": "next",
        "scientific_status": "BLOCKED",
        "environment": {"x": 1},
        "cold_main_worker_context": {
            "x": 1
        },
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "source_binding": {"x": 1},
        "objective_control_prediction_sha256":
            {"10": "a"},
        "translation_invariance": {
            "x": 1
        },
        "baseline_means": {"x": 1},
        "candidate_records": [{"x": 1}],
        "objective_train_selected_configuration":
            None,
        "permutation_control": None,
        "locked_holdout_evaluation": None,
        "classification": {"x": 1},
        "direction_surrogate_contract": {
            "x": 1
        },
        "selection": {"x": 1},
    }


def test_identity_projection():
    result = staged.identity_projection(
        fake_result()
    )
    assert result[
        "candidate_records"
    ] == [{"x": 1}]


def test_compare_workers_exact():
    left = fake_result()
    right = copy.deepcopy(left)
    result = staged.compare_worker_results(
        left,
        right,
    )
    assert result["exact"]
    assert result[
        "candidate_records_exact"
    ]


def test_compare_workers_detects_change():
    left = fake_result()
    right = copy.deepcopy(left)
    right["candidate_records"] = [
        {"x": 2}
    ]
    result = staged.compare_worker_results(
        left,
        right,
    )
    assert not result["exact"]
    assert not result[
        "candidate_records_exact"
    ]


def test_state_candidate_uses_all_raw_dimensions():
    standardizer = make_target_standardizer()
    standardizer.active[0, 0] = False
    control = np.zeros((2, 4, 48), np.float32)
    control[:, 0, 0] = 7.0
    residual = np.ones((2, 4, 48), np.float64)
    candidate = staged.state_candidate_from_residual(
        control=control,
        predicted_residual=residual,
        scale=1.0,
        target_standardizer=standardizer,
    )
    assert np.all(
        candidate[:, 0, 0] == 8.0
    )


def test_evaluate_scale_grid_selects_witness(
    monkeypatch,
):
    def fake_population(*args, **kwargs):
        return fake_evaluation()

    monkeypatch.setattr(
        staged.stageb258,
        "evaluate_output_population",
        fake_population,
    )
    rows = 4
    context = {
        "target_standardizer":
            make_target_standardizer(),
        "objective_contract":
            object(),
        "upper_gate": object(),
        "stage_d_contract": object(),
        "historical_geometry": object(),
    }
    result = staged.evaluate_scale_grid(
        predicted_residual=
            np.ones((rows, 4, 48)),
        control=
            np.zeros((rows, 4, 48), np.float32),
        target=
            np.ones((rows, 4, 48), np.float32),
        groups=
            np.asarray(
                ["g{}".format(index)
                 for index in range(rows)]
            ),
        condition_name=make_names(rows),
        timestep=10,
        context=context,
        spec=staged.DirectionSurrogateSpec(),
    )
    assert len(result["records"]) == 6
    assert result["scientific_reachable"]
    assert result[
        "scientific_witness"
    ]["scale"] == 0.5


def test_selection_key_is_deterministic():
    record = fake_candidate_record()
    first = staged._selection_key(record)
    second = staged._selection_key(
        copy.deepcopy(record)
    )
    assert first == second


def test_holdout_direction_gate_fails_condition():
    metrics = direction_metrics(0.3)
    metrics[
        "by_condition"
    ][
        staged.CONDITION_VALUES[1]
    ]["cosine"]["mean"] = 0.0
    gates = staged.holdout_direction_gate(
        metrics=metrics,
        spec=staged.DirectionSurrogateSpec(),
    )
    assert not gates["all"]
