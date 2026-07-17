from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r258_stagee_constrained_integrator as stagee


def make_cable(rows: int = 4, length: float = 0.10) -> np.ndarray:
    x = (np.arange(stageb.BEADS, dtype=np.float32) - 11.5) * float(length)
    points = np.zeros((rows, stageb.FUTURE_STEPS, stageb.BEADS, 2), dtype=np.float32)
    points[..., 0] = x[None, None]
    for row in range(rows):
        points[row, :, :, 1] = 0.01 * row
    return points.reshape(rows, stageb.FUTURE_STEPS, stageb.CABLE_DIM)


def make_reference(length: float = 0.10, scale: float = 0.25):
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    center = np.full(shape, np.log(length), dtype=np.float64)
    scale_array = np.full(shape, scale, dtype=np.float64)
    return staged.SegmentReference(
        center_log=center,
        scale_log=scale_array,
        mad_scale=scale_array.copy(),
        iqr_scale=scale_array.copy(),
        scale_floor=1.0e-6,
    )


def make_context() -> dict:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    historical = stageb.GeometryContract(
        segment_lower=np.full(shape, 0.05, dtype=np.float32),
        segment_upper=np.full(shape, 0.20, dtype=np.float32),
        coordinate_abs_max=5.0,
        target_intersection_max=0,
    )
    reference = make_reference()
    return {
        "historical_geometry": historical,
        "stage_d_contract": SimpleNamespace(
            reference=reference,
            lower_threshold=4.0,
        ),
        "upper_gate": SimpleNamespace(
            upper_threshold=4.0,
        ),
    }


def selectable_definition() -> stagee.IntegratorDefinition:
    return next(
        item
        for item in stagee.INTEGRATOR_DEFINITIONS
        if item.candidate_id == "centered_a10_z4_m1_r25"
    )


def fake_evaluation(pass_state: bool = True) -> dict:
    return {
        "normalized_mse_ratio": 0.80 if pass_state else 1.20,
        "historical_physical_row_any_rate": 0.98,
        "target_direction_cosine": {"mean": 0.60},
        "target_distance_reduction_fraction": {"mean": 0.40},
        "segment_length": {
            "collapse_fraction": 0.01,
            "stretch_fraction": 0.01,
        },
        "upper_row_pass_rate": 0.90,
        "lower_row_pass_rate": 0.95,
    }


def fake_timestep_record(
    *,
    scientific: bool,
    observable: bool = True,
    direction: bool = True,
    reduction: float = 0.40,
) -> dict:
    evaluation = fake_evaluation(scientific)
    evaluation["target_distance_reduction_fraction"]["mean"] = reduction
    return {
        "scientific_pass": scientific,
        "observable_gates": {"all": observable},
        "state_gates": {"all": scientific},
        "direction_source_record": {
            "direction_gates": {"all": direction},
            "direction_metrics": {"cosine": {"mean": 0.5}},
        },
        "integration": {"fallback_rate": 0.0},
        "evaluation": evaluation,
    }


def fake_candidate(
    candidate_id: str,
    *,
    role: str = "selectable",
    scientific: bool = False,
    observable: bool = True,
    eligible: bool = False,
    source: str = "centered_ridge_a10",
    reduction: float = 0.40,
) -> dict:
    records = {
        str(t): fake_timestep_record(
            scientific=scientific,
            observable=observable,
            reduction=reduction,
        )
        for t in (10, 25, 50)
    }
    if role in ("oracle_control", "diagnostic_control", "negative_control"):
        for item in records.values():
            item["direction_source_record"] = None
    return {
        "candidate_id": candidate_id,
        "definition": {
            "candidate_id": candidate_id,
            "direction_source_id": source,
            "integration_mode": "segment_reconstruct",
            "bound_mode": "robust_intersection",
            "lower_z": 4.0,
            "upper_z": 4.0,
            "retention_min": 0.25,
            "maximum_scale": 1.0,
            "role": role,
        },
        "role": role,
        "timestep_records": records,
        "eligible": eligible,
    }


def test_phase_constant():
    assert stagee.PHASE == "Phase3.14b-r2.5.8 Stage E"


def test_base_commit_constant():
    assert stagee.BASE_EVIDENCE_COMMIT == "61be1c377eeda6da4c25e21399e160beb9a33249"


def test_submodule_constant():
    assert stagee.EXPECTED_SUBMODULE_COMMIT == "633a88752445cf5d6776ed374fdbbdb35f93050c"


def test_direction_source_population():
    assert stagee.DIRECTION_SOURCE_IDS == (
        "centered_ridge_a10",
        "centered_rridge_k32_a10",
        "segment_rridge_k32_a10",
    )


def test_candidate_population_and_order():
    spec = stagee.ConstrainedIntegratorSpec()
    spec.validate()
    assert len(stagee.INTEGRATOR_DEFINITIONS) == 13
    assert stagee.INTEGRATOR_DEFINITIONS[0].candidate_id == "linear_centered_a10_m2"
    assert stagee.INTEGRATOR_DEFINITIONS[-1].candidate_id == "oracle_z4_m2_r25"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"integration_mode": "bad"},
        {"bound_mode": "bad"},
        {"role": "bad"},
        {"direction_source_id": "bad"},
        {"retention_min": -0.1},
        {"retention_min": 1.1},
        {"maximum_scale": 0.0},
    ],
)
def test_definition_rejects_invalid_common_fields(kwargs):
    values = dict(
        candidate_id="x",
        direction_source_id="centered_ridge_a10",
        integration_mode="segment_reconstruct",
        bound_mode="robust_intersection",
        lower_z=4.0,
        upper_z=4.0,
        retention_min=0.25,
        maximum_scale=1.0,
        role="selectable",
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(**values).validate()


def test_definition_rejects_linear_bounds():
    value = copy.copy(selectable_definition())
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(
            **{**value.__dict__, "integration_mode": "linear"}
        ).validate()


def test_definition_rejects_reconstruction_without_bounds():
    value = copy.copy(selectable_definition())
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(
            **{**value.__dict__, "bound_mode": "none"}
        ).validate()


def test_definition_rejects_nonrobust_z():
    value = stagee.INTEGRATOR_DEFINITIONS[1]
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(
            **{**value.__dict__, "lower_z": 1.0}
        ).validate()


def test_definition_rejects_non_deployable_selectable():
    value = selectable_definition()
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(
            **{**value.__dict__, "direction_source_id": "oracle"}
        ).validate()


def test_definition_rejects_mislabeled_oracle():
    value = stagee.INTEGRATOR_DEFINITIONS[-1]
    with pytest.raises(ValueError):
        stagee.IntegratorDefinition(
            **{**value.__dict__, "direction_source_id": "zero"}
        ).validate()


def test_spec_rejects_unsorted_scale_grid():
    with pytest.raises(ValueError):
        stagee.ConstrainedIntegratorSpec(scale_grid=(0.5, 0.25)).validate()


def test_spec_rejects_wrong_fold_count():
    with pytest.raises(ValueError):
        stagee.ConstrainedIntegratorSpec(grouped_cv_folds=5).validate()


def test_spec_rejects_wrong_seed():
    with pytest.raises(ValueError):
        stagee.ConstrainedIntegratorSpec(objective_train_noise_seed_offset=1).validate()


def test_sha_array_binds_dtype_and_shape():
    a = np.zeros((2, 3), dtype=np.float32)
    assert stagee.sha256_array(a) != stagee.sha256_array(a.astype(np.float64))
    assert stagee.sha256_array(a) != stagee.sha256_array(a.reshape(3, 2))


def test_stable_json_deterministic():
    assert stagee.stable_json_bytes({"b": 2, "a": 1}) == stagee.stable_json_bytes({"a": 1, "b": 2})


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        stagee.stable_json_bytes({"x": float("nan")})


def test_atomic_write_once(tmp_path):
    path = tmp_path / "a.json"
    stagee.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        stagee.atomic_write_once(path, b"{}\n")


def test_safe_stats_empty():
    assert stagee._safe_stats(np.asarray([]))["count"] == 0


def test_safe_stats_values():
    result = stagee._safe_stats(np.asarray([1.0, 2.0, 3.0]))
    assert result["mean"] == 2.0
    assert result["median"] == 2.0


def test_row_cosine_identity():
    value = np.eye(3, dtype=np.float64).reshape(3, 1, 3)
    assert np.allclose(stagee._row_cosine(value, value, epsilon=1e-12), 1.0)


def test_row_cosine_opposite():
    value = np.ones((2, 1, 3), dtype=np.float64)
    assert np.allclose(stagee._row_cosine(value, -value, epsilon=1e-12), -1.0)


def test_definition_by_id():
    assert stagee.definition_by_id("centered_ridge_a10").candidate_id == "centered_ridge_a10"


def test_definition_by_id_rejects_unknown():
    with pytest.raises(stagee.ConstrainedIntegratorError):
        stagee.definition_by_id("unknown")


def test_segment_bounds_historical():
    result = stagee.segment_bounds(
        definition=stagee.INTEGRATOR_DEFINITIONS[1],
        context=make_context(),
    )
    assert np.allclose(result["lower"], 0.05)
    assert np.allclose(result["upper"], 0.20)


def test_segment_bounds_robust_intersection():
    result = stagee.segment_bounds(
        definition=selectable_definition(),
        context=make_context(),
    )
    assert np.all(result["lower"] >= 0.05)
    assert np.all(result["upper"] <= 0.20000001)
    assert np.all(result["upper"] > result["lower"])


def test_coordinate_recenter_possible():
    points = np.asarray([[[[-2.0, 0.0], [2.0, 0.0]]]], dtype=np.float64)
    shifted, possible = stagee._coordinate_recenter(points, absolute_limit=3.0)
    assert possible.tolist() == [True]
    assert np.max(np.abs(shifted)) <= 3.0


def test_coordinate_recenter_impossible():
    points = np.asarray([[[[-4.0, 0.0], [4.0, 0.0]]]], dtype=np.float64)
    _, possible = stagee._coordinate_recenter(points, absolute_limit=3.0)
    assert possible.tolist() == [False]


def test_reconstruct_preserves_shape_and_finiteness():
    context = make_context()
    control = make_cable()
    proposed = control.copy()
    proposed[:, :, 1::2] += 0.01
    result = stagee.reconstruct_segment_vectors(
        proposed=proposed,
        control=control,
        lower=np.full((4, 23), 0.05),
        upper=np.full((4, 23), 0.15),
        coordinate_abs_max=5.0,
        epsilon=1e-8,
    )
    assert result["candidate"].shape == control.shape
    assert np.all(np.isfinite(result["candidate"]))
    assert np.all(result["bound_pass"])


def test_reconstruct_clips_long_segments():
    control = make_cable(rows=1, length=0.10)
    proposed = make_cable(rows=1, length=0.30)
    result = stagee.reconstruct_segment_vectors(
        proposed=proposed,
        control=control,
        lower=np.full((4, 23), 0.05),
        upper=np.full((4, 23), 0.15),
        coordinate_abs_max=10.0,
        epsilon=1e-8,
    )
    assert np.max(stageb.segment_lengths(result["candidate"])) <= 0.150001
    assert result["clipped_segment_rate"] > 0.99


def test_reconstruct_uses_control_direction_for_degenerate_segment():
    control = make_cable(rows=1)
    proposed = control.copy()
    points = proposed.reshape(1, 4, 24, 2)
    points[:, :, 1] = points[:, :, 0]
    result = stagee.reconstruct_segment_vectors(
        proposed=proposed,
        control=control,
        lower=np.full((4, 23), 0.05),
        upper=np.full((4, 23), 0.15),
        coordinate_abs_max=10.0,
        epsilon=1e-8,
    )
    assert result["degenerate_segment_rate"] == 0.0
    assert np.all(result["bound_pass"])


def test_reconstruct_preserves_proposed_centroid_without_recenter():
    control = make_cable(rows=1)
    proposed = control + 0.25
    result = stagee.reconstruct_segment_vectors(
        proposed=proposed,
        control=control,
        lower=np.full((4, 23), 0.05),
        upper=np.full((4, 23), 0.15),
        coordinate_abs_max=10.0,
        epsilon=1e-8,
    )
    proposed_centroid = proposed.reshape(1, 4, 24, 2).mean(axis=2)
    actual_centroid = result["candidate"].reshape(1, 4, 24, 2).mean(axis=2)
    assert np.allclose(actual_centroid, proposed_centroid, atol=1e-6)


def test_observable_fast_path_does_not_call_topology(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("topology called")
    monkeypatch.setattr(stagee.stageb, "physical_validity", forbidden)
    control = make_cable()
    direction = np.full_like(control, 0.001, dtype=np.float64)
    result = stagee.observable_row_metrics(
        candidate=control + 0.001,
        raw_proposal=control + 0.001,
        control=control,
        direction=direction,
        definition=selectable_definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
        include_topology=False,
    )
    assert not result["topology_evaluated"]
    assert np.all(result["topology_pass"])


def test_observable_full_path_calls_topology(monkeypatch):
    calls = {"count": 0}
    def fake(value, contract):
        calls["count"] += 1
        rows = value.shape[0]
        return {"topology": np.ones((rows, 1), dtype=np.bool_)}
    monkeypatch.setattr(stagee.stageb, "physical_validity", fake)
    control = make_cable()
    direction = np.full_like(control, 0.001, dtype=np.float64)
    result = stagee.observable_row_metrics(
        candidate=control + 0.001,
        raw_proposal=control + 0.001,
        control=control,
        direction=direction,
        definition=selectable_definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
        include_topology=True,
    )
    assert calls["count"] == 1
    assert result["topology_evaluated"]


def test_integrate_rowwise_selects_largest_feasible_scale(monkeypatch):
    control = make_cable(rows=2)
    direction = np.full_like(control, 0.001, dtype=np.float64)
    result = stagee.integrate_rowwise(
        control=control,
        direction=direction,
        definition=selectable_definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
    )
    assert np.allclose(result["selected_scale"], 1.0)
    assert result["fallback_rate"] == 0.0


def test_integrate_rowwise_retries_after_topology_rejection(monkeypatch):
    control = make_cable(rows=2)
    direction = np.full_like(control, 0.001, dtype=np.float64)
    calls = {"count": 0}
    original = stagee.stageb.physical_validity
    def fake(value, contract):
        calls["count"] += 1
        result = original(value, contract)
        if calls["count"] == 1:
            result["topology"][:] = False
            result["valid"][:] = False
        return result
    monkeypatch.setattr(stagee.stageb, "physical_validity", fake)
    result = stagee.integrate_rowwise(
        control=control,
        direction=direction,
        definition=selectable_definition(),
        context=make_context(),
        spec=stagee.ConstrainedIntegratorSpec(),
    )
    assert calls["count"] >= 3  # rejected scale, accepted scale, final audit
    assert np.all(result["selected_scale"] < 1.0)
    assert result["fallback_rate"] == 0.0


def test_integrate_rowwise_target_independent_signature():
    assert "target" not in stagee.integrate_rowwise.__annotations__


@pytest.mark.parametrize(
    "field,value",
    [
        ("observable_feasible_rate", 0.1),
        ("fallback_rate", 0.5),
        ("direction_retention", {"mean": 0.1}),
        ("correction_ratio", {"p95": 3.0}),
        ("final_upper_pass_rate", 0.1),
        ("final_physical_pass_rate", 0.1),
    ],
)
def test_observable_gates_reject_each_failure(field, value):
    integration = {
        "observable_feasible_rate": 1.0,
        "fallback_rate": 0.0,
        "direction_retention": {"mean": 1.0},
        "correction_ratio": {"p95": 0.0},
        "final_upper_pass_rate": 1.0,
        "final_physical_pass_rate": 1.0,
    }
    integration[field] = value
    assert not stagee.integrator_observable_gates(
        integration,
        spec=stagee.ConstrainedIntegratorSpec(),
    )["all"]


def test_observable_gates_pass():
    integration = {
        "observable_feasible_rate": 1.0,
        "fallback_rate": 0.0,
        "direction_retention": {"mean": 1.0},
        "correction_ratio": {"p95": 0.0},
        "final_upper_pass_rate": 1.0,
        "final_physical_pass_rate": 1.0,
    }
    assert stagee.integrator_observable_gates(
        integration,
        spec=stagee.ConstrainedIntegratorSpec(),
    )["all"]


def test_direction_for_zero():
    control = make_cable()
    value = stagee._direction_for_definition(
        definition=stagee.INTEGRATOR_DEFINITIONS[-2],
        timestep=10,
        direction_bundle={},
        control=control,
        target=control + 1,
    )
    assert np.count_nonzero(value) == 0


def test_direction_for_oracle():
    control = make_cable()
    target = control + 0.5
    value = stagee._direction_for_definition(
        definition=stagee.INTEGRATOR_DEFINITIONS[-1],
        timestep=10,
        direction_bundle={},
        control=control,
        target=target,
    )
    assert np.allclose(value, 0.5)


def test_direction_for_surrogate():
    control = make_cable()
    expected = np.ones_like(control)
    bundle = {"predictions": {"centered_ridge_a10": {10: expected}}}
    value = stagee._direction_for_definition(
        definition=selectable_definition(),
        timestep=10,
        direction_bundle=bundle,
        control=control,
        target=control,
    )
    assert np.array_equal(value, expected)


def test_select_integrator_none():
    assert stagee.select_integrator([fake_candidate("x")]) is None


def test_select_integrator_uses_selection_key():
    first = fake_candidate("centered_a10_z4_m1_r25", scientific=True, eligible=True, reduction=0.3)
    second = fake_candidate("centered_rr32_z4_m1_r25", scientific=True, eligible=True, source="centered_rridge_k32_a10", reduction=0.5)
    selected = stagee.select_integrator([first, second])
    assert selected["candidate_id"] == second["candidate_id"]
    assert not selected["holdout_used"]


def test_selection_key_rejects_ineligible():
    with pytest.raises(stagee.ConstrainedIntegratorError):
        stagee._selection_key(fake_candidate("x"))


def classification_records(*, oracle: bool, selectable_state: bool, selectable_observable: bool = True):
    return [
        fake_candidate("selectable", scientific=selectable_state, observable=selectable_observable, eligible=selectable_state),
        fake_candidate("oracle", role="oracle_control", scientific=oracle, source="oracle"),
    ]


def test_classify_oracle_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=False, selectable_state=False),
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "integrator_family_not_oracle_reachable"


def test_classify_surrogate_precision_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=False),
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "surrogate_precision_under_constraints"


def test_classify_observable_policy_failure():
    records = classification_records(oracle=True, selectable_state=True, selectable_observable=False)
    records[0]["eligible"] = False
    result = stagee.classify_integrator(
        records=records,
        selected=None,
        permutation=None,
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "observable_policy_gate"


def selected_value():
    candidate = fake_candidate("selectable", scientific=True, eligible=True)
    return {
        "candidate_id": candidate["candidate_id"],
        "definition": candidate["definition"],
    }


def test_classify_permutation_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=True),
        selected=selected_value(),
        permutation={"all_pass": False},
        projection_only=None,
        holdout=None,
    )
    assert result["primary_failure_locus"] == "direction_permutation_shortcut"


def test_classify_projection_only_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=True),
        selected=selected_value(),
        permutation={"all_pass": True},
        projection_only={"pass": False},
        holdout=None,
    )
    assert result["primary_failure_locus"] == "projection_only_shortcut"


def test_classify_requires_holdout():
    with pytest.raises(stagee.ConstrainedIntegratorError):
        stagee.classify_integrator(
            records=classification_records(oracle=True, selectable_state=True),
            selected=selected_value(),
            permutation={"all_pass": True},
            projection_only={"pass": True},
            holdout=None,
        )


def test_classify_holdout_direction_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=True),
        selected=selected_value(),
        permutation={"all_pass": True},
        projection_only={"pass": True},
        holdout={"direction_all_timesteps": False, "state_all_timesteps": False, "scientific_pass": False},
    )
    assert result["primary_failure_locus"] == "holdout_direction_generalization"


def test_classify_holdout_state_failure():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=True),
        selected=selected_value(),
        permutation={"all_pass": True},
        projection_only={"pass": True},
        holdout={"direction_all_timesteps": True, "state_all_timesteps": False, "scientific_pass": False},
    )
    assert result["primary_failure_locus"] == "holdout_state_generalization"


def test_classify_success():
    result = stagee.classify_integrator(
        records=classification_records(oracle=True, selectable_state=True),
        selected=selected_value(),
        permutation={"all_pass": True},
        projection_only={"pass": True},
        holdout={"direction_all_timesteps": True, "state_all_timesteps": True, "scientific_pass": True},
    )
    assert result["primary_failure_locus"] == "constrained_integrator_validated"


def test_identity_projection_preserves_complete_worker_identity():
    value = {
        "environment": {"observation_sha256": "x", "hardware_observation": {"gpu": "x"}},
        "other": 1,
    }
    projected = stagee.identity_projection(value)
    assert projected == value
    assert projected is not value


def test_compare_worker_results_exact():
    result = stagee.compare_worker_results({"a": 1}, {"a": 1})
    assert result["exact"]
    assert result["left_sha256"] == result["right_sha256"]


def test_compare_worker_results_detects_observation_change():
    left = {"environment": {"compatibility_sha256": "same", "observation_sha256": "a"}}
    right = {"environment": {"compatibility_sha256": "same", "observation_sha256": "b"}}
    assert not stagee.compare_worker_results(left, right)["exact"]


def test_compare_worker_results_detects_scientific_change():
    assert not stagee.compare_worker_results({"a": 1}, {"a": 2})["exact"]


def test_status_paths_sorted(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    assert stagee.status_paths(tmp_path) == ("a.txt", "b.txt")


def test_initial_paths_population():
    assert len(stagee.STAGEE_IMPLEMENTATION_PATHS) == 7


def test_source_binding_population():
    assert len(stagee.SOURCE_BOUND_SHA256) == 7


def test_contract_generation_has_no_target_argument():
    names = stagee.integrate_rowwise.__code__.co_varnames[: stagee.integrate_rowwise.__code__.co_argcount]
    assert "target" not in names


def test_projection_only_candidate_is_unselectable():
    definition = next(item for item in stagee.INTEGRATOR_DEFINITIONS if item.direction_source_id == "zero")
    assert definition.role == "diagnostic_control"


def test_oracle_candidate_is_unselectable():
    definition = next(item for item in stagee.INTEGRATOR_DEFINITIONS if item.direction_source_id == "oracle")
    assert definition.role == "oracle_control"


def test_linear_candidate_is_negative_control():
    assert stagee.INTEGRATOR_DEFINITIONS[0].role == "negative_control"


def test_all_selectable_use_robust_intersection():
    values = [item for item in stagee.INTEGRATOR_DEFINITIONS if item.role == "selectable"]
    assert len(values) == 9
    assert all(item.bound_mode == "robust_intersection" for item in values)


def test_scale_grid_is_pre_registered():
    assert stagee.ConstrainedIntegratorSpec().scale_grid == (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def _commit_all(path: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=path, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def test_commit_helpers(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    first = _commit_all(tmp_path, "first")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    second = _commit_all(tmp_path, "second")
    assert stagee.commit_parent(tmp_path, second) == first
    assert stagee.commit_subject(tmp_path, second) == "second"
    assert stagee.commit_paths(tmp_path, second) == ("b.txt",)


def test_commit_paths_sorted(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / "base.txt").write_text("x", encoding="utf-8")
    _commit_all(tmp_path, "base")
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    commit = _commit_all(tmp_path, "two")
    assert stagee.commit_paths(tmp_path, commit) == ("a.txt", "z.txt")


def test_zero_direction_control_exempts_only_retention_gate():
    integration = {
        "direction_source_id": "zero",
        "observable_feasible_rate": 1.0,
        "fallback_rate": 0.0,
        "direction_retention": {"mean": 0.0},
        "correction_ratio": {"p95": 0.0},
        "final_upper_pass_rate": 1.0,
        "final_physical_pass_rate": 1.0,
    }
    gates = stagee.integrator_observable_gates(
        integration,
        spec=stagee.ConstrainedIntegratorSpec(),
    )
    assert gates["retention"]
    assert gates["all"]
