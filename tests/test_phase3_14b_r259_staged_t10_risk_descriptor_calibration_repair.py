from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


FALSE_BOUNDARIES = (
    "selection_holdout_evaluated",
    "selection_holdout_used_for_fit_or_selection",
    "selection_holdout_reaccessed",
    "selection_holdout_target_loaded",
    "frozen_probe_accessed",
    "formal_training_run",
    "reverse_sampling_run",
    "idm_run",
    "candidate_execution",
    "deformable_ravens_executed",
    "phase4",
    "cps",
    "checkpoint_saved",
    "weights_persisted",
    "prediction_tensor_persisted",
    "candidate_tensor_persisted",
    "risk_probability_tensor_persisted",
    "descriptor_tensor_persisted",
    "npz_saved",
    "cache_saved",
    "image_saved",
    "video_saved",
)


def _install_dependency_stubs() -> None:
    stagex_name = "ccda_phase3.phase314b_r258_stagex_tail_robust_nested_oof"
    if stagex_name not in sys.modules:
        stagex = types.ModuleType(stagex_name)
        stagex.LOCKED_BACKBONE = "segment_target_rr64_feasible"
        stagex.EXPECTED_INTERNAL_SCALE_COUNT = 7

        class StageXSpec:
            def __init__(self, risk_l2=1.0):
                self.risk_l2 = float(risk_l2)

            def validate(self):
                if self.risk_l2 <= 0:
                    raise ValueError("risk_l2")

        stagex.StageXSpec = StageXSpec

        def compact(control, direction, candidate, selected_scale):
            control = np.asarray(control, dtype=np.float64)
            direction = np.asarray(direction, dtype=np.float64)
            candidate = np.asarray(candidate, dtype=np.float64)
            scale = np.asarray(selected_scale, dtype=np.float64)
            rows = control.shape[0]
            movement = candidate - control
            flat_d = direction.reshape(rows, -1)
            flat_m = movement.reshape(rows, -1)
            dnorm = np.linalg.norm(flat_d, axis=1)
            mnorm = np.linalg.norm(flat_m, axis=1)
            cnorm = np.linalg.norm(control.reshape(rows, -1), axis=1)
            cosine = np.sum(flat_d * flat_m, axis=1) / np.maximum(dnorm * mnorm, 1e-12)
            columns = [
                scale,
                (scale > 0).astype(np.float64),
                dnorm,
                mnorm,
                cnorm,
                np.max(np.abs(flat_d), axis=1),
                np.max(np.abs(flat_m), axis=1),
                cosine,
                mnorm / np.maximum(dnorm, 1e-12),
            ]
            columns.extend(np.linalg.norm(direction[:, i].reshape(rows, -1), axis=1) for i in range(4))
            columns.extend(np.linalg.norm(movement[:, i].reshape(rows, -1), axis=1) for i in range(4))
            return np.stack(columns, axis=1)

        stagex.compact_risk_descriptors = compact
        stagex.fit_risk_model = lambda *args, **kwargs: {
            "mode": "constant",
            "converged": True,
            "prevalence": 0.5,
        }
        stagex.predict_risk = lambda model, features: np.full(len(features), model["prevalence"])
        stagex.sha256_array = lambda value: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
        sys.modules[stagex_name] = stagex

    kernel_name = "ccda_phase3.phase314b_r258_stagex_resume1_import_crossfit_recovery"
    if kernel_name not in sys.modules:
        kernel = types.ModuleType(kernel_name)
        kernel.require_clean_generation = lambda value, label=None: None
        kernel.evaluate_variable_threshold_policy = lambda *args, **kwargs: {}
        sys.modules[kernel_name] = kernel

    stagec_name = "ccda_phase3.phase314b_r259_stagec_fold_resolved_tail_attribution"
    if stagec_name not in sys.modules:
        stagec = types.ModuleType(stagec_name)
        stagec.child_environment = lambda root: {}
        sys.modules[stagec_name] = stagec

    resume_name = (
        "ccda_phase3.phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery"
    )
    if resume_name not in sys.modules:
        resume = types.ModuleType(resume_name)
        resume.FALSE_BOUNDARIES = FALSE_BOUNDARIES
        resume.validate_summary = lambda value: None
        resume.make_probe_evidence = lambda root: {}
        resume.validate_probe_evidence = lambda value: value["environment"]
        resume.base = types.SimpleNamespace(
            stagea=types.SimpleNamespace(
                resume2a=types.SimpleNamespace(
                    stable_json_bytes=lambda value: json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    ).encode(),
                    sha256_bytes=lambda value: hashlib.sha256(value).hexdigest(),
                )
            )
        )
        sys.modules[resume_name] = resume


_install_dependency_stubs()

from ccda_phase3 import (  # noqa: E402
    phase314b_r259_staged_t10_risk_descriptor_calibration_repair as d,
)


def good_record(**updates):
    value = {
        "acceptance_rate": 0.60,
        "overall_mse_ratio": 0.90,
        "accepted_row_mse_ratio": 0.85,
        "positive_distance_reduction_rate": 0.60,
        "relative_distance_reduction_mean": 0.05,
        "adverse_sse_mass": 7.0,
        "group_tail": {"worst_fraction_cvar_mse_ratio": 0.90},
        "risk_brier_nonworse": True,
        "risk_brier_score": 0.15,
    }
    value.update(updates)
    return value


def baseline_record():
    return {
        "adverse_sse_mass": 10.0,
        "group_tail": {"worst_fraction_cvar_mse_ratio": 1.0},
    }


def synthetic_probe(pid=10):
    environment = {"compatibility_pass": True, "hardware_observation": {"gpu": "test"}}
    payload = {
        "phase": d.PHASE,
        "schema": d.PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "process_id": pid,
        "environment": environment,
        "environment_sha256": d.stagec_resume1.base.stagea.resume2a.sha256_bytes(
            d.stagec_resume1.base.stagea.resume2a.stable_json_bytes(environment)
        ),
        "portable_environment_audit": {
            "portable_contract_passed": True,
            "hardware_identity_required": False,
        },
        "source_stagec_resume1_probe_schema": "source",
        "source_stagec_resume1_probe_sha256": "a" * 64,
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    payload["probe_evidence_sha256"] = d.sha256_bytes(d.stable_json_bytes(payload))
    return payload


def synthetic_worker(slot="A", status="BLOCKED"):
    recommendation = None
    if status == "READY":
        recommendation = {"t10_descriptor_id": "compact_geometry_v2"}
    payload = {
        "phase": d.PHASE,
        "schema": d.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": status,
        "root_cause": "root",
        "required_next_path": "next",
        "primary_failure_locus": "locus",
        "process_id": 20 if slot == "A" else 21,
        "worker_slot": slot,
        "repository_head": "h" * 40,
        "environment_sha256": "e" * 64,
        "execution_counts": dict(d.EXPECTED_EXECUTION_COUNTS),
        "population": {"row_count": 638, "group_count": 126},
        "recipe_population": [
            {**vars(item)} for item in d.recipe_population()
        ],
        "legacy_stagec_control_contract": {
            "10": {"all_pass": False, "failed_checks": ["acceptance"]},
            "25": {"all_pass": True, "failed_checks": []},
            "50": {"all_pass": True, "failed_checks": []},
        },
        "outer_fold_selections": [
            {
                "outer_fold": fold,
                "selected_recipe_id": d.recipe_population()[0].recipe_id,
                "selected_recipe": vars(d.recipe_population()[0]),
                "selected_recipe_inner_eligible": True,
                "diagnostic_fallback_used": False,
            }
            for fold in range(6)
        ],
        "outer_fold_selected_recipe_metrics": [],
        "outer_selection_modal_recipe_diagnostic": {
            "recipe_id": d.recipe_population()[0].recipe_id,
            "support_count": 6,
        },
        "outer_crossfit_fold_selected_recipe_record": good_record(),
        "outer_crossfit_no_abstention_baseline_record": baseline_record(),
        "outer_crossfit_procedure_eligibility": {
            "pass": status == "READY",
            "checks": d.t10_checks(good_record(), baseline_record(), d.StageDSpec()),
        },
        "full_objective_oof_fixed_recipe_selection": {
            "selected_recipe_id": d.recipe_population()[0].recipe_id,
            "selected_recipe_inner_eligible": status == "READY",
        },
        "procedure_gate_totals": {
            "length_log_z_element_mismatch_count": 0,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": recommendation,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_accessed": False,
        "rerun_authorized": False,
        "durable_write_ahead_evidence": True,
        "write_ahead_location_outside_git_worktree": True,
    }
    payload.update({key: False for key in d.FALSE_BOUNDARIES})
    payload["worker_result_sha256"] = d.sha256_bytes(d.stable_json_bytes(payload))
    return payload


def rehash_worker(value):
    value["worker_result_sha256"] = d.sha256_bytes(
        d.stable_json_bytes({k: v for k, v in value.items() if k != "worker_result_sha256"})
    )
    return value


def test_phase_and_schema():
    assert d.PHASE.endswith("Stage D")
    assert d.SCHEMA.endswith("_v1")


def test_starting_commit_is_stagec_resume1_evidence():
    assert d.BASE_STAGEC_RESUME1_EVIDENCE_COMMIT == "591d1fd0814f7bd50b8e580274462908512a745f"


def test_implementation_parent_is_frozen():
    assert d.BASE_STAGEC_RESUME1_IMPLEMENTATION_COMMIT.startswith("b6b35467")


def test_remote_states_are_predeclared():
    assert d.EXPECTED_REMOTE_HEADS == (
        "4f8a73fb5cc9b60c92c440b6e61bc1765704f3e5",
        "591d1fd0814f7bd50b8e580274462908512a745f",
    )


def test_submodule_is_frozen():
    assert d.EXPECTED_SUBMODULE.startswith("633a8875")


def test_add_only_four_files():
    assert len(d.IMPLEMENTATION_PATHS) == 4
    assert {status for status, _ in d.IMPLEMENTATION_PATHS} == {"A"}


def test_spec_validates():
    d.StageDSpec().validate()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_acceptance_rate": 0.49},
        {"minimum_positive_reduction_rate": 0.49},
        {"adverse_sse_reduction_fraction": 0.20},
        {"group_cvar_fraction": 0.25},
        {"modal_outer_fold_minimum": 3},
        {"epsilon": 1e-9},
    ],
)
def test_spec_rejects_changes(kwargs):
    with pytest.raises(d.StageDError):
        d.StageDSpec(**kwargs).validate()


def test_recipe_population_count():
    assert len(d.recipe_population()) == 360


def test_recipe_ids_unique():
    values = d.recipe_population()
    assert len({value.recipe_id for value in values}) == len(values)


def test_recipe_bank_dimensions():
    assert len(d.DESCRIPTOR_IDS) * len(d.RISK_L2_VALUES) * len(d.TEMPERATURES) * len(d.DIRECTION_SHRINKAGES) * len(d.RISK_THRESHOLDS) == 360


def test_first_recipe_is_deterministic():
    assert d.recipe_population()[0].recipe_id == "desc_compact_v1__l2_0.25__temp_0.75__shrink_0.50__risk_0.25"


@pytest.mark.parametrize("field,value", [
    ("descriptor_id", "bad"),
    ("risk_l2", 2.0),
    ("temperature", 3.0),
    ("shrinkage", 0.25),
    ("risk_threshold", 0.1),
])
def test_recipe_rejects_out_of_bank(field, value):
    kwargs = vars(d.recipe_population()[0]).copy()
    kwargs[field] = value
    with pytest.raises(d.StageDError):
        d.RiskRecipe(**kwargs).validate()


def test_descriptor_widths():
    assert len(d.descriptor_names("compact_v1")) == 17
    assert len(d.descriptor_names("compact_geometry_v2")) == 66


def test_descriptor_unknown_rejected():
    with pytest.raises(d.StageDError):
        d.descriptor_names("unknown")


def tensor_population(rows=3):
    control = np.zeros((rows, 4, 67), dtype=np.float32)
    xy = np.linspace(0.0, 0.22, 23)
    control[:, :, :46] = np.stack([xy, np.zeros_like(xy)], axis=1).reshape(46)
    direction = np.zeros_like(control)
    direction[..., :46] = 0.001
    candidate = control + direction
    scale = np.ones(rows, dtype=np.float64)
    return control, direction, candidate, scale


def test_compact_descriptor_shape():
    values = tensor_population()
    result = d.build_risk_descriptors("compact_v1", *values)
    assert result.shape == (3, 17)


def test_geometry_descriptor_shape():
    values = tensor_population()
    result = d.build_risk_descriptors("compact_geometry_v2", *values)
    assert result.shape == (3, 66)
    assert np.all(np.isfinite(result))


def test_geometry_descriptor_preserves_legacy_prefix():
    values = tensor_population()
    compact = d.build_risk_descriptors("compact_v1", *values)
    geometry = d.build_risk_descriptors("compact_geometry_v2", *values)
    np.testing.assert_allclose(geometry[:, :17], compact)


def test_geometry_descriptor_rejects_shape():
    with pytest.raises(d.StageDError, match="shape"):
        d.build_risk_descriptors(
            "compact_geometry_v2",
            np.zeros((2, 4, 66)),
            np.zeros((2, 4, 66)),
            np.zeros((2, 4, 66)),
            np.ones(2),
        )


def test_geometry_descriptor_rejects_mismatched_shape():
    control, direction, candidate, scale = tensor_population()
    with pytest.raises(d.StageDError):
        d.build_risk_descriptors(
            "compact_geometry_v2", control, direction[:2], candidate, scale
        )


def test_temperature_one_is_identity():
    p = np.array([0.1, 0.5, 0.9])
    result = d.prevalence_centered_temperature(p, np.full(3, 0.3), 1.0)
    np.testing.assert_allclose(result, p)


def test_temperature_preserves_prevalence_fixed_point():
    prior = np.full(4, 0.3)
    result = d.prevalence_centered_temperature(prior, prior, 2.0)
    np.testing.assert_allclose(result, prior)


def test_temperature_above_one_shrinks_to_prevalence():
    result = d.prevalence_centered_temperature(
        np.array([0.9]), np.array([0.2]), 2.0
    )
    assert 0.2 < result[0] < 0.9


def test_temperature_below_one_sharpens_from_prevalence():
    result = d.prevalence_centered_temperature(
        np.array([0.8]), np.array([0.2]), 0.75
    )
    assert result[0] > 0.8


def test_temperature_rejects_unknown():
    with pytest.raises(d.StageDError):
        d.prevalence_centered_temperature(np.array([0.5]), np.array([0.5]), 3.0)


def test_temperature_rejects_shape_mismatch():
    with pytest.raises(d.StageDError):
        d.prevalence_centered_temperature(np.ones(2) * 0.5, np.ones(3) * 0.5, 1.0)


def test_good_record_passes_all_checks():
    assert all(d.t10_checks(good_record(), baseline_record(), d.StageDSpec()).values())


@pytest.mark.parametrize(
    "update,failed",
    [
        ({"acceptance_rate": 0.49}, "acceptance"),
        ({"overall_mse_ratio": 1.0}, "overall_mse"),
        ({"accepted_row_mse_ratio": 1.0}, "accepted_mse"),
        ({"positive_distance_reduction_rate": 0.5}, "positive_reduction"),
        ({"relative_distance_reduction_mean": 0.0}, "relative_reduction"),
        ({"adverse_sse_mass": 9.1}, "adverse_sse_reduction"),
        ({"group_tail": {"worst_fraction_cvar_mse_ratio": 1.0}}, "group_cvar_improvement"),
        ({"risk_brier_nonworse": False}, "risk_brier_nonworse"),
    ],
)
def test_each_gate_fails_closed(update, failed):
    checks = d.t10_checks(good_record(**update), baseline_record(), d.StageDSpec())
    assert checks[failed] is False


def test_recipe_eligibility():
    assert d.recipe_is_eligible(good_record(), baseline_record(), d.StageDSpec())["pass"] is True


def test_recipe_selection_chooses_first_on_tie():
    records = {item.recipe_id: good_record() for item in d.recipe_population()}
    result = d.select_recipe(records, baseline_record(), d.StageDSpec())
    assert result["selected_recipe_id"] == d.recipe_population()[0].recipe_id
    assert result["selected_recipe_inner_eligible"] is True


def test_recipe_selection_fallback_when_none_eligible():
    records = {
        item.recipe_id: good_record(acceptance_rate=0.1)
        for item in d.recipe_population()
    }
    result = d.select_recipe(records, baseline_record(), d.StageDSpec())
    assert result["diagnostic_fallback_used"] is True
    assert result["eligible_recipe_ids"] == []


def test_recipe_selection_rejects_missing_population():
    with pytest.raises(d.StageDError):
        d.select_recipe({}, baseline_record(), d.StageDSpec())


def test_modal_recipe_support():
    a = d.recipe_population()[0].recipe_id
    b = d.recipe_population()[1].recipe_id
    result = d.modal_recipe([a, a, a, a, b, b])
    assert result["recipe_id"] == a
    assert result["support_count"] == 4


def test_modal_recipe_tie_uses_population_order():
    a = d.recipe_population()[0].recipe_id
    b = d.recipe_population()[1].recipe_id
    assert d.modal_recipe([b, a, b, a, b, a])["recipe_id"] == a


def test_modal_recipe_rejects_wrong_count():
    with pytest.raises(d.StageDError):
        d.modal_recipe([d.recipe_population()[0].recipe_id])


def test_stable_json_order():
    assert d.stable_json_bytes({"b": 1, "a": 2}).startswith(b'{\n  "a"')


def test_compact_json():
    assert d.compact_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_sha256_bytes():
    assert d.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_atomic_write_once(tmp_path: Path):
    path = tmp_path / "x.json"
    d.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(d.StageDError):
        d.atomic_write_once(path, b"again")


def test_promote_write_ahead(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text("{}\n")
    d.promote_write_ahead(source, destination, lambda value: None)
    assert destination.read_bytes() == source.read_bytes()


def test_promote_rejects_existing_destination(tmp_path: Path):
    source = tmp_path / "source.json"
    destination = tmp_path / "destination.json"
    source.write_text("{}\n")
    destination.write_text("existing")
    with pytest.raises(d.StageDError):
        d.promote_write_ahead(source, destination, lambda value: None)


def test_probe_validation():
    assert d.validate_probe_evidence(synthetic_probe())["compatibility_pass"] is True


def test_probe_rejects_hardware_identity_gate():
    value = synthetic_probe()
    value["portable_environment_audit"]["hardware_identity_required"] = True
    value["probe_evidence_sha256"] = d.sha256_bytes(
        d.stable_json_bytes({k: v for k, v in value.items() if k != "probe_evidence_sha256"})
    )
    with pytest.raises(d.StageDError):
        d.validate_probe_evidence(value)


def test_probe_rejects_bad_self_hash():
    value = synthetic_probe()
    value["probe_evidence_sha256"] = "x" * 64
    with pytest.raises(d.StageDError):
        d.validate_probe_evidence(value)


def test_worker_validation_blocked():
    d.validate_worker_evidence(synthetic_worker())


def test_worker_validation_ready():
    d.validate_worker_evidence(synthetic_worker(status="READY"))


def test_worker_rejects_execution_count_change():
    value = synthetic_worker()
    value["execution_counts"]["risk_fit_count"] += 1
    rehash_worker(value)
    with pytest.raises(d.StageDError):
        d.validate_worker_evidence(value)


def test_worker_rejects_holdout_access():
    value = synthetic_worker()
    value["selection_holdout_evaluation_count_added"] = 1
    rehash_worker(value)
    with pytest.raises(d.StageDError):
        d.validate_worker_evidence(value)


def test_worker_rejects_blocked_recommendation():
    value = synthetic_worker()
    value["train_only_recommendation"] = {"bad": True}
    rehash_worker(value)
    with pytest.raises(d.StageDError):
        d.validate_worker_evidence(value)


def test_worker_projection_removes_process_fields():
    result = d.worker_scientific_projection(synthetic_worker())
    assert "process_id" not in result
    assert "worker_slot" not in result
    assert "worker_result_sha256" not in result


def test_same_device_workers_exact():
    result = d.compare_same_device_workers(synthetic_worker("A"), synthetic_worker("B"))
    assert result["same_device_scientific_projection_byte_exact"] is True


def test_same_device_workers_reject_difference():
    a = synthetic_worker("A")
    b = synthetic_worker("B")
    b["root_cause"] = "different"
    rehash_worker(b)
    with pytest.raises(d.StageDError):
        d.compare_same_device_workers(a, b)


def test_blocked_before_worker():
    result = d.blocked_report(repository=None, error=RuntimeError("x"))
    assert result["required_next_path"] == "DESIGN_ADD_ONLY_R259_STAGED_EXECUTION_RECOVERY_BEFORE_SCIENCE"
    assert result["science_reexecution_authorized"] is False


def test_blocked_after_worker(tmp_path: Path):
    path = tmp_path / "worker.json"
    path.write_text("{}")
    result = d.blocked_report(repository={}, error=RuntimeError("x"), worker_a_path=path)
    assert result["primary_failure_locus"] == "controller_after_durable_worker_evidence"
    assert "WITHOUT_SCIENCE_REEXECUTION" in result["required_next_path"]


def test_false_boundaries_are_frozen():
    value = synthetic_worker()
    assert all(value[key] is False for key in d.FALSE_BOUNDARIES)


def test_output_paths_are_staged_namespaced():
    assert "staged" in d.PROBE_EVIDENCE
    assert "staged" in d.WORKER_A_EVIDENCE
    assert "staged" in d.WORKER_B_EVIDENCE
    assert "staged" in d.SUCCESS_REPORT


def test_workers_have_distinct_paths():
    assert d.WORKER_A_EVIDENCE != d.WORKER_B_EVIDENCE


def test_execution_counts_exact():
    assert d.EXPECTED_EXECUTION_COUNTS == {
        "direction_fit_count": 36,
        "candidate_generation_count": 108,
        "risk_fit_count": 648,
        "internal_scale_attempt_count": 756,
        "inner_recipe_evaluation_count": 2160,
        "full_objective_recipe_evaluation_count": 360,
        "nonconverged_risk_fit_count": 0,
        "descriptor_build_count": 72,
    }


def test_only_t10_is_mutable_scientific_scope():
    assert d.LOCKED_TIMESTEP == 10
    assert d.CONTROL_TIMESTEPS == (25, 50)


def test_joint_timestep_policy_preserved_by_ready_recommendation():
    value = synthetic_worker(status="READY")
    assert value["selected_configuration"] is None
    assert value["train_only_recommendation"] is not None
