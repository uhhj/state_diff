from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r259_stageh_frozen_probe_transfer_audit as h


@pytest.mark.parametrize(
    "name,expected",
    [
        ("LOCKED_TIMESTEPS", (10, 25, 50)),
        ("EXPECTED_FROZEN_PROBE_ROWS", 126),
        ("BASE_HEAD", "127cf4c59edcd0a6052cb04c969f86f393b23616"),
        ("BASE_IMPLEMENTATION", "048adcb8c375fd8431b14c9130612fb837f8509f"),
        ("EXPECTED_SUBMODULE", "633a88752445cf5d6776ed374fdbbdb35f93050c"),
        (
            "EXPECTED_CANONICAL_RECIPE",
            "desc_compact_v1__l2_4.00__temp_1.00__shrink_1.00__risk_0.50",
        ),
    ],
)
def test_frozen_constants(name, expected):
    assert getattr(h, name) == expected


@pytest.mark.parametrize(
    "value",
    [h.BASE_HEAD, h.BASE_IMPLEMENTATION, h.EXPECTED_SUBMODULE],
)
def test_commit_values_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_implementation_is_add_only_three_files():
    assert len(h.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in h.IMPLEMENTATION_PATHS} == {"A"}


def test_stable_json_order_independent():
    assert h.stable_json_bytes({"b": 1, "a": 2}) == h.stable_json_bytes({"a": 2, "b": 1})


def test_sha256_bytes_known():
    assert h.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    h.atomic_write_once(path, b"x")
    assert path.read_bytes() == b"x"


def test_atomic_write_refuses_overwrite(tmp_path):
    path = tmp_path / "x.json"
    path.write_bytes(b"old")
    with pytest.raises(h.StageHAuditError, match="write-once"):
        h.atomic_write_once(path, b"new")


def test_load_json_requires_mapping(tmp_path):
    path = tmp_path / "x.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(h.StageHAuditError, match="not a mapping"):
        h.load_json(path)


def test_safe_ratio():
    assert h._safe_ratio(2.0, 4.0) == 0.5
    assert h._safe_ratio(1.0, 0.0) is None


def test_constant_brier_identity():
    observed = 0.4 * (1.0 - 0.3) ** 2 + 0.6 * 0.3**2
    assert h._verify_constant_brier(
        probe_prevalence=0.4, training_prevalence=0.3, observed=observed
    ) == pytest.approx(observed)
    assert observed - 0.4 * 0.6 == pytest.approx((0.4 - 0.3) ** 2)


def test_constant_brier_rejects_inconsistent_value():
    with pytest.raises(h.StageHAuditError, match="inconsistent"):
        h._verify_constant_brier(
            probe_prevalence=0.4, training_prevalence=0.3, observed=0.1
        )


def make_record(
    *,
    timestep: int,
    training_prevalence: float,
    raw_rows: int,
    adverse_rows: int,
    selected_rows: int,
    model_brier: float,
    policy_overall: float,
    baseline_overall: float,
    acceptance_pass: bool,
    brier_pass: bool,
    gate_value: int = 0,
):
    p = adverse_rows / raw_rows
    constant_brier = p * (1.0 - training_prevalence) ** 2 + (1.0 - p) * training_prevalence**2
    acceptance = selected_rows / 126.0
    baseline_adverse = 0.05
    policy_adverse = 0.035
    baseline_beneficial = 0.08
    policy_beneficial = 0.06
    base = {
        "row_count": 126,
        "selected_row_count": raw_rows,
        "acceptance_rate": raw_rows / 126.0,
        "overall_mse_ratio": baseline_overall,
        "accepted_row_mse_ratio": baseline_overall,
        "positive_distance_reduction_rate": 0.60,
        "relative_distance_reduction_mean": 0.03,
        "adverse_sse_mass": baseline_adverse,
        "beneficial_sse_mass": baseline_beneficial,
        "net_sse_reduction_mass": baseline_beneficial - baseline_adverse,
        "risk_brier_score": 0.0,
        "risk_constant_brier_score": 0.0,
        "risk_brier_nonworse": True,
        "group_tail": {"worst_fraction_cvar_mse_ratio": 1.10},
        "adverse_row_count": adverse_rows,
        "raw_accepted_row_count": raw_rows,
    }
    record = {
        "row_count": 126,
        "selected_row_count": selected_rows,
        "acceptance_rate": acceptance,
        "overall_mse_ratio": policy_overall,
        "accepted_row_mse_ratio": 0.95,
        "positive_distance_reduction_rate": 0.65,
        "relative_distance_reduction_mean": 0.04,
        "adverse_sse_mass": policy_adverse,
        "beneficial_sse_mass": policy_beneficial,
        "net_sse_reduction_mass": policy_beneficial - policy_adverse,
        "risk_brier_score": model_brier,
        "risk_constant_brier_score": constant_brier,
        "risk_brier_nonworse": brier_pass,
        "group_tail": {"worst_fraction_cvar_mse_ratio": 1.05},
        "adverse_row_count": adverse_rows,
        "raw_accepted_row_count": raw_rows,
    }
    checks = {
        "acceptance": acceptance_pass,
        "overall_mse": True,
        "accepted_mse": True,
        "positive_reduction": True,
        "relative_reduction": True,
        "adverse_sse_reduction": True,
        "group_cvar_improvement": True,
        "risk_brier_nonworse": brier_pass,
    }
    return {
        "timestep": timestep,
        "risk_path": "stagef_canonical_t10" if timestep == 10 else "frozen_stagec_resume1_legacy_control",
        "descriptor_id": "compact_v1",
        "risk_l2": 4.0 if timestep == 10 else 1.0,
        "temperature": 1.0,
        "direction_shrinkage": 1.0,
        "risk_threshold": 0.5,
        "record": record,
        "no_abstention_baseline": base,
        "eligibility": {"pass": all(checks.values()), "checks": checks},
        "gate_counts": {
            "length_log_z_element_mismatch_count": gate_value,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        },
        "direction_sha256": "1" * 64,
        "candidate_sha256": "2" * 64,
        "selected_scale_sha256": "3" * 64,
        "descriptor_sha256": "4" * 64,
        "risk_probability_sha256": "5" * 64,
    }


def synthetic_sources(*, geometry_regression=False, raw_candidate_failure=False):
    training = {
        "10": {
            "risk_prevalence": 0.30,
            "adverse_label_rate": 0.28,
            "risk_fit_row_count": 600,
        },
        "25": {
            "risk_prevalence": 0.35,
            "adverse_label_rate": 0.33,
            "risk_fit_row_count": 610,
        },
        "50": {
            "risk_prevalence": 0.40,
            "adverse_label_rate": 0.38,
            "risk_fit_row_count": 612,
        },
    }
    baseline_overall = 1.01 if raw_candidate_failure else 0.98
    records = {
        "10": make_record(
            timestep=10,
            training_prevalence=0.30,
            raw_rows=120,
            adverse_rows=48,
            selected_rows=33,
            model_brier=0.35,
            policy_overall=0.9916,
            baseline_overall=baseline_overall,
            acceptance_pass=False,
            brier_pass=False,
            gate_value=1 if geometry_regression else 0,
        ),
        "25": make_record(
            timestep=25,
            training_prevalence=0.35,
            raw_rows=120,
            adverse_rows=54,
            selected_rows=63,
            model_brier=0.31,
            policy_overall=0.9819,
            baseline_overall=baseline_overall,
            acceptance_pass=True,
            brier_pass=False,
        ),
        "50": make_record(
            timestep=50,
            training_prevalence=0.40,
            raw_rows=119,
            adverse_rows=57,
            selected_rows=90,
            model_brier=0.30,
            policy_overall=0.9811,
            baseline_overall=baseline_overall,
            acceptance_pass=True,
            brier_pass=False,
        ),
    }
    probe = {
        "schema": h.SOURCE_PROBE_SCHEMA,
        "execution_verdict": "PASS",
        "portable_environment_audit": {"portable_contract_passed": True},
        "required_operation_dry_run": {"pass": True},
        "frozen_probe_accessed": False,
    }
    probe["probe_evidence_sha256"] = h.sha256_bytes(h.stable_json_bytes(probe))
    worker = {
        "schema": h.SOURCE_WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r259_stageg_canonical_joint_policy_fails_one_shot_frozen_probe",
        "required_next_path": "AUDIT_R259_FROZEN_PROBE_TRANSFER_FAILURE_WITHOUT_REACCESS_OR_RETUNING",
        "preregistration_contract": {
            "canonical_recipe_id": h.EXPECTED_CANONICAL_RECIPE,
            "one_shot_frozen_probe": True,
            "rerun_forbidden": True,
        },
        "training_diagnostics": training,
        "frozen_probe_timestep_records": records,
        "procedure_gate_totals": {
            "length_log_z_element_mismatch_count": 1 if geometry_regression else 0,
            "aligned_upper_element_failure_count": 0,
            "strict_pass_aligned_fail_row_count": 0,
        },
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
    }
    worker["worker_result_sha256"] = h.sha256_bytes(h.stable_json_bytes(worker))
    summary = {
        "schema": h.SOURCE_SUMMARY_SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": worker["root_cause"],
        "required_next_path": worker["required_next_path"],
        "all_timesteps_pass": False,
        "training_diagnostics": copy.deepcopy(training),
        "frozen_probe_timestep_records": copy.deepcopy(records),
        "procedure_gate_totals": copy.deepcopy(worker["procedure_gate_totals"]),
        "frozen_probe_evaluation_count_added": 1,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed": True,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "durable_evidence_protocol": {
            "worker_result_sha256": worker["worker_result_sha256"],
            "worker_file_sha256": "a" * 64,
            "probe_file_sha256": "b" * 64,
        },
    }
    summary["summary_sha256"] = h.sha256_bytes(h.stable_json_bytes(summary))
    return probe, worker, summary


def test_timestep_audit_decomposes_brier():
    _, worker, _ = synthetic_sources()
    audit = h._timestep_audit(10, worker["training_diagnostics"]["10"], worker["frozen_probe_timestep_records"]["10"])
    transfer = audit["prevalence_transfer"]
    probability = audit["probability_transfer"]
    assert transfer["brier_penalty_from_prevalence_shift"] == pytest.approx((0.4 - 0.3) ** 2)
    assert probability["model_brier_excess_over_training_constant"] > 0.0
    assert probability["base_rate_shift_alone_explains_model_failure"] is False


def test_timestep_audit_acceptance_attribution():
    _, worker, _ = synthetic_sources()
    audit = h._timestep_audit(10, worker["training_diagnostics"]["10"], worker["frozen_probe_timestep_records"]["10"])
    assert audit["frozen_probe_raw_candidate_population"]["raw_candidate_admission_rate"] == pytest.approx(120 / 126)
    assert audit["abstention"]["conditional_keep_rate_given_raw_candidate"] == pytest.approx(33 / 120)
    assert audit["eligibility"]["failed_checks"] == ["acceptance", "risk_brier_nonworse"]


def test_timestep_audit_candidate_effect():
    _, worker, _ = synthetic_sources()
    audit = h._timestep_audit(25, worker["training_diagnostics"]["25"], worker["frozen_probe_timestep_records"]["25"])
    effect = audit["candidate_and_policy_effect"]
    assert effect["raw_candidate_mean_improvement"] is True
    assert effect["adverse_sse_reduction_fraction"] == pytest.approx(0.3)
    assert effect["beneficial_sse_retention_fraction"] == pytest.approx(0.75)


@pytest.mark.parametrize("timestep", [10, 25, 50])
def test_timestep_audit_geometry_preserved(timestep):
    _, worker, _ = synthetic_sources()
    audit = h._timestep_audit(timestep, worker["training_diagnostics"][str(timestep)], worker["frozen_probe_timestep_records"][str(timestep)])
    assert audit["geometry"]["all_aligned_gate_counts_zero"] is True


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("row_count", 125, "population"),
        ("raw_accepted_row_count", 0, "no raw candidate"),
        ("adverse_row_count", 121, "inconsistent"),
        ("selected_row_count", 121, "exceeds"),
    ],
)
def test_timestep_audit_rejects_bad_counts(field, value, match):
    _, worker, _ = synthetic_sources()
    item = copy.deepcopy(worker["frozen_probe_timestep_records"]["10"])
    item["record"][field] = value
    with pytest.raises(h.StageHAuditError, match=match):
        h._timestep_audit(10, worker["training_diagnostics"]["10"], item)


def test_build_audit_current_classification():
    probe, worker, summary = synthetic_sources()
    audit = h.build_audit(probe, worker, summary)
    assert audit["classification"] == (
        "risk_probability_transfer_failure_beyond_base_rate_shift_with_"
        "raw_candidate_mean_improvement_preserved"
    )
    assert audit["required_next_path"] == (
        "DESIGN_R259_RISK_TRANSFER_REPAIR_ON_OBJECTIVE_TRAIN_WITH_FRESH_"
        "UNTOUCHED_EVALUATION_SET"
    )
    assert audit["joint_findings"]["all_timesteps_risk_brier_worse_than_training_prevalence_constant"] is True
    assert audit["joint_findings"]["any_timestep_acceptance_gate_failed"] is True


def test_build_audit_geometry_branch():
    probe, worker, summary = synthetic_sources(geometry_regression=True)
    audit = h.build_audit(probe, worker, summary)
    assert audit["classification"] == "frozen_probe_transfer_includes_geometry_gate_regression"


def test_build_audit_mixed_candidate_branch():
    probe, worker, summary = synthetic_sources(raw_candidate_failure=True)
    audit = h.build_audit(probe, worker, summary)
    assert audit["classification"] == (
        "risk_probability_transfer_failure_beyond_base_rate_shift_with_mixed_candidate_transfer"
    )


def test_source_limitations_are_fail_closed():
    probe, worker, summary = synthetic_sources()
    limits = h.build_audit(probe, worker, summary)["source_evidence_limitations"]
    for key in (
        "auroc_computable",
        "average_precision_computable",
        "ece_computable",
        "reliability_bins_computable",
        "risk_score_distribution_shift_computable",
        "descriptor_distribution_shift_computable",
        "accept_mask_hamming_attribution_computable",
        "calibration_vs_ranking_failure_separable",
    ):
        assert limits[key] is False


def test_governance_forbids_reaccess_and_retuning():
    probe, worker, summary = synthetic_sources()
    governance = h.build_audit(probe, worker, summary)["governance_conclusion"]
    assert governance == {
        "frozen_probe_access_consumed": True,
        "frozen_probe_reaccess_authorized": False,
        "retuning_against_frozen_probe_authorized": False,
        "recipe_fallback_authorized": False,
        "timestep_cherry_pick_authorized": False,
        "future_final_evaluation_requires_fresh_untouched_data": True,
    }


def synthetic_stageh_summary():
    probe, worker, source_summary = synthetic_sources()
    audit = h.build_audit(probe, worker, source_summary)
    payload = {
        "schema": h.SCHEMA,
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": audit["root_cause"],
        "required_next_path": audit["required_next_path"],
        "audit": audit,
        "execution_counts": {
            "report_read_count": 3,
            "environment_probe_count": 0,
            "science_worker_count": 0,
            "cuda_operation_count": 0,
            "dataset_load_count": 0,
            "direction_fit_count": 0,
            "risk_fit_count": 0,
            "candidate_generation_count": 0,
            "recipe_evaluation_count": 0,
            "frozen_probe_evaluation_count": 0,
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "frozen_probe_evaluation_count_added": 0,
        "cumulative_frozen_probe_evaluation_count": 1,
        "frozen_probe_accessed_historically": True,
        "frozen_probe_reaccessed": False,
        "rerun_authorized": False,
    }
    for key in h.FALSE_BOUNDARIES:
        payload[key] = False
    payload["summary_sha256"] = h.sha256_bytes(h.stable_json_bytes(payload))
    return payload


def test_validate_summary_accepts():
    h.validate_summary(synthetic_stageh_summary())


def rehash(payload):
    payload["summary_sha256"] = h.sha256_bytes(
        h.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("scientific_status", "READY", "BLOCKED"),
        ("selected_configuration", {"x": 1}, "selected"),
        ("train_only_recommendation", {"x": 1}, "recommendation"),
        ("selection_holdout_evaluation_count_added", 1, "selection holdout"),
        ("cumulative_selection_holdout_evaluation_count", 2, "count"),
        ("frozen_probe_evaluation_count_added", 1, "re-accessed"),
        ("cumulative_frozen_probe_evaluation_count", 2, "count"),
        ("frozen_probe_accessed_historically", False, "historical"),
        ("frozen_probe_reaccessed", True, "reaccess"),
        ("rerun_authorized", True, "rerun"),
    ],
)
def test_validate_summary_rejects_boundary_changes(field, value, match):
    payload = synthetic_stageh_summary()
    payload[field] = value
    rehash(payload)
    with pytest.raises(h.StageHAuditError, match=match):
        h.validate_summary(payload)


def test_validate_summary_rejects_forbidden_boundary():
    payload = synthetic_stageh_summary()
    payload["cuda_used"] = True
    rehash(payload)
    with pytest.raises(h.StageHAuditError, match="forbidden"):
        h.validate_summary(payload)


def test_validate_summary_rejects_counts():
    payload = synthetic_stageh_summary()
    payload["execution_counts"]["report_read_count"] = 2
    rehash(payload)
    with pytest.raises(h.StageHAuditError, match="execution counts"):
        h.validate_summary(payload)


def test_validate_summary_rejects_bad_hash():
    payload = synthetic_stageh_summary()
    payload["summary_sha256"] = "x" * 64
    with pytest.raises(h.StageHAuditError, match="self-hash"):
        h.validate_summary(payload)


def test_blocked_report_never_authorizes_reaccess():
    payload = h.blocked_report(None, RuntimeError("x"))
    assert payload["frozen_probe_evaluation_count_added"] == 0
    assert payload["frozen_probe_reaccessed"] is False
    assert payload["rerun_authorized"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None


@pytest.mark.parametrize("key", h.FALSE_BOUNDARIES)
def test_blocked_report_false_boundaries(key):
    assert h.blocked_report(None, RuntimeError("x"))[key] is False
