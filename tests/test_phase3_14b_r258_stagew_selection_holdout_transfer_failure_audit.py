from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from ccda_phase3 import (
    phase314b_r258_stagew_selection_holdout_transfer_failure_audit as stagew,
)


def _stats(count: int, *, mean: float = 0.1, minimum: float = -0.2) -> dict:
    if minimum > mean:
        raise ValueError("minimum must not exceed mean")
    span = max(abs(mean - minimum), 0.1)
    return {
        "count": count,
        "mean": mean,
        "std": 0.2,
        "min": minimum,
        "p05": minimum + 0.10 * span,
        "median": mean,
        "p95": mean + 0.50 * span,
        "max": mean + span,
    }


def _stageu_cell(
    timestep: int,
    selected: int,
    overall: float,
    accepted: float,
    positive: float,
    relative: float,
) -> dict:
    return {
        "accepted_row_mse_ratio": accepted,
        "aligned_acceptance_rate": selected / stagew.EXPECTED_OBJECTIVE_ROWS,
        "aligned_candidate_sha256": f"{timestep:064x}"[-64:],
        "aligned_selected_scale_histogram": {
            "0": stagew.EXPECTED_OBJECTIVE_ROWS - selected,
            "2": selected,
        },
        "aligned_selected_scale_sha256": f"{timestep + 1:064x}"[-64:],
        "aligned_upper_element_failure_count": 0,
        "base_direction_id": stagew.LOCKED_BACKBONE,
        "feature_mode": "full_segment_constraint",
        "fidelity_eligible": True,
        "global_rank": 1,
        "legacy_functional_identity_exact": True,
        "length_log_z_element_mismatch_count": 0,
        "matrix_reconstruction_exact": True,
        "mechanism_eligible": True,
        "overall_mse_ratio": overall,
        "per_timestep_rank": 1,
        "positive_distance_reduction_rate": positive,
        "relative_distance_reduction_mean": relative,
        "row_count": stagew.EXPECTED_OBJECTIVE_ROWS,
        "selected_row_count": selected,
        "strict_pass_aligned_fail_row_count": 0,
        "timestep": timestep,
    }


def make_stageu_contract() -> dict:
    cells = [
        _stageu_cell(10, 613, 0.9568765463967528, 0.954716946762974, 0.6182707993474714, 0.03731045290964939),
        _stageu_cell(25, 626, 0.8631487559799662, 0.859470986196053, 0.7332268370607029, 0.09348730898776085),
        _stageu_cell(50, 625, 0.8989194158335131, 0.8950161130212291, 0.6464, 0.059029166016273255),
    ]
    locked_sha = stagew.sha256_bytes(stagew.stable_json_bytes(cells))
    value = {
        "phase": "Phase3.14b-r2.5.8 Stage U",
        "schema": "phase314b_r258_stageu_candidate_frontier_contract_v1",
        "frontier_lock": {
            "backbone_id": stagew.LOCKED_BACKBONE,
            "timesteps": list(stagew.LOCKED_TIMESTEPS),
            "timestep_policy": "joint_all_timesteps_no_cherry_pick",
            "backbone_search_reopened": False,
            "timestep_search_reopened": False,
            "fallback_backbone_allowed": False,
            "fallback_timestep_allowed": False,
            "frontier_is_unique": True,
            "all_locked_timesteps_fidelity_eligible": True,
            "locked_cells": cells,
            "locked_cells_sha256": locked_sha,
        },
        "aligned_gate_lock": {"inner_gate_contract_sha256": "a" * 64},
        "next_stage_holdout_policy": {
            "evaluation_count": 1,
            "population": "locked_backbone_all_three_timesteps",
            "backbone_id": stagew.LOCKED_BACKBONE,
            "timesteps": list(stagew.LOCKED_TIMESTEPS),
            "all_timesteps_must_pass": True,
            "post_holdout_backbone_selection_forbidden": True,
            "post_holdout_timestep_selection_forbidden": True,
            "threshold_changes_after_holdout_forbidden": True,
            "rerun_after_failure_forbidden": True,
            "aggregate_metrics_are_diagnostic_only": True,
            "frozen_probe_remains_closed": True,
        },
    }
    value["contract_payload_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(value))
    return value


def _record(
    timestep: int,
    selected: int,
    positive_count: int,
    overall_ratio: float,
    accepted_ratio: float,
    relative_mean: float,
) -> dict:
    positive_rate = positive_count / selected
    accepted = {
        "control_distance": _stats(selected, mean=1.0, minimum=0.2),
        "candidate_distance": _stats(selected, mean=0.99, minimum=0.1),
        "distance_reduction": _stats(selected, mean=0.01, minimum=-0.5),
        "relative_distance_reduction": _stats(selected, mean=relative_mean, minimum=-0.4),
        "motion_norm": _stats(selected, mean=0.05, minimum=0.001),
        "mse_ratio": accepted_ratio,
        "positive_distance_reduction_rate": positive_rate,
    }
    fidelity = {
        "row_count": stagew.EXPECTED_HOLDOUT_ROWS,
        "selected_row_count": selected,
        "acceptance_rate": selected / stagew.EXPECTED_HOLDOUT_ROWS,
        "overall_control_distance": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=1.0, minimum=0.1),
        "overall_candidate_distance": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=0.99, minimum=0.1),
        "overall_distance_reduction": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=0.01, minimum=-0.5),
        "overall_relative_distance_reduction": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=0.01, minimum=-0.4),
        "overall_motion_norm": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=0.04, minimum=0.0),
        "overall_mse_ratio": overall_ratio,
        "accepted_rows": accepted,
        "selected_scale_stats": _stats(stagew.EXPECTED_HOLDOUT_ROWS, mean=1.8, minimum=0.0),
        "candidate_motion_positive_rate": selected / stagew.EXPECTED_HOLDOUT_ROWS,
        "mechanism_eligible": True,
        "fidelity_eligible": False,
        "eligibility_policy": {},
    }
    return {
        "base_direction_id": stagew.LOCKED_BACKBONE,
        "timestep": timestep,
        "feature_mode": "full_segment_constraint",
        "selection_holdout_rows": stagew.EXPECTED_HOLDOUT_ROWS,
        "objective_fit_rows": 500,
        "full_fit_repeat_count": 2,
        "model_identity": {"sha": f"{timestep:064x}"[-64:]},
        "holdout_feature_sha256": "b" * 64,
        "holdout_direction_sha256": "c" * 64,
        "candidate_sha256": "d" * 64,
        "selected_scale_sha256": "e" * 64,
        "selected_scale_histogram": {
            "0": stagew.EXPECTED_HOLDOUT_ROWS - selected,
            "2": selected,
        },
        "acceptance_rate": selected / stagew.EXPECTED_HOLDOUT_ROWS,
        "internal_scale_attempt_order": [2.0, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        "internal_scale_attempt_count": 7,
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "candidate_finalized_before_holdout_target_access": True,
        "holdout_target_used_for_fit": False,
        "holdout_target_used_for_selection": False,
        "holdout_target_used_only_for_final_metrics": True,
        "holdout_target_access_count": 1,
        "candidate_fidelity": fidelity,
        "pass_checks": {
            "mechanism_eligible": True,
            "overall_mse_ratio": False,
            "accepted_row_mse_ratio": False,
            "positive_distance_reduction_rate": True,
            "relative_distance_reduction_mean": True,
            "length_log_z_element_mismatch_count": True,
            "aligned_upper_element_failure_count": True,
            "strict_pass_aligned_fail_row_count": True,
            "all": False,
        },
        "scientific_pass": False,
        "candidate_tensor_persisted": False,
        "selected_scale_tensor_persisted": False,
        "direction_tensor_persisted": False,
        "holdout_target_tensor_persisted": False,
    }


def make_stagev_summary() -> dict:
    records = [
        _record(10, 211, 112, 1.0261, 1.0315, 0.00660),
        _record(25, 210, 132, 1.0014, 1.0017, 0.02465),
        _record(50, 217, 126, 1.0084, 1.0098, 0.01414),
    ]
    value = {
        "phase": "Phase3.14b-r2.5.8 Stage V Resume1",
        "schema": "phase314b_r258_stagev_resume1_preholdout_cuda_recovery_v1",
        "execution_verdict": "PASS",
        "resume1_execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagev_locked_frontier_fails_preregistered_selection_holdout_contract",
        "required_next_path": "AUDIT_SELECTION_HOLDOUT_TRANSFER_FAILURE_WITHOUT_RERUN_RETUNING_OR_FALLBACK",
        "primary_failure_locus": "all_timestep_fidelity_transfer_failure",
        "passed_timesteps": [],
        "failed_timesteps": [10, 25, 50],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "selection_holdout_evaluated": True,
        "selection_holdout_used_for_fit_or_selection": False,
        "evaluation_count": 1,
        "rerun_authorized": False,
        "resume1_rerun_authorized": False,
        "holdout_access_before_resume1": False,
        "holdout_access_before_resume1_proven": True,
        "prior_selection_holdout_evaluation_count": 0,
        "resume1_selection_holdout_evaluation_count": 1,
        "cumulative_selection_holdout_evaluation_count": 1,
        "original_stagev_blocked_report_preserved": True,
        "original_stagev_blocked_report_sha256": stagew.EXPECTED_ORIGINAL_STAGEV_BLOCKED_REPORT_SHA256,
        "stagev_execution": {
            "environment_probe_count": 1,
            "cold_science_worker_count": 1,
            "objective_full_fit_count": 6,
            "holdout_direction_prediction_count": 6,
            "candidate_generation_count": 3,
            "joint_selection_holdout_evaluation_count": 1,
            "holdout_target_metric_access_count": 3,
            "internal_scale_attempt_count": 21,
            "oof_fit_count": 0,
            "callback_pair_count": 0,
            "worker_output_persisted": False,
        },
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_process_count": 1,
            "probe_process_id": 2497,
            "science_worker_process_id": 2574,
            "processes_distinct": True,
            "workers_launched_sequentially": True,
            "temporary_payloads_deleted": True,
        },
        "recovery_scope": {
            "execution_contract_only": True,
            "cuda_unavailable_failure_recovered": True,
            "science_code_changed": False,
            "frontier_changed": False,
            "timestep_policy_changed": False,
            "gate_changed": False,
            "threshold_changed": False,
            "fallback_opened": False,
            "holdout_reaccessed": False,
        },
        "timestep_records": records,
    }
    for key in (
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
        "surrogate_weights_persisted",
        "prediction_tensor_persisted",
        "direction_tensor_persisted",
        "candidate_tensor_persisted",
        "predicate_tensor_persisted",
        "callback_event_persisted",
        "npz_saved",
        "cache_saved",
        "image_saved",
        "video_saved",
    ):
        value[key] = False
    value["scientific_result_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(value))
    return value


@pytest.fixture(autouse=True)
def deterministic_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in stagew.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)


def _patch_stageu_hashes(monkeypatch: pytest.MonkeyPatch, contract: dict) -> None:
    monkeypatch.setattr(stagew, "EXPECTED_STAGEU_LOCKED_CELLS_SHA256", contract["frontier_lock"]["locked_cells_sha256"])
    monkeypatch.setattr(stagew, "EXPECTED_STAGEU_CONTRACT_PAYLOAD_SHA256", contract["contract_payload_sha256"])


def _patch_stagev_hash(monkeypatch: pytest.MonkeyPatch, summary: dict) -> None:
    monkeypatch.setattr(stagew, "EXPECTED_STAGEV_RESUME1_SCIENTIFIC_RESULT_SHA256", summary["scientific_result_sha256"])


def test_environment_validation_passes() -> None:
    assert stagew.validate_environment_variables() == dict(stagew.EXPECTED_ENV)


def test_environment_validation_rejects_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(stagew.StageWError):
        stagew.validate_environment_variables()


def test_stable_json_is_deterministic() -> None:
    assert stagew.stable_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_stable_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        stagew.stable_json_bytes({"x": math.nan})


def test_scalar_stats_accepts_ordered_values() -> None:
    assert stagew.validate_scalar_stats(_stats(3), "stats")["count"] == 3


def test_scalar_stats_rejects_bad_quantiles() -> None:
    value = _stats(3)
    value["p05"] = 2.0
    with pytest.raises(stagew.StageWError):
        stagew.validate_scalar_stats(value, "stats")


def test_stageu_contract_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = make_stageu_contract()
    _patch_stageu_hashes(monkeypatch, contract)
    result = stagew.validate_stageu_contract(contract)
    assert [cell["timestep"] for cell in result["locked_cells"]] == [10, 25, 50]


def test_stageu_contract_rejects_timestep_search(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = make_stageu_contract()
    contract["frontier_lock"]["timestep_search_reopened"] = True
    contract.pop("contract_payload_sha256")
    contract["contract_payload_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(contract))
    _patch_stageu_hashes(monkeypatch, contract)
    with pytest.raises(stagew.StageWError):
        stagew.validate_stageu_contract(contract)


def test_stagev_summary_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = make_stagev_summary()
    _patch_stagev_hash(monkeypatch, summary)
    result = stagew.validate_stagev_resume1_summary(summary)
    assert [record["selected_row_count"] for record in result["records"]] == [211, 210, 217]


def test_stagev_summary_rejects_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = make_stagev_summary()
    summary["rerun_authorized"] = True
    summary.pop("scientific_result_sha256")
    summary["scientific_result_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(summary))
    _patch_stagev_hash(monkeypatch, summary)
    with pytest.raises(stagew.StageWError):
        stagew.validate_stagev_resume1_summary(summary)


def test_stagev_summary_rejects_changed_failure_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = make_stagev_summary()
    summary["timestep_records"][0]["pass_checks"]["overall_mse_ratio"] = True
    summary.pop("scientific_result_sha256")
    summary["scientific_result_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(summary))
    _patch_stagev_hash(monkeypatch, summary)
    with pytest.raises(stagew.StageWError):
        stagew.validate_stagev_resume1_summary(summary)


def test_histogram_total_variation_zero() -> None:
    assert stagew.histogram_total_variation({"0": 1, "2": 3}, {"0": 2, "2": 6}) == 0.0


def test_histogram_total_variation_disjoint() -> None:
    assert stagew.histogram_total_variation({"0": 4}, {"2": 4}) == 1.0


def test_integer_positive_count() -> None:
    assert stagew.integer_positive_count(112 / 211, 211, "t10") == 112


def test_integer_positive_count_rejects_noninteger_rate() -> None:
    with pytest.raises(stagew.StageWError):
        stagew.integer_positive_count(0.531, 211, "t10")


def test_accepted_sse_share_identity() -> None:
    share = stagew.accepted_baseline_sse_share(1.0261, 1.0315)
    assert share == pytest.approx(0.8285714285714307)


def test_accepted_sse_share_rejects_impossible() -> None:
    with pytest.raises(stagew.StageWError):
        stagew.accepted_baseline_sse_share(1.2, 1.1)


def _validated_payloads(monkeypatch: pytest.MonkeyPatch):
    contract = make_stageu_contract()
    summary = make_stagev_summary()
    _patch_stageu_hashes(monkeypatch, contract)
    _patch_stagev_hash(monkeypatch, summary)
    return stagew.validate_stageu_contract(contract), stagew.validate_stagev_resume1_summary(summary)


def test_timestep_diagnostic_proves_minority_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    train, holdout = _validated_payloads(monkeypatch)
    result = stagew.build_timestep_diagnostic(train["locked_cells"][0], holdout["records"][0])
    counts = result["accepted_population_counts"]
    assert counts["strictly_improving_rows"] == 112
    assert counts["non_improving_rows"] == 99
    assert result["sse_identity"]["minority_non_improving_side_dominates_net_accepted_sse"] is True


def test_timestep_diagnostic_computes_transfer_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    train, holdout = _validated_payloads(monkeypatch)
    result = stagew.build_timestep_diagnostic(train["locked_cells"][1], holdout["records"][1])
    assert result["transfer_gaps"]["overall_mse_ratio"] > 0.0
    assert result["contract_failure_margins"]["overall_mse_ratio_minus_one"] == pytest.approx(0.0014)


def test_timestep_diagnostic_rejects_no_adverse_min(monkeypatch: pytest.MonkeyPatch) -> None:
    train, holdout = _validated_payloads(monkeypatch)
    altered = copy.deepcopy(holdout["records"][0])
    altered["accepted_distribution"]["distance_reduction"]["min"] = 0.0
    altered["accepted_distribution"]["distance_reduction"]["p05"] = 0.0
    with pytest.raises(stagew.StageWError):
        stagew.build_timestep_diagnostic(train["locked_cells"][0], altered)


def test_cross_timestep_diagnostic_selects_t25_only_diagnostically(monkeypatch: pytest.MonkeyPatch) -> None:
    train, holdout = _validated_payloads(monkeypatch)
    records = [
        stagew.build_timestep_diagnostic(a, b)
        for a, b in zip(train["locked_cells"], holdout["records"])
    ]
    result = stagew.build_cross_timestep_diagnostic(records)
    assert result["diagnostic_closest_to_threshold_timestep"] == 25
    assert result["closest_timestep_selection_authorized"] is False


def test_evidence_sufficiency_closes_row_attribution() -> None:
    result = stagew.build_evidence_sufficiency()
    assert result["aggregate_localisation_complete"] is True
    assert result["row_level_causal_localisation_claimed"] is False
    assert result["unavailable"]["condition_attribution"] is True


def test_negative_result_contract_is_self_hashed() -> None:
    contract = stagew.build_negative_result_contract()
    observed = contract["contract_payload_sha256"]
    copy_value = dict(contract)
    copy_value.pop("contract_payload_sha256")
    assert observed == stagew.sha256_bytes(stagew.stable_json_bytes(copy_value))


def test_negative_result_contract_forbids_t25_near_pass() -> None:
    contract = stagew.build_negative_result_contract()
    assert contract["forbidden"]["choose_t25_as_near_pass"] is True


def test_negative_result_contract_keeps_probe_closed() -> None:
    contract = stagew.build_negative_result_contract()
    assert contract["allowed_next_research"]["frozen_probe_remains_closed_until_a_separate_preregistered_final_test"] is True


def test_execute_audit_builds_passed_audit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = make_stageu_contract()
    summary = make_stagev_summary()
    _patch_stageu_hashes(monkeypatch, contract)
    _patch_stagev_hash(monkeypatch, summary)
    (tmp_path / "reports").mkdir()
    (tmp_path / stagew.STAGEU_CONTRACT).write_text(json.dumps(contract), encoding="utf-8")
    (tmp_path / stagew.STAGEV_RESUME1_SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    repository = {"head": "x", "source_semantics": {"unselected_candidate_equals_control": True}}
    result = stagew.execute_audit(root=tmp_path, repository=repository)
    assert result["execution_verdict"] == "PASS"
    assert result["scientific_status"] == "BLOCKED"
    assert result["selection_holdout_evaluation_count_added"] == 0
    assert result["cross_timestep_diagnostic"]["failure_is_cross_timestep_systematic"] is True


def test_validate_output_rejects_selection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    contract = make_stageu_contract()
    summary = make_stagev_summary()
    _patch_stageu_hashes(monkeypatch, contract)
    _patch_stagev_hash(monkeypatch, summary)
    (tmp_path / "reports").mkdir()
    (tmp_path / stagew.STAGEU_CONTRACT).write_text(json.dumps(contract), encoding="utf-8")
    (tmp_path / stagew.STAGEV_RESUME1_SUMMARY).write_text(json.dumps(summary), encoding="utf-8")
    result = stagew.execute_audit(root=tmp_path, repository={"source_semantics": {}})
    result["selected_configuration"] = {"timestep": 25}
    result.pop("audit_result_sha256")
    result["audit_result_sha256"] = stagew.sha256_bytes(stagew.stable_json_bytes(result))
    with pytest.raises(stagew.StageWError):
        stagew.validate_output_payload(result)


def test_blocked_report_adds_no_holdout_evaluation() -> None:
    result = stagew.blocked_report(repository=None, error=RuntimeError("boom"))
    assert result["selection_holdout_evaluation_count_added"] == 0
    assert result["selection_holdout_reaccessed"] is False


def test_write_once(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    stagew.write_once(path, b"{}")
    assert path.read_bytes() == b"{}"
    with pytest.raises(stagew.StageWError):
        stagew.write_once(path, b"{}")


def test_source_has_no_torch_or_numpy_import() -> None:
    source = Path(stagew.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source
    assert "from numpy" not in source


def test_source_does_not_import_science_modules() -> None:
    source = Path(stagew.__file__).read_text(encoding="utf-8")
    assert "from ccda_phase3 import" not in source
    assert "import ccda_phase3" not in source


def test_audit_false_boundaries_are_complete() -> None:
    contract = stagew.build_negative_result_contract()
    for key in stagew.FALSE_BOUNDARIES:
        assert contract[key] is False


def test_exact_expected_timesteps() -> None:
    assert stagew.LOCKED_TIMESTEPS == (10, 25, 50)


def test_current_evidence_commit_is_frozen() -> None:
    assert stagew.BASE_STAGEV_RESUME1_EVIDENCE_COMMIT == "55e3cfdcbc4ed8d3392f9c307f793fdd1799de6c"


def test_current_evidence_file_sha_is_frozen() -> None:
    assert stagew.EXPECTED_STAGEV_RESUME1_SUMMARY_SHA256 == "1b27e2bd20f1022cf2baa8017b995c27e3c9fc486ea72761c7305415990f181b"
