from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r259_stagee_recipe_instability_audit as e


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "value",
    [e.BASE_HEAD, e.BASE_IMPLEMENTATION, e.EXPECTED_SUBMODULE],
)
def test_commit_identities_are_sha1(value):
    assert len(value) == 40
    int(value, 16)


def test_stage_is_read_only_audit():
    assert e.PHASE.endswith("Stage E")
    assert e.SUCCESS_REPORT.endswith("recipe_instability_audit_summary.json")
    assert e.BLOCKED_REPORT.endswith("recipe_instability_audit_blocked_summary.json")


def test_implementation_is_add_only_three_files():
    assert len(e.IMPLEMENTATION_PATHS) == 3
    assert {status for status, _ in e.IMPLEMENTATION_PATHS} == {"A"}


def test_remote_is_not_a_gate():
    source = Path(e.__file__).read_text(encoding="utf-8")
    assert "EXPECTED_REMOTE" not in source


def test_stable_json_order_independent():
    assert e.stable_json_bytes({"b": 1, "a": 2}) == e.stable_json_bytes({"a": 2, "b": 1})


def test_sha256_known_value():
    assert e.sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_atomic_write_once(tmp_path):
    target = tmp_path / "x.json"
    e.atomic_write_once(target, b"x")
    assert target.read_bytes() == b"x"


def test_atomic_write_refuses_overwrite(tmp_path):
    target = tmp_path / "x.json"
    target.write_bytes(b"old")
    with pytest.raises(e.StageEAuditError, match="write-once"):
        e.atomic_write_once(target, b"new")


def test_recipe_id_round_trip():
    recipe = {
        "descriptor_id": "compact_geometry_v2",
        "risk_l2": 4.0,
        "temperature": 0.75,
        "shrinkage": 1.0,
        "risk_threshold": 0.5,
    }
    assert e.recipe_id(recipe) == (
        "desc_compact_geometry_v2__l2_4.00__temp_0.75__shrink_1.00__risk_0.50"
    )


def test_operational_family_id():
    assert e.operational_family_id({"shrinkage": 1.0, "risk_threshold": 0.5}) == "shrink_1.00__risk_0.50"


def test_calibration_family_id():
    assert e.calibration_family_id({
        "descriptor_id": "compact_v1", "risk_l2": 4.0, "temperature": 1.0
    }) == "desc_compact_v1__l2_4.00__temp_1.00"


def test_counter_report():
    report = e._counter_report(["b", "a", "b"])
    assert report == {
        "counts": {"a": 1, "b": 2},
        "modal_value": "b",
        "modal_support": 2,
        "population": 3,
        "unique_count": 2,
    }


def test_counter_report_empty_fails():
    with pytest.raises(e.StageEAuditError, match="empty"):
        e._counter_report([])


def test_metric_spread_nested():
    records = [{"x": {"y": 1.0}}, {"x": {"y": 1.5}}]
    assert e._metric_spread(records, ("x", "y")) == {"min": 1.0, "max": 1.5, "spread": 0.5}


def test_metric_spread_missing_fails():
    with pytest.raises(e.StageEAuditError, match="missing"):
        e._metric_spread([{"x": 1}], ("x", "y"))


def real_evidence():
    return e.validate_source_evidence(ROOT)


def test_real_source_evidence_validates():
    worker, summary = real_evidence()
    assert worker["worker_result_sha256"] == e.SOURCE_WORKER_RESULT_SHA256
    assert summary["summary_sha256"] == e.SOURCE_SUMMARY_SELF_SHA256


def test_real_source_file_hashes():
    assert e.sha256_file(ROOT / e.SOURCE_WORKER) == e.SOURCE_WORKER_SHA256
    assert e.sha256_file(ROOT / e.SOURCE_SUMMARY) == e.SOURCE_SUMMARY_SHA256


def real_audit():
    worker, summary = real_evidence()
    return e.audit_recipe_instability(worker, summary)


def test_real_classification():
    audit = real_audit()
    assert audit["classification"] == "operational_family_stable_calibration_recipe_realizations_differ"


def test_real_next_path():
    audit = real_audit()
    assert audit["required_next_path"] == (
        "DESIGN_R259_STAGED_T10_OPERATIONAL_FAMILY_LOCK_WITH_NESTED_CALIBRATION_CANONICALIZATION"
    )


def test_real_exact_recipe_support():
    support = real_audit()["exact_recipe_support"]
    assert support["modal_support"] == 3
    assert support["unique_count"] == 4
    assert support["population"] == 6


@pytest.mark.parametrize(
    "axis,modal,support",
    [
        ("descriptor_id", "compact_geometry_v2", 5),
        ("risk_l2", "4.00", 5),
        ("temperature", "0.75", 4),
        ("shrinkage", "1.00", 6),
        ("risk_threshold", "0.50", 6),
    ],
)
def test_real_factor_support(axis, modal, support):
    value = real_audit()["factor_support"][axis]
    assert value["modal_value"] == modal
    assert value["modal_support"] == support


def test_real_operational_family_support_is_six_of_six():
    value = real_audit()["operational_family_support"]
    assert value["modal_value"] == "shrink_1.00__risk_0.50"
    assert value["modal_support"] == 6


def test_real_eligible_counts():
    value = real_audit()["eligible_surface"]
    assert value["eligible_recipe_count_by_fold"] == [62, 60, 57, 60, 62, 65]
    assert value["intersection_count"] == 57
    assert value["union_count"] == 67
    assert value["recipes_eligible_in_all_six_folds"] == 57


def test_real_selected_family_all_eligible():
    value = real_audit()["selected_operational_family"]
    assert value["recipe_population"] == 30
    assert value["eligible_recipe_count_by_fold"] == [30, 30, 30, 30, 30, 30]
    assert value["all_family_recipes_eligible_in_every_fold"] is True


def test_real_common_surface_all_selected_eligible():
    value = real_audit()["common_full_objective_surface"]
    assert value["unique_selected_recipe_count"] == 4
    assert value["all_selected_recipes_eligible"] is True


def test_real_common_surface_not_exact_output_identity():
    value = real_audit()["common_full_objective_surface"]
    assert value["unique_selected_scale_sha_count"] == 4
    assert value["unique_candidate_sha_count"] == 4
    assert value["exact_output_identity"] is False


@pytest.mark.parametrize(
    "metric,maximum_spread",
    [
        ("acceptance_rate", 0.04),
        ("overall_mse_ratio", 0.02),
        ("accepted_row_mse_ratio", 0.02),
        ("positive_distance_reduction_rate", 0.04),
        ("relative_distance_reduction_mean", 0.02),
        ("risk_brier_score", 0.02),
        ("worst_fraction_cvar_mse_ratio", 0.01),
    ],
)
def test_real_common_surface_spreads_are_recorded(metric, maximum_spread):
    spread = real_audit()["common_full_objective_surface"]["metric_spreads"][metric]["spread"]
    assert 0.0 <= spread < maximum_spread


def test_evidence_limitations_are_explicit():
    value = real_audit()["source_evidence_limitations"]
    assert value["per_recipe_inner_selection_scores_persisted"] is False
    assert value["near_tie_rank_margin_computable"] is False
    assert value["row_level_accept_masks_persisted"] is False
    assert value["accept_mask_hamming_distance_computable"] is False


def synthetic_repository():
    return {
        "root": "/tmp/repo",
        "head": "a" * 40,
        "parent": e.BASE_HEAD,
        "stage_d_resume3_implementation": e.BASE_IMPLEMENTATION,
        "stage_d_resume3_evidence": e.BASE_HEAD,
        "submodule_commit": e.EXPECTED_SUBMODULE,
        "remote_identity_used_as_gate": False,
    }


def test_build_real_summary():
    worker, source = real_evidence()
    payload = e.build_summary(synthetic_repository(), worker, source)
    assert payload["execution_verdict"] == "PASS"
    assert payload["scientific_status"] == "BLOCKED"
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None
    e.validate_summary(payload)


def real_summary_payload():
    worker, source = real_evidence()
    return e.build_summary(synthetic_repository(), worker, source)


def rehash(payload):
    payload["summary_sha256"] = e.sha256_bytes(
        e.stable_json_bytes({k: v for k, v in payload.items() if k != "summary_sha256"})
    )


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("selected_configuration", {"x": 1}, "selected"),
        ("train_only_recommendation", {"x": 1}, "recommendation"),
        ("selection_holdout_evaluation_count_added", 1, "holdout"),
        ("cumulative_selection_holdout_evaluation_count", 2, "holdout"),
        ("frozen_probe_accessed", True, "frozen probe"),
        ("rerun_authorized", True, "rerun"),
    ],
)
def test_summary_boundary_changes_fail(field, value, match):
    payload = real_summary_payload()
    payload[field] = value
    rehash(payload)
    with pytest.raises(e.StageEAuditError, match=match):
        e.validate_summary(payload)


@pytest.mark.parametrize("field", e.FALSE_BOUNDARIES[:8])
def test_false_boundaries_fail(field):
    payload = real_summary_payload()
    payload[field] = True
    rehash(payload)
    with pytest.raises(e.StageEAuditError):
        e.validate_summary(payload)


def test_nonzero_science_execution_count_fails():
    payload = real_summary_payload()
    payload["execution_counts"]["science_worker_count"] = 1
    rehash(payload)
    with pytest.raises(e.StageEAuditError, match="scientific execution"):
        e.validate_summary(payload)


def test_bad_summary_hash_fails():
    payload = real_summary_payload()
    payload["summary_sha256"] = "0" * 64
    with pytest.raises(e.StageEAuditError, match="self-hash"):
        e.validate_summary(payload)


def test_blocked_report_boundaries():
    payload = e.blocked_report(None, RuntimeError("x"))
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None
    assert payload["rerun_authorized"] is False
    assert all(payload[key] is False for key in e.FALSE_BOUNDARIES)


def test_remove_one_fold_fails():
    worker, summary = real_evidence()
    changed = copy.deepcopy(worker)
    changed["outer_fold_selections"] = changed["outer_fold_selections"][:-1]
    with pytest.raises(e.StageEAuditError, match="exactly 6"):
        e.audit_recipe_instability(changed, summary)


def test_fallback_fails():
    worker, summary = real_evidence()
    changed = copy.deepcopy(worker)
    changed["outer_fold_selections"][0]["diagnostic_fallback_used"] = True
    with pytest.raises(e.StageEAuditError, match="fallback"):
        e.audit_recipe_instability(changed, summary)


def test_selected_recipe_not_eligible_fails():
    worker, summary = real_evidence()
    changed = copy.deepcopy(worker)
    rid = changed["outer_fold_selections"][0]["selected_recipe_id"]
    changed["outer_fold_selections"][0]["eligible_recipe_ids"].remove(rid)
    with pytest.raises(e.StageEAuditError, match="identity mismatch"):
        e.audit_recipe_instability(changed, summary)
