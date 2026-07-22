from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest

from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages


def fake_repeat(index: int = 0):
    return {
        "current_prediction_sha256": f"p{index}",
        "repeat_prediction_sha256": f"p{index}",
        "repeat_prediction_byte_exact": True,
        "fold_records_sha256": f"f{index}",
        "model_identities_sha256": f"m{index}",
        "fold_count": 6,
    }


def fake_cell(index: int, acceptance: float = 0.8):
    return {
        "base_direction_id": f"b{index // 3}",
        "timestep": (10, 25, 50)[index % 3],
        "feature_mode": "point",
        "feature_sha256": f"feature{index}",
        "legacy_identity": {
            "all_functional_exact": True,
            "portable_repeat_fit": fake_repeat(index),
        },
        "aligned_acceptance_rate": acceptance,
        "aligned_acceptance_locus": "partial_admission",
        "aligned_selected_scale_sha256": f"scale{index}",
        "aligned_selected_scale_histogram": {"1": 1},
        "aligned_candidate_sha256": f"candidate{index}",
        "internal_scale_attempt_order": [1.0, .75, .5, .25],
        "aligned_assembly": {"x": index},
        "raw_oracle_control": {"x": index},
        "projected_oracle_control": {"x": index},
        "comparator_source": "raw_oracle",
        "comparator_discriminator_predicate": None,
        "dual_oracle_discriminator_predicate": None,
        "strict_pass_aligned_fail_row_count": 0,
        "aligned_upper_element_failure_count": 0,
        "length_log_z_element_mismatch_count": 0,
    }


def fake_classification():
    return {
        "cell_count": 27,
        "backbone_count": 9,
        "timestep_count": 3,
        "oracle_like_admission_cell_count": 7,
        "meaningful_support_cell_count": 27,
        "zero_acceptance_cell_count": 0,
        "length_log_z_element_mismatch_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "aligned_upper_element_failure_count": 0,
        "dominant_dual_oracle_discriminator": "direction_retention",
        "dominant_discriminator_support_count": 2,
        "dominant_discriminator_backbone_coverage": 1,
        "dominant_discriminator_timestep_coverage": 2,
    }


def fake_wrapper():
    cells = [fake_cell(i, 0.95 if i < 7 else 0.8) for i in range(27)]
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stager_tolerance_alignment_restores_broad_oof_support",
        "required_next_path": "CONFIRM_TOLERANCE_ALIGNED_OOF_SUPPORT_ON_OBJECTIVE_TRAIN_ONLY",
        "primary_failure_locus": "broad_oof_support",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "stage_r_result": {
            "root_cause": "phase314b_r258_stager_tolerance_alignment_restores_broad_oof_support",
            "required_next_path": "CONFIRM_TOLERANCE_ALIGNED_OOF_SUPPORT_ON_OBJECTIVE_TRAIN_ONLY",
            "primary_failure_locus": "broad_oof_support",
            "tolerance_aligned_oof_post_upper_audit": {
                "cell_records": cells,
                "scientific_oof_fit_count": 27,
                "portable_repeat_fit_count": 27,
                "total_oof_fit_count": 54,
                "legacy_callback_off_on_pair_count": 27,
                "oracle_callback_rerun_count": 0,
                "historical_prediction_sha_exact_cell_count": 0,
                "historical_prediction_sha_mismatch_cell_count": 27,
                "classification": fake_classification(),
            },
        },
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }


def fake_worker(worker_id: str = "w1"):
    wrapper = fake_wrapper()
    functional = stages.functional_projection(wrapper)
    fit = stages.current_fit_projection(wrapper)
    summary = stages.validate_confirmation_projection(functional)
    return {
        "worker_id": worker_id,
        "execution_verdict": "PASS",
        "functional_projection_matches_base": True,
        "functional_projection_sha256": stages.sha256_bytes(stages.stable_json_bytes(functional)),
        "current_fit_projection_sha256": stages.sha256_bytes(stages.stable_json_bytes(fit)),
        "environment_sha256": "env",
        "confirmation_summary": summary,
        "functional_projection": functional,
        "current_fit_projection": fit,
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }


def test_phase_and_schema():
    assert stages.PHASE.endswith("Stage S")
    assert stages.SCHEMA.endswith("_v1")


def test_frozen_commit_chain():
    assert stages.BASE_IMPLEMENTATION_COMMIT == "5668d053ef267bbf8cc5c12d7a70656fe09fe697"
    assert stages.BASE_EVIDENCE_COMMIT == "35285e99304a926723379a3cf2a22acd17f9c0ac"


def test_frozen_report_hashes():
    assert stages.EXPECTED_BASE_REPORT_SHA256 == "4900dacf43affd7f9aa097d71978d481b9cb944ea5b19711a26f196a87a87d54"
    assert stages.EXPECTED_BASE_SCIENTIFIC_SHA256 == "04cd8feaeed1d8db16b8c4c8d3ba51242e0691e2cc139bc5b8749440d3f47fd3"


def test_add_only_four_paths():
    assert len(stages.IMPLEMENTATION_PATHS) == 4
    assert all(status == "A" for status, _ in stages.IMPLEMENTATION_PATHS)


def test_execution_counts():
    assert stages.EXPECTED_TOTAL_FITS == 108
    assert stages.EXPECTED_TOTAL_CALLBACK_PAIRS == 54
    assert stages.EXPECTED_TOTAL_INTERNAL_ATTEMPTS == 378


def test_current_fit_projection_count():
    value = stages.current_fit_projection(fake_wrapper())
    assert value["cell_count"] == 27
    assert len(value["records"]) == 27


def test_current_fit_projection_excludes_functional_candidate():
    value = stages.current_fit_projection(fake_wrapper())
    assert "aligned_candidate_sha256" not in value["records"][0]


def test_functional_projection_count():
    value = stages.functional_projection(fake_wrapper())
    assert value["cell_count"] == 27


def test_functional_projection_excludes_current_prediction_sha():
    value = stages.functional_projection(fake_wrapper())
    assert "oof_prediction_sha256" not in value["cell_records"][0]
    assert "current_prediction_sha256" not in value["cell_records"][0]


def test_functional_projection_includes_candidate_and_assembly():
    value = stages.functional_projection(fake_wrapper())
    assert value["cell_records"][0]["aligned_candidate_sha256"] == "candidate0"
    assert value["cell_records"][0]["aligned_assembly"] == {"x": 0}


def test_functional_projection_is_order_stable():
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"]["cell_records"].reverse()
    first = stages.functional_projection(wrapper)
    second = stages.functional_projection(fake_wrapper())
    assert stages.stable_json_bytes(first) == stages.stable_json_bytes(second)


def test_validate_confirmation_summary():
    value = stages.validate_confirmation_projection(stages.functional_projection(fake_wrapper()))
    assert value["nonzero_support_cell_count"] == 27
    assert value["oracle_like_cell_count"] == 7


@pytest.mark.parametrize("field,bad", [
    ("meaningful_support_cell_count", 26),
    ("zero_acceptance_cell_count", 1),
    ("oracle_like_admission_cell_count", 6),
    ("dominant_dual_oracle_discriminator", "topology"),
    ("dominant_discriminator_support_count", 3),
    ("dominant_discriminator_backbone_coverage", 2),
    ("dominant_discriminator_timestep_coverage", 3),
])
def test_validate_confirmation_rejects_classification_change(field, bad):
    projection = stages.functional_projection(fake_wrapper())
    projection["classification"][field] = bad
    with pytest.raises(stages.StageSError):
        stages.validate_confirmation_projection(projection)


@pytest.mark.parametrize("field", [
    "length_log_z_element_mismatch_count",
    "strict_pass_aligned_fail_row_count",
    "aligned_upper_element_failure_count",
])
def test_validate_confirmation_rejects_gate_regression(field):
    projection = stages.functional_projection(fake_wrapper())
    projection["classification"][field] = 1
    with pytest.raises(stages.StageSError):
        stages.validate_confirmation_projection(projection)


@pytest.mark.parametrize("field,bad", [
    ("scientific_oof_fit_count", 26),
    ("portable_repeat_fit_count", 26),
    ("total_oof_fit_count", 53),
    ("legacy_callback_off_on_pair_count", 26),
    ("oracle_callback_rerun_count", 1),
])
def test_validate_confirmation_rejects_execution_population_change(field, bad):
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"][field] = bad
    projection = stages.functional_projection(wrapper)
    with pytest.raises(stages.StageSError):
        stages.validate_confirmation_projection(projection)


def test_compare_workers_accepts_exact():
    assert stages.compare_workers(fake_worker("a"), fake_worker("b"))["all_exact"]


@pytest.mark.parametrize("field", [
    "functional_projection_sha256",
    "current_fit_projection_sha256",
    "environment_sha256",
    "confirmation_summary",
    "functional_projection",
    "current_fit_projection",
])
def test_compare_workers_rejects_difference(field):
    first = fake_worker("a")
    second = fake_worker("b")
    second[field] = "different"
    with pytest.raises(stages.StageSError):
        stages.compare_workers(first, second)


def test_compare_workers_ignores_worker_id():
    first = fake_worker("a")
    second = fake_worker("b")
    assert stages.compare_workers(first, second)["all_exact"]


def test_classify_confirmed_support():
    result = stages.classify_confirmation(fake_worker()["confirmation_summary"])
    assert result["root_cause"] == "phase314b_r258_stages_tolerance_aligned_oof_support_confirmed"
    assert result["required_next_path"].startswith("FREEZE_TOLERANCE_ALIGNED_GATE")


def test_classify_stable_discriminator():
    summary = dict(fake_worker()["confirmation_summary"])
    summary.update({
        "dominant_support_count": 24,
        "dominant_backbone_coverage": 8,
        "dominant_timestep_coverage": 3,
    })
    result = stages.classify_confirmation(summary)
    assert result["primary_failure_locus"] == "stable_post_upper_discriminator"


def test_classify_partial_support():
    summary = dict(fake_worker()["confirmation_summary"])
    summary["nonzero_support_cell_count"] = 24
    summary["zero_acceptance_cell_count"] = 3
    result = stages.classify_confirmation(summary)
    assert result["primary_failure_locus"] == "partial_oof_support"


def test_classify_failed_support():
    summary = dict(fake_worker()["confirmation_summary"])
    summary["nonzero_support_cell_count"] = 20
    summary["zero_acceptance_cell_count"] = 7
    result = stages.classify_confirmation(summary)
    assert result["primary_failure_locus"] == "oof_support_confirmation_failure"


def test_functional_projection_rejects_nonfunctional_cell():
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"]["cell_records"][0]["legacy_identity"]["all_functional_exact"] = False
    with pytest.raises(stages.StageSError):
        stages.functional_projection(wrapper)


def test_current_fit_projection_rejects_repeat_failure():
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"]["cell_records"][0]["legacy_identity"]["portable_repeat_fit"]["repeat_prediction_byte_exact"] = False
    with pytest.raises(stages.StageSError):
        stages.current_fit_projection(wrapper)


def test_functional_projection_rejects_selected_configuration():
    wrapper = fake_wrapper()
    wrapper["selected_configuration"] = "x"
    with pytest.raises(stages.StageSError):
        stages.functional_projection(wrapper)


def test_functional_projection_rejects_boundary_change():
    wrapper = fake_wrapper()
    wrapper[stages.FALSE_BOUNDARIES[0]] = True
    with pytest.raises(stages.StageSError):
        stages.functional_projection(wrapper)


def test_blocked_payload_preserves_base():
    payload = stages.blocked_report(repository={"head": "x", "base_report": {}}, error=RuntimeError("x"))
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["base_report_sha256"] == stages.EXPECTED_BASE_REPORT_SHA256
    assert payload["selected_configuration"] is None


def test_false_boundaries_unique():
    assert len(stages.FALSE_BOUNDARIES) == len(set(stages.FALSE_BOUNDARIES))


def test_environment_contract_complete():
    assert stages.EXPECTED_ENV["PYTHONHASHSEED"] == "0"
    assert stages.EXPECTED_ENV["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_hash_is_stable():
    assert stages.sha256_bytes(stages.stable_json_bytes({"b": 2, "a": 1})) == stages.sha256_bytes(stages.stable_json_bytes({"a": 1, "b": 2}))
