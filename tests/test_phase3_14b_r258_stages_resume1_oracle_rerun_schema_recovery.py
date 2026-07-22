from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import phase314b_r258_stages_oof_support_confirmation as stages
from ccda_phase3 import (
    phase314b_r258_stages_resume1_oracle_rerun_schema_recovery as resume1,
)


def fake_classification():
    return {
        "meaningful_support_cell_count": 27,
        "zero_acceptance_cell_count": 0,
        "oracle_like_admission_cell_count": 7,
        "dominant_dual_oracle_discriminator": "direction_retention",
        "dominant_discriminator_support_count": 2,
        "dominant_discriminator_backbone_coverage": 1,
        "dominant_discriminator_timestep_coverage": 2,
        "length_log_z_element_mismatch_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "aligned_upper_element_failure_count": 0,
    }


def fake_cell(index: int):
    timestep = (10, 25, 50)[index % 3]
    return {
        "base_direction_id": f"backbone_{index // 3}",
        "timestep": timestep,
        "feature_mode": "point",
        "feature_sha256": f"feature{index}",
        "aligned_acceptance_rate": 0.96 if index < 7 else 0.80,
        "aligned_acceptance_locus": "accepted",
        "aligned_selected_scale_sha256": f"scale{index}",
        "aligned_selected_scale_histogram": {"0.25": 1},
        "aligned_candidate_sha256": f"candidate{index}",
        "internal_scale_attempt_order": [1.0, 0.75, 0.5, 0.25],
        "aligned_assembly": {"cell": index},
        "raw_oracle_control": {"aligned_acceptance_rate": 1.0},
        "projected_oracle_control": {"aligned_acceptance_rate": 0.98},
        "comparator_source": "raw_oracle",
        "comparator_discriminator_predicate": None,
        "dual_oracle_discriminator_predicate": (
            "direction_retention" if index < 2 else None
        ),
        "strict_pass_aligned_fail_row_count": 0,
        "aligned_upper_element_failure_count": 0,
        "length_log_z_element_mismatch_count": 0,
        "legacy_identity": {
            "all_functional_exact": True,
            "portable_repeat_fit": {
                "repeat_prediction_byte_exact": True,
                "current_prediction_sha256": f"pred{index}",
                "repeat_prediction_sha256": f"pred{index}",
                "fold_records_sha256": f"fold{index}",
                "model_identities_sha256": f"model{index}",
                "fold_count": 6,
            },
        },
    }


def fake_wrapper(*, real_key=True, alias_key=False, value=0):
    audit = {
        "cell_records": [fake_cell(index) for index in range(27)],
        "scientific_oof_fit_count": 27,
        "portable_repeat_fit_count": 27,
        "total_oof_fit_count": 54,
        "legacy_callback_off_on_pair_count": 27,
        "historical_prediction_sha_exact_cell_count": 0,
        "historical_prediction_sha_mismatch_cell_count": 27,
        "classification": fake_classification(),
    }
    if real_key:
        audit["oracle_callback_pairs_rerun"] = value
    if alias_key:
        audit["oracle_callback_rerun_count"] = value
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
            "tolerance_aligned_oof_post_upper_audit": audit,
        },
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }


def fake_worker(worker_id: str, wrapper=None):
    wrapper = fake_wrapper() if wrapper is None else wrapper
    functional = resume1.corrected_functional_projection(wrapper, stages=stages)
    fit = stages.current_fit_projection(wrapper)
    summary = stages.validate_confirmation_projection(functional)
    return {
        "worker_id": worker_id,
        "execution_verdict": "PASS",
        "functional_projection_matches_base": True,
        "functional_projection_sha256": stages.sha256_bytes(
            stages.stable_json_bytes(functional)
        ),
        "current_fit_projection_sha256": stages.sha256_bytes(
            stages.stable_json_bytes(fit)
        ),
        "environment_sha256": "environment",
        "confirmation_summary": summary,
        "functional_projection": functional,
        "current_fit_projection": fit,
        **{key: False for key in stages.FALSE_BOUNDARIES},
    }


def test_phase_and_schema():
    assert resume1.PHASE.endswith("Stage S Resume1")
    assert resume1.SCHEMA.endswith("_v1")


def test_implementation_population():
    assert len(resume1.IMPLEMENTATION_PATHS) == 4


def test_original_source_population():
    assert set(resume1.ORIGINAL_SOURCE_SHA256) == {
        resume1.STAGES_SOURCE,
        resume1.STAGES_WORKER,
        resume1.STAGES_EXECUTE,
        resume1.STAGES_TEST,
    }


@pytest.mark.parametrize("value", [0, 1, 27])
def test_required_int_accepts_integer(value):
    assert resume1._required_int({"x": value}, "x", "value") == value


def test_required_int_rejects_missing():
    with pytest.raises(resume1.StageSResume1Error, match="is missing: x"):
        resume1._required_int({}, "x", "value")


@pytest.mark.parametrize("value", [None, "0", 0.0, True, False, [], {}])
def test_required_int_rejects_non_integer(value):
    with pytest.raises(resume1.StageSResume1Error, match="non-boolean integer"):
        resume1._required_int({"x": value}, "x", "value")


def test_corrected_projection_accepts_real_stage_r_key():
    value = resume1.corrected_functional_projection(fake_wrapper(), stages=stages)
    assert value["oracle_callback_rerun_count"] == 0


def test_corrected_projection_rejects_alias_only():
    with pytest.raises(resume1.StageSResume1Error, match="oracle_callback_pairs_rerun"):
        resume1.corrected_functional_projection(
            fake_wrapper(real_key=False, alias_key=True), stages=stages
        )


def test_corrected_projection_rejects_missing_key_without_typeerror():
    with pytest.raises(resume1.StageSResume1Error) as captured:
        resume1.corrected_functional_projection(
            fake_wrapper(real_key=False), stages=stages
        )
    assert not isinstance(captured.value, TypeError)


@pytest.mark.parametrize("value", [None, "0", 0.0, True])
def test_corrected_projection_rejects_bad_real_value(value):
    with pytest.raises(resume1.StageSResume1Error):
        resume1.corrected_functional_projection(
            fake_wrapper(value=value), stages=stages
        )


def test_projection_normalizes_only_output_name():
    projection = resume1.corrected_functional_projection(fake_wrapper(), stages=stages)
    assert "oracle_callback_rerun_count" in projection
    assert "oracle_callback_pairs_rerun" not in projection


def test_projection_matches_original_when_both_schema_keys_are_present():
    wrapper = fake_wrapper(real_key=True, alias_key=True)
    corrected = resume1.corrected_functional_projection(wrapper, stages=stages)
    original = stages.functional_projection(wrapper)
    assert stages.stable_json_bytes(corrected) == stages.stable_json_bytes(original)


def test_positive_rerun_survives_projection_but_validation_rejects():
    projection = resume1.corrected_functional_projection(
        fake_wrapper(value=1), stages=stages
    )
    assert projection["oracle_callback_rerun_count"] == 1
    with pytest.raises(stages.StageSError, match="oracle controls were rerun"):
        stages.validate_confirmation_projection(projection)


def test_projection_order_is_stable():
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"][
        "cell_records"
    ].reverse()
    first = resume1.corrected_functional_projection(wrapper, stages=stages)
    second = resume1.corrected_functional_projection(fake_wrapper(), stages=stages)
    assert stages.stable_json_bytes(first) == stages.stable_json_bytes(second)


def test_projection_rejects_nonfunctional_cell():
    wrapper = fake_wrapper()
    wrapper["stage_r_result"]["tolerance_aligned_oof_post_upper_audit"][
        "cell_records"
    ][0]["legacy_identity"]["all_functional_exact"] = False
    with pytest.raises(resume1.StageSResume1Error):
        resume1.corrected_functional_projection(wrapper, stages=stages)


def test_projection_rejects_selection():
    wrapper = fake_wrapper()
    wrapper["selected_configuration"] = "x"
    with pytest.raises(resume1.StageSResume1Error):
        resume1.corrected_functional_projection(wrapper, stages=stages)


def test_projection_rejects_boundary_change():
    wrapper = fake_wrapper()
    wrapper[stages.FALSE_BOUNDARIES[0]] = True
    with pytest.raises(resume1.StageSResume1Error):
        resume1.corrected_functional_projection(wrapper, stages=stages)


def test_projection_preserves_candidate_and_assembly():
    value = resume1.corrected_functional_projection(fake_wrapper(), stages=stages)
    assert value["cell_records"][0]["aligned_candidate_sha256"] == "candidate0"
    assert value["cell_records"][0]["aligned_assembly"] == {"cell": 0}


def test_projection_excludes_prediction_sha():
    value = resume1.corrected_functional_projection(fake_wrapper(), stages=stages)
    assert "current_prediction_sha256" not in value["cell_records"][0]


def test_worker_pair_remains_exact():
    assert stages.compare_workers(fake_worker("a"), fake_worker("b"))["all_exact"]


def test_worker_difference_still_rejected():
    first = fake_worker("a")
    second = fake_worker("b")
    second["functional_projection_sha256"] = "different"
    with pytest.raises(stages.StageSError):
        stages.compare_workers(first, second)


def test_no_worker_started_when_base_schema_is_invalid(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(resume1, "validate_environment_variables", lambda: {})

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("worker must not start")

    monkeypatch.setattr(resume1.subprocess, "run", fake_run)
    repository = {"base_report": fake_wrapper(real_key=False)}
    with pytest.raises(resume1.StageSResume1Error):
        resume1.run_recovered_confirmation(
            root=tmp_path,
            repository=repository,
            python_bin="python",
        )
    assert calls == []


def test_workers_are_started_sequentially(monkeypatch, tmp_path):
    starts = []
    monkeypatch.setattr(resume1, "validate_environment_variables", lambda: {})
    worker_value = fake_worker("placeholder")

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, **kwargs):
        starts.append(command[-3])
        output = Path(command[-1])
        value = copy.deepcopy(worker_value)
        value["worker_id"] = command[command.index("--worker-id") + 1]
        output.write_text(json.dumps(value), encoding="utf-8")
        return Completed()

    monkeypatch.setattr(resume1.subprocess, "run", fake_run)
    repository = {"base_report": fake_wrapper()}
    result = resume1.run_recovered_confirmation(
        root=tmp_path,
        repository=repository,
        python_bin="python",
    )
    assert len(starts) == 2
    assert result["confirmation_execution"]["worker_count"] == 2


def test_blocked_report_preserves_original_stage_s():
    report = resume1.blocked_report(
        repository={"head": "x", "base_report": {}},
        error=RuntimeError("x"),
    )
    assert report["execution_verdict"] == "BLOCKED"
    assert report["base_stage_s_blocked_report_sha256"] == (
        resume1.EXPECTED_STAGES_BLOCKED_REPORT_SHA256
    )
    assert report["selected_configuration"] is None


def test_false_boundaries_are_unique():
    assert len(resume1.FALSE_BOUNDARIES) == len(set(resume1.FALSE_BOUNDARIES))
