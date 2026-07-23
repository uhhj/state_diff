from __future__ import annotations

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import (
    phase314b_r258_stagev_resume1_preholdout_cuda_recovery as resume1,
)


def original_blocked_payload() -> dict:
    return {
        "phase": "Phase3.14b-r2.5.8 Stage V",
        "schema": (
            "phase314b_r258_stagev_locked_selection_holdout_evaluation_v1_"
            "blocked_v1"
        ),
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagev_execution_contract_failed",
        "required_next_path": (
            "DESIGN_ADD_ONLY_STAGEV_EXECUTION_CONTRACT_CORRECTION_"
            "WITHOUT_REACCESSING_HOLDOUT_IF_ACCESS_OCCURRED"
        ),
        "primary_failure_locus": "execution_contract",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "repository": {
            "head": resume1.BASE_STAGEV_IMPLEMENTATION_COMMIT,
            "parent": resume1.EXPECTED_STAGEV_IMPLEMENTATION_PARENT,
            "origin_experiment1": resume1.EXPECTED_REMOTE,
            "submodule_commit": resume1.EXPECTED_SUBMODULE,
        },
        "error_type": "StageVError",
        "error_message": (
            "environment probe rc=2 stdout='' "
            "stderr='BLOCKED: no CUDA device is available\\n'"
        ),
        "selection_holdout_access_state": (
            "unknown_if_failure_after_worker_start"
        ),
        "complete_worker_population_claimed": False,
        "complete_fit_count_claimed": False,
        "complete_candidate_count_claimed": False,
        "complete_holdout_evaluation_count_claimed": False,
        "rerun_authorized": False,
        **{key: False for key in resume1.ORIGINAL_FALSE_BOUNDARIES},
    }


def control_flow() -> dict:
    return {
        "environment_probe_precedes_science_worker": True,
        "science_worker_precedes_worker_payload_load": True,
        "controller_has_no_in_process_holdout_evaluation": True,
    }


def recovered_summary() -> dict:
    execution = {
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
    }
    return {
        "phase": "Phase3.14b-r2.5.8 Stage V",
        "schema": "base-schema",
        "execution_verdict": "PASS",
        "scientific_status": "READY",
        "root_cause": "base-root",
        "required_next_path": "base-next",
        "failed_timesteps": [],
        "selection_holdout_evaluated": True,
        "selection_holdout_used_for_fit_or_selection": False,
        "evaluation_count": 1,
        "rerun_authorized": False,
        "stagev_execution": execution,
        "selected_configuration": {"backbone_id": "x"},
        "train_only_recommendation": None,
        "scientific_result_sha256": "old",
    }


def fake_stagev() -> SimpleNamespace:
    return SimpleNamespace(
        PHASE="Phase3.14b-r2.5.8 Stage V",
        SCHEMA="base-schema",
        WORKER_SCHEMA="base-worker-schema",
        validate_worker_payload=lambda value: value,
    )


def test_add_only_three_paths() -> None:
    assert len(resume1.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.IMPLEMENTATION_PATHS)


def test_original_commits_are_exact() -> None:
    assert resume1.BASE_STAGEV_IMPLEMENTATION_COMMIT.startswith("5bb1140e")
    assert resume1.BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT.startswith("e02d8131")


def test_original_blocked_sha_is_exact() -> None:
    assert resume1.EXPECTED_ORIGINAL_BLOCKED_REPORT_SHA256 == (
        "f9facbe30f6292256451ec94ceaaf87654b5ea5cd95892956e4433ba90bb033c"
    )


def test_validate_original_blocked_proves_no_holdout() -> None:
    result = resume1.validate_original_blocked_payload(
        original_blocked_payload(), control_flow=control_flow()
    )
    assert result["failure_stage"] == "environment_probe"
    assert result["science_worker_started"] is False
    assert result["selection_holdout_target_access_count"] == 0
    assert result["selection_holdout_evaluation_count"] == 0
    assert result["holdout_access_before_resume1_proven_false"] is True


@pytest.mark.parametrize(
    "message",
    [
        "single cold science worker rc=2 stdout='' stderr='x'",
        "environment probe rc=2 stdout='' stderr='some other failure'",
        "no CUDA device is available",
    ],
)
def test_validate_original_blocked_rejects_unproven_message(message: str) -> None:
    payload = original_blocked_payload()
    payload["error_message"] = message
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_original_blocked_payload(payload, control_flow=control_flow())


def test_validate_original_blocked_rejects_changed_repository() -> None:
    payload = original_blocked_payload()
    payload["repository"]["head"] = "changed"
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_original_blocked_payload(payload, control_flow=control_flow())


def test_validate_original_blocked_rejects_true_boundary() -> None:
    payload = original_blocked_payload()
    payload[resume1.ORIGINAL_FALSE_BOUNDARIES[0]] = True
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_original_blocked_payload(payload, control_flow=control_flow())


def test_validate_original_blocked_rejects_incomplete_control_flow() -> None:
    bad = control_flow()
    bad["environment_probe_precedes_science_worker"] = False
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_original_blocked_payload(
            original_blocked_payload(), control_flow=bad
        )




def test_real_original_control_flow_proof() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / resume1.ORIGINAL_STAGEV_SOURCE
    )
    proof = resume1.prove_original_control_flow(source)
    assert proof["environment_probe_precedes_science_worker"] is True
    assert proof["controller_has_no_in_process_holdout_evaluation"] is True


def test_original_environment_probe_worker_hash_frozen() -> None:
    assert resume1.ORIGINAL_SOURCE_SHA256[
        "scripts/phase3_14b_r258_stages_resume2a_worker.py"
    ].startswith("096ffbed")


def test_validate_recovered_summary_accepts() -> None:
    result = resume1.validate_recovered_summary(
        recovered_summary(), stagev=fake_stagev()
    )
    assert result["selection_holdout_evaluated"] is True


@pytest.mark.parametrize(
    "field,bad",
    [
        ("selection_holdout_evaluated", False),
        ("selection_holdout_used_for_fit_or_selection", True),
        ("evaluation_count", 2),
        ("rerun_authorized", True),
    ],
)
def test_validate_recovered_summary_rejects_boundary(field: str, bad: object) -> None:
    payload = recovered_summary()
    payload[field] = bad
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_recovered_summary(payload, stagev=fake_stagev())


def test_validate_recovered_summary_rejects_execution_count_change() -> None:
    payload = recovered_summary()
    payload["stagev_execution"]["candidate_generation_count"] = 4
    with pytest.raises(resume1.StageVResume1Error):
        resume1.validate_recovered_summary(payload, stagev=fake_stagev())


def test_classify_failure_environment_probe() -> None:
    result = resume1.classify_failure_stage(
        RuntimeError("environment probe rc=2 stdout='' stderr='x'")
    )
    assert result["holdout_access_state"] == "proven_not_accessed"
    assert result["science_worker_started"] is False


def test_classify_failure_science_worker_is_unknown() -> None:
    result = resume1.classify_failure_stage(
        RuntimeError("single cold science worker rc=2 stdout='' stderr='x'")
    )
    assert result["holdout_access_state"] == "unknown_after_worker_start"
    assert result["science_worker_started"] is True


def test_blocked_report_never_authorizes_rerun() -> None:
    report = resume1.blocked_report(repository=None, error=RuntimeError("x"))
    assert report["rerun_authorized"] is False
    assert report["resume1_rerun_authorized"] is False
    assert report["selected_configuration"] is None


def test_success_and_blocked_paths_are_distinct_from_original() -> None:
    assert resume1.RESUME1_SUCCESS_REPORT != resume1.RESUME1_BLOCKED_REPORT
    assert resume1.RESUME1_SUCCESS_REPORT != resume1.ORIGINAL_STAGEV_SUCCESS_REPORT
    assert resume1.RESUME1_BLOCKED_REPORT != resume1.ORIGINAL_STAGEV_BLOCKED_REPORT


def test_execute_script_reuses_original_run_not_original_execute() -> None:
    source = inspect.getsource(resume1.run_recovery)
    assert "stagev.run_evaluation(" in source
    assert resume1.ORIGINAL_STAGEV_EXECUTE not in source


def test_resume1_does_not_name_holdout_target_data_key() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "ccda_phase3"
        / "phase314b_r258_stagev_resume1_preholdout_cuda_recovery.py"
    )
    text = path.read_text(encoding="utf-8")
    assert 'context["holdout_target"]' not in text
    assert "context['holdout_target']" not in text


def test_run_recovery_calls_original_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    stagev = fake_stagev()

    def run_evaluation(**kwargs):
        calls.append(kwargs)
        return recovered_summary()

    stagev.run_evaluation = run_evaluation
    monkeypatch.setattr(resume1, "_original_stagev_module", lambda: stagev)
    repository = {
        "head": "resume1-implementation",
        "parent": resume1.BASE_STAGEV_BLOCKED_EVIDENCE_COMMIT,
        "original_failure_proof": {
            "holdout_access_before_resume1_proven_false": True
        },
    }
    result = resume1.run_recovery(
        root=Path("/tmp/example"),
        repository=repository,
        python_bin="python",
        cuda_admission={"cuda_available": True, "cuda_device_count": 1},
    )
    assert len(calls) == 1
    assert result["prior_selection_holdout_evaluation_count"] == 0
    assert result["resume1_selection_holdout_evaluation_count"] == 1
    assert result["cumulative_selection_holdout_evaluation_count"] == 1
    assert result["resume1_rerun_authorized"] is False


def test_run_recovery_rejects_unproven_prior_closure() -> None:
    with pytest.raises(resume1.StageVResume1Error):
        resume1.run_recovery(
            root=Path("/tmp/example"),
            repository={
                "head": "x",
                "parent": "y",
                "original_failure_proof": {
                    "holdout_access_before_resume1_proven_false": False
                },
            },
            python_bin="python",
            cuda_admission={"cuda_available": True},
        )


def test_run_recovery_rejects_missing_cuda() -> None:
    with pytest.raises(resume1.StageVResume1Error):
        resume1.run_recovery(
            root=Path("/tmp/example"),
            repository={
                "head": "x",
                "parent": "y",
                "original_failure_proof": {
                    "holdout_access_before_resume1_proven_false": True
                },
            },
            python_bin="python",
            cuda_admission={"cuda_available": False},
        )


def test_deterministic_environment_contract_is_complete() -> None:
    assert resume1.EXPECTED_ENV == {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
