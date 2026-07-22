from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import (
    phase314b_r258_staget_resume1_process_topology_schema_recovery as resume1,
)


def false_boundaries() -> dict:
    return {key: False for key in resume1.FALSE_BOUNDARIES}


def real_stage_s_report() -> dict:
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": (
            "phase314b_r258_stages_resume3_"
            "tolerance_aligned_oof_support_confirmed"
        ),
        "required_next_path": (
            "FREEZE_TOLERANCE_ALIGNED_GATE_AND_CONFIRM_"
            "OOF_CANDIDATE_MATRIX_ON_OBJECTIVE_TRAIN_ONLY"
        ),
        "scientific_result_sha256": (
            "9316b92cc349a3b689f3336066f6f768e0f6e7022e48368e964ab6211ae2da64"
        ),
        "selected_configuration": None,
        "train_only_recommendation": None,
        "process_topology": {
            "environment_probe_process_count": 1,
            "cold_science_worker_process_count": 2,
            "probe_process_id": 3085,
            "science_worker_process_ids": [3165, 3384],
            "probe_distinct_from_each_science_worker": True,
            "workers_launched_sequentially": True,
            "environment_probe_rerun_in_science_workers": False,
            "environment_probe_payload_sha256": "probe",
            "environment_sha256": "environment",
            "temporary_payloads_deleted": True,
        },
        "confirmation_execution": {
            "worker_count": 2,
            "total_oof_fit_count": 108,
            "total_callback_pair_count": 54,
            "total_internal_scale_attempt_count": 378,
        },
        "confirmation_summary": {
            "cell_count": 27,
            "nonzero_support_cell_count": 27,
            "zero_acceptance_cell_count": 0,
            "oracle_like_cell_count": 7,
            "dominant_discriminator": "direction_retention",
            "dominant_support_count": 2,
        },
        **false_boundaries(),
    }


def fake_original_validator(report: dict) -> dict:
    execution = report["confirmation_execution"]
    assert execution["processes_distinct"] is True
    assert execution["workers_sequential"] is True
    return report


def fake_staget() -> SimpleNamespace:
    return SimpleNamespace(
        EXPECTED_WORKER_COUNT=2,
        validate_base_report=fake_original_validator,
    )


def recovered_summary() -> dict:
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "matrix-result",
        "required_next_path": "next-matrix-stage",
        "primary_failure_locus": "candidate-matrix",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "confirmation_execution": {
            "total_oof_fit_count": 108,
            "total_callback_pair_count": 54,
        },
        "candidate_matrix": {
            "mechanism_eligible_cell_count": 27,
            "fidelity_eligible_cell_count": 11,
            "matrix_eligible_backbone_count": 2,
        },
        **false_boundaries(),
    }


def recovered_gate() -> dict:
    return {
        "deployment_authorized": False,
        "selected_configuration": None,
        "train_only_recommendation": None,
        **false_boundaries(),
    }


def test_extract_real_schema() -> None:
    result = resume1.extract_process_topology(
        real_stage_s_report(), expected_worker_count=2
    )
    assert result["probe_process_id"] == 3085
    assert result["science_worker_process_ids"] == [3165, 3384]
    assert result["all_process_ids_distinct"] is True


def test_corrected_validator_does_not_mutate_report() -> None:
    report = real_stage_s_report()
    before = copy.deepcopy(report)
    result = resume1.corrected_validate_base_report(
        report, staget=fake_staget()
    )
    assert result is report
    assert report == before
    assert "processes_distinct" not in report["confirmation_execution"]


def test_corrected_validator_calls_original_once() -> None:
    calls = []

    def validator(report: dict) -> dict:
        calls.append(report)
        return fake_original_validator(report)

    staget = fake_staget()
    staget.validate_base_report = validator
    resume1.corrected_validate_base_report(real_stage_s_report(), staget=staget)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "environment_probe_process_count",
        "cold_science_worker_process_count",
        "probe_process_id",
        "science_worker_process_ids",
        "probe_distinct_from_each_science_worker",
        "workers_launched_sequentially",
        "environment_probe_rerun_in_science_workers",
        "temporary_payloads_deleted",
    ],
)
def test_missing_real_topology_field_is_rejected(field: str) -> None:
    report = real_stage_s_report()
    del report["process_topology"][field]
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("probe_distinct_from_each_science_worker", None),
        ("probe_distinct_from_each_science_worker", 1),
        ("probe_distinct_from_each_science_worker", "true"),
        ("workers_launched_sequentially", None),
        ("workers_launched_sequentially", 1),
        ("workers_launched_sequentially", "true"),
        ("environment_probe_rerun_in_science_workers", 0),
        ("temporary_payloads_deleted", 1),
    ],
)
def test_topology_boolean_type_is_strict(field: str, bad: object) -> None:
    report = real_stage_s_report()
    report["process_topology"][field] = bad
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("environment_probe_process_count", None),
        ("environment_probe_process_count", True),
        ("environment_probe_process_count", 1.0),
        ("cold_science_worker_process_count", "2"),
        ("probe_process_id", False),
        ("probe_process_id", 3.5),
    ],
)
def test_topology_integer_type_is_strict(field: str, bad: object) -> None:
    report = real_stage_s_report()
    report["process_topology"][field] = bad
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


@pytest.mark.parametrize(
    "field",
    [
        "probe_distinct_from_each_science_worker",
        "workers_launched_sequentially",
        "temporary_payloads_deleted",
    ],
)
def test_required_true_topology_flags_are_enforced(field: str) -> None:
    report = real_stage_s_report()
    report["process_topology"][field] = False
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


def test_probe_rerun_flag_must_be_false() -> None:
    report = real_stage_s_report()
    report["process_topology"][
        "environment_probe_rerun_in_science_workers"
    ] = True
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


@pytest.mark.parametrize(
    "worker_pids",
    [[3085, 3384], [3165, 3165]],
)
def test_process_ids_must_be_distinct(worker_pids: list[int]) -> None:
    report = real_stage_s_report()
    report["process_topology"]["science_worker_process_ids"] = worker_pids
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


@pytest.mark.parametrize("worker_pids", [[], [3165], [3165, 3384, 3500]])
def test_worker_pid_population_is_exact(worker_pids: list[int]) -> None:
    report = real_stage_s_report()
    report["process_topology"]["science_worker_process_ids"] = worker_pids
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


def test_invented_aliases_are_rejected() -> None:
    report = real_stage_s_report()
    report["confirmation_execution"]["processes_distinct"] = True
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


def test_alias_only_report_is_rejected() -> None:
    report = real_stage_s_report()
    del report["process_topology"]
    report["confirmation_execution"].update(
        {"processes_distinct": True, "workers_sequential": True}
    )
    with pytest.raises(resume1.StageTResume1Error):
        resume1.extract_process_topology(report, expected_worker_count=2)


def test_context_manager_restores_validator() -> None:
    staget = fake_staget()
    original = staget.validate_base_report
    with resume1.patched_base_report_validator(staget):
        assert staget.validate_base_report is not original
        staget.validate_base_report(real_stage_s_report())
    assert staget.validate_base_report is original


def test_context_manager_restores_after_exception() -> None:
    staget = fake_staget()
    original = staget.validate_base_report
    with pytest.raises(RuntimeError):
        with resume1.patched_base_report_validator(staget):
            raise RuntimeError("stop")
    assert staget.validate_base_report is original


def test_recovered_summary_validation() -> None:
    resume1._validate_recovered_summary(recovered_summary())


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "READY"),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_recovered_summary_rejects_boundary_change(
    field: str, bad: object
) -> None:
    summary = recovered_summary()
    summary[field] = bad
    with pytest.raises(resume1.StageTResume1Error):
        resume1._validate_recovered_summary(summary)


def test_recovered_gate_validation() -> None:
    resume1._validate_recovered_gate(recovered_gate())


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("deployment_authorized", True),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_recovered_gate_rejects_boundary_change(field: str, bad: object) -> None:
    gate = recovered_gate()
    gate[field] = bad
    with pytest.raises(resume1.StageTResume1Error):
        resume1._validate_recovered_gate(gate)


def test_execute_recovery_calls_original_stage_t_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / resume1.BASE_STAGES_RESUME3_REPORT
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(real_stage_s_report()), encoding="utf-8")
    calls = []
    fake = fake_staget()

    def run_confirmation(**kwargs):
        calls.append(kwargs)
        return recovered_summary(), recovered_gate()

    fake.run_confirmation = run_confirmation
    import ccda_phase3

    monkeypatch.setattr(
        ccda_phase3,
        "phase314b_r258_staget_aligned_gate_candidate_matrix",
        fake,
        raising=False,
    )
    repository = {
        "head": "implementation",
        "historical_process_topology": resume1.extract_process_topology(
            real_stage_s_report(), expected_worker_count=2
        ),
    }
    result, gate = resume1.execute_recovery(
        root=tmp_path,
        repository=repository,
        python_bin="python",
    )
    assert len(calls) == 1
    assert result["execution_verdict"] == "PASS"
    assert result["recovered_stage_t_result"]["root_cause"] == "matrix-result"
    assert gate["deployment_authorized"] is False
    assert gate["recovered_stage_t_gate_contract"] == recovered_gate()


def test_execute_recovery_restores_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / resume1.BASE_STAGES_RESUME3_REPORT
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(real_stage_s_report()), encoding="utf-8")
    fake = fake_staget()
    original = fake.validate_base_report
    fake.run_confirmation = lambda **kwargs: (recovered_summary(), recovered_gate())
    import ccda_phase3

    monkeypatch.setattr(
        ccda_phase3,
        "phase314b_r258_staget_aligned_gate_candidate_matrix",
        fake,
        raising=False,
    )
    repository = {
        "head": "implementation",
        "historical_process_topology": resume1.extract_process_topology(
            real_stage_s_report(), expected_worker_count=2
        ),
    }
    resume1.execute_recovery(
        root=tmp_path,
        repository=repository,
        python_bin="python",
    )
    assert fake.validate_base_report is original


def test_blocked_report_claims_no_science() -> None:
    report = resume1.blocked_report(repository=None, error=RuntimeError("x"))
    assert report["execution_verdict"] == "BLOCKED"
    assert report["stage_t_worker_population_started"] is None
    assert report["completed_oof_fit_count_claimed"] is None
    assert report["completed_callback_pair_count_claimed"] is None
    assert report["completed_matrix_reconstruction_attempt_count_claimed"] is None
    assert report["execution_counts_not_claimed_without_success_report"] is True
    assert report["selected_configuration"] is None
    assert report["train_only_recommendation"] is None


def test_stable_json_is_deterministic() -> None:
    assert resume1.stable_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_success_and_blocked_paths_are_distinct() -> None:
    assert len(
        {
            resume1.SUCCESS_REPORT,
            resume1.GATE_CONTRACT_REPORT,
            resume1.BLOCKED_REPORT,
            resume1.BASE_STAGET_BLOCKED_REPORT,
        }
    ) == 4
