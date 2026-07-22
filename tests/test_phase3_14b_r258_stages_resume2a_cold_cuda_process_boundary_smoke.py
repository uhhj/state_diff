from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccda_phase3 import (
    phase314b_r258_stages_resume2a_cold_cuda_process_boundary_smoke as resume2a,
)


def env_payload():
    environment = {
        "compatibility_pass": True,
        "compatibility": {"required_operation_pass": True},
        "required_operation_dry_run": {"pass": True},
        "compatibility_sha256": "compat",
        "observation_sha256": "obs",
    }
    return {
        "phase": resume2a.PHASE,
        "schema": "phase314b_r258_stages_resume2a_environment_probe_v1",
        "mode": "environment_probe",
        "execution_verdict": "PASS",
        "process_id": 101,
        "environment": environment,
        "environment_sha256": resume2a.sha256_bytes(
            resume2a.stable_json_bytes(environment)
        ),
        "cuda_initialization_allowed_in_this_process": True,
        "process_disposable": True,
    }


def smoke_payload(probe=None, *, pid=202):
    probe = env_payload() if probe is None else probe
    environment = probe["environment"]
    control = {"reference_equivalence_pass": True, "sha": "control"}
    ulp = {"selected_factor": 4.0}
    return {
        "phase": resume2a.PHASE,
        "schema": "phase314b_r258_stages_resume2a_cold_control_smoke_v1",
        "mode": "cold_control_smoke",
        "execution_verdict": "PASS",
        "process_id": pid,
        "environment_sha256": resume2a.sha256_bytes(
            resume2a.stable_json_bytes(environment)
        ),
        "cold_cuda_precheck": {"torch_cuda_is_initialized": False},
        "runtime_cold_cuda_contract": {"torch_cuda_is_initialized": False},
        "frozen_control_replay_completed": True,
        "control_capture": control,
        "control_capture_sha256": resume2a.sha256_bytes(
            resume2a.stable_json_bytes(control)
        ),
        "ulp_policy": ulp,
        "ulp_policy_sha256": resume2a.sha256_bytes(
            resume2a.stable_json_bytes(ulp)
        ),
        "oof_fit_count": 0,
        "callback_pair_count": 0,
        "internal_scale_attempt_count": 0,
        "oracle_callback_rerun_count": 0,
        "selection_holdout_evaluated": False,
        "frozen_probe_accessed": False,
        "formal_training_run": False,
        "reverse_sampling_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "worker_output_persisted": False,
        "environment_probe_rerun_in_smoke_process": False,
    }


def install_fake_stage_modules(monkeypatch):
    cold_calls = []
    validate_calls = []

    class StageA:
        @staticmethod
        def validate_environment_payload(value):
            validate_calls.append(copy.deepcopy(value))

        @staticmethod
        def assert_cold_cuda_context_portable():
            cold_calls.append(True)
            return {"checked": True, "torch_cuda_is_initialized": False}

    stagea = StageA()
    stagef = SimpleNamespace(stagec258=SimpleNamespace(stagea258=stagea))
    stagel = SimpleNamespace(stagef=stagef)

    stageo = SimpleNamespace(
        _runtime_modules=lambda: {"stagel": stagel},
        _prepare_runtime=lambda root, environment: {
            "cold_main_worker_context": {
                "checked": True,
                "torch_cuda_is_initialized": False,
            },
            "control_capture": {
                "reference_equivalence_pass": True,
                "root": str(root),
            },
            "ulp_policy": {"selected_factor": 4.0},
        },
    )
    stager = SimpleNamespace(probe_environment=lambda root: env_payload()["environment"])
    monkeypatch.setitem(
        sys.modules,
        "ccda_phase3.phase314b_r258_stageo_dual_oracle_lower_multiplier_admission",
        stageo,
    )
    monkeypatch.setitem(
        sys.modules,
        "ccda_phase3.phase314b_r258_stager_tolerance_aligned_oof_post_upper_audit",
        stager,
    )
    return cold_calls, validate_calls


def test_phase_and_schema_are_resume2a():
    assert resume2a.PHASE.endswith("Stage S Resume2A")
    assert resume2a.SCHEMA.endswith("_v1")


def test_stage_is_smoke_only():
    assert "smoke" in resume2a.SUCCESS_REPORT
    assert "smoke" in resume2a.BLOCKED_REPORT


def test_parent_is_resume1_blocked_evidence():
    assert resume2a.BASE_RESUME1_BLOCKED_EVIDENCE_COMMIT == (
        "28cfba64980ed311810ca314dcc249a6a5681fb3"
    )


def test_resume1_blocked_report_sha_is_frozen():
    assert resume2a.EXPECTED_RESUME1_BLOCKED_REPORT_SHA256 == (
        "55f509a6e5320e42b2d0e4b5004c5cdc402a06174e073dc3c812132e278ad9b3"
    )


def test_implementation_is_add_only_four_files():
    assert len(resume2a.IMPLEMENTATION_PATHS) == 4
    assert {status for status, _ in resume2a.IMPLEMENTATION_PATHS} == {"A"}


def test_environment_contract_has_no_cuda_visible_device_gate():
    assert "CUDA_VISIBLE_DEVICES" not in resume2a.EXPECTED_ENV


def test_stable_json_is_order_independent():
    assert resume2a.stable_json_bytes({"b": 1, "a": 2}) == resume2a.stable_json_bytes(
        {"a": 2, "b": 1}
    )


def test_validate_environment_accepts_exact(monkeypatch):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    assert resume2a.validate_environment_variables() == dict(resume2a.EXPECTED_ENV)


def test_validate_environment_rejects_mismatch(monkeypatch):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(resume2a.StageSResume2AError, match="deterministic"):
        resume2a.validate_environment_variables()


def test_validate_probe_payload_accepts_real_shape():
    assert resume2a.validate_probe_payload(env_payload())["compatibility_pass"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "wrong"),
        ("execution_verdict", "BLOCKED"),
        ("environment_sha256", "bad"),
    ],
)
def test_validate_probe_payload_rejects_wrapper_change(field, value):
    payload = env_payload()
    payload[field] = value
    with pytest.raises(resume2a.StageSResume2AError):
        resume2a.validate_probe_payload(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("compatibility_pass",), False),
        (("compatibility", "required_operation_pass"), False),
        (("required_operation_dry_run", "pass"), False),
    ],
)
def test_validate_probe_payload_rejects_failed_contract(path, value):
    payload = env_payload()
    target = payload["environment"]
    if len(path) == 1:
        target[path[0]] = value
    else:
        target[path[0]][path[1]] = value
    payload["environment_sha256"] = resume2a.sha256_bytes(
        resume2a.stable_json_bytes(target)
    )
    with pytest.raises(resume2a.StageSResume2AError):
        resume2a.validate_probe_payload(payload)


def test_environment_probe_payload_uses_disposable_process(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    install_fake_stage_modules(monkeypatch)
    result = resume2a.environment_probe_payload(tmp_path)
    assert result["process_disposable"] is True
    assert result["cuda_initialization_allowed_in_this_process"] is True
    assert resume2a.validate_probe_payload(result)["compatibility_pass"] is True


def test_cold_smoke_calls_cold_gate_before_runtime(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    cold_calls, validate_calls = install_fake_stage_modules(monkeypatch)
    result = resume2a.cold_control_smoke_payload(
        root=tmp_path,
        probe_payload=env_payload(),
    )
    assert len(cold_calls) == 1
    assert len(validate_calls) == 1
    assert result["cold_cuda_precheck"]["torch_cuda_is_initialized"] is False
    assert result["frozen_control_replay_completed"] is True


def test_cold_smoke_runs_no_science(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    install_fake_stage_modules(monkeypatch)
    result = resume2a.cold_control_smoke_payload(
        root=tmp_path,
        probe_payload=env_payload(),
    )
    assert result["oof_fit_count"] == 0
    assert result["callback_pair_count"] == 0
    assert result["internal_scale_attempt_count"] == 0
    assert result["oracle_callback_rerun_count"] == 0


def test_validate_smoke_payload_accepts_valid():
    probe = env_payload()
    result = smoke_payload(probe)
    assert resume2a.validate_smoke_payload(
        result,
        expected_environment_sha256=probe["environment_sha256"],
    )["frozen_control_replay_completed"] is True


@pytest.mark.parametrize(
    "field,bad",
    [
        ("oof_fit_count", 1),
        ("callback_pair_count", 1),
        ("internal_scale_attempt_count", 1),
        ("oracle_callback_rerun_count", 1),
    ],
)
def test_validate_smoke_rejects_science_execution(field, bad):
    probe = env_payload()
    result = smoke_payload(probe)
    result[field] = bad
    with pytest.raises(resume2a.StageSResume2AError, match="unexpectedly ran science"):
        resume2a.validate_smoke_payload(
            result,
            expected_environment_sha256=probe["environment_sha256"],
        )


def test_validate_smoke_rejects_warm_process():
    probe = env_payload()
    result = smoke_payload(probe)
    result["cold_cuda_precheck"]["torch_cuda_is_initialized"] = True
    with pytest.raises(resume2a.StageSResume2AError, match="not cold"):
        resume2a.validate_smoke_payload(
            result,
            expected_environment_sha256=probe["environment_sha256"],
        )


def test_run_smoke_orders_probe_then_cold(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    calls = []

    def fake_run(command, *, root, label):
        calls.append((label, list(command)))
        output = Path(command[command.index("--output") + 1])
        if label == "environment probe":
            output.write_bytes(resume2a.stable_json_bytes(env_payload()))
        else:
            probe = resume2a.load_json(Path(command[command.index("--probe") + 1]))
            output.write_bytes(resume2a.stable_json_bytes(smoke_payload(probe)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(resume2a, "_run_child", fake_run)
    result = resume2a.run_process_boundary_smoke(
        root=tmp_path,
        repository={"head": "implementation"},
        python_bin=sys.executable,
    )
    assert [label for label, _ in calls] == [
        "environment probe",
        "cold-control smoke",
    ]
    assert result["process_boundary_smoke"]["oof_fit_count"] == 0


def test_run_smoke_does_not_start_cold_after_probe_failure(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)
    calls = []

    def fail_probe(command, *, root, label):
        calls.append(label)
        raise resume2a.StageSResume2AError("probe failed")

    monkeypatch.setattr(resume2a, "_run_child", fail_probe)
    with pytest.raises(resume2a.StageSResume2AError, match="probe failed"):
        resume2a.run_process_boundary_smoke(
            root=tmp_path,
            repository={},
            python_bin=sys.executable,
        )
    assert calls == ["environment probe"]


def test_run_smoke_rejects_same_pid(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)

    def fake_run(command, *, root, label):
        output = Path(command[command.index("--output") + 1])
        if label == "environment probe":
            output.write_bytes(resume2a.stable_json_bytes(env_payload()))
        else:
            probe = resume2a.load_json(Path(command[command.index("--probe") + 1]))
            output.write_bytes(
                resume2a.stable_json_bytes(smoke_payload(probe, pid=101))
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(resume2a, "_run_child", fake_run)
    with pytest.raises(resume2a.StageSResume2AError, match="same process"):
        resume2a.run_process_boundary_smoke(
            root=tmp_path,
            repository={},
            python_bin=sys.executable,
        )


def test_success_result_keeps_scientific_status_blocked(monkeypatch, tmp_path):
    for key, value in resume2a.EXPECTED_ENV.items():
        monkeypatch.setenv(key, value)

    def fake_run(command, *, root, label):
        output = Path(command[command.index("--output") + 1])
        if label == "environment probe":
            output.write_bytes(resume2a.stable_json_bytes(env_payload()))
        else:
            probe = resume2a.load_json(Path(command[command.index("--probe") + 1]))
            output.write_bytes(resume2a.stable_json_bytes(smoke_payload(probe)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(resume2a, "_run_child", fake_run)
    result = resume2a.run_process_boundary_smoke(
        root=tmp_path,
        repository={},
        python_bin=sys.executable,
    )
    assert result["execution_verdict"] == "PASS"
    assert result["scientific_status"] == "BLOCKED"
    assert result["selected_configuration"] is None
    assert result["train_only_recommendation"] is None
    assert result["mechanism_boundary"]["science_worker_started"] is False


def test_blocked_report_claims_no_counts():
    result = resume2a.blocked_report(repository={"head": "x"}, error=RuntimeError("x"))
    assert result["execution_verdict"] == "BLOCKED"
    assert result["oof_fit_count_claimed"] is False
    assert result["callback_pair_count_claimed"] is False
    assert result["internal_scale_attempt_count_claimed"] is False


def test_blocked_report_preserves_resume1_sha():
    result = resume2a.blocked_report(repository=None, error=RuntimeError("x"))
    assert result["resume1_blocked_report_sha256"] == (
        resume2a.EXPECTED_RESUME1_BLOCKED_REPORT_SHA256
    )


def test_resume1_source_documents_fault_order():
    root = Path(__file__).resolve().parents[1]
    source = root / (
        "ccda_phase3/phase314b_r258_stages_resume1_"
        "oracle_rerun_schema_recovery.py"
    )
    text = source.read_text(encoding="utf-8")
    probe = text.index("current_environment = stager.probe_environment")
    replay = text.index("replay = resume1.execute_recovery")
    assert probe < replay


def test_worker_has_only_probe_and_smoke_modes():
    root = Path(__file__).resolve().parents[1]
    source = root / "scripts/phase3_14b_r258_stages_resume2a_worker.py"
    text = source.read_text(encoding="utf-8")
    assert 'choices=("environment-probe", "cold-control-smoke")' in text
    assert "science-worker" not in text


def test_no_formal_science_import_at_module_top_level():
    root = Path(__file__).resolve().parents[1]
    source = root / (
        "ccda_phase3/phase314b_r258_stages_resume2a_"
        "cold_cuda_process_boundary_smoke.py"
    )
    first_80 = "\n".join(source.read_text(encoding="utf-8").splitlines()[:80])
    assert "import torch" not in first_80
    assert "import numpy" not in first_80
