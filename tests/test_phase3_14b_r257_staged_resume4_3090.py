from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r257_staged_resume4_3090 as resume4


def environment_payload():
    payload = {
        "schema":
            "phase314b_r257_staged_resume4_3090_environment_v1",
        "python_version": [3, 9, 15],
        "python_implementation": "CPython",
        "numpy_version": "1.23.3",
        "torch_version": "1.12.1.post200",
        "torch_cuda_version": "11.2",
        "cudnn_version": 8302,
        "cuda_device_count": 1,
        "cuda_device_index": 0,
        "torch_device_name":
            "NVIDIA GeForce RTX 3090",
        "compute_capability": [8, 6],
        "total_memory_bytes": 25447170048,
        "multi_processor_count": 82,
        "nvidia_smi": {
            "name": "NVIDIA GeForce RTX 3090",
            "uuid": "GPU-test",
            "pci_bus_id": "00000000:01:00.0",
            "driver_version": "535.0",
            "memory_total_mib": "24576",
        },
        "cuda_visible_devices": "0",
        "expected_gpu_name_token": "RTX 3090",
        "expected_compute_capability": [8, 6],
        "contract_pass": True,
    }
    payload["environment_sha256"] = (
        resume4.sha256_bytes(
            resume4.stable_json_bytes(payload)
        )
    )
    return payload


def minimal_resume3_result():
    return {
        "phase": "Resume3",
        "phase_id": "resume3",
        "schema": "resume3",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": "r",
        "required_next_path": "n",
        "resume3_calibration_attribution": {
            "manual_calibration_diff": {
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
            "stagec_entrypoint_capture": {
                "calibration_exact": True,
                "control_record_exact": True,
            },
        },
        "control_trace_correction": {
            "training_exact": True,
        },
        "split": {"frozen_probe_accessed": False},
        "calibration_contract": {
            "contract_sha256": "x" * 64,
            "value": 1,
        },
        "selection": {
            "selection_sha256": "s" * 64,
        },
        "classification": {
            "primary_failure_locus": "x",
        },
        "selected_configuration": None,
        "train_only_recommendation": None,
        "control_replay_exact": True,
        "new_hyperparameter_candidate_run": True,
        "new_objective_variant_run": True,
    }


def test_commit_constants_are_frozen():
    assert resume4.RESUME3_IMPLEMENTATION_COMMIT == (
        "ef5eebb964ac7a8198506962646e048ccf0f6605"
    )
    assert resume4.RESUME2_BLOCKED_EVIDENCE_COMMIT == (
        "18bad1f2d542875b788e1959e048717ec39b5a49"
    )


def test_resume3_report_shas_are_frozen():
    assert resume4.EXPECTED_RESUME3_BLOCKED_SHA256 == (
        "c769fa1f0af69dfe24612cf9bdb7f43c8ce03fb93f039c069b84b07d088a89cd"
    )
    assert resume4.EXPECTED_RESUME3_TEST_GATE_SHA256 == (
        "c1cf8558bc0d5a44018812b034e4ce5d3498c290c68c20d0ef8a249938b05adc"
    )


def test_environment_constants_are_frozen():
    assert resume4.EXPECTED_PYTHON == (3, 9, 15)
    assert resume4.EXPECTED_NUMPY == "1.23.3"
    assert resume4.EXPECTED_TORCH == "1.12.1.post200"
    assert resume4.EXPECTED_TORCH_CUDA == "11.2"
    assert resume4.EXPECTED_COMPUTE_CAPABILITY == (8, 6)


def test_stable_json_is_deterministic():
    assert resume4.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume4.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_supports_numpy():
    payload = resume4.stable_json_bytes(
        {
            "array": np.asarray(
                [1.0, 2.0],
                dtype=np.float32,
            ),
            "scalar": np.float64(3.0),
        }
    )
    loaded = json.loads(payload)
    assert loaded["array"] == [1.0, 2.0]
    assert loaded["scalar"] == 3.0


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume4.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume4.atomic_write_once(path, b"{}\n")


def test_load_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(resume4.Resume4EnvironmentError):
        resume4.load_json(path)


def test_normalize_gpu_name():
    assert resume4.normalize_gpu_name(
        " NVIDIA   GeForce RTX 3090 "
    ) == "NVIDIA GeForce RTX 3090"


def test_parse_nvidia_smi_row():
    result = resume4.parse_nvidia_smi_row(
        "NVIDIA GeForce RTX 3090, GPU-a, "
        "00000000:01:00.0, 535.1, 24576"
    )
    assert result["name"] == (
        "NVIDIA GeForce RTX 3090"
    )
    assert result["uuid"] == "GPU-a"
    assert result["memory_total_mib"] == "24576"


def test_parse_nvidia_smi_row_rejects_bad_shape():
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="nvidia-smi",
    ):
        resume4.parse_nvidia_smi_row(
            "a,b,c"
        )


def test_environment_payload_sha_is_valid():
    payload = environment_payload()
    expected = resume4.sha256_bytes(
        resume4.stable_json_bytes(
            {
                key: value
                for key, value in payload.items()
                if key != "environment_sha256"
            }
        )
    )
    assert payload["environment_sha256"] == expected


def test_run_resume4_rejects_failed_environment(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    payload = environment_payload()
    payload["contract_pass"] = False
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="did not pass",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=payload,
            manual_diff={
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_wrong_gpu_name(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    payload = environment_payload()
    payload["torch_device_name"] = (
        "NVIDIA GeForce RTX 4090"
    )
    payload["environment_sha256"] = (
        resume4.sha256_bytes(
            resume4.stable_json_bytes(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "environment_sha256"
                }
            )
        )
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="RTX 3090",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=payload,
            manual_diff={
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_wrong_capability(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    payload = environment_payload()
    payload["compute_capability"] = [8, 9]
    payload["environment_sha256"] = (
        resume4.sha256_bytes(
            resume4.stable_json_bytes(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "environment_sha256"
                }
            )
        )
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="capability",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=payload,
            manual_diff={
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_invalid_environment_sha(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    payload = environment_payload()
    payload["environment_sha256"] = "x" * 64
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="fingerprint",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=payload,
            manual_diff={
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_incomplete_manual_diff(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="did not complete",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=environment_payload(),
            manual_diff={
                "completed": False,
                "training_performed": False,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_manual_training(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="unexpectedly trained",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=environment_payload(),
            manual_diff={
                "completed": True,
                "training_performed": True,
                "frozen_probe_accessed": False,
            },
        )


def test_run_resume4_rejects_manual_probe_access(
    monkeypatch,
):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {},
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="frozen probe",
    ):
        resume4.run_resume4(
            root=Path("/tmp"),
            environment=environment_payload(),
            manual_diff={
                "completed": True,
                "training_performed": False,
                "frozen_probe_accessed": True,
            },
        )


def test_assert_cold_cuda_context_success(
    monkeypatch,
):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: False,
    )
    result = resume4.assert_cold_cuda_context()
    assert result["checked"]
    assert not result["torch_cuda_is_initialized"]


def test_assert_cold_cuda_context_rejects_warm(
    monkeypatch,
):
    import torch

    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: True,
    )
    with pytest.raises(
        resume4.Resume4EnvironmentError,
        match="initialized CUDA",
    ):
        resume4.assert_cold_cuda_context()


def test_run_resume4_success(monkeypatch):
    monkeypatch.setattr(
        resume4,
        "validate_resume3_failure",
        lambda root: {"blocked": True},
    )
    monkeypatch.setattr(
        resume4,
        "assert_cold_cuda_context",
        lambda: {
            "checked": True,
            "torch_cuda_is_initialized": False,
        },
    )
    base = minimal_resume3_result()
    monkeypatch.setattr(
        resume4.resume3,
        "run_resume3",
        lambda **kwargs: copy.deepcopy(base),
    )
    result = resume4.run_resume4(
        root=Path("/tmp"),
        environment=environment_payload(),
        manual_diff={
            "completed": True,
            "training_performed": False,
            "frozen_probe_accessed": False,
        },
    )
    assert result["phase"] == resume4.PHASE
    assert result[
        "resume4_3090_correction_applied"
    ]
    assert result["resume3_files_modified"] is False
    assert result[
        "resume4_3090_environment"
    ]["new_write_once_namespace"] == (
        "phase3_14b_r257_staged_resume4_3090"
    )
    assert result["calibration_contract"][
        "resume4_3090_environment"
    ]["cold_main_worker_context"][
        "torch_cuda_is_initialized"
    ] is False


def test_identity_projection_contains_environment():
    result = minimal_resume3_result()
    result["resume4_3090_environment"] = {
        "x": 1
    }
    projection = resume4.identity_projection(
        result
    )
    assert projection[
        "resume4_3090_environment"
    ] == {"x": 1}


def test_compare_worker_results_exact():
    base = minimal_resume3_result()
    base["resume4_3090_environment"] = {
        "environment_probe": {"x": 1},
    }
    result = resume4.compare_worker_results(
        base,
        copy.deepcopy(base),
    )
    assert result["exact"]
    assert result["environment_exact"]


def test_compare_worker_results_detects_environment():
    left = minimal_resume3_result()
    left["resume4_3090_environment"] = {
        "environment_probe": {"x": 1},
    }
    right = copy.deepcopy(left)
    right["resume4_3090_environment"] = {
        "environment_probe": {"x": 2},
    }
    result = resume4.compare_worker_results(
        left,
        right,
    )
    assert not result["exact"]
    assert not result["environment_exact"]


def test_compare_worker_results_detects_capture():
    left = minimal_resume3_result()
    left["resume4_3090_environment"] = {
        "environment_probe": {"x": 1},
    }
    right = copy.deepcopy(left)
    right["resume3_calibration_attribution"][
        "stagec_entrypoint_capture"
    ] = {"calibration_exact": False}
    result = resume4.compare_worker_results(
        left,
        right,
    )
    assert not result["exact"]
    assert not result["stagec_capture_exact"]
