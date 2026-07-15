from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r257_staged_resume1_control_trace as resume1


def test_failed_commit_is_frozen():
    assert resume1.FAILED_IMPLEMENTATION_COMMIT == (
        "ccff30947a85028a2c7974edb5b008f3b3de2757"
    )


def test_failed_blocked_sha_is_frozen():
    assert resume1.EXPECTED_FAILED_BLOCKED_SHA256 == (
        "bd9f3f63df92605cf7425ebecdcd80cb62c4cc33c9909ee3d51680909057da4f"
    )


def test_stable_json_is_deterministic():
    assert resume1.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == resume1.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_supports_numpy():
    payload = resume1.stable_json_bytes(
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


def test_load_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(resume1.Resume1ControlTraceError):
        resume1.load_json(path)


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    resume1.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        resume1.atomic_write_once(path, b"{}\n")


def test_source_trace_audit_accepts_failed_sources():
    repository_root = Path(__file__).resolve().parents[1]
    result = resume1.source_trace_audit(repository_root)
    assert result["all_confirmed"]
    assert result["checks"][
        "failed_stage_d_calibration_before_control"
    ]
    assert result["checks"][
        "stage_c_calibration_before_control"
    ]


def test_stagec_control_preamble_exact(monkeypatch):
    diagnostic = {
        "indices_sha256": "a" * 64,
    }
    calibration = {
        "calibration_sha256": "b" * 64,
        "candidate_order": ["x"],
    }
    candidates = [
        type("Candidate", (), {"candidate_id": "x"})()
    ]

    monkeypatch.setattr(
        resume1.stageb_mechanism,
        "fixed_diagnostic_batch",
        lambda **kwargs: diagnostic,
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "calibrate_candidates",
        lambda **kwargs: (candidates, calibration),
    )
    result = resume1.stagec_control_preamble(
        condition=np.zeros((2, 1), dtype=np.float32),
        target=np.zeros((2, 1), dtype=np.float32),
        stageb_spec=object(),
        condition_standardizer=object(),
        target_standardizer=object(),
        objective_contract=object(),
        stagec_contract={"calibration": calibration},
    )
    assert result["calibration_exact"]
    assert result["candidate_order"] == ["x"]


def test_stagec_control_preamble_rejects_mismatch(
    monkeypatch,
):
    monkeypatch.setattr(
        resume1.stageb_mechanism,
        "fixed_diagnostic_batch",
        lambda **kwargs: {},
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "calibrate_candidates",
        lambda **kwargs: (
            [],
            {"calibration_sha256": "a" * 64},
        ),
    )
    with pytest.raises(resume1.Resume1ControlTraceError):
        resume1.stagec_control_preamble(
            condition=np.zeros((2, 1), dtype=np.float32),
            target=np.zeros((2, 1), dtype=np.float32),
            stageb_spec=object(),
            condition_standardizer=object(),
            target_standardizer=object(),
            objective_contract=object(),
            stagec_contract={
                "calibration": {
                    "calibration_sha256": "b" * 64,
                }
            },
        )


def _patch_exact_control_dependencies(
    monkeypatch,
    *,
    training=None,
    train_control=None,
    one_step=None,
    profiles=None,
):
    expected_training = (
        {
            "lambda_upper": 0.0,
            "initial_model_sha256": "i" * 64,
            "final_model_sha256": "m" * 64,
            "initial_optimizer_sha256": "j" * 64,
            "final_optimizer_sha256": "o" * 64,
            "total_loss_history_sha256": "l" * 64,
            "diffusion_loss_history_sha256": "d" * 64,
            "geometry_loss_history_sha256": "g" * 64,
            "gradient_history_sha256": "r" * 64,
            "source_exposure_sha256": "e" * 64,
            "total_loss_first": 1.0,
            "total_loss_final": 0.1,
            "total_loss_tail_mean": 0.1,
            "diffusion_loss_tail_mean": 0.1,
            "geometry_loss_tail_mean": 0.0,
            "geometry_row_loss_tail_mean": 0.0,
            "geometry_element_loss_tail_mean": 0.0,
            "batch_row_violation_rate_tail_mean": 0.0,
            "maximum_log_excess_tail_max": 0.0,
            "gradient_tail_mean": 1.0,
            "loss_finite": True,
            "gradient_finite": True,
            "training_rows": 2,
            "training_steps": 8000,
        }
        if training is None
        else training
    )
    expected_train_control = (
        {"normalized_mse": 1.0}
        if train_control is None
        else train_control
    )
    expected_one_step = (
        {"prediction_sha256": "p" * 64}
        if one_step is None
        else one_step
    )
    expected_profiles = (
        {"10": {"row_any_rate": 0.0}}
        if profiles is None
        else profiles
    )
    monkeypatch.setattr(
        resume1,
        "stagec_control_preamble",
        lambda **kwargs: {
            "diagnostic_batch": {},
            "calibration_sha256": "c" * 64,
            "candidate_order": ["a", "b"],
        },
    )
    monkeypatch.setattr(
        resume1.stageb,
        "set_deterministic_runtime",
        lambda seed: {},
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "train_candidate",
        lambda **kwargs: (
            object(),
            expected_training,
            {"checkpoint_record_sha256": "d" * 64},
        ),
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "compatible_training_record",
        lambda value: value,
    )
    monkeypatch.setattr(
        resume1.stagea,
        "train_control_audit",
        lambda **kwargs: expected_train_control,
    )
    monkeypatch.setattr(
        resume1.stagea,
        "one_step_audit",
        lambda **kwargs: expected_one_step,
    )
    monkeypatch.setattr(
        resume1.stagec,
        "one_step_predictions",
        lambda **kwargs: (
            {10: np.zeros((1,), dtype=np.float32)},
            expected_one_step["prediction_sha256"],
        ),
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "holdout_profiles",
        lambda *args, **kwargs: expected_profiles,
    )
    monkeypatch.setattr(
        resume1.stageb,
        "fit_geometry_contract",
        lambda target: object(),
    )
    return {
        "training": expected_training,
        "train_control": expected_train_control,
        "one_step": expected_one_step,
        "holdout_profiles": expected_profiles,
    }


def _run_exact_control(expected):
    return resume1.exact_control_replay(
        condition=np.zeros((4, 1), dtype=np.float32),
        target=np.zeros((4, 1), dtype=np.float32),
        groups=np.asarray(["a", "b", "c", "d"]),
        condition_name=np.asarray(
            ["free", "free", "hidden", "hidden"]
        ),
        stageb_train=np.asarray(
            [True, True, True, False]
        ),
        objective_train=np.asarray(
            [True, True, False, False]
        ),
        selection_holdout=np.asarray(
            [False, False, True, False]
        ),
        stageb_spec=type("Spec", (), {"seed": 1})(),
        condition_standardizer=object(),
        target_standardizer=object(),
        objective_contract=object(),
        upper_gate=object(),
        stage_d_contract=object(),
        control_expected=expected,
        stagec_contract={"calibration": {}},
        holdout_noise_seed_offset=4101,
    )


def test_exact_control_success(monkeypatch):
    expected = _patch_exact_control_dependencies(
        monkeypatch
    )
    bundle, identity = _run_exact_control(expected)
    assert bundle["candidate_id"] == (
        resume1.CONTROL_CANDIDATE_ID
    )
    assert identity["training_exact"]
    assert identity["profiles_exact"]


def test_exact_control_reports_training_keys(monkeypatch):
    expected = _patch_exact_control_dependencies(
        monkeypatch,
        training={"final_model_sha256": "m" * 64},
    )
    monkeypatch.setattr(
        resume1.stagec_topk,
        "compatible_training_record",
        lambda value: {
            "final_model_sha256": "x" * 64,
        },
    )
    with pytest.raises(
        resume1.Resume1ControlTraceError,
        match="final_model_sha256",
    ):
        _run_exact_control(expected)


def test_exact_control_rejects_train_control(monkeypatch):
    expected = _patch_exact_control_dependencies(
        monkeypatch
    )
    monkeypatch.setattr(
        resume1.stagea,
        "train_control_audit",
        lambda **kwargs: {"normalized_mse": 2.0},
    )
    with pytest.raises(
        resume1.Resume1ControlTraceError,
        match="train-control",
    ):
        _run_exact_control(expected)


def test_compare_worker_results_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "failed_attempt": {
            "failed_implementation_commit": "a",
            "failed_blocked_sha256": "b",
            "failed_test_gate_sha256": "c",
        },
        "source_trace_audit": {"x": 1},
        "split": {"x": 1},
        "control_trace_correction": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    result = resume1.compare_worker_results(
        base,
        dict(base),
    )
    assert result["exact"]
    assert result["control_trace_exact"]


def test_compare_worker_results_detects_difference():
    left = {
        "root_cause": "r",
        "required_next_path": "n",
        "failed_attempt": {
            "failed_implementation_commit": "a",
            "failed_blocked_sha256": "b",
            "failed_test_gate_sha256": "c",
        },
        "source_trace_audit": {"x": 1},
        "split": {"x": 1},
        "control_trace_correction": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    right = dict(left)
    right["control_trace_correction"] = {"x": 2}
    assert not resume1.compare_worker_results(
        left,
        right,
    )["exact"]
