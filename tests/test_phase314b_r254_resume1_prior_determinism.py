from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r254_resume1_prior_determinism import (
    REPEAT_RUN_IDS,
    SOURCE_PATHS,
    PriorEquivalenceSpec,
    classify_prior_audit,
    compare_histories,
    compare_predictions,
    compare_prior_z_mse,
    compare_tensor_states,
    decode_prediction,
    decode_tensor_state,
    encode_prediction,
    encode_tensor_state,
    historical_functional_fingerprint,
    pairwise_repeat_audit,
    strip_worker_payload,
    tensor_state_sha256_numpy,
)


def history(scale: float = 1.0):
    return [
        {"step": 1, "loss": 1.0 * scale, "gradient_norm": 2.0 * scale},
        {"step": 250, "loss": 0.2 * scale, "gradient_norm": 0.4 * scale},
        {"step": 5000, "loss": 0.01 * scale, "gradient_norm": 0.02 * scale},
    ]


def state(offset: float = 0.0):
    return {
        "layer.bias": torch.tensor([0.1 + offset, -0.2], dtype=torch.float32),
        "layer.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0 + offset]], dtype=torch.float32
        ),
    }


def record(run_id: str, *, offset: float = 0.0, history_scale: float = 1.0):
    tensor_state = state(offset)
    arrays = {key: value.numpy() for key, value in tensor_state.items()}
    prediction = np.asarray([[[0.1 + offset, 0.2]]], dtype=np.float32)
    state_hash = tensor_state_sha256_numpy(arrays)
    prediction_hash = hashlib.sha256(prediction.tobytes()).hexdigest()
    return {
        "run_id": run_id,
        "prior_state_sha256": state_hash,
        "prior_prediction_sha256": prediction_hash,
        "prior_z_mse": 0.01 * history_scale,
        "prior_history": history(history_scale),
        "_state_payload": encode_tensor_state(tensor_state),
        "_prediction_payload": encode_prediction(prediction),
        "environment_after_fit": {"gpu_name": "GPU-A"},
    }


def base_report():
    return {
        "runs": [record(name) for name in REPEAT_RUN_IDS],
        "same_device_repeats": {
            "functional_equivalence_pass": True,
            "all_state_sha_exact": True,
            "observed_state_sha256": ["new"] * 3,
        },
        "historical_functional_fingerprint": {
            "functional_fingerprint_pass": True,
            "expected_prior_state_sha256": "old",
        },
        "environment": {"gpu_name": "GPU-B"},
        "historical_prior": {"gpu_name": "GPU-A"},
    }


def test_spec_validates():
    PriorEquivalenceSpec().validate()


def test_spec_rejects_single_repeat():
    with pytest.raises(ValueError):
        PriorEquivalenceSpec(repeat_count=1).validate()


def test_prediction_payload_roundtrip():
    value = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    restored = decode_prediction(encode_prediction(value))
    assert restored.dtype == np.float32
    assert np.array_equal(restored, value)


def test_prediction_payload_detects_tamper():
    payload = encode_prediction(np.ones((2, 2), dtype=np.float32))
    payload["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        decode_prediction(payload)


def test_tensor_state_roundtrip():
    payload = encode_tensor_state(state())
    restored = decode_tensor_state(payload)
    assert set(restored) == set(state())
    for key, value in state().items():
        assert np.array_equal(restored[key], value.numpy())


def test_tensor_state_hash_is_stable():
    arrays = {key: value.numpy() for key, value in state().items()}
    first = tensor_state_sha256_numpy(arrays)
    second = tensor_state_sha256_numpy(dict(reversed(list(arrays.items()))))
    assert first == second


def test_compare_tensor_states_exact():
    arrays = {key: value.numpy() for key, value in state().items()}
    result = compare_tensor_states(arrays, arrays, spec=PriorEquivalenceSpec())
    assert result["exact_equal"]
    assert result["functional_parameter_bound_pass"]


def test_compare_tensor_states_small_perturbation_passes_bound():
    first = {key: value.numpy() for key, value in state().items()}
    second = {key: value.numpy() for key, value in state(1e-8).items()}
    result = compare_tensor_states(first, second, spec=PriorEquivalenceSpec())
    assert result["functional_parameter_bound_pass"]


def test_compare_tensor_states_large_perturbation_fails_bound():
    first = {key: value.numpy() for key, value in state().items()}
    second = {key: value.numpy() for key, value in state(1e-3).items()}
    result = compare_tensor_states(first, second, spec=PriorEquivalenceSpec())
    assert not result["functional_parameter_bound_pass"]


def test_compare_tensor_states_key_mismatch():
    first = {"a": np.zeros(1, dtype=np.float32)}
    second = {"b": np.zeros(1, dtype=np.float32)}
    result = compare_tensor_states(first, second, spec=PriorEquivalenceSpec())
    assert not result["key_match"]
    assert not result["functional_parameter_bound_pass"]


def test_compare_predictions_exact():
    value = np.ones((2, 4, 87), dtype=np.float32)
    result = compare_predictions(value, value, spec=PriorEquivalenceSpec())
    assert result["pass"]
    assert result["exact_equal"]


def test_compare_predictions_large_difference_fails():
    first = np.zeros((2, 4, 87), dtype=np.float32)
    second = np.ones((2, 4, 87), dtype=np.float32) * 1e-3
    result = compare_predictions(first, second, spec=PriorEquivalenceSpec())
    assert not result["pass"]


def test_compare_prior_z_mse_exact():
    assert compare_prior_z_mse(0.1, 0.1, spec=PriorEquivalenceSpec())["pass"]


def test_compare_prior_z_mse_large_difference_fails():
    assert not compare_prior_z_mse(0.1, 0.2, spec=PriorEquivalenceSpec())["pass"]


def test_compare_histories_exact():
    result = compare_histories(history(), history())
    assert result["step_match"]
    assert result["loss_relative_error_max"] == 0


def test_compare_histories_step_mismatch():
    result = compare_histories(history(), history()[:-1])
    assert not result["step_match"]
    assert not result["pass"]


def test_pairwise_repeat_audit_exact():
    records = [record(name) for name in REPEAT_RUN_IDS]
    result = pairwise_repeat_audit(records, spec=PriorEquivalenceSpec())
    assert result["all_state_sha_exact"]
    assert result["all_prediction_sha_exact"]
    assert result["functional_equivalence_pass"]


def test_pairwise_repeat_audit_rejects_duplicate_ids():
    records = [record("same"), record("same"), record("third")]
    with pytest.raises(ValueError):
        pairwise_repeat_audit(records, spec=PriorEquivalenceSpec())


def test_pairwise_repeat_audit_detects_functional_difference():
    records = [record("a"), record("b"), record("c", offset=1e-3)]
    result = pairwise_repeat_audit(records, spec=PriorEquivalenceSpec())
    assert not result["functional_equivalence_pass"]


def test_historical_fingerprint_exact():
    records = [record(name) for name in REPEAT_RUN_IDS]
    expected = {
        "prior_state_sha256": records[0]["prior_state_sha256"],
        "prior_z_mse": 0.01,
        "prior_history": history(),
    }
    result = historical_functional_fingerprint(
        records, expected, spec=PriorEquivalenceSpec()
    )
    assert result["functional_fingerprint_pass"]
    assert result["exact_state_sha_match_count"] == 3


def test_historical_fingerprint_rejects_large_loss_difference():
    records = [record(name, history_scale=2.0) for name in REPEAT_RUN_IDS]
    expected = {
        "prior_state_sha256": "old",
        "prior_z_mse": 0.01,
        "prior_history": history(),
    }
    result = historical_functional_fingerprint(
        records, expected, spec=PriorEquivalenceSpec()
    )
    assert not result["functional_fingerprint_pass"]


def test_strip_worker_payload():
    value = record("a")
    stripped = strip_worker_payload(value)
    assert "_state_payload" not in stripped
    assert "_prediction_payload" not in stripped
    assert stripped["prior_state_sha256"] == value["prior_state_sha256"]


def test_classifier_matrix_incomplete():
    report = base_report()
    report["runs"] = report["runs"][:1]
    result = classify_prior_audit(report)
    assert result["root_cause"] == "phase314b_r254_resume1_prior_repeat_matrix_incomplete"
    assert not result["functional_prior_contract_supported"]


def test_classifier_same_device_functional_failure():
    report = base_report()
    report["same_device_repeats"]["functional_equivalence_pass"] = False
    result = classify_prior_audit(report)
    assert result["root_cause"] == (
        "phase314b_r254_resume1_same_device_prior_functional_nondeterminism"
    )


def test_classifier_historical_functional_failure():
    report = base_report()
    report["historical_functional_fingerprint"]["functional_fingerprint_pass"] = False
    result = classify_prior_audit(report)
    assert result["root_cause"] == (
        "phase314b_r254_resume1_prior_functional_equivalence_not_supported"
    )


def test_classifier_exact_reproduction():
    report = base_report()
    report["historical_functional_fingerprint"]["expected_prior_state_sha256"] = "new"
    result = classify_prior_audit(report)
    assert result["root_cause"] == (
        "phase314b_r254_resume1_shared_prior_exact_reproduction_supported"
    )
    assert result["functional_prior_contract_supported"]


def test_classifier_cross_device_overstrict():
    result = classify_prior_audit(base_report())
    assert result["root_cause"] == (
        "phase314b_r254_resume1_cross_device_bitwise_sha_contract_overstrict"
    )
    assert result["functional_prior_contract_supported"]


def test_classifier_same_device_historical_overstrict():
    report = base_report()
    report["environment"]["gpu_name"] = "GPU-A"
    result = classify_prior_audit(report)
    assert result["root_cause"] == (
        "phase314b_r254_resume1_historical_bitwise_sha_contract_overstrict"
    )


def test_classifier_bounded_bitwise_nondeterminism():
    report = base_report()
    report["same_device_repeats"]["all_state_sha_exact"] = False
    report["same_device_repeats"]["observed_state_sha256"] = ["a", "b", "c"]
    result = classify_prior_audit(report)
    assert result["root_cause"] == (
        "phase314b_r254_resume1_bitwise_nondeterminism_functionally_bounded"
    )


def test_classifier_never_recommends_or_selects():
    result = classify_prior_audit(base_report())
    assert result["train_only_recommendation"] is None
    assert result["selected_configuration"] is None


def test_source_paths_are_unique():
    assert len(SOURCE_PATHS) == len(set(SOURCE_PATHS))


def test_resume_shell_uses_exact_interpreter_and_new_paths():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r254_resume1_run.sh").read_text()
    assert "/miniforge3/envs/coord_bimanual/bin/python" in text
    assert "phase3_14b_r254_resume1_preflight_summary.json" in text
    assert "phase3_14b_r254_resume1_prior_repeat_evidence.json" in text
    assert "phase3_14b_r254_resume1_prior_audit_summary.json" in text
    assert "phase3_14b_r254_resume1_blocked_summary.json" in text


def test_resume_shell_does_not_overwrite_original_r254_paths():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r254_resume1_run.sh").read_text()
    assert 'PILOT="reports/phase3_14b_r254_pilot_summary.json"' not in text
    assert 'BLOCKED_JSON="reports/phase3_14b_r254_blocked_summary.json"' not in text


def test_resume_sources_do_not_call_robot_training_or_reverse():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "ccda_phase3/phase314b_r254_resume1_prior_determinism.py",
        root / "scripts/phase3_14b_r254_resume1_preflight.py",
        root / "scripts/phase3_14b_r254_resume1_run_audit.py",
        root / "scripts/phase3_14b_r254_resume1_finalize.py",
    ]
    text = "\n".join(path.read_text() for path in paths)
    assert "train_calibrated_geometry_variant(" not in text
    assert "paired_reverse_pool_metrics(" not in text
    assert "reverse_sample_pool(" not in text
