from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "ccda_phase3"
    / "phase314b_r254_resume2_functional_prior.py"
)
spec = importlib.util.spec_from_file_location("r254_resume2", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def resume1_documents():
    runs = [
        {
            "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
            "prior_prediction_sha256": module.CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
            "prior_z_mse": module.CURRENT_DEVICE_PRIOR_Z_MSE,
        }
        for _ in range(3)
    ]
    summary = {
        "verdict": "PASS",
        "root_cause": module.EXPECTED_RESUME1_ROOT_CAUSE,
        "next_stage": module.EXPECTED_NEXT_STAGE,
        "functional_prior_contract_supported": True,
        "same_device_exact_sha": True,
        "same_device_functional_equivalence": True,
        "historical_functional_fingerprint": True,
        "expected_prior_state_sha256": module.HISTORICAL_PRIOR_STATE_SHA256,
        "runs": runs,
    }
    audit = {
        "same_device_repeats": {
            "all_state_sha_exact": True,
            "all_prediction_sha_exact": True,
            "observed_state_sha256": [
                module.CURRENT_DEVICE_PRIOR_STATE_SHA256
            ] * 3,
            "observed_prediction_sha256": [
                module.CURRENT_DEVICE_PRIOR_PREDICTION_SHA256
            ] * 3,
            "functional_equivalence_pass": True,
        },
        "historical_functional_fingerprint": {
            "functional_fingerprint_pass": True,
            "expected_prior_state_sha256": module.HISTORICAL_PRIOR_STATE_SHA256,
            "prior_z_mse_observed_values": [module.CURRENT_DEVICE_PRIOR_Z_MSE] * 3,
        },
    }
    evidence = {"runs": runs}
    return summary, audit, evidence


def snapshot():
    prediction = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = module.prediction_sha256(prediction)
    old = module.CURRENT_DEVICE_PRIOR_PREDICTION_SHA256
    return prediction, expected, old


def test_prediction_sha_is_float32_raw_bytes():
    value = np.arange(8, dtype=np.float64)
    expected = module.sha256_bytes(np.asarray(value, dtype=np.float32).tobytes())
    assert module.prediction_sha256(value) == expected


def test_prediction_sha_tensor_and_numpy_match():
    value = np.arange(8, dtype=np.float32)
    assert module.prediction_sha256(value) == module.prediction_sha256(torch.from_numpy(value))


def test_resume1_contract_passes():
    result = module.verify_resume1_functional_contract(*resume1_documents())
    assert result["functional_prior_contract_supported"] is True
    assert result["state_sha_evidence_count"] >= 3


def test_resume1_wrong_verdict_fails():
    summary, audit, evidence = resume1_documents()
    summary["verdict"] = "BLOCKED"
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_wrong_root_fails():
    summary, audit, evidence = resume1_documents()
    summary["root_cause"] = "wrong"
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_wrong_next_fails():
    summary, audit, evidence = resume1_documents()
    summary["next_stage"] = "wrong"
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_false_functional_contract_fails():
    summary, audit, evidence = resume1_documents()
    summary["functional_prior_contract_supported"] = False
    audit.pop("historical_functional_fingerprint")
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_incomplete_state_hashes_fail():
    summary, audit, evidence = resume1_documents()
    summary["runs"] = summary["runs"][:1]
    audit["same_device_repeats"]["observed_state_sha256"] = []
    evidence["runs"] = []
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_incomplete_prediction_hashes_fail():
    summary, audit, evidence = resume1_documents()
    for document in (summary, evidence):
        for row in document["runs"]:
            row["prior_prediction_sha256"] = "wrong"
    audit["same_device_repeats"]["observed_prediction_sha256"] = []
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_incomplete_mse_fails():
    summary, audit, evidence = resume1_documents()
    for document in (summary, evidence):
        for row in document["runs"]:
            row["prior_z_mse"] = 999.0
    audit["historical_functional_fingerprint"]["prior_z_mse_observed_values"] = []
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_resume1_historical_hash_required():
    summary, audit, evidence = resume1_documents()
    summary.pop("expected_prior_state_sha256")
    audit["historical_functional_fingerprint"].pop("expected_prior_state_sha256")
    with pytest.raises(RuntimeError):
        module.verify_resume1_functional_contract(summary, audit, evidence)


def test_symmetric_relative_error_zero():
    assert module.symmetric_relative_error(1.0, 1.0) == 0.0


def test_spec_rejects_wrong_repeat_count():
    with pytest.raises(ValueError):
        module.FunctionalPriorSpec(repeat_count=2).validate()


def test_spec_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        module.FunctionalPriorSpec(prior_z_mse_absolute_tolerance=-1.0).validate()


def test_validate_snapshot_passes_with_monkeypatched_prediction_sha(monkeypatch):
    prediction = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", module.prediction_sha256(prediction))
    value = {
        "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "prior_z_mse": module.CURRENT_DEVICE_PRIOR_Z_MSE,
        "_prior_prediction_z": prediction,
    }
    result = module.validate_fresh_prior_snapshot(value)
    assert result["functional_prior_contract_pass"] is True
    assert value["prior_prediction_sha256"] == module.prediction_sha256(prediction)


def test_validate_snapshot_wrong_state_fails(monkeypatch):
    prediction = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", module.prediction_sha256(prediction))
    value = {
        "prior_state_sha256": "wrong",
        "prior_z_mse": module.CURRENT_DEVICE_PRIOR_Z_MSE,
        "_prior_prediction_z": prediction,
    }
    with pytest.raises(RuntimeError):
        module.validate_fresh_prior_snapshot(value)


def test_validate_snapshot_missing_prediction_fails():
    value = {
        "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "prior_z_mse": module.CURRENT_DEVICE_PRIOR_Z_MSE,
    }
    with pytest.raises(RuntimeError):
        module.validate_fresh_prior_snapshot(value)


def test_validate_snapshot_wrong_prediction_fails():
    value = {
        "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "prior_z_mse": module.CURRENT_DEVICE_PRIOR_Z_MSE,
        "_prior_prediction_z": np.arange(8, dtype=np.float32),
    }
    with pytest.raises(RuntimeError):
        module.validate_fresh_prior_snapshot(value)


def test_validate_snapshot_wrong_mse_fails(monkeypatch):
    prediction = np.arange(8, dtype=np.float32)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", module.prediction_sha256(prediction))
    value = {
        "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
        "prior_z_mse": 9.0,
        "_prior_prediction_z": prediction,
    }
    with pytest.raises(RuntimeError):
        module.validate_fresh_prior_snapshot(value)


def good_contract():
    return module.verify_resume1_functional_contract(*resume1_documents())


def good_fresh():
    return {
        "functional_prior_contract_pass": True,
        "state_sha_exact": True,
        "prediction_sha_exact": True,
    }


def good_pilot():
    return {
        "resume_generation": 2,
        "resume2_schema": module.RESUME2_SCHEMA,
        "functional_prior_contract": good_contract(),
        "fresh_prior_validation": good_fresh(),
        "shared_prior": {
            "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
            "prior_prediction_sha256": module.CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
        },
        "train_only_recommendation": None,
        "selected_configuration": None,
        "reverse_sampling_rerun": False,
    }


def test_validate_pilot_passes():
    module.validate_resume2_pilot_contract(good_pilot())


@pytest.mark.parametrize(
    "path,value",
    [
        (("resume_generation",), 1),
        (("resume2_schema",), "wrong"),
        (("functional_prior_contract", "functional_prior_contract_supported"), False),
        (("fresh_prior_validation", "functional_prior_contract_pass"), False),
        (("shared_prior", "prior_state_sha256"), "wrong"),
        (("shared_prior", "prior_prediction_sha256"), "wrong"),
        (("train_only_recommendation",), "bad"),
        (("selected_configuration",), "bad"),
        (("reverse_sampling_rerun",), True),
    ],
)
def test_validate_pilot_rejects_contract_changes(path, value):
    payload = good_pilot()
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(RuntimeError):
        module.validate_resume2_pilot_contract(payload)


def test_augment_pilot_preserves_original_fields():
    original = {"phase": "x", "verdict": "PASS"}
    result = module.augment_pilot_payload(
        original,
        functional_contract=good_contract(),
        fresh_prior_validation=good_fresh(),
        resume2_source_hashes={"a": "b"},
    )
    assert result["phase"] == "x"
    assert result["resume_generation"] == 2
    assert result["robot_proxy_attribution_run"] is True


def test_augment_does_not_mutate_original():
    original = {"phase": "x"}
    before = copy.deepcopy(original)
    module.augment_pilot_payload(
        original,
        functional_contract=good_contract(),
        fresh_prior_validation=good_fresh(),
        resume2_source_hashes={},
    )
    assert original == before


def test_compact_contract_has_no_evidence_payloads():
    compact = module.compact_functional_prior_contract(good_contract())
    assert compact["functional_prior_contract_supported"] is True
    assert "source_documents" not in compact


def test_source_paths_are_additive_resume2_only():
    assert module.SOURCE_PATHS
    assert all("resume2" in value for value in module.SOURCE_PATHS)


def test_final_paths_are_standard_and_resume2_pilot_is_distinct():
    assert module.FINAL_SUMMARY_PATH == "reports/phase3_14b_r254_summary.json"
    assert module.RESUME2_PILOT_PATH != "reports/phase3_14b_r254_pilot_summary.json"


def test_cross_device_historical_sha_differs_from_current():
    assert module.HISTORICAL_PRIOR_STATE_SHA256 != module.CURRENT_DEVICE_PRIOR_STATE_SHA256


def test_no_selection_constants():
    contract = good_contract()
    assert contract.get("train_only_recommendation") is None
    assert contract.get("selected_configuration") is None
