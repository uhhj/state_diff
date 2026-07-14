from __future__ import annotations

import ast
import json
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import pytest
import torch

import ccda_phase3.phase314b_r254_resume3_prediction_adapter as module


class FakeScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(1.0, 0.1, 100)


class FakeModel:
    def __init__(self, prediction: torch.Tensor, *, disturb_rng: bool = False) -> None:
        self.prediction = prediction
        self.disturb_rng = disturb_rng
        self.eval_called = False
        self.predict_calls = 0

    def eval(self) -> "FakeModel":
        self.eval_called = True
        return self

    def predict_base_x0(self, condition: torch.Tensor) -> torch.Tensor:
        self.predict_calls += 1
        if self.disturb_rng:
            random.random()
            np.random.rand()
            torch.rand(3)
        assert condition.shape[0] == self.prediction.shape[0]
        return self.prediction.clone()


def snapshot_for(state: Dict[str, torch.Tensor], *, z_mse: float = 0.25) -> Dict[str, Any]:
    return {
        "prior_state_sha256": module.tensor_state_sha256(state),
        "prior_z_mse": z_mse,
        "_prior_state": state,
    }


def fake_instantiate_factory(model: FakeModel, calls: Dict[str, Any]):
    def fake_instantiate(**kwargs: Any) -> FakeModel:
        calls.update(kwargs)
        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        return model

    return fake_instantiate


def test_spec_defaults_validate() -> None:
    module.Resume3PredictionSpec().validate()


@pytest.mark.parametrize("field", ["prior_z_mse_absolute_tolerance", "prior_z_mse_relative_tolerance"])
def test_spec_rejects_negative(field: str) -> None:
    kwargs = {field: -1.0}
    with pytest.raises(ValueError):
        module.Resume3PredictionSpec(**kwargs).validate()


def test_source_paths_are_unique_and_resume3_only() -> None:
    assert len(module.SOURCE_PATHS) == len(set(module.SOURCE_PATHS))
    assert all("resume3" in value for value in module.SOURCE_PATHS)


def test_reconstruct_requires_retained_state() -> None:
    with pytest.raises(RuntimeError, match="_prior_state"):
        module.reconstruct_prior_prediction(
            {"prior_state_sha256": "x"},
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
        )


def test_reconstruct_requires_tensor_condition() -> None:
    state = {"w": torch.ones(1)}
    with pytest.raises(TypeError, match="condition_z"):
        module.reconstruct_prior_prediction(
            snapshot_for(state),
            scheduler=FakeScheduler(),
            condition_z=np.zeros((2, 3)),  # type: ignore[arg-type]
            seed=1,
        )


@pytest.mark.parametrize("shape", [(3,), (0, 3)])
def test_reconstruct_rejects_invalid_condition_shape(shape: tuple[int, ...]) -> None:
    state = {"w": torch.ones(1)}
    with pytest.raises(ValueError, match="shape"):
        module.reconstruct_prior_prediction(
            snapshot_for(state),
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(shape),
            seed=1,
        )


def test_reconstruct_requires_scheduler_contract() -> None:
    state = {"w": torch.ones(1)}
    with pytest.raises(ValueError, match="alphas_cumprod"):
        module.reconstruct_prior_prediction(
            snapshot_for(state),
            scheduler=object(),
            condition_z=torch.zeros(2, 3),
            seed=1,
        )


def test_reconstruct_rejects_state_hash_mismatch() -> None:
    state = {"w": torch.ones(1)}
    snapshot = snapshot_for(state)
    snapshot["prior_state_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="retained prior state"):
        module.reconstruct_prior_prediction(
            snapshot,
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
        )


def test_reconstruct_uses_resume1_model_path() -> None:
    state = {"w": torch.ones(1)}
    prediction = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    model = FakeModel(prediction)
    calls: Dict[str, Any] = {}
    condition = torch.zeros(2, 5)
    snapshot = snapshot_for(state)
    observed, shape, dtype = module.reconstruct_prior_prediction(
        snapshot,
        scheduler=FakeScheduler(),
        condition_z=condition,
        seed=102000,
        instantiate_fn=fake_instantiate_factory(model, calls),
    )
    assert observed == module.prediction_sha256(prediction)
    assert shape == (2, 3, 4)
    assert dtype == "float32"
    assert calls["condition_dim"] == 5
    assert calls["device"] == condition.device
    assert calls["snapshot"] is snapshot
    assert calls["seed"] == 102000
    assert model.eval_called
    assert model.predict_calls == 1


def test_reconstruct_converts_prediction_to_float32() -> None:
    state = {"w": torch.ones(1)}
    prediction = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    model = FakeModel(prediction)
    observed, _, dtype = module.reconstruct_prior_prediction(
        snapshot_for(state),
        scheduler=FakeScheduler(),
        condition_z=torch.zeros(2, 4),
        seed=7,
        instantiate_fn=fake_instantiate_factory(model, {}),
    )
    expected = module.prediction_sha256(prediction.numpy().astype(np.float32))
    assert observed == expected
    assert dtype == "float32"


def test_reconstruct_restores_all_rng_streams() -> None:
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    before = module.stochastic_state_fingerprint()
    state = {"w": torch.ones(1)}
    model = FakeModel(torch.ones(2, 4, 3), disturb_rng=True)
    module.reconstruct_prior_prediction(
        snapshot_for(state),
        scheduler=FakeScheduler(),
        condition_z=torch.zeros(2, 5),
        seed=102000,
        instantiate_fn=fake_instantiate_factory(model, {}),
    )
    after = module.stochastic_state_fingerprint()
    assert before == after


def test_reconstruct_does_not_mutate_state() -> None:
    state = {"w": torch.tensor([1.0, 2.0])}
    before = state["w"].clone()
    model = FakeModel(torch.ones(2, 2, 2))
    module.reconstruct_prior_prediction(
        snapshot_for(state),
        scheduler=FakeScheduler(),
        condition_z=torch.zeros(2, 5),
        seed=1,
        instantiate_fn=fake_instantiate_factory(model, {}),
    )
    assert torch.equal(state["w"], before)


def test_validate_rejects_unexpected_prediction_field() -> None:
    state = {"w": torch.ones(1)}
    snapshot = snapshot_for(state)
    snapshot["_prior_prediction_z"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="historical snapshot API"):
        module.validate_fresh_prior_snapshot(
            snapshot,
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
        )


def test_validate_checks_state_before_instantiation(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"w": torch.ones(1)}
    snapshot = snapshot_for(state)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_STATE_SHA256", "other")
    called = False

    def instantiate(**_: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(RuntimeError, match="state SHA"):
        module.validate_fresh_prior_snapshot(
            snapshot,
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
            instantiate_fn=instantiate,
        )
    assert not called


def test_validate_rejects_prediction_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"w": torch.ones(1)}
    snapshot = snapshot_for(state)
    prediction = torch.ones(2, 2, 2)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_STATE_SHA256", snapshot["prior_state_sha256"])
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", "bad")
    with pytest.raises(RuntimeError, match="prediction SHA"):
        module.validate_fresh_prior_snapshot(
            snapshot,
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
            instantiate_fn=fake_instantiate_factory(FakeModel(prediction), {}),
        )


def test_validate_rejects_z_mse(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"w": torch.ones(1)}
    snapshot = snapshot_for(state, z_mse=0.9)
    prediction = torch.ones(2, 2, 2)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_STATE_SHA256", snapshot["prior_state_sha256"])
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", module.prediction_sha256(prediction))
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_Z_MSE", 0.1)
    with pytest.raises(RuntimeError, match="z-MSE"):
        module.validate_fresh_prior_snapshot(
            snapshot,
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(2, 3),
            seed=1,
            instantiate_fn=fake_instantiate_factory(FakeModel(prediction), {}),
        )


def test_validate_success_is_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"w": torch.ones(1)}
    prediction = torch.ones(2, 2, 2)
    snapshot = snapshot_for(state, z_mse=0.125)
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_STATE_SHA256", snapshot["prior_state_sha256"])
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_PREDICTION_SHA256", module.prediction_sha256(prediction))
    monkeypatch.setattr(module, "CURRENT_DEVICE_PRIOR_Z_MSE", 0.125)
    result = module.validate_fresh_prior_snapshot(
        snapshot,
        scheduler=FakeScheduler(),
        condition_z=torch.zeros(2, 3),
        seed=1,
        instantiate_fn=fake_instantiate_factory(FakeModel(prediction), {}),
    )
    assert result["functional_prior_contract_pass"]
    assert result["prediction_reconstructed_from_prior_state"]
    assert result["rng_state_restored"]
    assert result["prediction_tensor_persisted"] is False
    assert snapshot["prior_prediction_sha256"] == module.prediction_sha256(prediction)
    assert "_prior_prediction_z" not in snapshot
    assert not any(isinstance(value, torch.Tensor) for key, value in snapshot.items() if key != "_prior_state")


def test_augment_pilot_payload_sets_resume3_contract() -> None:
    payload = module.augment_pilot_payload(
        {"verdict": "PASS"},
        functional_contract={"functional_prior_contract_supported": True},
        fresh_prior_validation={"functional_prior_contract_pass": True},
        resume3_source_hashes={"x": "y"},
    )
    assert payload["resume_generation"] == 3
    assert payload["resume3_schema"] == module.RESUME3_SCHEMA
    assert payload["prior_prediction_reconstructed_in_memory"] is True
    assert payload["prior_prediction_tensor_persisted"] is False
    assert payload["train_only_recommendation"] is None
    assert payload["selected_configuration"] is None


def valid_pilot() -> Dict[str, Any]:
    return {
        "resume_generation": 3,
        "resume3_schema": module.RESUME3_SCHEMA,
        "functional_prior_contract": {"functional_prior_contract_supported": True},
        "fresh_prior_validation": {
            "functional_prior_contract_pass": True,
            "state_sha_exact": True,
            "prediction_sha_exact": True,
            "prior_z_mse_pass": True,
            "prediction_reconstructed_from_prior_state": True,
            "rng_state_restored": True,
            "prediction_tensor_persisted": False,
        },
        "shared_prior": {
            "prior_state_sha256": module.CURRENT_DEVICE_PRIOR_STATE_SHA256,
            "prior_prediction_sha256": module.CURRENT_DEVICE_PRIOR_PREDICTION_SHA256,
        },
        "train_only_recommendation": None,
        "selected_configuration": None,
        "reverse_sampling_rerun": False,
    }


def test_validate_resume3_pilot_accepts_complete_contract() -> None:
    module.validate_resume3_pilot_contract(valid_pilot())


@pytest.mark.parametrize(
    "path,value",
    [
        (("resume_generation",), 2),
        (("resume3_schema",), "bad"),
        (("functional_prior_contract", "functional_prior_contract_supported"), False),
        (("fresh_prior_validation", "functional_prior_contract_pass"), False),
        (("fresh_prior_validation", "state_sha_exact"), False),
        (("fresh_prior_validation", "prediction_sha_exact"), False),
        (("fresh_prior_validation", "prior_z_mse_pass"), False),
        (("fresh_prior_validation", "prediction_reconstructed_from_prior_state"), False),
        (("fresh_prior_validation", "rng_state_restored"), False),
        (("fresh_prior_validation", "prediction_tensor_persisted"), True),
        (("shared_prior", "prior_state_sha256"), "bad"),
        (("shared_prior", "prior_prediction_sha256"), "bad"),
        (("train_only_recommendation",), "x"),
        (("selected_configuration",), "x"),
        (("reverse_sampling_rerun",), True),
    ],
)
def test_validate_resume3_pilot_rejects_contract_violation(path: tuple[str, ...], value: Any) -> None:
    payload = valid_pilot()
    target: Dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError):
        module.validate_resume3_pilot_contract(payload)


def test_compact_functional_prior_contract() -> None:
    compact = module.compact_functional_prior_contract(
        {
            "resume1_root_cause": "root",
            "functional_prior_contract_supported": True,
            "historical_prior_state_sha256": "h",
            "current_device_prior_state_sha256": "s",
            "current_device_prior_prediction_sha256": "p",
            "current_device_prior_z_mse": 0.2,
            "same_device_exact_state_sha": True,
            "same_device_exact_prediction_sha": True,
            "same_device_functional_equivalence": True,
            "historical_functional_fingerprint": True,
        }
    )
    assert compact["resume_generation"] == 3
    assert compact["schema"] == module.RESUME3_SCHEMA
    assert compact["current_device_prior_prediction_sha256"] == "p"


def valid_blocked() -> Dict[str, Any]:
    return {
        "verdict": "BLOCKED",
        "root_cause": "phase314b_r254_resume2_execution_failed",
        "pilot_report": None,
        "pilot_sha256": None,
        "preflight_report": "reports/preflight.json",
        "preflight_sha256": "abc",
        "robot_proxy_attribution_interpretable": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }


def test_verify_resume2_blocked_contract() -> None:
    result = module.verify_resume2_blocked_contract(valid_blocked())
    assert result["pilot_created"] is False
    assert result["boundaries_closed"] is True


@pytest.mark.parametrize(
    "key,value",
    [
        ("verdict", "PASS"),
        ("root_cause", "bad"),
        ("pilot_report", "pilot.json"),
        ("preflight_report", None),
        ("robot_proxy_attribution_interpretable", True),
        ("train_only_recommendation", "x"),
        ("selected_configuration", "x"),
        ("formal_training", True),
    ],
)
def test_verify_resume2_blocked_contract_rejects(key: str, value: Any) -> None:
    payload = valid_blocked()
    payload[key] = value
    with pytest.raises(RuntimeError):
        module.verify_resume2_blocked_contract(payload)


def test_run_wrapper_has_no_bare_python() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r254_resume3_run.sh").read_text(encoding="utf-8")
    for line in text.splitlines():
        assert not line.lstrip().startswith("python ")
        assert not line.lstrip().startswith("python3 ")
    assert text.count('"${PYTHON_BIN}"') >= 5


def test_run_pilot_calls_original_fit_once_and_uses_new_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/phase3_14b_r254_resume3_run_pilot.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "original_fit"
    ]
    assert len(calls) == 1
    assert "RESUME3_PILOT_PATH" in text
    assert "validate_fresh_prior_snapshot" in text
    assert "inspect.signature" in text


def test_new_code_does_not_call_reverse_sampling() -> None:
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "paired_reverse_pool_metrics(",
        "reverse_sample_pool(",
        "reverse_predicted_x0_trajectory(",
    )
    for path in list((root / "ccda_phase3").glob("*resume3*.py")) + list(
        (root / "scripts").glob("*r254_resume3*.py")
    ):
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden)


def test_new_code_does_not_persist_torch_payloads() -> None:
    root = Path(__file__).resolve().parents[1]
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in list((root / "ccda_phase3").glob("*resume3*.py"))
        + list((root / "scripts").glob("*r254_resume3*.py"))
    )
    assert "torch.save(" not in text
    assert "np.save(" not in text
    assert "np.savez(" not in text
