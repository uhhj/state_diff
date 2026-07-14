from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r254_resume5_residual_determinism import (
    CONTRACT_KEYS,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    IDENTITY_FIELDS,
    NUMERIC_KEYS,
    RESUME5_TEST_PATH,
    assert_json_summary_safe,
    build_resume5_audit,
    canonical_sha256,
    compare_repeat_matrix,
    functional_reproduction_contract,
    instrument_residual_training,
    module_state_sha256,
    optimizer_state_sha256,
    parse_worker_stdout,
    tensor_fingerprint,
)


def identity_payload() -> dict:
    value = {name: "a" * 64 for name in IDENTITY_FIELDS}
    value.update({
        "prediction_fingerprint": {
            "sha256": "a" * 64,
            "shape": [2, 4, 87],
            "dtype": "torch.float32",
        },
        "training_loss_history": [{"step": 1, "total_loss": 1.0}],
        "gradient_history": [],
        "source_exposure_counts": [4, 4],
        "training_rng": {"global_after_model_init": "b" * 64},
    })
    return value


def contracts_for(name: str, historical: bool = False) -> dict:
    value = {key: True for key in CONTRACT_KEYS}
    if name != "v_only_frozen_control":
        value["historical_exact_reconstruction_pass"] = False
    if historical and name == "v_only_frozen_control":
        value["historical_exact_reconstruction_pass"] = False
        value["one_step_physical_pass"] = False
        value["all_earlier_horizon_cable_geometry_pass"] = False
    return value


def worker_repeat(index: int) -> dict:
    variants = {}
    for model_index, name in enumerate(DIAGNOSTIC_OBJECTIVE_NAMES):
        numeric = {key: float(model_index + offset + 1) for offset, key in enumerate(NUMERIC_KEYS)}
        variants[name] = {
            "identity": identity_payload(),
            "contracts": contracts_for(name),
            "state_group_z_metrics": numeric,
            "variant_fingerprint_sha256": "c" * 64,
        }
    return {
        "repeat_index": index,
        "runtime": {"gpu_name": "NVIDIA GeForce RTX 4090"},
        "prior": {
            "state_sha256": "8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904",
            "prediction_sha256": "70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9",
            "prior_z_mse": 0.17507688701152802,
        },
        "variants": variants,
    }


def resume4_fixture() -> dict:
    models = {}
    for model_index, name in enumerate(DIAGNOSTIC_OBJECTIVE_NAMES):
        historical = contracts_for(name, historical=True)
        current = contracts_for(name)
        checks = []
        for key in CONTRACT_KEYS:
            checks.append({
                "check": f"contract.{key}",
                "expected": historical[key],
                "observed": current[key],
            })
        for offset, key in enumerate(NUMERIC_KEYS):
            observed = float(model_index + offset + 1)
            checks.append({
                "check": f"state_group_z_metrics.{key}",
                "expected": observed + 0.5,
                "observed": observed,
            })
        models[name] = {"checks": checks}
    return {"models": models}


def test_canonical_hash_is_mapping_order_independent() -> None:
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_canonical_hash_preserves_float_bits_and_types() -> None:
    assert canonical_sha256(1.0) != canonical_sha256(1)
    assert canonical_sha256(np.float32(1.0)) == canonical_sha256(1.0)


def test_tensor_fingerprint_binds_shape_dtype_and_bytes() -> None:
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    value = tensor_fingerprint(tensor)
    assert value["shape"] == [2, 3]
    assert value["dtype"] == "torch.float32"
    assert value["sha256"] != tensor_fingerprint(tensor.double())["sha256"]


def test_module_and_optimizer_fingerprints_change_after_step() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    model_before = module_state_sha256(model)
    optimizer_before = optimizer_state_sha256(optimizer)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert module_state_sha256(model) != model_before
    assert optimizer_state_sha256(optimizer) != optimizer_before

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.residual_v = torch.nn.Linear(2, 1)

        def residual_parameters(self):
            return list(self.residual_v.parameters())

    class TinySampler:
        def __init__(self, source_count, batch_size, mode, *, device, seed):
            self.source_count = source_count
            self.batch_size = batch_size
            self.generator = torch.Generator(device=device).manual_seed(seed)
            self.counts = torch.zeros(source_count, dtype=torch.long, device=device)

        def next_indices(self):
            value = torch.randint(
                self.source_count,
                (self.batch_size,),
                generator=self.generator,
                device=self.counts.device,
            )
            self.counts += torch.bincount(value, minlength=self.source_count)
            return value

        def count_report(self):
            return {"counts": self.counts.cpu().tolist()}

    def make_model(*args, **kwargs):
        return TinyModel()

    def make_optimizer(parameters, learning_rate, weight_decay):
        return torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=weight_decay
        )

    def sample_timesteps(*, batch_size, values, generator, device):
        choices = torch.tensor(values, dtype=torch.long, device=device)
        index = torch.randint(
            len(values), (batch_size,), generator=generator, device=device
        )
        return choices[index]

    fake = SimpleNamespace(
        instantiate_snapshot_model=make_model,
        _optimizer=make_optimizer,
        SourceBatchSampler=TinySampler,
        _sample_timesteps=sample_timesteps,
    )
    with instrument_residual_training(fake) as recorder:
        trained = fake.instantiate_snapshot_model()
        trained_optimizer = fake._optimizer(
            trained.residual_parameters(), 1.0e-3, 0.0
        )
        sampler = fake.SourceBatchSampler(
            2, 2, "balanced", device=torch.device("cpu"), seed=7
        )
        sampler.next_indices()
        generator = torch.Generator().manual_seed(8)
        fake._sample_timesteps(
            batch_size=2, values=(0, 1), generator=generator, device=torch.device("cpu")
        )
        torch.randn((2, 2), generator=generator)
        trained.residual_v(torch.ones(2, 2)).sum().backward()
        trained_optimizer.step()
        recorded = recorder.finish({
            "_true_prediction_z": torch.ones(2, 1),
            "residual_history": [{"step": 1, "total_loss": 1.0}],
            "gradient_audit": [],
            "source_exposure": sampler.count_report(),
        })
    assert all(recorded[name] for name in IDENTITY_FIELDS)


def test_json_summary_safety_accepts_scalar_tree() -> None:
    assert_json_summary_safe({"a": [1, 2.0, True, None, "x"]})


def test_json_summary_safety_rejects_runtime_tensor() -> None:
    with pytest.raises(TypeError, match="runtime tensor"):
        assert_json_summary_safe({"bad": torch.zeros(1)})


def test_worker_stdout_requires_one_sentinel() -> None:
    assert parse_worker_stdout('log\nRESUME5_RESULT_JSON={"repeat_index": 0}\n') == {
        "repeat_index": 0
    }


def test_worker_stdout_rejects_multiple_sentinels() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        parse_worker_stdout("RESUME5_RESULT_JSON={}\nRESUME5_RESULT_JSON={}")


def test_functional_contract_excludes_mixed_robot_physical_gate() -> None:
    historical = {key: True for key in CONTRACT_KEYS}
    historical["one_step_physical_pass"] = False
    current = dict(historical)
    current["one_step_physical_pass"] = True
    result = functional_reproduction_contract(historical, current)
    assert result["pass"] is True
    assert result["new_numeric_tolerance_introduced"] is False
    assert "one_step_physical_pass" in result["excluded_from_gate"]
    assert all(
        item["contract"] != "one_step_physical_pass" for item in result["checks"]
    )


def test_functional_contract_rejects_true_to_false_regression() -> None:
    historical = {key: True for key in CONTRACT_KEYS}
    current = dict(historical)
    current["ordered_topology_pass"] = False
    result = functional_reproduction_contract(historical, current)
    assert result["pass"] is False
    assert any(item["historical_true_regressed"] for item in result["checks"])


def test_repeat_matrix_accepts_three_exact_variants() -> None:
    result = compare_repeat_matrix([worker_repeat(index) for index in range(3)])
    assert result["same_device_exact"] is True
    assert all(item["exact"] for item in result["models"].values())


def test_repeat_matrix_localizes_identity_mismatch() -> None:
    repeats = [worker_repeat(index) for index in range(3)]
    repeats[2]["variants"]["ordered_mean_raw_g100"]["identity"][
        "residual_final_state_sha256"
    ] = "d" * 64
    result = compare_repeat_matrix(repeats)
    assert result["same_device_exact"] is False
    assert result["models"]["ordered_mean_raw_g100"]["identity_fields"][
        "residual_final_state_sha256"
    ]["exact"] is False


def test_resume5_build_and_closed_resume4_glob_contract() -> None:
    assert not RESUME5_TEST_PATH.startswith("tests/test_phase314b_r2")
    report = build_resume5_audit(
        repeats=[worker_repeat(index) for index in range(3)],
        resume4_audit=resume4_fixture(),
    )
    assert report["same_device_residual_determinism_pass"] is True
    assert report["fresh_exact_match_to_resume3_pass"] is True
    assert report["functional_cable_nonregression_pass"] is True
    assert report["functional_reproduction_contract_pass"] is True
    assert report["historical_boolean_exact_agreement_pass"] is False
    assert report["robot_proxy_metrics_used_for_gate"] is False
    assert report["legacy_cache_loader_eagerly_materializes_full_npz"] is True
    assert report["validation_target_rows_indexed"] is False
    assert report["formal_target_rows_indexed"] is False
    assert report["selected_configuration"] is None
