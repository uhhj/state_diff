from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ccda_phase3 import phase314b_r258_stagea_conflict_projected_k16 as stagea258


def make_named_parameters():
    names = (
        "noisy_projection.weight",
        "noisy_projection.bias",
        "condition_projection.weight",
        "condition_projection.bias",
        "time_projection.0.weight",
        "time_projection.0.bias",
        "time_projection.2.weight",
        "time_projection.2.bias",
        "blocks.0.norm.weight",
        "blocks.0.norm.bias",
        "blocks.0.fc1.weight",
        "blocks.0.fc1.bias",
        "blocks.0.fc2.weight",
        "blocks.0.fc2.bias",
        "blocks.1.norm.weight",
        "blocks.1.norm.bias",
        "blocks.1.fc1.weight",
        "blocks.1.fc1.bias",
        "blocks.1.fc2.weight",
        "blocks.1.fc2.bias",
        "blocks.2.norm.weight",
        "blocks.2.norm.bias",
        "blocks.2.fc1.weight",
        "blocks.2.fc1.bias",
        "blocks.2.fc2.weight",
        "blocks.2.fc2.bias",
        "blocks.3.norm.weight",
        "blocks.3.norm.bias",
        "blocks.3.fc1.weight",
        "blocks.3.fc1.bias",
        "blocks.3.fc2.weight",
        "blocks.3.fc2.bias",
        "output_norm.weight",
        "output_norm.bias",
        "output.weight",
        "output.bias",
    )
    return [
        (
            name,
            torch.nn.Parameter(
                torch.ones(
                    2,
                    dtype=torch.float32,
                )
            ),
        )
        for name in names
    ]


def fake_candidate(
    *,
    candidate_id="block_t50_r0p25",
    mode="blockwise",
    cutoff=50,
    ratio=0.25,
):
    return stagea258.ProjectedCandidate(
        candidate_id=candidate_id,
        projection_mode=mode,
        timestep_cutoff=cutoff,
        top_k=16,
        target_gradient_ratio=ratio,
        lambda_upper=0.1,
        initial_model_sha256="a" * 64,
        diffusion_gradient_norm=1.0,
        geometry_gradient_norm=2.0,
        projected_geometry_gradient_norm=1.5,
        pre_projection_cosine=-0.2,
        post_projection_cosine=0.0,
        projection_triggered_block_fraction=0.5,
        removed_geometry_norm_fraction=0.25,
    )


def base_selection_record(
    *,
    candidate=None,
    continuous=False,
    fidelity=False,
    binary=False,
    stability=True,
    projection=True,
    reduction=0.0,
    train_ratio=1.0,
):
    candidate = fake_candidate() if candidate is None else candidate
    return {
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.to_dict(),
        "continuous_geometry_pass": continuous,
        "fidelity_pass": fidelity,
        "binary_pass": binary,
        "stability_pass": stability,
        "projection_contract_pass": projection,
        "eligible_for_selection": bool(
            continuous
            and fidelity
            and binary
            and stability
            and projection
            and candidate.projection_mode == "blockwise"
        ),
        "binary_rate": {
            "10": 0.0,
            "25": 0.0,
            "50": 0.0,
        },
        "holdout_profiles": {
            "10": {
                "row_best_max_log_excess": {
                    "mean": 1.0,
                }
            }
        },
        "relative_to_control": {
            "train_control_nmse_ratio": train_ratio,
        },
        "continuous_response": {
            "t10_mean_excess_reduction": reduction,
        },
        "training": {
            "final_model_sha256": "f" * 64,
        },
        "training_diagnostics": {
            "projection_trigger_frequency": 0.5,
            "removed_norm_fraction_mean": 0.2,
        },
    }




def portable_environment_payload(gpu_name="NVIDIA GeForce RTX 4090"):
    spec = stagea258.PortableEnvironmentSpec()
    compatibility = {
        "schema": "phase314b_r258_stagea_portable_compatibility_v1",
        "python_version": list(spec.python_version),
        "python_implementation": "CPython",
        "numpy_version": spec.numpy_version,
        "torch_version": spec.torch_version,
        "torch_cuda_version": spec.torch_cuda_version,
        "cuda_available": True,
        "required_operation_schema": spec.required_operation_schema,
        "required_operation_pass": True,
        "deterministic_runtime": {"x": 1},
    }
    observation = {
        "schema": "phase314b_r258_stagea_hardware_observation_v1",
        "torch_device_name": gpu_name,
        "compute_capability": [8, 9],
        "total_memory_bytes": 24 * 1024 ** 3,
    }
    dry_run = {
        "schema": spec.required_operation_schema,
        "pass": True,
        "batch_size": 64,
        "calibration_batch_size": 100,
    }
    payload = {
        "schema": "phase314b_r258_stagea_portable_environment_v1",
        "compatibility": compatibility,
        "hardware_observation": observation,
        "required_operation_dry_run": dry_run,
        "compatibility_pass": True,
    }
    payload["compatibility_sha256"] = stagea258.sha256_bytes(
        stagea258.stable_json_bytes(compatibility)
    )
    payload["observation_sha256"] = stagea258.sha256_bytes(
        stagea258.stable_json_bytes(
            {
                "hardware_observation": observation,
                "required_operation_dry_run": dry_run,
            }
        )
    )
    return payload


def test_phase_and_base_commit_are_frozen():
    assert stagea258.PHASE == (
        "Phase3.14b-r2.5.8 Stage A"
    )
    assert stagea258.BASE_EVIDENCE_COMMIT == (
        "f0a5bec1f89f625e74150e55e2da884413d72eb9"
    )


def test_resume4_identity_constants_are_frozen():
    assert stagea258.EXPECTED_RESUME4_WORKER_SHA256 == (
        "0e87be2ae637ad1dd3680535e40ab8b29f7d7e54bf8c6555817d4da4617cac63"
    )
    assert stagea258.EXPECTED_RESUME4_ENVIRONMENT_SHA256 == (
        "7b51113fb013a7f24716049fe4e11464261af62b0b3c4950cdf454fc21aad260"
    )
    assert stagea258.EXPECTED_STAGEC_CONTROL_SHA256 == (
        "dee61b8e31455eeb0e7e0eceae5c1500e3ce57ca8bd42989378d9f76fd779ae7"
    )


def test_stable_json_is_deterministic():
    assert stagea258.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == stagea258.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_stable_json_supports_numpy():
    payload = stagea258.stable_json_bytes(
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
    stagea258.atomic_write_once(
        path,
        b"{}\n",
    )
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        stagea258.atomic_write_once(
            path,
            b"{}\n",
        )


def test_load_json_rejects_non_object(tmp_path):
    path = tmp_path / "value.json"
    path.write_text(
        "[1]\n",
        encoding="utf-8",
    )
    with pytest.raises(
        stagea258.ConflictProjectionError
    ):
        stagea258.load_json(path)


def test_projection_spec_validates():
    stagea258.ConflictProjectionSpec().validate()


def test_projection_spec_rejects_changed_candidates():
    spec = stagea258.ConflictProjectionSpec(
        candidate_definitions=(
            stagea258.ProjectionCandidateDefinition(
                "blockwise",
                50,
                0.25,
            ),
        )
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_candidate_definition_rejects_mode():
    with pytest.raises(ValueError):
        stagea258.ProjectionCandidateDefinition(
            "tensorwise",
            50,
            0.25,
        ).validate()


def test_candidate_definition_rejects_cutoff():
    with pytest.raises(ValueError):
        stagea258.ProjectionCandidateDefinition(
            "blockwise",
            10,
            0.25,
        ).validate()


def test_candidate_definition_rejects_ratio():
    with pytest.raises(ValueError):
        stagea258.ProjectionCandidateDefinition(
            "blockwise",
            50,
            1.0,
        ).validate()


def test_candidate_id_tokens():
    assert stagea258.candidate_id(
        stagea258.ProjectionCandidateDefinition(
            "global",
            50,
            0.25,
        )
    ) == "global_t50_r0p25"
    assert stagea258.candidate_id(
        stagea258.ProjectionCandidateDefinition(
            "blockwise",
            25,
            0.10,
        )
    ) == "block_t25_r0p10"


def test_projected_candidate_validates():
    fake_candidate().validate()


def test_projected_candidate_rejects_sha():
    candidate = fake_candidate()
    value = copy.deepcopy(
        candidate.to_dict()
    )
    value["initial_model_sha256"] = "x"
    with pytest.raises(ValueError):
        stagea258.ProjectedCandidate(
            **value
        ).validate()


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "noisy_projection.weight",
            "noisy_projection",
        ),
        (
            "condition_projection.bias",
            "condition_projection",
        ),
        (
            "time_projection.2.weight",
            "time_projection",
        ),
        (
            "blocks.3.fc2.bias",
            "residual_block_3",
        ),
        (
            "output_norm.weight",
            "output_norm",
        ),
        (
            "output.bias",
            "output_head",
        ),
    ],
)
def test_parameter_block_name(name, expected):
    assert (
        stagea258.parameter_block_name(name)
        == expected
    )


def test_parameter_block_name_rejects_unknown():
    with pytest.raises(
        stagea258.ConflictProjectionError
    ):
        stagea258.parameter_block_name(
            "unknown.weight"
        )


def test_parameter_block_map_exact_order():
    mapping = stagea258.parameter_block_map(
        make_named_parameters()
    )
    assert tuple(mapping) == (
        stagea258.EXPECTED_BLOCK_ORDER
    )
    assert sum(
        len(value)
        for value in mapping.values()
    ) == len(make_named_parameters())


def test_gradient_sequence_sha_is_deterministic():
    named = make_named_parameters()
    gradients = [
        torch.arange(
            parameter.numel(),
            dtype=torch.float32,
        ).reshape_as(parameter)
        for _name, parameter in named
    ]
    first = stagea258.gradient_sequence_sha256(
        named,
        gradients,
    )
    second = stagea258.gradient_sequence_sha256(
        named,
        gradients,
    )
    assert first == second
    assert len(first) == 64


def test_global_projection_removes_negative_dot():
    named = make_named_parameters()
    diffusion = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    geometry = [
        -torch.ones_like(parameter)
        for _name, parameter in named
    ]
    combined, record = (
        stagea258.project_gradient_pair(
            named,
            diffusion,
            geometry,
            projection_mode="global",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=True,
        )
    )
    assert len(combined) == len(named)
    assert record["global"][
        "projection_triggered_group_count"
    ] == 1
    assert record["global"][
        "post_projection_violation_count"
    ] == 0
    assert record["global"][
        "post_dot"
    ] >= -1.0e-5
    assert record["global"][
        "diffusion_gradient_unchanged"
    ]


def test_global_projection_keeps_positive_dot():
    named = make_named_parameters()
    diffusion = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    geometry = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    _combined, record = (
        stagea258.project_gradient_pair(
            named,
            diffusion,
            geometry,
            projection_mode="global",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=False,
        )
    )
    assert record["global"][
        "projection_triggered_group_count"
    ] == 0
    assert record["global"][
        "removed_norm_fraction"
    ] == pytest.approx(0.0)


def test_blockwise_projection_only_triggers_conflicting_blocks():
    named = make_named_parameters()
    diffusion = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    geometry = []
    for name, parameter in named:
        sign = (
            -1.0
            if name.startswith(
                "condition_projection."
            )
            else 1.0
        )
        geometry.append(
            torch.full_like(
                parameter,
                sign,
            )
        )
    _combined, record = (
        stagea258.project_gradient_pair(
            named,
            diffusion,
            geometry,
            projection_mode="blockwise",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=False,
        )
    )
    assert record["per_group"][
        "condition_projection"
    ]["projection_triggered"]
    assert not record["per_group"][
        "noisy_projection"
    ]["projection_triggered"]
    assert record["global"][
        "projection_triggered_group_count"
    ] == 1


def test_blockwise_projection_covers_nine_groups():
    named = make_named_parameters()
    diffusion = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    geometry = [
        -torch.ones_like(parameter)
        for _name, parameter in named
    ]
    _combined, record = (
        stagea258.project_gradient_pair(
            named,
            diffusion,
            geometry,
            projection_mode="blockwise",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=False,
        )
    )
    assert record["global"][
        "projection_group_count"
    ] == 9
    assert record["global"][
        "projection_triggered_group_count"
    ] == 9
    assert tuple(
        record["per_group"]
    ) == stagea258.EXPECTED_BLOCK_ORDER


def test_projection_rejects_unknown_mode():
    named = make_named_parameters()
    gradients = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    with pytest.raises(ValueError):
        stagea258.project_gradient_pair(
            named,
            gradients,
            gradients,
            projection_mode="other",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=False,
        )


def test_projection_reports_per_parameter_metrics():
    named = make_named_parameters()
    diffusion = [
        torch.ones_like(parameter)
        for _name, parameter in named
    ]
    geometry = [
        -torch.ones_like(parameter)
        for _name, parameter in named
    ]
    _combined, record = (
        stagea258.project_gradient_pair(
            named,
            diffusion,
            geometry,
            projection_mode="blockwise",
            lambda_upper=0.25,
            epsilon=1.0e-12,
            post_dot_tolerance=1.0e-6,
            include_per_parameter=True,
        )
    )
    assert len(
        record["per_parameter"]
    ) == len(named)
    first = next(
        iter(
            record["per_parameter"].values()
        )
    )
    assert "block" in first
    assert "post_dot" in first


def test_projection_contract_gates_pass():
    record = {
        "training_diagnostics": {
            "post_projection_violation_frequency": 0.0,
            "diffusion_gradient_change_count": 0,
            "checkpoint_records": {
                "8000": {
                    "gradient": {
                        "post_projection_violation_count": 0,
                        "diffusion_gradient_unchanged": True,
                    }
                }
            },
        },
        "training": {
            "projection_finite": True,
        },
    }
    gates = (
        stagea258.projection_contract_gates(
            record,
            spec=
                stagea258.ConflictProjectionSpec(),
        )
    )
    assert all(gates.values())


def test_projection_contract_gates_detect_violation():
    record = {
        "training_diagnostics": {
            "post_projection_violation_frequency": 0.1,
            "diffusion_gradient_change_count": 0,
            "checkpoint_records": {
                "8000": {
                    "gradient": {
                        "post_projection_violation_count": 0,
                        "diffusion_gradient_unchanged": True,
                    }
                }
            },
        },
        "training": {
            "projection_finite": True,
        },
    }
    gates = (
        stagea258.projection_contract_gates(
            record,
            spec=
                stagea258.ConflictProjectionSpec(),
        )
    )
    assert not gates[
        "post_projection_training"
    ]


def test_global_negative_control_cannot_be_selected():
    candidate = fake_candidate(
        candidate_id="global_t50_r0p25",
        mode="global",
    )
    record = base_selection_record(
        candidate=candidate,
        continuous=True,
        fidelity=True,
        binary=True,
    )
    record["eligible_for_selection"] = False
    assert (
        stagea258.select_configuration(
            [record]
        )
        is None
    )


def test_blockwise_candidate_can_be_selected():
    record = base_selection_record(
        continuous=True,
        fidelity=True,
        binary=True,
    )
    selected = (
        stagea258.select_configuration(
            [record]
        )
    )
    assert selected is not None
    assert selected[
        "candidate_id"
    ] == "block_t50_r0p25"


def test_classification_selected():
    global_record = base_selection_record(
        candidate=fake_candidate(
            candidate_id="global_t50_r0p25",
            mode="global",
        ),
        reduction=0.0,
    )
    block_records = [
        base_selection_record(
            candidate=fake_candidate(
                candidate_id=name,
                mode="blockwise",
                cutoff=cutoff,
                ratio=ratio,
            ),
            continuous=True,
            fidelity=True,
            binary=True,
            reduction=0.3,
        )
        for name, cutoff, ratio in (
            ("block_t25_r0p25", 25, 0.25),
            ("block_t50_r0p10", 50, 0.10),
            ("block_t50_r0p25", 50, 0.25),
            ("block_t50_r0p50", 50, 0.50),
        )
    ]
    selected = {
        "candidate_id":
            "block_t50_r0p25"
    }
    result = stagea258.classify_selection(
        records=[
            global_record,
            *block_records,
        ],
        selected=selected,
        spec=stagea258.ConflictProjectionSpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "none"


def test_classification_inactive_projection():
    global_record = base_selection_record(
        candidate=fake_candidate(
            candidate_id="global_t50_r0p25",
            mode="global",
        )
    )
    block_records = []
    for name, cutoff, ratio in (
        ("block_t25_r0p25", 25, 0.25),
        ("block_t50_r0p10", 50, 0.10),
        ("block_t50_r0p25", 50, 0.25),
        ("block_t50_r0p50", 50, 0.50),
    ):
        record = base_selection_record(
            candidate=fake_candidate(
                candidate_id=name,
                mode="blockwise",
                cutoff=cutoff,
                ratio=ratio,
            )
        )
        record[
            "training_diagnostics"
        ][
            "projection_trigger_frequency"
        ] = 0.0
        record[
            "training_diagnostics"
        ][
            "removed_norm_fraction_mean"
        ] = 0.0
        block_records.append(record)
    result = stagea258.classify_selection(
        records=[
            global_record,
            *block_records,
        ],
        selected=None,
        spec=stagea258.ConflictProjectionSpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "projection_inactive"


def test_classification_joint_tradeoff():
    global_record = base_selection_record(
        candidate=fake_candidate(
            candidate_id="global_t50_r0p25",
            mode="global",
        )
    )
    block_records = [
        base_selection_record(
            candidate=fake_candidate(
                candidate_id=name,
                mode="blockwise",
                cutoff=cutoff,
                ratio=ratio,
            ),
            continuous=False,
            fidelity=False,
            reduction=0.05,
            train_ratio=3.0,
        )
        for name, cutoff, ratio in (
            ("block_t25_r0p25", 25, 0.25),
            ("block_t50_r0p10", 50, 0.10),
            ("block_t50_r0p25", 50, 0.25),
            ("block_t50_r0p50", 50, 0.50),
        )
    ]
    result = stagea258.classify_selection(
        records=[
            global_record,
            *block_records,
        ],
        selected=None,
        spec=stagea258.ConflictProjectionSpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "joint_tradeoff"


def test_classification_geometry_underpowered():
    global_record = base_selection_record(
        candidate=fake_candidate(
            candidate_id="global_t50_r0p25",
            mode="global",
        )
    )
    block_records = [
        base_selection_record(
            candidate=fake_candidate(
                candidate_id=name,
                mode="blockwise",
                cutoff=cutoff,
                ratio=ratio,
            ),
            continuous=False,
            fidelity=True,
        )
        for name, cutoff, ratio in (
            ("block_t25_r0p25", 25, 0.25),
            ("block_t50_r0p10", 50, 0.10),
            ("block_t50_r0p25", 50, 0.25),
            ("block_t50_r0p50", 50, 0.50),
        )
    ]
    result = stagea258.classify_selection(
        records=[
            global_record,
            *block_records,
        ],
        selected=None,
        spec=stagea258.ConflictProjectionSpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "geometry_underpowered"


def test_classification_fidelity_tradeoff():
    global_record = base_selection_record(
        candidate=fake_candidate(
            candidate_id="global_t50_r0p25",
            mode="global",
        )
    )
    block_records = [
        base_selection_record(
            candidate=fake_candidate(
                candidate_id=name,
                mode="blockwise",
                cutoff=cutoff,
                ratio=ratio,
            ),
            continuous=True,
            fidelity=False,
            reduction=0.3,
            train_ratio=3.0,
        )
        for name, cutoff, ratio in (
            ("block_t25_r0p25", 25, 0.25),
            ("block_t50_r0p10", 50, 0.10),
            ("block_t50_r0p25", 50, 0.25),
            ("block_t50_r0p50", 50, 0.50),
        )
    ]
    result = stagea258.classify_selection(
        records=[
            global_record,
            *block_records,
        ],
        selected=None,
        spec=stagea258.ConflictProjectionSpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "fidelity_tradeoff"


def test_calibration_seed_matches_frozen_stage_d():
    assert (
        stagea258.ConflictProjectionSpec().calibration_seed_offset
        == 6301
    )


def test_portable_environment_spec_validates():
    stagea258.PortableEnvironmentSpec().validate()


def test_portable_numerical_spec_validates():
    stagea258.PortableNumericalEquivalenceSpec().validate()


def test_environment_payload_accepts_4090():
    stagea258.validate_environment_payload(
        portable_environment_payload(
            "NVIDIA GeForce RTX 4090"
        )
    )


def test_environment_payload_accepts_a100():
    stagea258.validate_environment_payload(
        portable_environment_payload(
            "NVIDIA A100-SXM4-40GB"
        )
    )


def test_environment_payload_rejects_compatibility_failure():
    payload = portable_environment_payload()
    payload["compatibility_pass"] = False
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.validate_environment_payload(payload)


def test_environment_payload_rejects_dry_run_failure():
    payload = portable_environment_payload()
    payload["required_operation_dry_run"]["pass"] = False
    payload["observation_sha256"] = stagea258.sha256_bytes(
        stagea258.stable_json_bytes(
            {
                "hardware_observation": payload["hardware_observation"],
                "required_operation_dry_run": payload["required_operation_dry_run"],
            }
        )
    )
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.validate_environment_payload(payload)


def test_environment_payload_rejects_compatibility_sha():
    payload = portable_environment_payload()
    payload["compatibility_sha256"] = "x" * 64
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.validate_environment_payload(payload)


def test_environment_payload_rejects_observation_sha():
    payload = portable_environment_payload()
    payload["observation_sha256"] = "x" * 64
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.validate_environment_payload(payload)


def test_environment_payload_rejects_software_mismatch():
    payload = portable_environment_payload()
    payload["compatibility"]["torch_version"] = "2.0.0"
    payload["compatibility_sha256"] = stagea258.sha256_bytes(
        stagea258.stable_json_bytes(payload["compatibility"])
    )
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.validate_environment_payload(payload)


def test_semanticize_removes_hardware_sensitive_sha():
    value = stagea258._semanticize_portable(
        {
            "final_model_sha256": "x" * 64,
            "initial_model_sha256": "a" * 64,
            "prediction_sha256": "b" * 64,
            "metric": 1.0,
        }
    )
    assert "final_model_sha256" not in value
    assert "prediction_sha256" not in value
    assert value["initial_model_sha256"] == "a" * 64
    assert value["metric"] == 1.0


def test_semanticize_removes_ranked_top_positions():
    value = stagea258._semanticize_portable(
        {
            "selected_positions": {
                "top_positions": [{"count": 1}],
                "horizon_selection_fraction": [0.5, 0.5],
            }
        }
    )
    assert "top_positions" not in value["selected_positions"]
    assert value["selected_positions"][
        "horizon_selection_fraction"
    ] == [0.5, 0.5]


def test_numeric_tree_compare_exact():
    result = stagea258._numeric_tree_compare(
        {"a": 1.0, "b": [1, "x"]},
        {"a": 1.0, "b": [1, "x"]},
        rtol=1.0e-3,
        atol=1.0e-6,
        maximum_records=16,
    )
    assert result["pass"]
    assert result["difference_count"] == 0


def test_numeric_tree_compare_accepts_float_tolerance():
    result = stagea258._numeric_tree_compare(
        {"a": 1.0},
        {"a": 1.0005},
        rtol=1.0e-3,
        atol=1.0e-6,
        maximum_records=16,
    )
    assert result["pass"]


def test_numeric_tree_compare_rejects_float_outside_tolerance():
    result = stagea258._numeric_tree_compare(
        {"a": 1.0},
        {"a": 1.1},
        rtol=1.0e-3,
        atol=1.0e-6,
        maximum_records=16,
    )
    assert not result["pass"]
    assert result["differences"][0]["kind"] == "numeric_tolerance"


def test_numeric_tree_compare_keeps_integer_counts_exact():
    result = stagea258._numeric_tree_compare(
        {"count": 100},
        {"count": 101},
        rtol=0.1,
        atol=10.0,
        maximum_records=16,
    )
    assert not result["pass"]
    assert result["differences"][0]["kind"] == "integer_mismatch"


def test_numeric_tree_compare_keeps_strings_exact():
    result = stagea258._numeric_tree_compare(
        {"candidate_id": "a"},
        {"candidate_id": "b"},
        rtol=1.0,
        atol=1.0,
        maximum_records=16,
    )
    assert not result["pass"]
    assert result["differences"][0]["kind"] == "string_mismatch"


def test_numeric_tree_compare_records_structure_difference():
    result = stagea258._numeric_tree_compare(
        {"a": 1},
        {"b": 1},
        rtol=1.0e-3,
        atol=1.0e-6,
        maximum_records=16,
    )
    assert not result["pass"]
    assert result["difference_count"] == 2


def test_cold_cuda_context_portable_success(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: False,
    )
    result = stagea258.assert_cold_cuda_context_portable()
    assert result["torch_cuda_is_initialized"] is False


def test_cold_cuda_context_portable_rejects_warm(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_initialized",
        lambda: True,
    )
    with pytest.raises(stagea258.ConflictProjectionError):
        stagea258.assert_cold_cuda_context_portable()


def test_portable_exact_sha_allowlist():
    assert "initial_model_sha256" in stagea258.PORTABLE_EXACT_SHA_KEYS
    assert "source_exposure_sha256" in stagea258.PORTABLE_EXACT_SHA_KEYS
    assert "final_model_sha256" in stagea258.PORTABLE_IGNORED_SHA_KEYS


def test_identity_projection_contains_contract():
    result = {
        "root_cause": "r",
        "required_next_path": "n",
        "environment": {"x": 1},
        "cold_main_worker_context": {
            "x": 1
        },
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    projection = (
        stagea258.identity_projection(
            result
        )
    )
    assert projection[
        "calibration_contract"
    ] == {"x": 1}


def test_compare_worker_results_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "environment": {"x": 1},
        "cold_main_worker_context": {
            "x": 1
        },
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    result = (
        stagea258.compare_worker_results(
            base,
            copy.deepcopy(base),
        )
    )
    assert result["exact"]
    assert result["environment_exact"]


def test_compare_worker_results_detects_selection():
    left = {
        "root_cause": "r",
        "required_next_path": "n",
        "environment": {"x": 1},
        "cold_main_worker_context": {
            "x": 1
        },
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    right = copy.deepcopy(left)
    right["selection"] = {"x": 2}
    result = (
        stagea258.compare_worker_results(
            left,
            right,
        )
    )
    assert not result["exact"]
    assert not result["selection_exact"]
