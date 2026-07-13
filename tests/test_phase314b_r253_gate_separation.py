from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r231_controls import ReconstructionGate
from ccda_phase3.phase314b_r241_multirow import LabeledTupleBank
from ccda_phase3.phase314b_r253_gate_separation import (
    CABLE_DIM,
    DIAGNOSTIC_OBJECTIVE_NAMES,
    GATE_SEPARATION_SCHEMA,
    RECONSTRUCTION_DECOMPOSITION_SCHEMA,
    GateSeparationSpec,
    _gate_separation_payload,
    _group_z_metrics,
    _single_horizon_metrics,
    classify_gate_separation,
    compare_reconstruction_to_training_result,
    evaluate_prediction_contract,
    evaluate_synthetic_controls,
    reconstruction_gate_components,
    sha256_file,
    strip_runtime_objects,
    synthetic_prediction_controls,
    variant_supported_mechanisms,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, N_BEADS, STATE_DIM


def gate() -> ReconstructionGate:
    return ReconstructionGate(
        z_mse_max=2.5e-3,
        ordered_rmse_p95_max=1.0e-2,
        segment_relative_error_p95_max=0.30,
        chain_relative_error_p95_max=0.15,
    )


def metric_payload(
    *,
    z: float = 0.0,
    ordered: float = 0.0,
    segment: float = 0.0,
    chain: float = 0.0,
) -> dict:
    return {
        "z_mse": z,
        "ordered_rmse": {"p95": ordered},
        "segment_relative_error": {"p95": segment},
        "chain_relative_error": {"p95": chain},
    }


def make_bank() -> tuple[LabeledTupleBank, list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    source_count = 4
    condition = torch.zeros(source_count, 5)
    raw = torch.zeros(source_count, DEFAULT_TF, STATE_DIM)
    for source in range(source_count):
        pair = source // 2
        branch = source % 2
        base_x = 0.05 * pair + 0.025 * branch
        for horizon in range(DEFAULT_TF):
            xy = torch.zeros(N_BEADS, 2)
            xy[:, 0] = base_x + torch.arange(N_BEADS) * 0.01
            xy[:, 1] = 0.002 * horizon + 0.004 * branch
            raw[source, horizon, :CABLE_DIM] = xy.reshape(-1)
            raw[source, horizon, CABLE_DIM:] = 0.1 * source + 0.01 * horizon
    z = raw.clone()
    zeros = torch.zeros_like(z)
    ids = torch.arange(source_count, dtype=torch.long)
    bank = LabeledTupleBank(
        condition_z=condition,
        clean_z=z,
        clean_raw=raw,
        noisy=z.clone(),
        noise=zeros,
        timesteps=torch.full((source_count,), 10, dtype=torch.long),
        noise_ids=torch.full((source_count,), 99000, dtype=torch.long),
        source_ids=ids,
    )
    bank.validate()
    active = torch.ones(DEFAULT_TF, STATE_DIM, dtype=torch.bool)
    mean = torch.zeros(DEFAULT_TF, STATE_DIM)
    scale = torch.ones(DEFAULT_TF, STATE_DIM)
    return bank, [0, 0, 1, 1], active, mean, scale


def monotonic_physical(raw: np.ndarray, _contract: object) -> dict:
    value = np.asarray(raw)
    xy = value[..., :CABLE_DIM].reshape(*value.shape[:-1], N_BEADS, 2)
    monotonic = np.all(np.diff(xy[..., 0], axis=-1) >= -1.0e-8, axis=(-1, -2))
    rate = float(np.mean(monotonic.astype(np.float64)))
    return {"sample_validity_rate": rate, "sample_valid_mask": monotonic}


def contracts(
    *,
    exact: bool = False,
    branch: bool = True,
    physical: bool = True,
    final_cable: bool = False,
    earlier: bool = False,
) -> dict:
    return {
        "historical_exact_reconstruction_pass": exact,
        "branch_transport_pass": branch,
        "one_step_physical_pass": physical,
        "separated_branch_physical_pass": branch and physical,
        "historical_composite_pass": exact and branch,
        "final_horizon_cable_geometry_pass": final_cable,
        "all_earlier_horizon_cable_geometry_pass": earlier,
    }


def separation_payload(
    *,
    exact: bool = False,
    branch: bool = True,
    z_pass: bool = False,
    cable_pass: bool = True,
    robot_fraction: float = 0.8,
    horizons: tuple[bool, bool, bool, bool] = (False, False, False, True),
    source_fail: bool = False,
) -> dict:
    return {
        "schema": GATE_SEPARATION_SCHEMA,
        "contracts": contracts(
            exact=exact,
            branch=branch,
            final_cable=horizons[-1],
            earlier=all(horizons[:-1]),
        ),
        "full_reconstruction_components": {
            "z_state_gate_pass": z_pass,
            "cable_geometry_gate_pass": cable_pass,
        },
        "state_group_z_metrics": {
            "robot_proxy_error_fraction": robot_fraction,
            "cable_error_fraction": 1.0 - robot_fraction,
        },
        "by_horizon": {
            str(index): {"cable_geometry_gate_pass": value}
            for index, value in enumerate(horizons)
        },
        "by_source": {
            "groups": {
                "0": {"gate_pass": not source_fail},
                "1": {"gate_pass": True},
            }
        },
    }


def variant(value: dict, *, training: bool = True, reproduction: bool = True) -> dict:
    return {
        "training_completed": training,
        "r252_one_step_reproduction": {"pass": reproduction},
        "gate_separation": value,
    }


def report_with(value: dict) -> dict:
    return {
        "variants": {
            name: variant(deepcopy(value)) for name in DIAGNOSTIC_OBJECTIVE_NAMES
        },
        "synthetic_controls": {"contract": {"pass": True}},
    }


def test_dimension_contract() -> None:
    assert DEFAULT_TF == 4
    assert STATE_DIM == 87
    assert CABLE_DIM == 48


def test_spec_validation() -> None:
    GateSeparationSpec().validate()
    with pytest.raises(ValueError):
        GateSeparationSpec(branch_own_fraction_min=1.1).validate()
    with pytest.raises(ValueError):
        GateSeparationSpec(cable_translation_m=-1).validate()


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("z_mse", {"z": 0.1}),
        ("ordered_rmse_p95", {"ordered": 0.1}),
        ("segment_relative_error_p95", {"segment": 0.5}),
        ("chain_relative_error_p95", {"chain": 0.5}),
    ],
)
def test_reconstruction_component_failure(name: str, kwargs: dict) -> None:
    value = reconstruction_gate_components(metric_payload(**kwargs), gate())
    assert name in value["failed_components"]
    assert value["historical_exact_reconstruction_pass"] is False


def test_reconstruction_components_exact_pass() -> None:
    value = reconstruction_gate_components(metric_payload(), gate())
    assert value["historical_exact_reconstruction_pass"] is True
    assert value["cable_geometry_gate_pass"] is True


def test_group_z_decomposition_splits_cable_and_robot() -> None:
    target = np.zeros((2, DEFAULT_TF, STATE_DIM), dtype=np.float32)
    predicted = target.copy()
    predicted[..., CABLE_DIM:] = 2.0
    active = np.ones((DEFAULT_TF, STATE_DIM), dtype=bool)
    value = _group_z_metrics(predicted, target, active)
    assert value["cable_z_mse"] == 0.0
    assert value["robot_proxy_z_mse"] == 4.0
    assert value["robot_proxy_error_fraction"] == pytest.approx(1.0)


def test_single_horizon_exact_geometry_passes() -> None:
    target = np.zeros((2, DEFAULT_TF, STATE_DIM), dtype=np.float32)
    for bead in range(N_BEADS):
        target[..., 2 * bead] = bead * 0.01
    value = _single_horizon_metrics(
        predicted_z=target,
        target_z=target,
        predicted_raw=target,
        target_raw=target,
        active_mask=np.ones((DEFAULT_TF, STATE_DIM), dtype=bool),
        horizon=3,
        gate=gate(),
    )
    assert value["full_gate_pass"] is True
    assert value["cable_geometry_gate_pass"] is True


def test_prediction_contract_separates_robot_error_from_branch() -> None:
    bank, pair_ids, active, mean, scale = make_bank()
    predicted = bank.clean_z.clone()
    predicted[..., CABLE_DIM:] += 1.0
    value = evaluate_prediction_contract(
        predicted_z=predicted,
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=active,
        future_mean=mean,
        future_scale=scale,
        physical_contract=None,
        physical_evaluator=monotonic_physical,
    )
    assert value["contracts"]["branch_transport_pass"] is True
    assert value["full_reconstruction_components"]["cable_geometry_gate_pass"] is True
    assert value["contracts"]["historical_exact_reconstruction_pass"] is False
    assert value["contracts"]["ordered_topology_pass"] is True
    assert value["state_group_z_metrics"]["robot_proxy_error_fraction"] > 0.99


def test_synthetic_controls_have_expected_shapes() -> None:
    bank, pair_ids, active, mean, scale = make_bank()
    controls = synthetic_prediction_controls(
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=active,
        future_mean=mean,
        future_scale=scale,
    )
    assert set(controls) == {
        "exact_oracle",
        "branch_swap",
        "pair_mean",
        "collapse_first_branch",
        "robot_proxy_offset",
        "cable_translation",
        "bead_order_reversal",
    }
    assert all(value.shape == bank.clean_z.shape for value in controls.values())



def test_bead_reversal_fails_frozen_order_topology_contract() -> None:
    bank, pair_ids, active, mean, scale = make_bank()
    controls = synthetic_prediction_controls(
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=active,
        future_mean=mean,
        future_scale=scale,
    )
    value = evaluate_prediction_contract(
        predicted_z=controls["bead_order_reversal"],
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=active,
        future_mean=mean,
        future_scale=scale,
        physical_contract=None,
        physical_evaluator=monotonic_physical,
    )
    assert value["ordered_topology"]["nearest_inversion"]["p95"] > 0.9
    assert value["contracts"]["ordered_topology_pass"] is False

def test_synthetic_control_contract_passes_real_branch_path() -> None:
    bank, pair_ids, active, mean, scale = make_bank()
    value = evaluate_synthetic_controls(
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=active,
        future_mean=mean,
        future_scale=scale,
        physical_contract=None,
        physical_evaluator=monotonic_physical,
    )
    assert value["contract"]["pass"] is True
    assert value["contract"]["branch_swap_rejected"] is True
    assert value["contract"]["robot_proxy_orthogonality_demonstrated"] is True
    assert value["contract"]["topology_corruption_rejected"] is True


def test_compare_reconstruction_result_uses_canonical_nested_metrics() -> None:
    observed = {"full_reconstruction_metrics": metric_payload(ordered=0.005)}
    training = {
        "evaluations": {
            "true": {"aggregate": {"metrics": metric_payload(ordered=0.005)}}
        }
    }
    value = compare_reconstruction_to_training_result(
        observed=observed, training_result=training
    )
    assert value["pass"] is True


def test_compare_reconstruction_result_detects_mismatch() -> None:
    observed = {"full_reconstruction_metrics": metric_payload(z=0.2)}
    training = {
        "evaluations": {"true": {"aggregate": {"metrics": metric_payload()}}}
    }
    value = compare_reconstruction_to_training_result(
        observed=observed, training_result=training
    )
    assert value["pass"] is False


def test_gate_payload_unwraps_runtime_variant() -> None:
    payload = separation_payload()
    assert _gate_separation_payload({"gate_separation": payload}) is payload
    assert _gate_separation_payload(payload) is payload
    with pytest.raises(ValueError):
        _gate_separation_payload({"gate_separation": 1})


def test_variant_mechanisms_read_nested_gate_separation() -> None:
    value = variant(separation_payload())
    mechanisms = variant_supported_mechanisms(value)
    assert "exact_reconstruction_vs_branch_identity_separation" in mechanisms
    assert "robot_proxy_reconstruction_conflation" in mechanisms
    assert "intermediate_horizon_fidelity_gap" in mechanisms


def test_classifier_matrix_incomplete() -> None:
    value = classify_gate_separation({"variants": {}, "synthetic_controls": {}})
    assert value["root_cause"] == "phase314b_r253_diagnostic_matrix_incomplete"


def test_classifier_training_failure_precedes_science() -> None:
    payload = report_with(separation_payload())
    payload["variants"][DIAGNOSTIC_OBJECTIVE_NAMES[0]]["training_completed"] = False
    value = classify_gate_separation(payload)
    assert value["root_cause"] == "phase314b_r253_training_reproduction_failed"


def test_classifier_reproduction_failure_precedes_science() -> None:
    payload = report_with(separation_payload())
    payload["variants"][DIAGNOSTIC_OBJECTIVE_NAMES[0]]["r252_one_step_reproduction"] = {"pass": False}
    value = classify_gate_separation(payload)
    assert value["root_cause"] == "phase314b_r253_r252_one_step_reproduction_failed"


def test_classifier_control_failure_precedes_science() -> None:
    payload = report_with(separation_payload())
    payload["synthetic_controls"]["contract"]["pass"] = False
    value = classify_gate_separation(payload)
    assert value["root_cause"] == "phase314b_r253_gate_separation_controls_failed"


def test_classifier_requires_branch_reproduction() -> None:
    payload = report_with(separation_payload(branch=False))
    value = classify_gate_separation(payload)
    assert value["root_cause"] == "phase314b_r253_branch_transport_not_reproduced"


def test_classifier_robot_proxy_root() -> None:
    value = classify_gate_separation(report_with(separation_payload()))
    assert value["root_cause"] == "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
    assert value["train_only_recommendation"] is None


def test_classifier_intermediate_horizon_root() -> None:
    separation = separation_payload(z_pass=True, robot_fraction=0.1)
    value = classify_gate_separation(report_with(separation))
    assert value["root_cause"] == "phase314b_r253_intermediate_horizon_fidelity_gap_supported"


def test_classifier_exact_cable_tail_root() -> None:
    separation = separation_payload(
        z_pass=True,
        cable_pass=False,
        robot_fraction=0.1,
        horizons=(False, False, False, False),
    )
    value = classify_gate_separation(report_with(separation))
    assert value["root_cause"] == "phase314b_r253_exact_cable_fidelity_tail_gap_supported"


def test_classifier_source_tail_root() -> None:
    separation = separation_payload(
        exact=True,
        z_pass=True,
        cable_pass=True,
        robot_fraction=0.1,
        horizons=(True, True, True, True),
        source_fail=True,
    )
    value = classify_gate_separation(report_with(separation))
    assert value["root_cause"] == "phase314b_r253_all_source_reconstruction_tail_failure_supported"


def test_classifier_generic_gate_separation_root() -> None:
    separation = separation_payload(
        z_pass=True,
        cable_pass=True,
        robot_fraction=0.1,
        horizons=(True, True, True, True),
    )
    value = classify_gate_separation(report_with(separation))
    assert value["root_cause"] == "phase314b_r253_gate_separation_contract_supported"


def test_strip_runtime_objects_removes_private_keys() -> None:
    value = strip_runtime_objects({"_model": object(), "a": np.array([1, 2]), "b": (3, 4)})
    assert value == {"a": [1, 2], "b": [3, 4]}


def test_strip_runtime_objects_rejects_vector_tensor() -> None:
    with pytest.raises(TypeError):
        strip_runtime_objects(torch.ones(2))
    assert strip_runtime_objects(torch.tensor(2.0)) == 2.0


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
