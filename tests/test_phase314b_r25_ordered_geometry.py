from __future__ import annotations

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r25_ordered_geometry import (
    COMMON_PAIRED_TRAINING_SEED,
    COMMON_REVERSE_SEED,
    COMMON_UNIQUE_TRAINING_SEED,
    GEOMETRY_OBJECTIVES,
    GeometryObjectiveConfig,
    GeometryScales,
    _top_fraction_mean,
    classify_pilot,
    combined_geometry_loss,
    compare_geometry_to_control,
    fit_geometry_scales,
    geometry_timestep_mask,
    loss_balance_factor,
    ordered_geometry_loss_components,
    paired_full_reverse_branch_support,
    torch_contract_from_payload,
)


def straight_future(batch: int = 1, spacing: float = 0.01) -> torch.Tensor:
    future = torch.zeros(batch, 4, 87, dtype=torch.float32)
    x = torch.arange(24, dtype=torch.float32) * float(spacing)
    xy = torch.stack((x, torch.zeros_like(x)), dim=-1)
    future[..., :48] = xy.reshape(1, 1, 48)
    return future


def contract_payload(spacing: float = 0.01) -> dict:
    return {
        "coordinate_lower": [-1.0, -1.0],
        "coordinate_upper": [1.0, 1.0],
        "segment_center": np.full((4, 23), spacing, dtype=np.float32).tolist(),
        "segment_scale": np.full((4, 23), 0.001, dtype=np.float32).tolist(),
        "segment_score_threshold": 5.0,
        "chain_center": np.full((4,), spacing * 23, dtype=np.float32).tolist(),
        "chain_scale": np.full((4,), 0.01, dtype=np.float32).tolist(),
        "chain_score_threshold": 5.0,
    }


def scales() -> GeometryScales:
    return GeometryScales(
        ordered_xy_m=0.01,
        edge_vector_m=0.01,
        segment_floor_m=0.005,
        chain_floor_m=0.05,
    )


def test_geometry_objective_matrix_is_fixed_and_valid() -> None:
    names = [value.name for value in GEOMETRY_OBJECTIVES]
    assert names == [
        "v_only_frozen_control",
        "ordered_mean_raw",
        "ordered_cvar_raw",
        "ordered_cvar_contract",
    ]
    assert not GEOMETRY_OBJECTIVES[0].selectable
    assert all(value.selectable for value in GEOMETRY_OBJECTIVES[1:])
    for value in GEOMETRY_OBJECTIVES:
        value.validate()


def test_invalid_selectable_zero_geometry_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be active"):
        GeometryObjectiveConfig(
            name="invalid",
            outer_weight=0.1,
            mean_weight=0.0,
            cvar_weight=0.0,
            contract_weight=0.0,
        ).validate()


def test_common_random_stream_constants_are_distinct() -> None:
    assert len(
        {
            COMMON_UNIQUE_TRAINING_SEED,
            COMMON_PAIRED_TRAINING_SEED,
            COMMON_REVERSE_SEED,
        }
    ) == 3


def test_fit_geometry_scales_uses_coordinate_gate_not_chain_length() -> None:
    future = straight_future(batch=8).numpy()
    result = fit_geometry_scales(future)
    assert result.ordered_xy_m == pytest.approx(0.01)
    assert result.edge_vector_m == pytest.approx(0.01, abs=1e-6)
    assert result.ordered_xy_m != pytest.approx(0.23)


def test_tail_loss_detects_sparse_outlier_more_than_mean() -> None:
    values = torch.zeros(2, 100)
    values[:, -1] = 10.0
    tail = _top_fraction_mean(values, 0.01)
    mean = values.mean(dim=1)
    assert torch.all(tail > mean)
    assert torch.allclose(tail, torch.full_like(tail, 10.0))


def test_geometry_components_zero_on_matching_contract_reference() -> None:
    target = straight_future(batch=2)
    contract = torch_contract_from_payload(contract_payload())
    result = ordered_geometry_loss_components(
        predicted_raw=target.clone(),
        target_raw=target,
        scales=scales(),
        contract=contract,
        tail_fraction=0.10,
    )
    assert torch.max(result["segment_contract_excess"]).item() == pytest.approx(0.0)
    assert torch.max(result["chain_contract_excess"]).item() == pytest.approx(0.0)
    assert torch.max(result["coordinate_contract_excess"]).item() == pytest.approx(0.0)
    assert torch.max(result["mean"]).item() < 1e-6


def test_geometry_components_penalize_one_extreme_edge() -> None:
    target = straight_future(batch=1)
    prediction = target.clone()
    xy = prediction[..., :48].reshape(1, 4, 24, 2)
    xy[:, :, 12:, 0] += 0.10
    contract = torch_contract_from_payload(contract_payload())
    result = ordered_geometry_loss_components(
        predicted_raw=prediction,
        target_raw=target,
        scales=scales(),
        contract=contract,
        tail_fraction=0.10,
    )
    assert result["segment_relative_max"].item() > 5.0
    assert result["segment_contract_excess"].item() > 0.0
    assert result["cvar"].item() > result["mean"].item()


def test_geometry_timestep_mask_limits_paired_target_loss() -> None:
    timesteps = torch.tensor([0, 10, 50, 51, 99])
    assert geometry_timestep_mask(timesteps, 50).tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_combined_geometry_loss_respects_sample_mask() -> None:
    components = {
        "mean": torch.tensor([1.0, 10.0]),
        "cvar": torch.tensor([2.0, 20.0]),
        "contract": torch.tensor([3.0, 30.0]),
    }
    config = GeometryObjectiveConfig(
        name="test",
        outer_weight=0.1,
        mean_weight=1.0,
        cvar_weight=1.0,
        contract_weight=1.0,
    )
    loss = combined_geometry_loss(
        components=components,
        config=config,
        sample_mask=torch.tensor([True, False]),
    )
    assert loss.item() == pytest.approx(6.0)


def test_loss_balance_factor_is_detached_and_bounded() -> None:
    config = GeometryObjectiveConfig(
        name="test",
        outer_weight=0.1,
        mean_weight=1.0,
        cvar_weight=0.0,
        contract_weight=0.0,
        loss_balance_min=0.05,
        loss_balance_max=20.0,
    )
    value = loss_balance_factor(
        torch.tensor(2.0, requires_grad=True),
        torch.tensor(0.5, requires_grad=True),
        config,
    )
    assert value.item() == pytest.approx(4.0)
    assert not value.requires_grad


def test_paired_branch_support_passes_when_pool_contains_both_targets() -> None:
    target_a = straight_future(batch=1).numpy()[0]
    target_b = target_a.copy()
    target_b[..., 1:48:2] += 0.02
    targets = np.stack((target_a, target_b), axis=0)
    pool = np.stack((target_a, target_b), axis=0)[:, None]
    result = paired_full_reverse_branch_support(
        pool_raw=pool,
        paired_target_raw=targets,
        reference_threshold=0.02,
    )
    assert result["pass"]
    assert result["both_branch_support_rate"] == pytest.approx(1.0)
    assert result["two_branch_occupancy_rate"] == pytest.approx(1.0)


def test_compare_geometry_to_control_requires_tail_gain_without_order_loss() -> None:
    control = {
        "reverse_metrics": {
            "calibrated": {
                "segment_score_p95": 10.0,
                "sample_validity_rate": 0.20,
            },
            "best_ordered_rmse_mean": 0.05,
        }
    }
    candidate = {
        "reverse_metrics": {
            "calibrated": {
                "segment_score_p95": 8.0,
                "sample_validity_rate": 0.25,
            },
            "best_ordered_rmse_mean": 0.052,
        }
    }
    result = compare_geometry_to_control(candidate=candidate, control=control)
    assert result["tail_improved"]
    assert result["ordered_preserved"]
    assert result["pass"]


def test_classifier_never_recommends_control() -> None:
    report = {
        "unique_free_variants": {
            "v_only_frozen_control": {"pass": True, "geometry_gradient_gate": True},
            "ordered_cvar_contract": {"pass": True, "geometry_gradient_gate": True},
        },
        "paired_variants": {
            "v_only_frozen_control": {
                "one_step_pass": True,
                "reverse_metrics": {
                    "calibrated": {
                        "query_has_valid_candidate_rate": 1.0,
                        "sample_validity_rate": 0.2,
                    },
                    "best_ordered_rmse_mean": 0.05,
                },
                "full_reverse_branch_support": {"pass": True},
                "comparison_to_control": {"pass": True},
            },
            "ordered_cvar_contract": {
                "one_step_pass": True,
                "reverse_metrics": {
                    "calibrated": {
                        "query_has_valid_candidate_rate": 1.0,
                        "sample_validity_rate": 0.9,
                    },
                    "best_ordered_rmse_mean": 0.03,
                },
                "full_reverse_branch_support": {
                    "pass": True,
                    "both_branch_support_rate": 1.0,
                },
                "comparison_to_control": {"pass": True},
            },
        },
    }
    result = classify_pilot(report)
    assert result["train_only_recommendation"] == "ordered_cvar_contract"
    assert result["train_only_recommendation"] != "v_only_frozen_control"
