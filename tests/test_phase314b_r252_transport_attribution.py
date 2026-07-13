"""Tests for Phase3.14b-r2.5.2 paired transport attribution."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest
import torch

import ccda_phase3.phase314b_r252_transport_attribution as r252


def _mechanism_variant(
    *,
    selectable: bool = True,
    training_completed: bool = True,
    pipeline_pass: bool = True,
    denoising_pass: bool = False,
    branch_pass: bool = False,
    final_gradient_pass: bool = True,
    poor_timesteps: int = 0,
    endpoint_branch_pass: bool = False,
    endpoint_valid_pass: bool = False,
    reproduction_pass: bool = True,
) -> Dict[str, Any]:
    by_timestep: Dict[str, Any] = {}
    for index, timestep in enumerate(r252.PAIR_TIMESTEPS):
        poor = index < poor_timesteps
        by_timestep[str(timestep)] = {
            "forward_signal_contract_pass": True,
            "model_separation_ratio": {"p50": 0.25 if poor else 0.75},
            "residual_delta_cosine": {"p50": 0.25 if poor else 0.75},
        }
    return {
        "geometry_objective_selectable": selectable,
        "training_completed": training_completed,
        "one_step_attribution": {
            "pipeline_controls": {"pass": pipeline_pass},
            "by_timestep": by_timestep,
        },
        "one_step_gate_decomposition": {
            "denoising_result_pass": denoising_pass,
            "branch_audit_pass": branch_pass,
            "composite_one_step_pass": bool(denoising_pass and branch_pass),
        },
        "per_timestep_gradient": {
            "final": {"10": {}, "25": {}, "50": {}},
            "final_all_timesteps_pass": final_gradient_pass,
        },
        "trajectory": {
            "attribution": {
                "endpoint_branch_support_pass": endpoint_branch_pass,
                "endpoint_valid_query_pass": endpoint_valid_pass,
            }
        },
        "r251_endpoint_reproduction": {"pass": reproduction_pass},
    }


def _report(**overrides: Dict[str, Any]) -> Dict[str, Any]:
    variants = {
        name: _mechanism_variant(selectable=(name != "v_only_frozen_control"))
        for name in r252.DIAGNOSTIC_OBJECTIVE_NAMES
    }
    variants.update(overrides)
    return {"variants": variants}


def test_diagnostic_objectives_have_fixed_order_and_identity() -> None:
    objectives = r252.diagnostic_objectives()
    assert tuple(value.name for value in objectives) == r252.DIAGNOSTIC_OBJECTIVE_NAMES
    assert objectives[0].selectable is False
    assert all(value.selectable for value in objectives[1:])


def test_attribution_gate_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        replace(r252.AttributionGate(), legacy_own_fraction=1.1).validate()
    with pytest.raises(ValueError):
        replace(r252.AttributionGate(), decode_tolerance=float("nan")).validate()
    with pytest.raises(ValueError):
        replace(r252.AttributionGate(), reverse_valid_query_min=-0.1).validate()


def test_stats_reports_expected_quantiles() -> None:
    result = r252._stats([0.0, 1.0, 2.0, 3.0])
    assert result["min"] == 0.0
    assert result["max"] == 3.0
    assert result["mean"] == 1.5
    assert result["p50"] == 1.5


@pytest.mark.parametrize("values", [[], [1.0, float("nan")], [float("inf")]])
def test_stats_rejects_empty_or_nonfinite(values) -> None:
    with pytest.raises(ValueError):
        r252._stats(values)


def test_cosine_and_rmse() -> None:
    assert r252._cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert r252._cosine(np.zeros(2), np.ones(2)) == 0.0
    assert r252._rmse(np.array([3.0, 4.0])) == pytest.approx(np.sqrt(12.5))


def _record(timestep: int, *, own: bool = True, separation: float = 0.75) -> Dict[str, Any]:
    return {
        "timestep": timestep,
        "own_target_closer": [own, own],
        "normalized_own_margin": [0.2, 0.3],
        "model_separation_ratio": separation,
        "model_delta_cosine": 0.8,
        "noisy_separation_ratio": 0.6,
        "expected_forward_alpha": 0.6,
        "residual_delta_gain": 0.7,
        "residual_delta_cosine": 0.75,
        "required_residual_delta_rms": 2.0,
        "predicted_residual_delta_rms": 1.4,
    }


def test_group_records_by_timestep_preserves_gate_components() -> None:
    records = [_record(timestep) for timestep in r252.PAIR_TIMESTEPS for _ in range(2)]
    result = r252._group_records_by_timestep(records, gate=r252.AttributionGate())
    assert set(result) == {"10", "25", "50"}
    assert all(value["legacy_gate_pass"] for value in result.values())
    assert all(value["forward_signal_contract_pass"] for value in result.values())
    assert result["10"]["comparison_count"] == 2


def test_group_records_requires_every_timestep() -> None:
    with pytest.raises(RuntimeError, match="missing paired transport records"):
        r252._group_records_by_timestep([_record(10)], gate=r252.AttributionGate())


def test_repeat_balanced_rows_cycles_sources() -> None:
    value = torch.tensor([[1.0], [2.0], [3.0]])
    result = r252._repeat_balanced_rows(value, 8)
    assert result.reshape(-1).tolist() == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]


def test_fixed_timestep_control_has_zero_gradient_ratio() -> None:
    objective = r252.diagnostic_objectives()[0]
    result = r252._fixed_timestep_ratio_samples(
        model=object(),
        scheduler=object(),
        condition_z=torch.zeros(2, 3),
        clean_z=torch.zeros(2, 4, 87),
        clean_raw=torch.zeros(2, 4, 87),
        active_mask=torch.ones(4, 87, dtype=torch.bool),
        future_mean=torch.zeros(4, 87),
        future_scale=torch.ones(4, 87),
        torch_contract=object(),
        scales=object(),
        objective=objective,
        multiplier=0.0,
        timestep=10,
        batch_size=2,
        noise_seeds=(1, 2),
    )
    assert result["tracking"]["pass"] is True
    assert result["weighted_ratios"] == []


def test_fixed_timestep_uses_actual_gradient_norms(monkeypatch: pytest.MonkeyPatch) -> None:
    objective = r252.diagnostic_objectives()[1]
    v_loss = torch.tensor(1.0, requires_grad=True)
    geometry_loss = torch.tensor(2.0, requires_grad=True)

    class FakeModel:
        def residual_parameters(self):
            return [torch.tensor(1.0, requires_grad=True)]

    monkeypatch.setattr(
        r252.r251,
        "_forward_losses",
        lambda **kwargs: {"v_loss": v_loss, "geometry_loss": geometry_loss},
    )
    monkeypatch.setattr(
        r252.r251,
        "_gradient_norm",
        lambda loss, parameters, retain_graph: 2.0 if loss is v_loss else 4.0,
    )
    monkeypatch.setattr(
        r252.r251,
        "gradient_tracking_gate",
        lambda target_ratio, observed_ratios, spec: {
            "target_gradient_ratio": target_ratio,
            "observed_ratio_median": float(np.median(observed_ratios)),
            "observed_ratio_p95": float(np.percentile(observed_ratios, 95)),
            "pass": True,
        },
    )
    result = r252._fixed_timestep_ratio_samples(
        model=FakeModel(),
        scheduler=object(),
        condition_z=torch.zeros(2, 3),
        clean_z=torch.zeros(2, 4, 87),
        clean_raw=torch.zeros(2, 4, 87),
        active_mask=torch.ones(4, 87, dtype=torch.bool),
        future_mean=torch.zeros(4, 87),
        future_scale=torch.ones(4, 87),
        torch_contract=object(),
        scales=object(),
        objective=objective,
        multiplier=0.25,
        timestep=25,
        batch_size=2,
        noise_seeds=(1, 2),
    )
    assert result["unweighted_ratios"] == [2.0, 2.0]
    assert result["weighted_ratios"] == [0.5, 0.5]
    assert result["tracking"]["pass"] is True


def test_reverse_trajectory_uses_matched_initial_noise_and_captures_x0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, sample, timesteps, condition, detach_prior_for_v):
            return torch.zeros_like(sample)

    class FakeScheduler:
        def __init__(self):
            self.timesteps = torch.tensor([], dtype=torch.long)

        def set_timesteps(self, count, device=None):
            assert count == 4
            self.timesteps = torch.tensor([3, 2, 1, 0], device=device)

        def step(self, output, timestep, sample, generator=None):
            return SimpleNamespace(prev_sample=sample * 0.5)

    monkeypatch.setattr(
        r252,
        "predict_original_sample",
        lambda scheduler, config, sample, model_output, timesteps: sample * 0.25,
    )
    result = r252.reverse_predicted_x0_trajectory(
        model=FakeModel(),
        scheduler=FakeScheduler(),
        condition_z=torch.zeros(3, 5),
        active_mask=torch.ones(4, 87, dtype=torch.bool),
        sample_count=2,
        inference_steps=4,
        seed=7,
        checkpoint_timesteps=(3, 2, 1, 0),
    )
    assert result["schema"] == r252.TRAJECTORY_SCHEMA
    assert result["endpoint_sample"].shape == (2, 3, 4, 87)
    assert set(result["checkpoint_predicted_x0"]) == {"3", "2", "1", "0"}
    first = result["checkpoint_predicted_x0"]["3"]
    assert torch.equal(first[:, 0], first[:, 1])
    assert result["checkpoint_iterations"] == {"3": 1, "2": 2, "1": 3, "0": 4}


def test_reverse_trajectory_rejects_missing_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeModel:
        def eval(self):
            return self

        def __call__(self, sample, timesteps, condition, detach_prior_for_v):
            return torch.zeros_like(sample)

    class FakeScheduler:
        timesteps = torch.tensor([], dtype=torch.long)

        def set_timesteps(self, count, device=None):
            self.timesteps = torch.tensor([2, 1, 0], device=device)

        def step(self, output, timestep, sample):
            return SimpleNamespace(prev_sample=sample)

    monkeypatch.setattr(
        r252,
        "predict_original_sample",
        lambda scheduler, config, sample, model_output, timesteps: sample,
    )
    with pytest.raises(RuntimeError, match="missing trajectory timesteps"):
        r252.reverse_predicted_x0_trajectory(
            model=FakeModel(),
            scheduler=FakeScheduler(),
            condition_z=torch.zeros(1, 2),
            active_mask=torch.ones(4, 87, dtype=torch.bool),
            sample_count=1,
            inference_steps=3,
            checkpoint_timesteps=(3, 2, 1, 0),
        )


def test_compare_endpoint_to_r251_accepts_matching_metrics() -> None:
    observed = {
        "metrics": {
            "sample_validity_rate": 1.0,
            "query_has_valid_candidate_rate": 1.0,
            "best_ordered_rmse_mean": 0.001,
            "nearest_inversion_p95": 0.02,
        },
        "branch_support": {"both_branch_support_rate": 1.0},
    }
    expected = {
        "reverse_sample_validity_rate": 1.0,
        "reverse_query_has_valid_candidate_rate": 1.0,
        "reverse_best_ordered_rmse_mean": 0.001,
        "reverse_nearest_inversion_p95": 0.02,
        "both_branch_support_rate": 1.0,
    }
    assert r252.compare_endpoint_to_r251(observed=observed, expected=expected)["pass"]


def test_compare_endpoint_to_r251_rejects_large_drift() -> None:
    observed = {
        "metrics": {
            "sample_validity_rate": 0.5,
            "query_has_valid_candidate_rate": 1.0,
            "best_ordered_rmse_mean": 0.002,
            "nearest_inversion_p95": 0.02,
        },
        "branch_support": {"both_branch_support_rate": 1.0},
    }
    expected = {
        "reverse_sample_validity_rate": 1.0,
        "reverse_query_has_valid_candidate_rate": 1.0,
        "reverse_best_ordered_rmse_mean": 0.001,
        "reverse_nearest_inversion_p95": 0.02,
        "both_branch_support_rate": 1.0,
    }
    assert not r252.compare_endpoint_to_r251(observed=observed, expected=expected)["pass"]


def test_strip_runtime_objects_removes_private_values() -> None:
    value = {
        "keep": np.asarray([1.0, 2.0]),
        "scalar": torch.tensor(3.0),
        "_model": object(),
        "nested": {"_tensor": torch.ones(2), "value": (1, 2)},
    }
    result = r252.strip_runtime_objects(value)
    assert result == {"keep": [1.0, 2.0], "scalar": 3.0, "nested": {"value": [1, 2]}}


def test_strip_runtime_objects_rejects_public_tensor_array() -> None:
    with pytest.raises(TypeError, match="runtime tensor"):
        r252.strip_runtime_objects({"tensor": torch.ones(2)})


def test_classifier_requires_complete_matrix() -> None:
    result = r252.classify_attribution({"variants": {}})
    assert result["root_cause"] == "phase314b_r252_diagnostic_matrix_incomplete"


def test_classifier_prioritizes_training_failure() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"]["training_completed"] = False
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_training_reproduction_failed"


def test_classifier_prioritizes_metric_pipeline_failure() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"]["one_step_attribution"]["pipeline_controls"]["pass"] = False
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_metric_pipeline_contract_failed"


def test_classifier_prioritizes_endpoint_reproduction_failure() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"]["r251_endpoint_reproduction"]["pass"] = False
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_r251_reverse_reproduction_failed"


def test_classifier_supports_selectable_composite_gate_conflation() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"] = _mechanism_variant(
        selectable=True,
        denoising_pass=False,
        branch_pass=True,
    )
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_composite_one_step_gate_conflation_supported"


def test_control_alone_cannot_trigger_composite_gate_root() -> None:
    report = _report()
    report["variants"]["v_only_frozen_control"] = _mechanism_variant(
        selectable=False,
        denoising_pass=False,
        branch_pass=True,
    )
    result = r252.classify_attribution(report)
    assert result["root_cause"] != "phase314b_r252_composite_one_step_gate_conflation_supported"


def test_classifier_requires_all_selectable_gradients_bad_for_gradient_root() -> None:
    report = _report()
    for name in r252.DIAGNOSTIC_OBJECTIVE_NAMES[1:]:
        report["variants"][name] = _mechanism_variant(
            selectable=True,
            final_gradient_pass=False,
        )
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_timestep_gradient_miscalibration_supported"


def test_single_gradient_failure_is_not_sufficient_primary_root() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"] = _mechanism_variant(
        selectable=True,
        final_gradient_pass=False,
    )
    report["variants"]["ordered_cvar_contract_g010"] = _mechanism_variant(
        selectable=True,
        final_gradient_pass=True,
        poor_timesteps=2,
    )
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_sigma_scaled_residual_gain_deficiency_supported"


def test_classifier_supports_sigma_scaled_residual_deficiency() -> None:
    report = _report()
    report["variants"]["ordered_mean_raw_g100"] = _mechanism_variant(
        selectable=True,
        final_gradient_pass=True,
        poor_timesteps=2,
    )
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_sigma_scaled_residual_gain_deficiency_supported"


def test_classifier_supports_selectable_reverse_trajectory_recovery() -> None:
    report = _report()
    report["variants"]["ordered_cvar_contract_g010"] = _mechanism_variant(
        selectable=True,
        branch_pass=False,
        endpoint_branch_pass=True,
        endpoint_valid_pass=True,
    )
    result = r252.classify_attribution(report)
    assert result["root_cause"] == "phase314b_r252_reverse_trajectory_recovery_supported"


def test_classifier_falls_back_to_confirmed_transport_failure() -> None:
    result = r252.classify_attribution(_report())
    assert result["root_cause"] == "phase314b_r252_paired_transport_failure_confirmed"
    assert result["train_only_recommendation"] is None


def test_paired_one_step_attribution_real_shape_path(monkeypatch: pytest.MonkeyPatch) -> None:
    source_count = 4
    pair_ids = [0, 0, 1, 1]
    noise_ids = (0, 1)
    rows = []
    for source in range(source_count):
        for timestep in r252.PAIR_TIMESTEPS:
            for noise_id in noise_ids:
                rows.append((source, timestep, noise_id))
    count = len(rows)
    clean = torch.zeros(count, 4, 87)
    condition = torch.zeros(count, 3)
    noise = torch.zeros_like(clean)
    source_ids = torch.tensor([row[0] for row in rows], dtype=torch.long)
    timesteps = torch.tensor([row[1] for row in rows], dtype=torch.long)
    tuple_noise_ids = torch.tensor([row[2] for row in rows], dtype=torch.long)
    for index, (source, timestep, noise_id) in enumerate(rows):
        pair = pair_ids[source]
        branch = source % 2
        clean[index, :, :48] = float(pair * 4 + branch + 1)
        condition[index] = torch.tensor([float(pair), 0.0, 1.0])
        noise[index] = float(noise_id + timestep / 100.0)
    noisy = 0.6 * clean + noise

    bank = SimpleNamespace(
        condition_z=condition,
        clean_z=clean,
        clean_raw=clean,
        noise=noise,
        noisy=noisy,
        source_ids=source_ids,
        timesteps=timesteps,
        noise_ids=tuple_noise_ids,
        validate=lambda: None,
    )

    class FakeModel:
        def eval(self):
            return self

        def __call__(self, sample, timesteps, condition, detach_prior_for_v):
            return clean

        def predict_base_x0(self, condition):
            return torch.zeros_like(clean)

        def predict_residual_v(self, sample, timesteps, condition):
            return -clean

        def coefficients(self, timesteps, dtype):
            shape = (timesteps.shape[0], 1, 1)
            return torch.full(shape, 0.6, dtype=dtype), torch.ones(shape, dtype=dtype)

    monkeypatch.setattr(
        r252,
        "predict_original_sample",
        lambda scheduler, config, sample, model_output, timesteps: model_output,
    )
    monkeypatch.setattr(
        r252,
        "training_target",
        lambda scheduler, config, clean_sample, noise, timesteps: clean_sample,
    )
    monkeypatch.setattr(
        r252,
        "paired_branch_audit",
        lambda **kwargs: {
            "own_target_closer_fraction": 1.0,
            "separation_ratio": {"p50": 1.0},
            "branch_delta_cosine": {"p50": 1.0},
            "pass": True,
        },
    )
    result = r252.paired_one_step_attribution(
        model=FakeModel(),
        scheduler=object(),
        bank=bank,
        source_pair_ids=pair_ids,
        active_mask=torch.ones(4, 87, dtype=torch.bool),
        future_mean=torch.zeros(4, 87),
        future_scale=torch.ones(4, 87),
    )
    assert result["schema"] == r252.ATTRIBUTION_SCHEMA
    assert result["record_count"] == 2 * 3 * 2
    assert result["pipeline_controls"]["pass"] is True
    assert all(item["legacy_gate_pass"] for item in result["by_timestep"].values())
    assert result["pipeline_controls"]["matched_noise_max_abs_difference"] == 0.0
