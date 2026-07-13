from __future__ import annotations

import numpy as np
import pytest
import torch

from ccda_phase3.phase314b_r241_multirow import LabeledTupleBank
from ccda_phase3.phase314b_r242_frozen_prior import (
    FactorizedAnalyticX0SkipDenoiser,
    FactorizedVariant,
    PRIOR_SEED_RUN_SCHEMA_VERSION,
    PRIOR_SEED_STABILITY_SCHEMA_VERSION,
    RECONSTRUCTION_METRICS_SCHEMA_VERSION,
    compact_reconstruction_metrics,
    reconstruction_metric_quantile,
    classify_pilot,
    corrected_r241_interpretation,
    paired_branch_audit,
    summarize_seed_stability,
)
from ccda_phase3.phase314b_r231_controls import target_reconstruction_metrics
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


def make_model(
    prior_hidden: int = 32,
    residual_hidden: int = 48,
) -> FactorizedAnalyticX0SkipDenoiser:
    alpha_bar = torch.linspace(0.99, 0.01, 100)
    return FactorizedAnalyticX0SkipDenoiser(
        condition_dim=7,
        alpha_bar=alpha_bar,
        prior_hidden_dim=prior_hidden,
        residual_hidden_dim=residual_hidden,
        time_dim=16,
    )


def test_factorized_model_output_shape_and_zero_residual() -> None:
    torch.manual_seed(1)
    model = make_model()
    condition = torch.randn(3, 7)
    noisy = torch.randn(3, DEFAULT_TF, STATE_DIM)
    timestep = torch.tensor([10, 50, 90], dtype=torch.long)

    base = model.predict_base_x0(condition)
    alpha, sigma = model.coefficients(timestep, noisy.dtype)
    expected = (alpha * noisy - base) / sigma
    observed = model(noisy, timestep, condition)

    assert observed.shape == noisy.shape
    assert torch.allclose(observed, expected, atol=1e-7, rtol=0)


def test_prior_and_residual_widths_are_independent() -> None:
    narrow_prior = make_model(prior_hidden=16, residual_hidden=64)
    wide_prior = make_model(prior_hidden=32, residual_hidden=64)
    wide_residual = make_model(prior_hidden=16, residual_hidden=128)

    narrow_prior_count = sum(
        p.numel() for p in narrow_prior.prior_parameters()
    )
    wide_prior_count = sum(
        p.numel() for p in wide_prior.prior_parameters()
    )
    narrow_residual_count = sum(
        p.numel() for p in narrow_prior.residual_parameters()
    )
    wide_residual_count = sum(
        p.numel() for p in wide_residual.residual_parameters()
    )

    assert wide_prior_count > narrow_prior_count
    assert wide_residual_count > narrow_residual_count
    assert sum(p.numel() for p in wide_prior.residual_parameters()) == (
        narrow_residual_count
    )


def test_detached_v_path_does_not_update_prior() -> None:
    torch.manual_seed(2)
    model = make_model()
    condition = torch.randn(4, 7)
    noisy = torch.randn(4, DEFAULT_TF, STATE_DIM)
    timestep = torch.tensor([10, 25, 50, 75], dtype=torch.long)
    target = torch.randn_like(noisy)

    output = model(
        noisy,
        timestep,
        condition,
        detach_prior_for_v=True,
    )
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    assert all(parameter.grad is None for parameter in model.prior_parameters())
    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in model.residual_parameters()
    )


def test_joint_v_path_updates_prior() -> None:
    torch.manual_seed(3)
    model = make_model()
    condition = torch.randn(4, 7)
    noisy = torch.randn(4, DEFAULT_TF, STATE_DIM)
    timestep = torch.tensor([10, 25, 50, 75], dtype=torch.long)
    target = torch.randn_like(noisy)

    output = model(
        noisy,
        timestep,
        condition,
        detach_prior_for_v=False,
    )
    loss = torch.mean((output - target) ** 2)
    loss.backward()

    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in model.prior_parameters()
    )


def test_direct_prior_loss_updates_prior_after_detached_v() -> None:
    torch.manual_seed(4)
    model = make_model()
    condition = torch.randn(4, 7)
    target_x0 = torch.randn(4, DEFAULT_TF, STATE_DIM)

    loss = torch.mean(
        (model.predict_base_x0(condition) - target_x0) ** 2
    )
    loss.backward()

    assert any(
        parameter.grad is not None
        and torch.any(parameter.grad != 0)
        for parameter in model.prior_parameters()
    )


def r241_summary_fixture():
    return {
        "root_cause": (
            "phase314b_r241_condition_encoder_width_limit_supported"
        ),
        "direct_controls": {
            "rows_16_width_512": {"pass": False},
            "rows_16_width_1024": {"pass": True},
        },
        "diffusion_variants": {
            "balanced_frozen_prior": {
                "pass": True,
                "prior_drift_ratio": 1.0,
            },
            "current_joint_random_reused_optimizer": {
                "pass": False,
                "prior_drift_ratio": 47708.0,
            },
            "balanced_joint_fresh_optimizer": {
                "pass": False,
                "prior_drift_ratio": 562.6,
            },
            "balanced_anchor_prior": {
                "pass": False,
                "prior_drift_ratio": 1962.9,
            },
            "random_anchor_prior": {
                "pass": False,
                "prior_drift_ratio": 8010.6,
            },
        },
    }


def test_corrected_r241_interpretation_prioritizes_diffusion_evidence() -> None:
    result = corrected_r241_interpretation(r241_summary_fixture())

    assert result["classifier_precedence_bug_supported"]
    assert result["corrected_primary_hypothesis"] == (
        "x0_prior_gradient_coupling_and_drift"
    )
    assert result["secondary_hypothesis"] == (
        "single_seed_condition_width_sensitivity"
    )


def test_corrected_r241_interpretation_rejects_missing_frozen_pass() -> None:
    summary = r241_summary_fixture()
    summary["diffusion_variants"]["balanced_frozen_prior"]["pass"] = False
    result = corrected_r241_interpretation(summary)

    assert not result["classifier_precedence_bug_supported"]


def make_prior_seed_run(
    *,
    seed: int,
    passed: bool,
    hidden_dim: int = 512,
    z_mse: float = 1.0e-5,
    ordered_rmse_p95: float = 1.0e-3,
):
    return {
        "schema_version": PRIOR_SEED_RUN_SCHEMA_VERSION,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "pass": passed,
        "gate_pass": passed,
        "all_source_gate_pass": passed,
        "metrics": {
            "z_mse": z_mse,
            "ordered_rmse": {"p95": ordered_rmse_p95},
            "segment_relative_error": {"p95": 1.0e-3},
            "chain_relative_error": {"p95": 1.0e-3},
        },
    }


def test_seed_stability_requires_two_of_three_nested_metric_runs() -> None:
    runs = [
        make_prior_seed_run(seed=95101, passed=True, z_mse=1.0e-5),
        make_prior_seed_run(seed=95102, passed=False, z_mse=2.0e-5),
        make_prior_seed_run(seed=95103, passed=True, z_mse=3.0e-5),
    ]
    result = summarize_seed_stability(runs)

    assert result["schema_version"] == PRIOR_SEED_STABILITY_SCHEMA_VERSION
    assert result["hidden_dim"] == 512
    assert result["seeds"] == [95101, 95102, 95103]
    assert result["pass_count"] == 2
    assert result["stable_2_of_3"]
    assert result["z_mse_median"] == pytest.approx(2.0e-5)
    assert len(result["per_seed"]) == 3


def test_seed_stability_rejects_obsolete_nested_schema() -> None:
    obsolete = {
        "schema_version": PRIOR_SEED_RUN_SCHEMA_VERSION,
        "hidden_dim": 512,
        "seed": 95101,
        "pass": True,
        "gate_pass": True,
        "all_source_gate_pass": True,
        "aggregate": {
            "metrics": {
                "z_mse": 1.0e-5,
                "ordered_rmse_p95": 1.0e-3,
            }
        },
    }
    with pytest.raises(ValueError, match="obsolete nested aggregate"):
        summarize_seed_stability([obsolete, obsolete, obsolete])


def test_reconstruction_metric_quantile_reads_canonical_nested_path() -> None:
    metrics = make_prior_seed_run(seed=95101, passed=True)["metrics"]
    assert reconstruction_metric_quantile(
        metrics,
        "ordered_rmse",
        "p95",
    ) == pytest.approx(1.0e-3)


def test_reconstruction_metric_quantile_rejects_flat_alias() -> None:
    metrics = make_prior_seed_run(seed=95101, passed=True)["metrics"]
    metrics["ordered_rmse_p95"] = 1.0e-3
    with pytest.raises(ValueError, match="flat reconstruction metric alias"):
        reconstruction_metric_quantile(metrics, "ordered_rmse", "p95")


def test_reconstruction_metric_quantile_rejects_missing_group() -> None:
    metrics = make_prior_seed_run(seed=95101, passed=True)["metrics"]
    metrics.pop("ordered_rmse")
    with pytest.raises(ValueError, match="missing nested group"):
        reconstruction_metric_quantile(metrics, "ordered_rmse", "p95")


def test_reconstruction_metric_quantile_rejects_missing_quantile() -> None:
    metrics = make_prior_seed_run(seed=95101, passed=True)["metrics"]
    metrics["ordered_rmse"] = {}
    with pytest.raises(ValueError, match="missing p95"):
        reconstruction_metric_quantile(metrics, "ordered_rmse", "p95")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_reconstruction_metric_quantile_rejects_invalid_values(value: float) -> None:
    metrics = make_prior_seed_run(seed=95101, passed=True)["metrics"]
    metrics["ordered_rmse"]["p95"] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        reconstruction_metric_quantile(metrics, "ordered_rmse", "p95")


def test_compact_metrics_accept_real_target_reconstruction_schema() -> None:
    predicted_z = torch.zeros(2, DEFAULT_TF, STATE_DIM)
    target_z = torch.zeros_like(predicted_z)
    predicted_raw = torch.zeros_like(predicted_z)
    target_raw = torch.zeros_like(predicted_z)
    active = torch.ones(DEFAULT_TF, STATE_DIM, dtype=torch.bool)
    metrics = target_reconstruction_metrics(
        predicted_z=predicted_z,
        target_z=target_z,
        predicted_raw=predicted_raw,
        target_raw=target_raw,
        active_mask=active,
    )
    compact = compact_reconstruction_metrics(metrics)
    assert compact["z_mse"] == pytest.approx(0.0)
    assert compact["ordered_rmse_p95"] == pytest.approx(0.0)
    assert compact["segment_relative_error_p95"] == pytest.approx(0.0)
    assert compact["chain_relative_error_p95"] == pytest.approx(0.0)


def test_seed_stability_accepts_real_producer_metric_schema() -> None:
    runs = [
        make_prior_seed_run(seed=95101, passed=True),
        make_prior_seed_run(seed=95102, passed=True),
        make_prior_seed_run(seed=95103, passed=False),
    ]
    result = summarize_seed_stability(runs)
    assert result["schema_version"] == PRIOR_SEED_STABILITY_SCHEMA_VERSION
    assert result["per_seed"][0]["metrics_schema_version"] == (
        RECONSTRUCTION_METRICS_SCHEMA_VERSION
    )
    assert result["pass_count"] == 2


def test_seed_stability_requires_exactly_three_runs() -> None:
    runs = [
        make_prior_seed_run(seed=95101, passed=True),
        make_prior_seed_run(seed=95102, passed=True),
    ]
    with pytest.raises(ValueError, match="exactly three"):
        summarize_seed_stability(runs)


def test_seed_stability_rejects_duplicate_seeds() -> None:
    runs = [
        make_prior_seed_run(seed=95101, passed=True),
        make_prior_seed_run(seed=95101, passed=True),
        make_prior_seed_run(seed=95103, passed=True),
    ]
    with pytest.raises(ValueError, match="duplicate seeds"):
        summarize_seed_stability(runs)


def test_seed_stability_rejects_mixed_hidden_dims() -> None:
    runs = [
        make_prior_seed_run(seed=95101, passed=True, hidden_dim=512),
        make_prior_seed_run(seed=95102, passed=True, hidden_dim=1024),
        make_prior_seed_run(seed=95103, passed=True, hidden_dim=512),
    ]
    with pytest.raises(ValueError, match="mixes hidden dimensions"):
        summarize_seed_stability(runs)


def test_seed_stability_rejects_inconsistent_pass_fields() -> None:
    run = make_prior_seed_run(seed=95101, passed=True)
    run["gate_pass"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        summarize_seed_stability(
            [
                run,
                make_prior_seed_run(seed=95102, passed=True),
                make_prior_seed_run(seed=95103, passed=True),
            ]
        )


def test_seed_stability_rejects_nonfinite_metrics() -> None:
    with pytest.raises(ValueError, match="z_mse"):
        summarize_seed_stability(
            [
                make_prior_seed_run(
                    seed=95101,
                    passed=True,
                    z_mse=float("nan"),
                ),
                make_prior_seed_run(seed=95102, passed=True),
                make_prior_seed_run(seed=95103, passed=True),
            ]
        )

def test_paired_branch_audit_passes_exact_predictions() -> None:
    batch = 2
    condition = torch.zeros(batch, 5)
    clean = torch.zeros(batch, DEFAULT_TF, STATE_DIM)
    clean[0, :, :48] = 0.1
    clean[1, :, :48] = -0.1
    bank = LabeledTupleBank(
        condition_z=condition,
        clean_z=clean,
        clean_raw=clean.clone(),
        noisy=clean.clone(),
        noise=torch.zeros_like(clean),
        timesteps=torch.tensor([25, 25], dtype=torch.long),
        noise_ids=torch.tensor([7, 7], dtype=torch.long),
        source_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    bank.validate()
    result = paired_branch_audit(
        predicted_z=clean.clone(),
        bank=bank,
        source_pair_ids=[0, 0],
        future_mean=torch.zeros(DEFAULT_TF, STATE_DIM),
        future_scale=torch.ones(DEFAULT_TF, STATE_DIM),
    )

    assert result["pass"]
    assert result["own_target_closer_fraction"] == pytest.approx(1.0)
    assert result["branch_delta_cosine"]["p50"] == pytest.approx(1.0)


def test_paired_branch_audit_rejects_collapsed_predictions() -> None:
    batch = 2
    condition = torch.zeros(batch, 5)
    clean = torch.zeros(batch, DEFAULT_TF, STATE_DIM)
    clean[0, :, :48] = 0.1
    clean[1, :, :48] = -0.1
    bank = LabeledTupleBank(
        condition_z=condition,
        clean_z=clean,
        clean_raw=clean.clone(),
        noisy=clean.clone(),
        noise=torch.zeros_like(clean),
        timesteps=torch.tensor([25, 25], dtype=torch.long),
        noise_ids=torch.tensor([7, 7], dtype=torch.long),
        source_ids=torch.tensor([0, 1], dtype=torch.long),
    )
    collapsed = torch.zeros_like(clean)
    result = paired_branch_audit(
        predicted_z=collapsed,
        bank=bank,
        source_pair_ids=[0, 0],
        future_mean=torch.zeros(DEFAULT_TF, STATE_DIM),
        future_scale=torch.ones(DEFAULT_TF, STATE_DIM),
    )

    assert not result["pass"]
    assert result["separation_ratio"]["p50"] == pytest.approx(0.0)


def test_factorized_variant_validation() -> None:
    with pytest.raises(ValueError, match="unsupported prior policy"):
        FactorizedVariant(
            name="bad",
            prior_hidden_dim=512,
            residual_hidden_dim=512,
            prior_policy="unknown",
        ).validate()

    with pytest.raises(ValueError, match="frozen prior"):
        FactorizedVariant(
            name="bad_anchor",
            prior_hidden_dim=512,
            residual_hidden_dim=512,
            prior_policy="frozen",
            prior_anchor_weight=1.0,
        ).validate()


def test_classify_pilot_requires_paired_branch_transport() -> None:
    report = {
        "r241_corrected_interpretation": {
            "classifier_precedence_bug_supported": True,
        },
        "direct_width_seed_stability": {
            "512": {"stable_2_of_3": True},
            "1024": {"stable_2_of_3": True},
        },
        "unique_free_variants": {
            "frozen_p512_r512": {"pass": True},
            "joint_p1024_r512_control": {"pass": False},
        },
        "paired_low_mid_variants": {
            "frozen_p512_r512": {
                "pass": True,
                "paired_branch_audit": {"pass": False},
            }
        },
    }
    result = classify_pilot(report)

    assert result["root_cause"] == (
        "phase314b_r242_paired_low_mid_branch_transport_failed"
    )


def test_classify_pilot_can_recommend_frozen_recipe() -> None:
    report = {
        "r241_corrected_interpretation": {
            "classifier_precedence_bug_supported": True,
        },
        "direct_width_seed_stability": {
            "512": {"stable_2_of_3": True},
            "1024": {"stable_2_of_3": True},
        },
        "unique_free_variants": {
            "frozen_p512_r512": {"pass": True},
            "joint_p1024_r512_control": {"pass": False},
        },
        "paired_low_mid_variants": {
            "frozen_p512_r512": {
                "pass": True,
                "paired_branch_audit": {"pass": True},
            }
        },
    }
    result = classify_pilot(report)

    assert result["root_cause"] == (
        "phase314b_r242_strict_frozen_prior_required"
    )
    assert result["train_only_recommendation"] == "frozen_p512_r512"
