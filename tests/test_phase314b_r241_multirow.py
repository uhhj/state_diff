from __future__ import annotations

import numpy as np
import torch

from ccda_phase3.phase314b_r241_multirow import (
    ConditionX0Prior,
    LabeledTupleBank,
    SourceBatchSampler,
    build_labeled_bank,
    classify_audit,
    condition_identifiability_audit,
    sample_aligned_source_batch,
    source_alignment_audit,
)
from ccda_phase3.schema_v2 import DEFAULT_TF, STATE_DIM


class FakeScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.01, 100)

    def add_noise(
        self,
        clean: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod.to(
            device=clean.device,
            dtype=clean.dtype,
        )[timesteps.to(clean.device)]
        alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
        sigma = torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1)
        return alpha * clean + sigma * noise

    def get_velocity(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod.to(
            device=sample.device,
            dtype=sample.dtype,
        )[timesteps.to(sample.device)]
        alpha = torch.sqrt(alpha_bar).reshape(-1, 1, 1)
        sigma = torch.sqrt(1.0 - alpha_bar).reshape(-1, 1, 1)
        return alpha * noise - sigma * sample


def make_sources(count: int = 4):
    condition = torch.arange(
        count * 6,
        dtype=torch.float32,
    ).reshape(count, 6)
    clean = torch.arange(
        count * DEFAULT_TF * STATE_DIM,
        dtype=torch.float32,
    ).reshape(count, DEFAULT_TF, STATE_DIM) / 1000.0
    raw = clean + 10.0
    active = torch.ones(DEFAULT_TF, STATE_DIM, dtype=torch.bool)
    return condition, clean, raw, active


def test_balanced_sampler_has_exact_alignment() -> None:
    condition, clean, raw, _ = make_sources(4)
    sampler = SourceBatchSampler(
        source_count=4,
        batch_size=8,
        mode="balanced",
        device=torch.device("cpu"),
        seed=7,
    )
    index, condition_batch, clean_batch, raw_batch = (
        sample_aligned_source_batch(
            condition_z=condition,
            clean_z=clean,
            clean_raw=raw,
            sampler=sampler,
        )
    )
    assert torch.equal(condition_batch, condition.index_select(0, index))
    assert torch.equal(clean_batch, clean.index_select(0, index))
    assert torch.equal(raw_batch, raw.index_select(0, index))
    report = sampler.count_report()
    assert report["counts"] == [2, 2, 2, 2]
    assert report["max_minus_min"] == 0


def test_random_sampler_records_all_counts() -> None:
    condition, clean, raw, _ = make_sources(4)
    sampler = SourceBatchSampler(
        source_count=4,
        batch_size=13,
        mode="random",
        device=torch.device("cpu"),
        seed=8,
    )
    index, _, _, _ = sample_aligned_source_batch(
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        sampler=sampler,
    )
    assert index.shape == (13,)
    assert sum(sampler.count_report()["counts"]) == 13


def test_labeled_bank_preserves_source_ids_and_alignment() -> None:
    condition, clean, raw, active = make_sources(3)
    bank = build_labeled_bank(
        scheduler=FakeScheduler(),
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(10, 50),
        noise_seeds=(100, 101),
        matched_noise_across_sources=True,
    )
    assert bank.row_count == 12
    assert bank.source_ids.tolist() == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
    ]
    audit = source_alignment_audit(
        bank=bank,
        source_condition=condition,
        source_clean_z=clean,
        source_clean_raw=raw,
    )
    assert audit["pass"] is True
    # Matched-noise contract: the same timestep/noise ID has equal noise.
    first = torch.nonzero(
        (bank.source_ids == 0)
        & (bank.timesteps == 10)
        & (bank.noise_ids == 100),
        as_tuple=False,
    ).reshape(-1)[0]
    second = torch.nonzero(
        (bank.source_ids == 1)
        & (bank.timesteps == 10)
        & (bank.noise_ids == 100),
        as_tuple=False,
    ).reshape(-1)[0]
    assert torch.equal(bank.noise[first], bank.noise[second])


def test_labeled_bank_index_select_keeps_tags() -> None:
    condition, clean, raw, active = make_sources(2)
    bank = build_labeled_bank(
        scheduler=FakeScheduler(),
        condition_z=condition,
        clean_z=clean,
        clean_raw=raw,
        active_mask=active,
        timesteps=(10, 50),
        noise_seeds=(1,),
    )
    selected = bank.index_select(torch.tensor([3, 0]))
    assert selected.source_ids.tolist() == [1, 0]
    assert selected.timesteps.tolist() == [50, 10]


def test_condition_identifiability_detects_duplicate_conflict() -> None:
    condition, clean, raw, _ = make_sources(3)
    condition[1] = condition[0]
    audit = condition_identifiability_audit(condition, clean, raw)
    assert audit["exact_duplicate_group_count"] == 1
    assert audit["exact_duplicate_conflict"] is True
    assert audit["pass"] is False


def test_condition_identifiability_accepts_distinct_rows() -> None:
    condition, clean, raw, _ = make_sources(4)
    audit = condition_identifiability_audit(condition, clean, raw)
    assert audit["exact_duplicate_conflict"] is False
    assert audit["centered_rank"] > 0
    assert audit["pass"] is True


def test_condition_prior_shape() -> None:
    model = ConditionX0Prior(condition_dim=7, hidden_dim=16)
    output = model(torch.randn(5, 7))
    assert output.shape == (5, DEFAULT_TF, STATE_DIM)


def _base_report():
    return {
        "source_alignment": {"pass": True},
        "condition_identifiability": {
            "exact_duplicate_conflict": False,
        },
        "direct_condition_controls": {
            "rows_16_width_512": {"pass": True},
            "rows_16_width_1024": {"pass": True},
        },
        "diffusion_variants": {},
    }


def _run(pass_value: bool, *, effect: bool = True, drift: float = 1.0):
    return {
        "pass": pass_value,
        "prior_drift_ratio": drift,
        "high_timestep_condition_effect": {
            "condition_effect_supported": effect,
        },
    }


def test_classifier_detects_source_alignment_bug() -> None:
    report = _base_report()
    report["source_alignment"] = {"pass": False}
    result = classify_audit(report)
    assert result["root_cause"] == (
        "phase314b_r241_source_row_alignment_bug_supported"
    )


def test_classifier_detects_condition_capacity_failure() -> None:
    report = _base_report()
    report["direct_condition_controls"] = {
        "rows_16_width_512": {"pass": False},
        "rows_16_width_1024": {"pass": False},
    }
    result = classify_audit(report)
    assert result["root_cause"] == (
        "phase314b_r241_direct_condition_multirow_capacity_failed"
    )


def test_classifier_detects_balanced_recovery() -> None:
    report = _base_report()
    report["diffusion_variants"] = {
        "current_joint_random_reused_optimizer": _run(False),
        "balanced_joint_fresh_optimizer": _run(True),
        "balanced_frozen_prior": _run(False),
        "balanced_anchor_prior": _run(False),
        "random_anchor_prior": _run(False),
    }
    result = classify_audit(report)
    assert result["root_cause"] == (
        "phase314b_r241_random_source_batch_or_optimizer_state_"
        "failure_supported"
    )
    assert result["train_only_debug_recommendation"] == (
        "balanced_joint_fresh_optimizer"
    )


def test_classifier_detects_prior_anchor_recovery() -> None:
    report = _base_report()
    report["diffusion_variants"] = {
        "current_joint_random_reused_optimizer": _run(False, drift=1000.0),
        "balanced_joint_fresh_optimizer": _run(False),
        "balanced_frozen_prior": _run(False),
        "balanced_anchor_prior": _run(True),
        "random_anchor_prior": _run(False),
    }
    result = classify_audit(report)
    assert result["root_cause"] == (
        "phase314b_r241_x0_prior_drift_under_joint_diffusion_supported"
    )
    assert result["train_only_debug_recommendation"] == (
        "balanced_anchor_prior"
    )


def test_classifier_detects_condition_underuse() -> None:
    report = _base_report()
    report["diffusion_variants"] = {
        "current_joint_random_reused_optimizer": _run(False, effect=False),
        "balanced_joint_fresh_optimizer": _run(False, effect=False),
    }
    result = classify_audit(report)
    assert result["root_cause"] == (
        "phase314b_r241_condition_path_underutilization_supported"
    )
