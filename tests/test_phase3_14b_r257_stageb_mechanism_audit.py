from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3.phase314b_r257_stageb_mechanism_audit import (
    MechanismAuditSpec,
    _gradient_pair_metrics,
    _top_k_mean,
    classify_mechanism,
    compare_worker_results,
    fixed_diagnostic_batch,
    load_objective_contract,
    safe_stats,
    sha256_array,
    source_logic_audit,
    stable_json_bytes,
    summarize_mechanism,
    upper_excess_torch,
    upper_violation_profile,
)


def reference() -> staged.SegmentReference:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    value = staged.SegmentReference(
        center_log=np.full(
            shape,
            np.log(0.04),
            dtype=np.float32,
        ),
        scale_log=np.full(shape, 0.01, dtype=np.float32),
        mad_scale=np.full(shape, 0.01, dtype=np.float32),
        iqr_scale=np.full(shape, 0.01, dtype=np.float32),
        scale_floor=0.01,
    )
    value.validate()
    return value


def upper_gate() -> staged3.OneSidedUpperContract:
    result = staged3.OneSidedUpperContract(
        reference=reference(),
        upper_threshold=3.0,
        within_group_row_coverage=0.95,
        conformal_group_coverage=0.95,
        fit_group_count=10,
        calibration_group_count=10,
        fit_group_sha256="a" * 64,
        calibration_group_sha256="b" * 64,
        calibration_group_score_sha256="c" * 64,
        stage_d_internal_contract_sha256="d" * 64,
        stage_d2_raw_xyz_sha256="e" * 64,
        stage_d2_projection_artifact_fraction=1.0,
    )
    result.validate()
    return result


def objective_contract() -> stagea.UpperObjectiveContract:
    result = stagea.UpperObjectiveContract(
        upper_gate_contract_sha256="f" * 64,
        upper_threshold=3.0,
        reference_center_log=reference().center_log.copy(),
        reference_scale_log=reference().scale_log.copy(),
        allowed_upper_log_length=(
            reference().center_log.astype(np.float64)
            + 3.0 * reference().scale_log.astype(np.float64)
        ).astype(np.float32),
        huber_delta_log_ratio=0.10,
        element_loss_weight=0.25,
        segment_length_epsilon=1.0e-8,
    )
    result.validate()
    return result


def standardizer() -> stageb.ArrayStandardizer:
    result = stageb.ArrayStandardizer(
        mean=np.zeros(
            (stageb.FUTURE_STEPS, stageb.CABLE_DIM),
            dtype=np.float32,
        ),
        scale=np.ones(
            (stageb.FUTURE_STEPS, stageb.CABLE_DIM),
            dtype=np.float32,
        ),
        active=np.ones(
            (stageb.FUTURE_STEPS, stageb.CABLE_DIM),
            dtype=np.bool_,
        ),
    )
    result.validate(
        (stageb.FUTURE_STEPS, stageb.CABLE_DIM)
    )
    return result


def cable_batch(
    spacing: float,
    *,
    rows: int = 2,
    candidates: Optional[int] = None,
) -> np.ndarray:
    x = (
        np.arange(stageb.BEADS, dtype=np.float32)
        * float(spacing)
    )
    y = np.zeros(stageb.BEADS, dtype=np.float32)
    flat = np.stack([x, y], axis=1).reshape(
        stageb.CABLE_DIM
    )
    trajectory = np.repeat(
        flat[None],
        stageb.FUTURE_STEPS,
        axis=0,
    )
    rows_value = np.repeat(
        trajectory[None],
        rows,
        axis=0,
    )
    if candidates is None:
        return rows_value
    return np.repeat(
        rows_value[:, None],
        candidates,
        axis=1,
    )


def synthetic_candidate_record(
    lambda_upper: float,
    *,
    mean_excess: float,
    gradient_ratio: float = 0.2,
    cosine: float = 0.0,
    geometry_norm: float = 1.0,
    linear: float = 0.5,
    concentration: float = 0.1,
    positive_count: float = 10.0,
    nmse_ratio: float = 1.0,
):
    checkpoint = {
        "gradient": {
            "scaled_geometry_to_diffusion_ratio":
                gradient_ratio,
            "cosine": cosine,
            "conflict_fraction_among_active": 0.2,
            "zero_geometry_fraction": 0.1,
            "geometry_norm": geometry_norm,
        },
        "huber": {
            "linear_fraction_among_positive": linear,
            "top_position_fraction": concentration,
        },
    }
    profile = {
        "row_best_max_log_excess": {
            "mean": mean_excess,
        },
        "top_position_fraction": concentration,
        "candidate_positive_segment_count": {
            "mean": positive_count,
        },
    }
    return {
        "lambda_upper": lambda_upper,
        "stagea_record": {
            "relative_to_control": {
                "t10_nmse": nmse_ratio,
            }
        },
        "training_attribution": {
            "checkpoint_records": {
                "1": checkpoint,
                "100": checkpoint,
            }
        },
        "holdout_profiles": {
            "10": profile,
            "25": profile,
            "50": profile,
        },
    }


def mechanism_summary(
    *,
    reduction: float = 0.1,
    gradient_ratio: float = 0.2,
    cosine: float = 0.0,
    geometry_norm: float = 1.0,
    linear: float = 0.5,
    concentration: float = 0.1,
    positive_count: float = 10.0,
    nmse_ratio: float = 1.1,
):
    return {
        "control_t10_mean_row_best_log_excess": 1.0,
        "best_lambda_by_t10_mean_excess": 0.1,
        "best_t10_mean_row_best_log_excess":
            1.0 - reduction,
        "best_t10_mean_excess_reduction": reduction,
        "strongest_lambda": 0.1,
        "strongest_scaled_gradient_ratio_median":
            gradient_ratio,
        "strongest_gradient_cosine_median": cosine,
        "strongest_gradient_conflict_fraction_median": 0.2,
        "strongest_zero_geometry_fraction_median": 0.1,
        "strongest_geometry_norm_median": geometry_norm,
        "strongest_huber_linear_fraction_median": linear,
        "strongest_rowmax_top_position_fraction_median":
            concentration,
        "strongest_t10_top_position_fraction":
            concentration,
        "strongest_t10_positive_segment_count_mean":
            positive_count,
        "strongest_t10_nmse_ratio": nmse_ratio,
    }


def test_spec_validates():
    MechanismAuditSpec().validate()


def test_spec_rejects_lambda_change():
    with pytest.raises(ValueError):
        MechanismAuditSpec(
            candidate_lambdas=(0.0, 0.01),
        ).validate()


def test_spec_rejects_missing_final_checkpoint():
    with pytest.raises(ValueError):
        MechanismAuditSpec(
            completed_step_checkpoints=(1, 100),
        ).validate()


def test_load_objective_contract_round_trip():
    payload = objective_contract().to_dict()
    loaded = load_objective_contract(payload)
    assert loaded.to_dict() == payload


def test_safe_stats_empty():
    result = safe_stats(np.asarray([], dtype=np.float32))
    assert result["count"] == 0
    assert result["mean"] == 0.0


def test_safe_stats_rejects_nan():
    with pytest.raises(Exception):
        safe_stats(np.asarray([np.nan]))


def test_fixed_diagnostic_batch_is_deterministic():
    condition = np.zeros((80, 201), dtype=np.float32)
    target = cable_batch(0.04, rows=80)
    first = fixed_diagnostic_batch(
        condition=condition,
        target=target,
        condition_standardizer=stageb.fit_standardizer(
            condition
        ),
        target_standardizer=stageb.fit_standardizer(
            target
        ),
        stageb_spec=stageb.DiagnosticSpec(),
        spec=MechanismAuditSpec(),
    )
    second = fixed_diagnostic_batch(
        condition=condition,
        target=target,
        condition_standardizer=stageb.fit_standardizer(
            condition
        ),
        target_standardizer=stageb.fit_standardizer(
            target
        ),
        stageb_spec=stageb.DiagnosticSpec(),
        spec=MechanismAuditSpec(),
    )
    assert first["noise_sha256"] == second["noise_sha256"]
    assert first["indices_sha256"] == second["indices_sha256"]


def test_fixed_diagnostic_batch_uses_registered_timesteps():
    condition = np.zeros((80, 201), dtype=np.float32)
    target = cable_batch(0.04, rows=80)
    result = fixed_diagnostic_batch(
        condition=condition,
        target=target,
        condition_standardizer=stageb.fit_standardizer(
            condition
        ),
        target_standardizer=stageb.fit_standardizer(
            target
        ),
        stageb_spec=stageb.DiagnosticSpec(),
        spec=MechanismAuditSpec(),
    )
    assert set(result["timesteps"].tolist()) == {
        10,
        25,
        50,
        75,
    }


def test_upper_excess_zero_for_valid_geometry():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.04),
        dtype=torch.float32,
    )
    result = upper_excess_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    assert float(torch.max(result["excess"]).item()) == 0.0


def test_upper_excess_positive_for_expansion():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    )
    result = upper_excess_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    assert float(torch.max(result["excess"]).item()) > 0.0


def test_top_k_mean():
    value = np.asarray(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        dtype=np.float64,
    )
    result = _top_k_mean(value, 2)
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(3.5)


def test_gradient_pair_aligned():
    torch, _ = stageb._torch_imports()
    parameter = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32)
    )
    gradient = torch.tensor([1.0, 2.0])
    result = _gradient_pair_metrics(
        [("p", parameter)],
        [gradient],
        [gradient],
        lambda_upper=0.1,
    )
    assert result["cosine"] == pytest.approx(1.0)
    assert result["conflict_fraction_among_active"] == 0.0


def test_gradient_pair_opposed():
    torch, _ = stageb._torch_imports()
    parameter = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32)
    )
    first = torch.tensor([1.0, 2.0])
    second = -first
    result = _gradient_pair_metrics(
        [("p", parameter)],
        [first],
        [second],
        lambda_upper=0.1,
    )
    assert result["cosine"] == pytest.approx(-1.0)
    assert result["conflict_fraction_among_active"] == 1.0


def test_gradient_pair_zero_geometry():
    torch, _ = stageb._torch_imports()
    parameter = torch.nn.Parameter(
        torch.zeros(2, dtype=torch.float32)
    )
    result = _gradient_pair_metrics(
        [("p", parameter)],
        [torch.ones(2)],
        [torch.zeros(2)],
        lambda_upper=0.1,
    )
    assert result["geometry_norm"] == 0.0
    assert result["zero_geometry_fraction"] == 1.0


def test_upper_profile_accepts_valid_geometry():
    result = upper_violation_profile(
        cable_batch(0.04, candidates=2),
        upper_gate=upper_gate(),
        objective_contract=objective_contract(),
        top_k=(1, 4, 8),
    )
    assert result["candidate_pass_rate"] == 1.0
    assert result["row_any_rate"] == 1.0
    assert result["candidate_positive_segment_count"]["mean"] == 0.0


def test_upper_profile_rejects_expansion():
    result = upper_violation_profile(
        cable_batch(0.08, candidates=2),
        upper_gate=upper_gate(),
        objective_contract=objective_contract(),
        top_k=(1, 4, 8),
    )
    assert result["candidate_pass_rate"] == 0.0
    assert result["row_any_rate"] == 0.0
    assert result["candidate_positive_segment_count"]["mean"] > 0.0


def test_upper_profile_ignores_contraction():
    result = upper_violation_profile(
        cable_batch(0.001),
        upper_gate=upper_gate(),
        objective_contract=objective_contract(),
        top_k=(1, 4, 8),
    )
    assert result["row_any_rate"] == 1.0


def test_upper_profile_reports_position_frequency():
    value = cable_batch(0.04)
    points = value.reshape(2, 4, 24, 2)
    points[:, :, 12:, 0] += 0.1
    result = upper_violation_profile(
        value,
        upper_gate=upper_gate(),
        objective_contract=objective_contract(),
        top_k=(1, 4, 8),
    )
    assert result["row_argmax_position_frequency"]
    assert result["top_position_fraction"] > 0.0


def test_summarize_mechanism_selects_best_response():
    records = [
        synthetic_candidate_record(0.0, mean_excess=1.0),
        synthetic_candidate_record(0.001, mean_excess=0.9),
        synthetic_candidate_record(0.01, mean_excess=0.8),
        synthetic_candidate_record(0.1, mean_excess=0.7),
    ]
    result = summarize_mechanism(records)
    assert result["best_lambda_by_t10_mean_excess"] == 0.1
    assert result["best_t10_mean_excess_reduction"] == pytest.approx(
        0.3
    )


def test_classification_detects_disconnected_gradient():
    result = classify_mechanism(
        summary=mechanism_summary(geometry_norm=0.0),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == "objective_graph"


def test_classification_detects_underpowered_gradient():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.01,
            gradient_ratio=0.01,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == "gradient_scale"


def test_classification_detects_gradient_conflict():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.01,
            gradient_ratio=1.0,
            cosine=-0.5,
            nmse_ratio=2.0,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == "gradient_conflict"


def test_classification_detects_rowmax_concentration():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.1,
            concentration=0.8,
            positive_count=20.0,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "rowmax_concentration"
    )


def test_classification_detects_huber_saturation():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.1,
            concentration=0.1,
            linear=0.99,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == "huber_saturation"


def test_classification_detects_substantial_response():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.3,
            concentration=0.1,
            linear=0.5,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "insufficient_gate_crossing"
    )


def test_classification_detects_weak_response():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.1,
            concentration=0.1,
            linear=0.5,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "weak_objective_response"
    )


def test_classification_detects_no_response():
    result = classify_mechanism(
        summary=mechanism_summary(
            reduction=0.01,
            gradient_ratio=1.0,
            cosine=0.0,
            nmse_ratio=1.0,
        ),
        spec=MechanismAuditSpec(),
    )
    assert result["primary_failure_locus"] == (
        "objective_response"
    )


def test_compare_worker_results_is_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "split": {"x": 1},
        "mechanism_contract": {"x": 1},
        "candidate_audits": {"x": 1},
        "mechanism_summary": {"x": 1},
        "classification": {"x": 1},
    }
    assert compare_worker_results(base, dict(base))["exact"]


def test_stable_json_is_deterministic():
    assert stable_json_bytes({"b": 2, "a": 1}) == (
        stable_json_bytes({"a": 1, "b": 2})
    )


def test_sha256_array_changes():
    assert sha256_array(
        np.asarray([1.0], dtype=np.float32)
    ) != sha256_array(
        np.asarray([2.0], dtype=np.float32)
    )


def test_source_logic_audit_accepts_frozen_sources():
    repository_root = Path(__file__).resolve().parents[1]
    result = source_logic_audit(repository_root)
    assert result["all_confirmed"]
    assert result["diagnostic_replay_safe"]
