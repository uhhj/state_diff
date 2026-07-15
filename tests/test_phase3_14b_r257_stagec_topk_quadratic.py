from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3.phase314b_r257_stagec_topk_quadratic import (
    CONTROL_CANDIDATE_ID,
    CalibratedCandidate,
    TopKCalibrationSpec,
    candidate_conflict_severe,
    classify_selection,
    compare_worker_results,
    compatible_training_record,
    k_token,
    ratio,
    ratio_token,
    reduction,
    row_contribution_profile,
    select_configuration,
    selected_position_profile,
    selection_record,
    source_logic_audit,
    stable_json_bytes,
    topk_quadratic_terms_torch,
)


def reference() -> staged.SegmentReference:
    shape = (
        stageb.FUTURE_STEPS,
        stageb.BEADS - 1,
    )
    result = staged.SegmentReference(
        center_log=np.full(
            shape,
            np.log(0.04),
            dtype=np.float32,
        ),
        scale_log=np.full(
            shape,
            0.01,
            dtype=np.float32,
        ),
        mad_scale=np.full(
            shape,
            0.01,
            dtype=np.float32,
        ),
        iqr_scale=np.full(
            shape,
            0.01,
            dtype=np.float32,
        ),
        scale_floor=0.01,
    )
    result.validate()
    return result


def objective_contract() -> stagea.UpperObjectiveContract:
    ref = reference()
    result = stagea.UpperObjectiveContract(
        upper_gate_contract_sha256="a" * 64,
        upper_threshold=3.0,
        reference_center_log=
            ref.center_log.copy(),
        reference_scale_log=
            ref.scale_log.copy(),
        allowed_upper_log_length=(
            ref.center_log.astype(np.float64)
            + 3.0 * ref.scale_log.astype(np.float64)
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
            (
                stageb.FUTURE_STEPS,
                stageb.CABLE_DIM,
            ),
            dtype=np.float32,
        ),
        scale=np.ones(
            (
                stageb.FUTURE_STEPS,
                stageb.CABLE_DIM,
            ),
            dtype=np.float32,
        ),
        active=np.ones(
            (
                stageb.FUTURE_STEPS,
                stageb.CABLE_DIM,
            ),
            dtype=np.bool_,
        ),
    )
    result.validate(
        (
            stageb.FUTURE_STEPS,
            stageb.CABLE_DIM,
        )
    )
    return result


def cable_batch(
    spacing: float,
    rows: int = 2,
) -> np.ndarray:
    x = (
        np.arange(
            stageb.BEADS,
            dtype=np.float32,
        )
        * float(spacing)
    )
    y = np.zeros(
        stageb.BEADS,
        dtype=np.float32,
    )
    flat = np.stack(
        [x, y],
        axis=1,
    ).reshape(stageb.CABLE_DIM)
    trajectory = np.repeat(
        flat[None],
        stageb.FUTURE_STEPS,
        axis=0,
    )
    return np.repeat(
        trajectory[None],
        rows,
        axis=0,
    )


def candidate(
    *,
    candidate_id: str = "topk8_r0p25",
    top_k: int = 8,
    target_ratio: float = 0.25,
    lambda_upper: float = 0.01,
) -> CalibratedCandidate:
    return CalibratedCandidate(
        candidate_id=candidate_id,
        top_k=top_k,
        target_gradient_ratio=target_ratio,
        lambda_upper=lambda_upper,
        initial_model_sha256="a" * 64,
        diffusion_gradient_norm=2.0,
        geometry_gradient_norm=50.0,
        geometry_gradient_cosine=0.25,
        geometry_gradient_conflict_fraction=0.40,
    )


def training_record() -> dict:
    return {
        "lambda_upper": 0.01,
        "initial_model_sha256": "a" * 64,
        "final_model_sha256": "b" * 64,
        "initial_optimizer_sha256": "c" * 64,
        "final_optimizer_sha256": "d" * 64,
        "total_loss_history_sha256": "e" * 64,
        "diffusion_loss_history_sha256": "f" * 64,
        "geometry_loss_history_sha256": "1" * 64,
        "gradient_history_sha256": "2" * 64,
        "source_exposure_sha256": "3" * 64,
        "total_loss_first": 1.0,
        "total_loss_final": 0.1,
        "total_loss_tail_mean": 0.1,
        "diffusion_loss_tail_mean": 0.1,
        "geometry_loss_tail_mean": 0.1,
        "geometry_row_loss_tail_mean": 0.1,
        "geometry_element_loss_tail_mean": 0.0,
        "batch_row_violation_rate_tail_mean": 1.0,
        "maximum_log_excess_tail_max": 1.0,
        "gradient_tail_mean": 1.0,
        "loss_finite": True,
        "gradient_finite": True,
        "training_rows": 638,
        "training_steps": 8000,
    }


def training_diagnostics(
    *,
    cosine: float = 0.2,
    conflict: float = 0.4,
    clip_frequency: float = 0.0,
    largest: float = 0.1,
    top_five: float = 0.2,
) -> dict:
    return {
        "candidate_id": "topk8_r0p25",
        "top_k": 8,
        "target_gradient_ratio": 0.25,
        "lambda_upper": 0.01,
        "gradient_clip_count": 0,
        "gradient_clip_frequency": clip_frequency,
        "checkpoint_records": {
            "8000": {
                "gradient": {
                    "cosine": cosine,
                    "conflict_fraction_among_active":
                        conflict,
                },
                "row_contribution": {
                    "largest_row_contribution_fraction":
                        largest,
                    "top_five_percent_contribution_fraction":
                        top_five,
                },
            }
        },
        "checkpoint_record_sha256": "4" * 64,
    }


def train_control(nmse: float = 1.0) -> dict:
    return {
        "rows": 64,
        "normalized_mse": nmse,
        "raw_rmse": nmse,
        "prediction_sha256": "5" * 64,
    }


def profile(
    *,
    mean_excess: float,
    p95_excess: float,
    top8: float,
    positive_count: float,
    binary_rate: float,
) -> dict:
    return {
        "row_best_max_log_excess": {
            "mean": mean_excess,
            "p95": p95_excess,
        },
        "top_k_mean_log_excess": {
            "8": {
                "mean": top8,
            }
        },
        "candidate_positive_segment_count": {
            "mean": positive_count,
        },
        "row_any_rate": binary_rate,
    }


def profiles(
    *,
    scale: float = 1.0,
    binary_rate: float = 0.0,
) -> dict:
    return {
        "10": profile(
            mean_excess=1.0 * scale,
            p95_excess=1.2 * scale,
            top8=0.8 * scale,
            positive_count=40.0 * scale,
            binary_rate=binary_rate,
        ),
        "25": profile(
            mean_excess=1.0 * scale,
            p95_excess=1.2 * scale,
            top8=0.8 * scale,
            positive_count=40.0 * scale,
            binary_rate=binary_rate,
        ),
        "50": profile(
            mean_excess=1.0 * scale,
            p95_excess=1.2 * scale,
            top8=0.8 * scale,
            positive_count=40.0 * scale,
            binary_rate=binary_rate,
        ),
    }


def one_step(nmse: float = 1.0) -> dict:
    return {
        "prediction_sha256": "6" * 64,
        "timesteps": {
            timestep: {
                "normalized_mse": nmse,
                "raw_rmse": nmse,
                "upper": {},
                "prediction_sha256": "7" * 64,
            }
            for timestep in ("10", "25", "50")
        },
    }


def control_record() -> dict:
    return selection_record(
        candidate=None,
        training=training_record(),
        training_diagnostics=
            training_diagnostics(),
        train_control=train_control(1.0),
        one_step=one_step(1.0),
        profiles=profiles(
            scale=1.0,
            binary_rate=0.0,
        ),
        control=None,
        spec=TopKCalibrationSpec(),
    )


def candidate_record(
    *,
    scale: float = 0.7,
    binary_rate: float = 0.8,
    nmse: float = 1.1,
    clip_frequency: float = 0.0,
    largest: float = 0.1,
    top_five: float = 0.2,
    cosine: float = 0.2,
    conflict: float = 0.4,
    candidate_value: CalibratedCandidate = None,
) -> dict:
    active = (
        candidate()
        if candidate_value is None
        else candidate_value
    )
    return selection_record(
        candidate=active,
        training=training_record(),
        training_diagnostics=training_diagnostics(
            cosine=cosine,
            conflict=conflict,
            clip_frequency=clip_frequency,
            largest=largest,
            top_five=top_five,
        ),
        train_control=train_control(nmse),
        one_step=one_step(nmse),
        profiles=profiles(
            scale=scale,
            binary_rate=binary_rate,
        ),
        control=control_record(),
        spec=TopKCalibrationSpec(),
    )


def test_spec_validates():
    TopKCalibrationSpec().validate()


def test_spec_rejects_unsorted_k():
    with pytest.raises(ValueError):
        TopKCalibrationSpec(
            top_k_values=(16, 8),
        ).validate()


def test_spec_rejects_large_k():
    with pytest.raises(ValueError):
        TopKCalibrationSpec(
            top_k_values=(8, 100),
        ).validate()


def test_spec_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        TopKCalibrationSpec(
            target_gradient_ratios=(0.25, 1.5),
        ).validate()


def test_candidate_validates():
    candidate().validate()


def test_candidate_rejects_negative_lambda():
    with pytest.raises(ValueError):
        candidate(lambda_upper=-0.1).validate()


def test_k_token():
    assert k_token(16) == "16"


def test_ratio_token():
    assert ratio_token(0.25) == "0p25"


def test_topk_loss_zero_for_valid_geometry():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.04),
        dtype=torch.float32,
    )
    result = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=8,
    )
    assert float(result["total"].item()) == 0.0


def test_topk_loss_positive_for_expansion():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    )
    result = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=8,
    )
    assert float(result["total"].item()) > 0.0


def test_topk_loss_ignores_contraction():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.001),
        dtype=torch.float32,
    )
    result = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=8,
    )
    assert float(result["total"].item()) == 0.0


def test_topk_selected_count():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    )
    result = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=8,
    )
    assert result["selected_excess"].shape == (2, 8)


def test_topk_larger_k_has_no_larger_mean_for_uniform_order():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    )
    eight = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=8,
    )
    sixteen = topk_quadratic_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        objective_contract=objective_contract(),
        top_k=16,
    )
    assert float(sixteen["total"].item()) <= (
        float(eight["total"].item()) + 1.0e-7
    )


def test_row_contribution_detects_dominant_row():
    result = row_contribution_profile(
        np.asarray([100.0, 1.0, 1.0]),
        epsilon=1.0e-12,
    )
    assert (
        result["largest_row_contribution_fraction"]
        > 0.9
    )


def test_row_contribution_uniform():
    result = row_contribution_profile(
        np.ones(10),
        epsilon=1.0e-12,
    )
    assert (
        result["largest_row_contribution_fraction"]
        == pytest.approx(0.1)
    )


def test_selected_position_profile():
    result = selected_position_profile(
        np.asarray(
            [[0, 1, 2], [0, 1, 3]],
            dtype=np.int64,
        )
    )
    assert result["top_positions"][0]["count"] == 2
    assert sum(
        result["horizon_selection_fraction"]
    ) == pytest.approx(1.0)


def test_compatible_training_record_has_expected_fields():
    result = compatible_training_record(
        training_record()
    )
    assert result["training_steps"] == 8000
    assert "final_model_sha256" in result


def test_reduction():
    assert reduction(0.7, 1.0) == pytest.approx(0.3)


def test_ratio():
    assert ratio(1.2, 1.0) == pytest.approx(1.2)


def test_control_record_is_not_eligible():
    result = control_record()
    assert result["candidate_id"] == CONTROL_CANDIDATE_ID
    assert not result["eligible_for_selection"]


def test_candidate_can_pass_all_gates():
    result = candidate_record(
        scale=0.7,
        binary_rate=0.8,
        nmse=1.1,
    )
    assert result["continuous_geometry_pass"]
    assert result["fidelity_pass"]
    assert result["binary_pass"]
    assert result["stability_pass"]
    assert result["eligible_for_selection"]


def test_candidate_fails_fidelity():
    result = candidate_record(
        scale=0.7,
        binary_rate=0.8,
        nmse=1.5,
    )
    assert result["continuous_geometry_pass"]
    assert not result["fidelity_pass"]
    assert not result["eligible_for_selection"]


def test_candidate_fails_binary():
    result = candidate_record(
        scale=0.7,
        binary_rate=0.0,
        nmse=1.1,
    )
    assert result["continuous_geometry_pass"]
    assert result["fidelity_pass"]
    assert not result["binary_pass"]


def test_candidate_fails_stability():
    result = candidate_record(
        scale=0.7,
        binary_rate=0.8,
        nmse=1.1,
        largest=0.5,
    )
    assert not result["stability_pass"]
    assert not result["eligible_for_selection"]


def test_candidate_fails_continuous_response():
    result = candidate_record(
        scale=0.95,
        binary_rate=0.8,
        nmse=1.1,
    )
    assert not result["continuous_geometry_pass"]


def test_select_configuration_prefers_binary_rate():
    first = candidate_record(
        scale=0.6,
        binary_rate=0.8,
        candidate_value=candidate(
            candidate_id="a",
        ),
    )
    second = candidate_record(
        scale=0.7,
        binary_rate=0.9,
        candidate_value=candidate(
            candidate_id="b",
            target_ratio=0.5,
        ),
    )
    selected = select_configuration(
        [control_record(), first, second]
    )
    assert selected is not None
    assert selected["candidate_id"] == "b"


def test_select_configuration_returns_none():
    selected = select_configuration(
        [
            control_record(),
            candidate_record(
                scale=0.95,
                binary_rate=0.0,
            ),
        ]
    )
    assert selected is None


def test_candidate_conflict_detects_cosine():
    record = candidate_record(
        scale=0.7,
        cosine=-0.5,
        conflict=0.2,
    )
    assert candidate_conflict_severe(
        record,
        TopKCalibrationSpec(),
    )


def test_candidate_conflict_detects_fraction():
    record = candidate_record(
        scale=0.7,
        cosine=0.2,
        conflict=0.8,
    )
    assert candidate_conflict_severe(
        record,
        TopKCalibrationSpec(),
    )


def test_classification_no_response():
    record = candidate_record(
        scale=0.95,
        binary_rate=0.0,
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=None,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == (
        "geometry_response"
    )


def test_classification_fidelity_tradeoff():
    record = candidate_record(
        scale=0.7,
        binary_rate=0.0,
        nmse=1.5,
        cosine=0.2,
        conflict=0.2,
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=None,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == (
        "fidelity_tradeoff"
    )


def test_classification_gradient_conflict():
    record = candidate_record(
        scale=0.7,
        binary_rate=0.0,
        nmse=1.5,
        cosine=-0.5,
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=None,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == (
        "gradient_conflict"
    )


def test_classification_outlier_dominance():
    record = candidate_record(
        scale=0.7,
        binary_rate=0.0,
        nmse=1.1,
        largest=0.5,
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=None,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == (
        "quadratic_outlier_dominance"
    )


def test_classification_continuous_without_gate():
    record = candidate_record(
        scale=0.7,
        binary_rate=0.0,
        nmse=1.1,
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=None,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == (
        "binary_gate_crossing"
    )


def test_classification_selected():
    record = candidate_record(
        scale=0.7,
        binary_rate=0.8,
        nmse=1.1,
    )
    selected = select_configuration(
        [control_record(), record]
    )
    result = classify_selection(
        records=[control_record(), record],
        selected=selected,
        spec=TopKCalibrationSpec(),
    )
    assert result["primary_failure_locus"] == "none"


def test_compare_worker_results_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "split": {"x": 1},
        "calibration_contract": {"x": 1},
        "selection": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    assert compare_worker_results(
        base,
        dict(base),
    )["exact"]


def test_stable_json_is_deterministic():
    assert stable_json_bytes(
        {"b": 2, "a": 1}
    ) == stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_source_logic_audit_accepts_frozen_sources():
    repository_root = (
        Path(__file__).resolve().parents[1]
    )
    result = source_logic_audit(
        repository_root
    )
    assert result["all_confirmed"]
    assert result["softplus_selected"] is False
