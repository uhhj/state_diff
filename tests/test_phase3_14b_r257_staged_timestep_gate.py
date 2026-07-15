from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r257_stagea_upper_objective as stagea
from ccda_phase3.phase314b_r257_staged_timestep_gate import (
    CONTROL_CANDIDATE_ID,
    GatedCandidate,
    TimestepGateSpec,
    candidate_conflict_severe,
    classify_selection,
    compare_worker_results,
    cutoff_token,
    gate_weights_torch,
    gated_topk_quadratic_terms_torch,
    ratio,
    ratio_token,
    reduction,
    select_configuration,
    selection_record,
    source_logic_audit,
    stable_json_bytes,
    stratified_calibration_batch,
)


def reference() -> staged.SegmentReference:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    result = staged.SegmentReference(
        center_log=np.full(shape, np.log(0.04), dtype=np.float32),
        scale_log=np.full(shape, 0.01, dtype=np.float32),
        mad_scale=np.full(shape, 0.01, dtype=np.float32),
        iqr_scale=np.full(shape, 0.01, dtype=np.float32),
        scale_floor=0.01,
    )
    result.validate()
    return result


def objective_contract() -> stagea.UpperObjectiveContract:
    ref = reference()
    result = stagea.UpperObjectiveContract(
        upper_gate_contract_sha256="a" * 64,
        upper_threshold=3.0,
        reference_center_log=ref.center_log.copy(),
        reference_scale_log=ref.scale_log.copy(),
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


def standardizer(shape) -> stageb.ArrayStandardizer:
    result = stageb.ArrayStandardizer(
        mean=np.zeros(shape, dtype=np.float32),
        scale=np.ones(shape, dtype=np.float32),
        active=np.ones(shape, dtype=np.bool_),
    )
    result.validate(shape)
    return result


def cable_batch(spacing: float, rows: int = 4) -> np.ndarray:
    x = np.arange(stageb.BEADS, dtype=np.float32) * float(spacing)
    y = np.zeros(stageb.BEADS, dtype=np.float32)
    flat = np.stack([x, y], axis=1).reshape(stageb.CABLE_DIM)
    trajectory = np.repeat(flat[None], stageb.FUTURE_STEPS, axis=0)
    return np.repeat(trajectory[None], rows, axis=0)


def candidate(
    *,
    candidate_id: str = "k16_t25_r0p50",
    cutoff: int = 25,
    target_ratio: float = 0.5,
    lambda_upper: float = 0.05,
) -> GatedCandidate:
    return GatedCandidate(
        candidate_id=candidate_id,
        timestep_cutoff=cutoff,
        top_k=16,
        target_gradient_ratio=target_ratio,
        lambda_upper=lambda_upper,
        initial_model_sha256="a" * 64,
        diffusion_gradient_norm=1.0,
        gated_geometry_gradient_norm=10.0,
        gated_geometry_gradient_cosine=0.2,
        gated_geometry_gradient_conflict_fraction=0.4,
        calibration_active_count=cutoff + 1,
        calibration_active_fraction=(cutoff + 1) / 100.0,
    )


def training() -> dict:
    return {
        "candidate_id": "k16_t25_r0p50",
        "lambda_upper": 0.05,
        "timestep_cutoff": 25,
        "top_k": 16,
        "initial_model_sha256": "a" * 64,
        "final_model_sha256": "b" * 64,
        "initial_optimizer_sha256": "c" * 64,
        "final_optimizer_sha256": "d" * 64,
        "total_loss_history_sha256": "e" * 64,
        "diffusion_loss_history_sha256": "f" * 64,
        "geometry_loss_history_sha256": "1" * 64,
        "gradient_history_sha256": "2" * 64,
        "source_exposure_sha256": "3" * 64,
        "timestep_exposure_sha256": "4" * 64,
        "active_count_history_sha256": "5" * 64,
        "active_fraction_history_sha256": "6" * 64,
        "total_loss_first": 1.0,
        "total_loss_final": 0.1,
        "total_loss_tail_mean": 0.1,
        "diffusion_loss_tail_mean": 0.1,
        "geometry_loss_tail_mean": 0.1,
        "gradient_tail_mean": 1.0,
        "batch_row_violation_rate_tail_mean": 1.0,
        "maximum_log_excess_tail_max": 1.0,
        "loss_finite": True,
        "gradient_finite": True,
        "training_rows": 638,
        "training_steps": 8000,
    }


def diagnostics(
    *,
    cosine: float = 0.2,
    conflict: float = 0.4,
    clip: float = 0.0,
    largest: float = 0.1,
    top_five: float = 0.2,
    observed: float = 0.26,
    expected: float = 0.26,
) -> dict:
    return {
        "candidate": candidate().to_dict(),
        "gradient_clip_count": 0,
        "gradient_clip_frequency": clip,
        "gate_active_count": {},
        "gate_active_fraction": {},
        "expected_gate_active_fraction": expected,
        "observed_global_active_fraction": observed,
        "zero_active_batch_count": 0,
        "zero_active_batch_fraction": 0.0,
        "timestep_exposure_sha256": "7" * 64,
        "timestep_exposure": [1] * 100,
        "checkpoint_records": {
            "8000": {
                "gradient": {
                    "cosine": cosine,
                    "conflict_fraction_among_active": conflict,
                },
                "row_contribution": {
                    "largest_row_contribution_fraction": largest,
                    "top_five_percent_contribution_fraction": top_five,
                },
            }
        },
        "checkpoint_record_sha256": "8" * 64,
    }


def train_control(nmse: float = 1.0) -> dict:
    return {
        "rows": 64,
        "normalized_mse": nmse,
        "raw_rmse": nmse,
        "prediction_sha256": "9" * 64,
    }


def one_step(nmse: float = 1.0) -> dict:
    return {
        "prediction_sha256": "a" * 64,
        "timesteps": {
            value: {
                "normalized_mse": nmse,
                "raw_rmse": nmse,
                "upper": {},
                "prediction_sha256": "b" * 64,
            }
            for value in ("10", "25", "50")
        },
    }


def profile(scale: float, binary: float) -> dict:
    return {
        "row_best_max_log_excess": {"mean": 1.0 * scale, "p95": 1.2 * scale},
        "top_k_mean_log_excess": {"8": {"mean": 0.8 * scale}},
        "candidate_positive_segment_count": {"mean": 40.0 * scale},
        "row_any_rate": binary,
    }


def profiles(scale: float = 1.0, binary: float = 0.0) -> dict:
    return {value: profile(scale, binary) for value in ("10", "25", "50")}


def control_record() -> dict:
    return {
        "candidate_id": CONTROL_CANDIDATE_ID,
        "candidate": None,
        "training": training(),
        "training_diagnostics": diagnostics(),
        "train_control": train_control(1.0),
        "one_step": one_step(1.0),
        "holdout_profiles": profiles(1.0, 0.0),
    }


def candidate_record(
    *,
    scale: float = 0.7,
    binary: float = 0.8,
    nmse: float = 1.1,
    clip: float = 0.0,
    largest: float = 0.1,
    top_five: float = 0.2,
    cosine: float = 0.2,
    conflict: float = 0.4,
    observed: float = 0.26,
    expected: float = 0.26,
    candidate_value: GatedCandidate = None,
) -> dict:
    active = candidate() if candidate_value is None else candidate_value
    return selection_record(
        candidate=active,
        training=training(),
        training_diagnostics=diagnostics(
            cosine=cosine,
            conflict=conflict,
            clip=clip,
            largest=largest,
            top_five=top_five,
            observed=observed,
            expected=expected,
        ),
        train_control=train_control(nmse),
        one_step=one_step(nmse),
        profiles=profiles(scale, binary),
        control=control_record(),
        spec=TimestepGateSpec(),
    )


def test_spec_validates():
    TimestepGateSpec().validate()


def test_spec_rejects_unsorted_cutoffs():
    with pytest.raises(ValueError):
        TimestepGateSpec(timestep_cutoffs=(25, 10)).validate()


def test_spec_rejects_cutoff_outside_range():
    with pytest.raises(ValueError):
        TimestepGateSpec(timestep_cutoffs=(10, 100)).validate()


def test_spec_rejects_unsorted_ratios():
    with pytest.raises(ValueError):
        TimestepGateSpec(target_gradient_ratios=(1.0, 0.5)).validate()


def test_spec_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        TimestepGateSpec(target_gradient_ratios=(0.5, 1.5)).validate()


def test_spec_rejects_non_stratified_batch_size():
    with pytest.raises(ValueError):
        TimestepGateSpec(calibration_batch_size=64).validate()


def test_candidate_validates():
    candidate().validate()


def test_candidate_rejects_negative_lambda():
    with pytest.raises(ValueError):
        candidate(lambda_upper=-0.1).validate()


def test_candidate_rejects_zero_active_count():
    value = candidate()
    broken = GatedCandidate(**{**value.to_dict(), "calibration_active_count": 0})
    with pytest.raises(ValueError):
        broken.validate()


def test_cutoff_token():
    assert cutoff_token(25) == "25"


def test_ratio_token():
    assert ratio_token(0.5) == "0p50"


def test_gate_weights_include_cutoff():
    torch, _ = stageb._torch_imports()
    timestep = torch.tensor([9, 10, 11])
    value = gate_weights_torch(timestep, 10, dtype=torch.float32)
    assert value.tolist() == [1.0, 1.0, 0.0]


def test_gated_loss_zero_for_valid_geometry():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(cable_batch(0.04), dtype=torch.float32)
    timestep = torch.tensor([10, 25, 50, 75])
    result = gated_topk_quadratic_terms_torch(
        prediction,
        timestep,
        target_standardizer=standardizer((4, 48)),
        objective_contract=objective_contract(),
        top_k=16,
        timestep_cutoff=25,
    )
    assert float(result["total"].item()) == 0.0


def test_gated_loss_positive_for_active_expansion():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(cable_batch(0.08), dtype=torch.float32)
    timestep = torch.tensor([10, 25, 50, 75])
    result = gated_topk_quadratic_terms_torch(
        prediction,
        timestep,
        target_standardizer=standardizer((4, 48)),
        objective_contract=objective_contract(),
        top_k=16,
        timestep_cutoff=25,
    )
    assert float(result["total"].item()) > 0.0
    assert int(result["active_count"].item()) == 2


def test_gated_loss_zero_when_all_rows_inactive():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(cable_batch(0.08), dtype=torch.float32)
    timestep = torch.tensor([50, 60, 70, 80])
    result = gated_topk_quadratic_terms_torch(
        prediction,
        timestep,
        target_standardizer=standardizer((4, 48)),
        objective_contract=objective_contract(),
        top_k=16,
        timestep_cutoff=25,
    )
    assert float(result["total"].item()) == 0.0
    assert bool(result["zero_active"].item())


def test_gated_loss_is_conditional_active_mean():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(cable_batch(0.08), dtype=torch.float32)
    timestep = torch.tensor([10, 50, 60, 70])
    gated = gated_topk_quadratic_terms_torch(
        prediction,
        timestep,
        target_standardizer=standardizer((4, 48)),
        objective_contract=objective_contract(),
        top_k=16,
        timestep_cutoff=10,
    )
    assert float(gated["total"].item()) == pytest.approx(
        float(gated["row_loss"][0].item())
    )


def test_gated_loss_ignores_contraction():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(cable_batch(0.001), dtype=torch.float32)
    timestep = torch.tensor([0, 10, 25, 50])
    result = gated_topk_quadratic_terms_torch(
        prediction,
        timestep,
        target_standardizer=standardizer((4, 48)),
        objective_contract=objective_contract(),
        top_k=16,
        timestep_cutoff=50,
    )
    assert float(result["total"].item()) == 0.0


def test_stratified_batch_is_deterministic():
    condition = np.zeros((638, 201), dtype=np.float32)
    target = cable_batch(0.04, rows=638)
    condition_std = standardizer((201,))
    target_std = standardizer((4, 48))
    first = stratified_calibration_batch(
        condition=condition,
        target=target,
        condition_standardizer=condition_std,
        target_standardizer=target_std,
        stageb_spec=stageb.DiagnosticSpec(),
        spec=TimestepGateSpec(),
    )
    second = stratified_calibration_batch(
        condition=condition,
        target=target,
        condition_standardizer=condition_std,
        target_standardizer=target_std,
        stageb_spec=stageb.DiagnosticSpec(),
        spec=TimestepGateSpec(),
    )
    assert first["noise_sha256"] == second["noise_sha256"]
    assert first["indices_sha256"] == second["indices_sha256"]


def test_stratified_batch_contains_every_timestep_once():
    condition = np.zeros((638, 201), dtype=np.float32)
    target = cable_batch(0.04, rows=638)
    result = stratified_calibration_batch(
        condition=condition,
        target=target,
        condition_standardizer=standardizer((201,)),
        target_standardizer=standardizer((4, 48)),
        stageb_spec=stageb.DiagnosticSpec(),
        spec=TimestepGateSpec(),
    )
    assert result["timesteps"].tolist() == list(range(100))


def test_reduction():
    assert reduction(0.7, 1.0) == pytest.approx(0.3)


def test_ratio():
    assert ratio(1.2, 1.0) == pytest.approx(1.2)


def test_candidate_can_pass_all_gates():
    result = candidate_record(scale=0.7, binary=0.8, nmse=1.1)
    assert result["continuous_geometry_pass"]
    assert result["fidelity_pass"]
    assert result["binary_pass"]
    assert result["stability_pass"]
    assert result["eligible_for_selection"]


def test_candidate_fails_continuous_gate():
    result = candidate_record(scale=0.95, binary=0.8, nmse=1.1)
    assert not result["continuous_geometry_pass"]


def test_candidate_fails_fidelity_gate():
    result = candidate_record(scale=0.7, binary=0.8, nmse=1.5)
    assert not result["fidelity_pass"]


def test_candidate_fails_binary_gate():
    result = candidate_record(scale=0.7, binary=0.0, nmse=1.1)
    assert not result["binary_pass"]


def test_candidate_fails_clip_stability():
    result = candidate_record(scale=0.7, binary=0.8, nmse=1.1, clip=0.5)
    assert not result["stability_pass"]


def test_candidate_fails_row_contribution_stability():
    result = candidate_record(scale=0.7, binary=0.8, nmse=1.1, largest=0.5)
    assert not result["stability_pass"]


def test_candidate_fails_active_fraction_contract():
    result = candidate_record(
        scale=0.7,
        binary=0.8,
        nmse=1.1,
        observed=0.40,
        expected=0.26,
    )
    assert not result["stability_pass"]


def test_select_configuration_prefers_binary_rate():
    first = candidate_record(
        scale=0.6,
        binary=0.8,
        candidate_value=candidate(candidate_id="a", cutoff=10),
    )
    second = candidate_record(
        scale=0.7,
        binary=0.9,
        candidate_value=candidate(candidate_id="b", cutoff=25),
    )
    selected = select_configuration([first, second])
    assert selected is not None
    assert selected["candidate_id"] == "b"


def test_select_configuration_prefers_narrower_cutoff_after_geometry_tie():
    first = candidate_record(
        scale=0.7,
        binary=0.8,
        candidate_value=candidate(candidate_id="a", cutoff=10),
    )
    second = candidate_record(
        scale=0.7,
        binary=0.8,
        candidate_value=candidate(candidate_id="b", cutoff=25),
    )
    selected = select_configuration([first, second])
    assert selected is not None
    assert selected["candidate_id"] == "a"


def test_select_configuration_returns_none():
    selected = select_configuration([candidate_record(scale=0.95, binary=0.0)])
    assert selected is None


def test_candidate_conflict_detects_cosine():
    record = candidate_record(scale=0.7, cosine=-0.5, conflict=0.2)
    assert candidate_conflict_severe(record, TimestepGateSpec())


def test_candidate_conflict_detects_fraction():
    record = candidate_record(scale=0.7, cosine=0.2, conflict=0.8)
    assert candidate_conflict_severe(record, TimestepGateSpec())


def test_classification_selected():
    record = candidate_record(scale=0.7, binary=0.8, nmse=1.1)
    selected = select_configuration([record])
    result = classify_selection(records=[record], selected=selected, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "none"


def test_classification_fidelity_restored_geometry_underpowered():
    record = candidate_record(scale=0.95, binary=0.0, nmse=1.1)
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "geometry_underpowered"


def test_classification_joint_tradeoff():
    record = candidate_record(scale=0.95, binary=0.0, nmse=1.5)
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "joint_tradeoff"


def test_classification_gradient_conflict():
    record = candidate_record(scale=0.7, binary=0.0, nmse=1.5, cosine=-0.5)
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "gradient_conflict"


def test_classification_persistent_fidelity_tradeoff():
    record = candidate_record(
        scale=0.7,
        binary=0.0,
        nmse=1.5,
        cosine=0.2,
        conflict=0.2,
    )
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "fidelity_tradeoff"


def test_classification_stability_failure():
    record = candidate_record(scale=0.7, binary=0.0, nmse=1.1, largest=0.5)
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "stability"


def test_classification_continuous_fidelity_without_binary():
    record = candidate_record(scale=0.7, binary=0.0, nmse=1.1)
    result = classify_selection(records=[record], selected=None, spec=TimestepGateSpec())
    assert result["primary_failure_locus"] == "binary_gate_crossing"


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
    assert compare_worker_results(base, dict(base))["exact"]


def test_stable_json_is_deterministic():
    assert stable_json_bytes({"b": 2, "a": 1}) == stable_json_bytes({"a": 1, "b": 2})


def test_source_logic_audit_accepts_frozen_sources():
    repository_root = Path(__file__).resolve().parents[1]
    result = source_logic_audit(repository_root)
    assert result["all_confirmed"]
    assert result["gate_type"] == "hard_low_noise_timestep_cutoff"
