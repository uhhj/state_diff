from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged3_resume1_upper_gate_freeze as staged3
from ccda_phase3.phase314b_r257_stagea_upper_objective import (
    UpperObjectiveContract,
    UpperObjectiveSpec,
    build_objective_contract,
    candidate_diversity,
    classify_final,
    compare_worker_results,
    deterministic_selection_split,
    huber_positive_torch,
    load_upper_gate_contract,
    select_configuration,
    selection_record,
    sha256_array,
    source_logic_audit,
    stable_json_bytes,
    upper_objective_terms_torch,
)


def reference() -> staged.SegmentReference:
    shape = (stageb.FUTURE_STEPS, stageb.BEADS - 1)
    value = staged.SegmentReference(
        center_log=np.full(shape, np.log(0.04), dtype=np.float32),
        scale_log=np.full(shape, 0.01, dtype=np.float32),
        mad_scale=np.full(shape, 0.01, dtype=np.float32),
        iqr_scale=np.full(shape, 0.01, dtype=np.float32),
        scale_floor=0.01,
    )
    value.validate()
    return value


def upper_gate() -> staged3.OneSidedUpperContract:
    value = staged3.OneSidedUpperContract(
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
    value.validate()
    return value


def upper_gate_payload():
    gate = upper_gate()
    payload = gate.to_dict()
    payload["contract_sha256"] = "f" * 64
    return payload


def objective_contract() -> UpperObjectiveContract:
    return build_objective_contract(
        upper_gate=upper_gate(),
        upper_gate_payload=upper_gate_payload(),
        spec=UpperObjectiveSpec(),
    )


def standardizer() -> stageb.ArrayStandardizer:
    value = stageb.ArrayStandardizer(
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
    value.validate((stageb.FUTURE_STEPS, stageb.CABLE_DIM))
    return value


def cable_batch(spacing: float, rows: int = 2) -> np.ndarray:
    x = np.arange(stageb.BEADS, dtype=np.float32) * float(spacing)
    y = np.zeros(stageb.BEADS, dtype=np.float32)
    flat = np.stack([x, y], axis=1).reshape(stageb.CABLE_DIM)
    trajectory = np.repeat(
        flat[None],
        stageb.FUTURE_STEPS,
        axis=0,
    )
    return np.repeat(trajectory[None], rows, axis=0)


def one_step_record(
    upper_rate: float,
    nmse: float,
):
    return {
        "prediction_sha256": "a" * 64,
        "timesteps": {
            timestep: {
                "normalized_mse": nmse,
                "raw_rmse": nmse,
                "upper": {
                    "upper_segment": {
                        "row_any_rate": upper_rate,
                    }
                },
                "prediction_sha256": "b" * 64,
            }
            for timestep in ("10", "25", "50")
        },
    }


def training_record():
    return {
        "loss_finite": True,
        "gradient_finite": True,
        "final_model_sha256": "a" * 64,
    }


def control_record(nmse: float = 0.001):
    return {
        "normalized_mse": nmse,
        "raw_rmse": nmse,
        "prediction_sha256": "c" * 64,
    }


def final_bundle(
    *,
    control_nmse: float = 0.001,
    t10_upper: float = 1.0,
    one_step_nmse: float = 0.01,
    reverse_upper: float = 1.0,
    reverse_combined: float = 1.0,
    reverse_nmse: float = 0.5,
    diversity: float = 1.0,
    eligible: float = 1.0,
    support: float = 1.0,
):
    one_step = one_step_record(t10_upper, one_step_nmse)
    reverse = {
        "best_of_k_normalized_mse": reverse_nmse,
        "diversity": {
            "row_mean_pairwise_distance": {
                "mean": diversity,
            }
        },
        "final": {
            "upper_segment": {"row_any_rate": reverse_upper},
            "combined": {"row_any_rate": reverse_combined},
        },
        "physical_branch": {
            "k8": {
                "eligible_row_rate": eligible,
                "support_rate_among_eligible": support,
            }
        },
    }
    return {
        "train_control": control_record(control_nmse),
        "one_step": one_step,
        "reverse": reverse,
    }


def selected():
    return {
        "lambda_upper": 0.01,
        "selection_rule": "x",
        "pilot_model_sha256": "a" * 64,
        "selection_record_sha256": "b" * 64,
    }


def test_spec_validates():
    UpperObjectiveSpec().validate()


def test_spec_rejects_missing_zero_control():
    with pytest.raises(ValueError):
        UpperObjectiveSpec(
            candidate_lambdas=(0.001, 0.01),
        ).validate()


def test_spec_rejects_unsorted_lambdas():
    with pytest.raises(ValueError):
        UpperObjectiveSpec(
            candidate_lambdas=(0.0, 0.1, 0.01),
        ).validate()


def test_load_upper_gate_contract_round_trip():
    loaded = load_upper_gate_contract(upper_gate_payload())
    assert loaded.to_dict() == upper_gate().to_dict()


def test_objective_contract_matches_gate_boundary():
    contract = objective_contract()
    expected = (
        contract.reference_center_log.astype(np.float64)
        + contract.upper_threshold
        * contract.reference_scale_log.astype(np.float64)
    ).astype(np.float32)
    np.testing.assert_array_equal(
        contract.allowed_upper_log_length,
        expected,
    )


def test_objective_contract_declares_no_robust_z_loss():
    payload = objective_contract().to_dict()
    assert payload["uses_robust_z_as_loss"] is False
    assert payload["uses_lower_xy_constraint"] is False


def test_selection_split_is_group_disjoint():
    groups = np.asarray(
        [f"g{index // 2}" for index in range(40)]
    )
    train = np.ones(40, dtype=np.bool_)
    left, right, record = deterministic_selection_split(
        groups,
        train,
        folds=4,
        holdout_fold=0,
    )
    assert not np.any(left & right)
    assert np.array_equal(left | right, train)
    assert record["group_overlap"] == 0


def test_selection_split_preserves_nontrain_rows():
    groups = np.asarray([f"g{index}" for index in range(12)])
    train = np.asarray([True] * 8 + [False] * 4)
    left, right, _ = deterministic_selection_split(
        groups,
        train,
        folds=2,
        holdout_fold=0,
    )
    assert not np.any((left | right)[~train])


def test_huber_positive_zero_is_zero():
    torch, _ = stageb._torch_imports()
    value = torch.tensor([-1.0, 0.0], dtype=torch.float32)
    result = huber_positive_torch(value, delta=0.1)
    assert torch.equal(result, torch.zeros_like(result))


def test_huber_positive_quadratic_region():
    torch, _ = stageb._torch_imports()
    value = torch.tensor([0.05], dtype=torch.float32)
    result = huber_positive_torch(value, delta=0.1)
    assert float(result.item()) == pytest.approx(0.0125)


def test_huber_positive_linear_region():
    torch, _ = stageb._torch_imports()
    value = torch.tensor([0.2], dtype=torch.float32)
    result = huber_positive_torch(value, delta=0.1)
    assert float(result.item()) == pytest.approx(0.15)


def test_upper_objective_zero_for_valid_geometry():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.04),
        dtype=torch.float32,
    )
    terms = upper_objective_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    assert float(terms["total"].item()) == 0.0


def test_upper_objective_positive_for_expansion():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    )
    terms = upper_objective_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    assert float(terms["total"].item()) > 0.0
    assert float(terms["row_violation_rate"].item()) == 1.0


def test_upper_objective_ignores_contraction():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.001),
        dtype=torch.float32,
    )
    terms = upper_objective_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    assert float(terms["total"].item()) == 0.0


def test_upper_objective_has_finite_gradient():
    torch, _ = stageb._torch_imports()
    prediction = torch.as_tensor(
        cable_batch(0.08),
        dtype=torch.float32,
    ).requires_grad_(True)
    terms = upper_objective_terms_torch(
        prediction,
        target_standardizer=standardizer(),
        contract=objective_contract(),
    )
    terms["total"].backward()
    assert prediction.grad is not None
    assert torch.all(torch.isfinite(prediction.grad))


def test_selection_control_is_not_feasible():
    record = selection_record(
        lambda_upper=0.0,
        training=training_record(),
        train_control=control_record(),
        one_step=one_step_record(1.0, 0.01),
        control=None,
        spec=UpperObjectiveSpec(),
    )
    assert not record["feasible_nonzero_configuration"]


def test_selection_nonzero_can_be_feasible():
    control = {
        "train_control": control_record(),
        "one_step": one_step_record(0.0, 0.01),
    }
    record = selection_record(
        lambda_upper=0.01,
        training=training_record(),
        train_control=control_record(),
        one_step=one_step_record(1.0, 0.01),
        control=control,
        spec=UpperObjectiveSpec(),
    )
    assert record["feasible_nonzero_configuration"]


def test_selection_rejects_fidelity_regression():
    control = {
        "train_control": control_record(),
        "one_step": one_step_record(0.0, 0.01),
    }
    record = selection_record(
        lambda_upper=0.01,
        training=training_record(),
        train_control=control_record(),
        one_step=one_step_record(1.0, 0.1),
        control=control,
        spec=UpperObjectiveSpec(),
    )
    assert not record["feasible_nonzero_configuration"]


def test_select_configuration_chooses_smallest_lambda():
    records = [
        {
            "lambda_upper": 0.1,
            "feasible_nonzero_configuration": True,
            "one_step": one_step_record(1.0, 0.01),
            "training": {"final_model_sha256": "b" * 64},
        },
        {
            "lambda_upper": 0.01,
            "feasible_nonzero_configuration": True,
            "one_step": one_step_record(1.0, 0.02),
            "training": {"final_model_sha256": "a" * 64},
        },
    ]
    result = select_configuration(records)
    assert result is not None
    assert result["lambda_upper"] == 0.01


def test_select_configuration_returns_none():
    result = select_configuration(
        [
            {
                "lambda_upper": 0.01,
                "feasible_nonzero_configuration": False,
            }
        ]
    )
    assert result is None


def test_candidate_diversity_detects_duplicates():
    candidates = np.repeat(
        cable_batch(0.04, rows=2)[:, None],
        4,
        axis=1,
    )
    result = candidate_diversity(candidates, standardizer())
    assert result["near_duplicate_pair_rate"] == 1.0


def test_candidate_diversity_detects_spread():
    base = cable_batch(0.04, rows=2)
    candidates = np.stack(
        [base + float(index) * 0.01 for index in range(4)],
        axis=1,
    )
    result = candidate_diversity(candidates, standardizer())
    assert result["row_mean_pairwise_distance"]["mean"] > 0.0
    assert result["near_duplicate_pair_rate"] == 0.0


def test_classification_handles_no_selection():
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=None,
        baseline=None,
        repaired=None,
    )
    assert result["primary_failure_locus"] == (
        "train_only_objective_selection"
    )


def test_classification_detects_train_control_regression():
    baseline = final_bundle(control_nmse=0.001)
    repaired = final_bundle(control_nmse=0.01)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == (
        "train_control_tradeoff"
    )


def test_classification_detects_x0_failure():
    baseline = final_bundle()
    repaired = final_bundle(t10_upper=0.0)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == "x0_upper_expansion"


def test_classification_detects_one_step_tradeoff():
    baseline = final_bundle(one_step_nmse=0.01)
    repaired = final_bundle(one_step_nmse=0.02)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == (
        "one_step_fidelity_tradeoff"
    )


def test_classification_detects_reverse_failure():
    baseline = final_bundle()
    repaired = final_bundle(reverse_upper=0.0)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == (
        "reverse_upper_transport"
    )


def test_classification_detects_diversity_collapse():
    baseline = final_bundle(diversity=1.0)
    repaired = final_bundle(diversity=0.1)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == (
        "candidate_diversity"
    )


def test_classification_detects_branch_failure():
    baseline = final_bundle()
    repaired = final_bundle(support=0.5)
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == "branch_transport"


def test_classification_can_reach_supported_path():
    baseline = final_bundle()
    repaired = final_bundle()
    result = classify_final(
        spec=UpperObjectiveSpec(),
        selected=selected(),
        baseline=baseline,
        repaired=repaired,
    )
    assert result["primary_failure_locus"] == "none"


def test_compare_worker_results_is_exact():
    base = {
        "root_cause": "r",
        "required_next_path": "n",
        "split": {"x": 1},
        "objective_contract": {"x": 1},
        "zero_weight_full_replay": {"x": 1},
        "train_only_selection": {"x": 1},
        "final_frozen_probe_audit": {"x": 1},
        "classification": {"x": 1},
        "selected_configuration": None,
    }
    assert compare_worker_results(base, dict(base))["exact"]


def test_stable_json_bytes_is_deterministic():
    assert stable_json_bytes({"b": 2, "a": 1}) == (
        stable_json_bytes({"a": 1, "b": 2})
    )


def test_sha256_array_changes_with_content():
    assert sha256_array(
        np.asarray([1.0], dtype=np.float32)
    ) != sha256_array(
        np.asarray([2.0], dtype=np.float32)
    )


def test_source_logic_audit_accepts_frozen_sources():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    result = source_logic_audit(root)
    assert result["all_confirmed"]
    assert result["epsilon_reconstruction_required"] is False
