from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from ccda_phase3 import phase314b_r258_stageb_direct_x0_reachability as stageb258


def make_standardizer():
    return stageb258.stageb.ArrayStandardizer(
        mean=np.zeros((4, 48), dtype=np.float32),
        scale=np.ones((4, 48), dtype=np.float32),
        active=np.ones((4, 48), dtype=np.bool_),
    )


def make_objective_contract(allowed=0.0):
    contract = stageb258.stagea.UpperObjectiveContract(
        upper_gate_contract_sha256="a" * 64,
        upper_threshold=1.0,
        reference_center_log=np.full(
            (4, 23),
            float(allowed - 1.0),
            dtype=np.float32,
        ),
        reference_scale_log=np.ones(
            (4, 23),
            dtype=np.float32,
        ),
        allowed_upper_log_length=np.full(
            (4, 23),
            float(allowed),
            dtype=np.float32,
        ),
        huber_delta_log_ratio=0.1,
        element_loss_weight=0.25,
        segment_length_epsilon=1.0e-8,
    )
    contract.validate()
    return contract


def simple_cable(length=1.0, rows=2):
    points = np.zeros(
        (rows, 4, 24, 2),
        dtype=np.float32,
    )
    points[..., 0] = (
        np.arange(24, dtype=np.float32)
        * float(length)
    )
    return points.reshape(rows, 4, 48)


def fake_timestep_record(
    *,
    timestep,
    target_pass=True,
    line=True,
    upper=False,
    valid=False,
    gradient_cosine=0.1,
    tangent=0.5,
    head=0.9,
    stable=True,
):
    return {
        "timestep": timestep,
        "target_line": {
            "target_anchor": {
                "upper_row_pass_rate": (
                    1.0 if target_pass else 0.0
                ),
                "historical_physical_row_any_rate": (
                    1.0 if target_pass else 0.0
                ),
            },
            "aggregate_line_reachable": line,
        },
        "direct_x0_oracle": {
            "upper_fidelity_reachable": upper,
            "cable_valid_reachable": valid,
        },
        "geometry_gradient_alignment": {
            "descent_target_cosine": {
                "mean": gradient_cosine,
            }
        },
        "tangent_reachability": {
            "parameter_groups": {
                "global": {
                    "authoritative_record": {
                        "explained_ratio": tangent,
                    },
                    "finite_difference_stable": stable,
                }
            },
            "iterative_global_tangent_lower_bound": {
                "final_explained_ratio": tangent,
            },
            "output_head_linear": {
                "explained_ratio": head,
                "relative_parameter_update_norm": 0.5,
            },
        },
    }


def fake_result():
    return {
        "root_cause": "root",
        "required_next_path": "next",
        "environment": {"x": 1},
        "cold_main_worker_context": {"x": 1},
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "reachability_contract": {"x": 1},
        "timestep_records": {
            "10": {
                "target_line": {"x": 1},
                "direct_x0_oracle": {"x": 1},
                "tangent_reachability": {"x": 1},
            }
        },
        "classification": {"x": 1},
        "selection": {"x": 1},
    }


def test_phase_and_commit_are_frozen():
    assert stageb258.PHASE == (
        "Phase3.14b-r2.5.8 Stage B"
    )
    assert stageb258.BASE_EVIDENCE_COMMIT == (
        "4d5773d070b99f0eccdb97c1bc9fa7c5cb3bc583"
    )


def test_base_identity_constants_are_frozen():
    assert stageb258.EXPECTED_BASE_WORKER_SHA256 == (
        "c71590b7557da309cc02b3fc9e6c4387"
        "f815eb7800d4307f76ce84d238c13149"
    )
    assert stageb258.EXPECTED_BASE_SELECTION_SHA256 == (
        "62e2800808620dc48858dec5105f97fca"
        "71ac25a5ca987242ae5dc2a9cc71a4f"
    )


def test_spec_validates():
    stageb258.ReachabilitySpec().validate()


def test_spec_rejects_changed_timesteps():
    spec = stageb258.ReachabilitySpec(
        timesteps=(10, 25),
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_spec_rejects_changed_topk():
    spec = stageb258.ReachabilitySpec(
        top_k=8,
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_spec_rejects_unordered_radii():
    spec = stageb258.ReachabilitySpec(
        oracle_radii=(0.1, 0.05),
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_spec_rejects_checkpoint_mismatch():
    spec = stageb258.ReachabilitySpec(
        oracle_checkpoints=(0, 1, 255),
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_stable_json_is_deterministic():
    assert stageb258.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == stageb258.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "value.json"
    stageb258.atomic_write_once(path, b"{}\n")
    assert path.read_bytes() == b"{}\n"
    with pytest.raises(FileExistsError):
        stageb258.atomic_write_once(path, b"{}\n")


def test_safe_stats_empty():
    result = stageb258._safe_stats(
        np.asarray([], dtype=np.float64)
    )
    assert result["count"] == 0
    assert result["mean"] == 0.0


def test_safe_stats_values():
    result = stageb258._safe_stats(
        np.asarray([1.0, 2.0, 3.0])
    )
    assert result["count"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["median"] == pytest.approx(2.0)


def test_safe_stats_rejects_nonfinite():
    with pytest.raises(ValueError):
        stageb258._safe_stats(
            np.asarray([1.0, np.nan])
        )


def test_row_cosine_parallel():
    result = stageb258._row_cosine(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[2.0, 0.0]]),
        epsilon=1.0e-12,
    )
    assert result[0] == pytest.approx(1.0)


def test_row_cosine_opposite():
    result = stageb258._row_cosine(
        np.asarray([[1.0, 0.0]]),
        np.asarray([[-2.0, 0.0]]),
        epsilon=1.0e-12,
    )
    assert result[0] == pytest.approx(-1.0)


def test_topk_upper_profile_zero_excess():
    value = simple_cable(
        length=1.0,
        rows=2,
    )
    contract = make_objective_contract(
        allowed=1.0,
    )
    result = (
        stageb258.topk_upper_profile_numpy(
            value,
            objective_contract=contract,
            top_k=16,
        )
    )
    assert result["total"] == pytest.approx(0.0)
    assert result["row_violation_rate"] == 0.0


def test_topk_upper_profile_positive_excess():
    value = simple_cable(
        length=2.0,
        rows=2,
    )
    contract = make_objective_contract(
        allowed=0.0,
    )
    result = (
        stageb258.topk_upper_profile_numpy(
            value,
            objective_contract=contract,
            top_k=16,
        )
    )
    assert result["total"] > 0.0
    assert result["row_violation_rate"] == 1.0


def test_explained_ratio_exact():
    result = stageb258._explained_ratio_along(
        np.asarray([1.0, 2.0]),
        np.asarray([2.0, 4.0]),
        epsilon=1.0e-12,
    )
    assert result["explained_ratio"] == pytest.approx(1.0)
    assert result["cosine"] == pytest.approx(1.0)


def test_explained_ratio_orthogonal():
    result = stageb258._explained_ratio_along(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        epsilon=1.0e-12,
    )
    assert result["explained_ratio"] == pytest.approx(0.0)
    assert result["cosine"] == pytest.approx(0.0)


def test_deterministic_tangent_rows_balances_conditions():
    eligible = np.asarray(
        [True, True, True, True],
        dtype=np.bool_,
    )
    groups = np.asarray(
        ["g2", "g1", "g4", "g3"]
    )
    conditions = np.asarray(
        ["a", "a", "b", "b"]
    )
    selected = (
        stageb258.deterministic_tangent_rows(
            eligible=eligible,
            groups=groups,
            condition_name=conditions,
            count=4,
        )
    )
    assert selected.tolist() == [1, 3, 0, 2]


def test_deterministic_tangent_rows_rejects_empty():
    with pytest.raises(
        stageb258.ReachabilityAuditError
    ):
        stageb258.deterministic_tangent_rows(
            eligible=np.zeros(3, dtype=np.bool_),
            groups=np.asarray(["a", "b", "c"]),
            condition_name=np.asarray(["x", "x", "x"]),
            count=2,
        )


def test_project_trust_region_limits_rms():
    value = torch.full(
        (2, 4, 48),
        2.0,
        dtype=torch.float32,
    )
    control = torch.zeros_like(value)
    active = torch.ones(
        4 * 48,
        dtype=torch.bool,
    )
    stageb258._project_normalized_trust_region(
        value=value,
        control=control,
        active_flat=active,
        radius=0.5,
        epsilon=1.0e-12,
    )
    rms = torch.sqrt(
        torch.mean(
            value.reshape(2, -1) ** 2,
            dim=1,
        )
    )
    assert torch.allclose(
        rms,
        torch.full_like(rms, 0.5),
        atol=1.0e-5,
    )


def test_project_trust_region_freezes_inactive():
    value = torch.ones(
        (1, 1, 4),
        dtype=torch.float32,
    )
    control = torch.zeros_like(value)
    active = torch.tensor(
        [True, False, True, False]
    )
    stageb258._project_normalized_trust_region(
        value=value,
        control=control,
        active_flat=active,
        radius=0.25,
        epsilon=1.0e-12,
    )
    flattened = value.reshape(-1)
    assert flattened[1] == 0.0
    assert flattened[3] == 0.0


@pytest.mark.parametrize(
    "timestep,threshold",
    [(10, 0.75), (25, 0.60), (50, 0.50)],
)
def test_timestep_thresholds(timestep, threshold):
    assert (
        stageb258.TIMESTEP_UPPER_MIN[timestep]
        == threshold
    )


def test_capture_portable_control_model(monkeypatch):
    model = object()
    original = (
        stageb258.stagec_topk.train_candidate
    )

    def fake_original(*args, **kwargs):
        return (
            model,
            {
                "initial_model_sha256":
                    stageb258
                    .EXPECTED_CONTROL_INITIAL_MODEL_SHA256,
            },
            {"x": 1},
        )

    monkeypatch.setattr(
        stageb258.stagec_topk,
        "train_candidate",
        fake_original,
    )

    def fake_replay(*, root):
        result = (
            stageb258.stagec_topk
            .train_candidate()
        )
        assert result[0] is model
        return {
            "control_bundle": {"x": 1},
            "control_identity": {
                "reference_equivalence_pass": True,
                "stagec_nonzero_candidate_training_count": 0,
                "observed_calibration_sha256":
                    stageb258
                    .EXPECTED_STAGEC_CALIBRATION_SHA256,
                "observed_control_sha256":
                    stageb258
                    .EXPECTED_STAGEC_CONTROL_SHA256,
            },
        }

    monkeypatch.setattr(
        stageb258.stagea258,
        "replay_stagec_control_portable",
        fake_replay,
    )
    result = (
        stageb258.capture_portable_control_model(
            root=Path("/tmp")
        )
    )
    assert result["model"] is model
    assert result["train_candidate_call_count"] == 1
    assert result["capture_wrapper_restored"]
    assert (
        stageb258.stagec_topk.train_candidate
        is fake_original
    )
    monkeypatch.setattr(
        stageb258.stagec_topk,
        "train_candidate",
        original,
    )


def test_capture_rejects_nonzero_candidate(monkeypatch):
    def fake_original(*args, **kwargs):
        return (
            object(),
            {
                "initial_model_sha256":
                    stageb258
                    .EXPECTED_CONTROL_INITIAL_MODEL_SHA256,
            },
            {},
        )

    monkeypatch.setattr(
        stageb258.stagec_topk,
        "train_candidate",
        fake_original,
    )

    def fake_replay(*, root):
        stageb258.stagec_topk.train_candidate()
        return {
            "control_bundle": {},
            "control_identity": {
                "reference_equivalence_pass": True,
                "stagec_nonzero_candidate_training_count": 1,
                "observed_calibration_sha256":
                    stageb258
                    .EXPECTED_STAGEC_CALIBRATION_SHA256,
                "observed_control_sha256":
                    stageb258
                    .EXPECTED_STAGEC_CONTROL_SHA256,
            },
        }

    monkeypatch.setattr(
        stageb258.stagea258,
        "replay_stagec_control_portable",
        fake_replay,
    )
    with pytest.raises(
        stageb258.ReachabilityAuditError
    ):
        stageb258.capture_portable_control_model(
            root=Path("/tmp")
        )


def records_for(**kwargs):
    return {
        str(t): fake_timestep_record(
            timestep=t,
            **kwargs,
        )
        for t in (10, 25, 50)
    }


def test_classification_target_failure():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            target_pass=False,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "target_or_gate"


def test_classification_line_failure():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            line=False,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "target_line_reachability"


def test_classification_nonphysical_shortcut():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=True,
            valid=False,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "one_sided_objective_shortcut"


def test_classification_objective_failure_even_when_tangent_low():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=False,
            valid=False,
            tangent=0.1,
            head=0.2,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "objective_direction_or_conditioning"


def test_classification_objective_failure():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=False,
            valid=False,
            tangent=0.6,
            head=0.9,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "objective_direction_or_conditioning"


def test_classification_tangent_underexpressive():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=True,
            valid=True,
            tangent=0.1,
            head=0.5,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "registered_tangent_lower_bound"


def test_classification_optimization_trajectory():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=True,
            valid=True,
            tangent=0.9,
            head=0.95,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "optimization_trajectory"


def test_classification_mixed_tangent():
    result = stageb258.classify_reachability(
        timestep_records=records_for(
            upper=True,
            valid=True,
            tangent=0.5,
            head=0.8,
        ),
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "mixed_tangent_support"


def test_classification_timestep_conditional():
    records = records_for(
        upper=True,
        valid=True,
        tangent=0.9,
        head=0.95,
    )
    records["25"][
        "direct_x0_oracle"
    ]["cable_valid_reachable"] = False
    records["25"][
        "direct_x0_oracle"
    ]["upper_fidelity_reachable"] = False
    result = stageb258.classify_reachability(
        timestep_records=records,
        spec=stageb258.ReachabilitySpec(),
    )
    assert result[
        "primary_failure_locus"
    ] == "timestep_conditional"


def test_public_target_line_removes_internal():
    result = stageb258._public_target_line(
        {
            "x": 1,
            "_row_first_alpha":
                np.asarray([1.0]),
        }
    )
    assert result == {"x": 1}


def test_public_oracle_removes_internal():
    result = stageb258._public_oracle(
        {
            "x": 1,
            "_outputs": {"a": np.zeros(1)},
        }
    )
    assert result == {"x": 1}


def test_identity_projection_contains_reachability():
    projection = stageb258.identity_projection(
        fake_result()
    )
    assert projection[
        "reachability_contract"
    ] == {"x": 1}
    assert projection[
        "timestep_records"
    ] == {
        "10": {
            "target_line": {"x": 1},
            "direct_x0_oracle": {"x": 1},
            "tangent_reachability": {"x": 1},
        }
    }


def test_compare_worker_results_exact():
    left = fake_result()
    right = copy.deepcopy(left)
    result = (
        stageb258.compare_worker_results(
            left,
            right,
        )
    )
    assert result["exact"]
    assert result["target_line_exact"]
    assert result["oracle_exact"]
    assert result["tangent_exact"]


def test_compare_worker_results_detects_tangent():
    left = fake_result()
    right = copy.deepcopy(left)
    left["timestep_records"] = {
        "10": {
            "target_line": {"x": 1},
            "direct_x0_oracle": {"x": 1},
            "tangent_reachability": {"x": 1},
        }
    }
    right["timestep_records"] = {
        "10": {
            "target_line": {"x": 1},
            "direct_x0_oracle": {"x": 1},
            "tangent_reachability": {"x": 2},
        }
    }
    result = (
        stageb258.compare_worker_results(
            left,
            right,
        )
    )
    assert not result["exact"]
    assert not result["tangent_exact"]


def test_output_head_lstsq_explains_full_rank():
    design = np.asarray(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    desired = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )
    solution, _, _, _ = np.linalg.lstsq(
        design,
        desired,
        rcond=1.0e-10,
    )
    predicted = design @ solution
    result = stageb258._explained_ratio_along(
        predicted,
        desired,
        epsilon=1.0e-12,
    )
    assert result["explained_ratio"] == pytest.approx(
        1.0
    )


def test_upper_objective_contract_is_one_sided():
    contract = make_objective_contract(
        allowed=0.0
    )
    assert contract.to_dict()[
        "uses_lower_xy_constraint"
    ] is False


def test_direct_x0_model_source_contract():
    source = Path(
        stageb258.stageb.__file__
    ).read_text(encoding="utf-8")
    assert (
        "predicted = flat + self.output(self.output_norm(hidden))"
        in source
    )


def test_output_head_grouped_cv_generalizes_linear_mapping(monkeypatch):
    rows = 8
    hidden = torch.stack(
        [
            torch.arange(rows, dtype=torch.float32),
            torch.ones(rows, dtype=torch.float32),
        ],
        dim=1,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(2, 4 * 48)
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

    model = Tiny()

    def forward_fn():
        return model.output(hidden).reshape(rows, 4, 48)

    desired = np.zeros((rows, 4, 48), dtype=np.float32)
    desired[:, 0, 0] = (
        2.0 * np.arange(rows, dtype=np.float32) + 3.0
    )
    monkeypatch.setattr(
        stageb258,
        "evaluate_output_population",
        lambda *args, **kwargs: {"pass": True},
    )
    result = stageb258.output_head_linear_reachability(
        model=model,
        forward_fn=forward_fn,
        control_z=np.zeros_like(desired),
        target=np.zeros_like(desired),
        control=np.zeros_like(desired),
        desired_z=desired,
        selected_rows=np.arange(rows, dtype=np.int64),
        context={
            "holdout_groups": np.asarray(
                ["g{}".format(index) for index in range(rows)]
            ),
            "target_standardizer": make_standardizer(),
            "holdout_condition_name": np.asarray(["a"] * rows),
            "objective_contract": make_objective_contract(),
            "upper_gate": object(),
            "stage_d_contract": object(),
            "historical_geometry": object(),
        },
        spec=stageb258.ReachabilitySpec(),
    )
    assert result["cross_validation_folds"] == 4
    assert result["explained_ratio"] > 0.999
    assert result["model_weights_modified"] is False


def test_output_head_grouped_cv_rejects_same_row_memorization(monkeypatch):
    rows = 8
    hidden = torch.eye(rows, dtype=torch.float32)

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(rows, 4 * 48)
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

    model = Tiny()

    def forward_fn():
        return model.output(hidden).reshape(rows, 4, 48)

    desired = np.zeros((rows, 4, 48), dtype=np.float32)
    for index in range(rows):
        desired[index, 0, index] = 1.0
    monkeypatch.setattr(
        stageb258,
        "evaluate_output_population",
        lambda *args, **kwargs: {"pass": True},
    )
    result = stageb258.output_head_linear_reachability(
        model=model,
        forward_fn=forward_fn,
        control_z=np.zeros_like(desired),
        target=np.zeros_like(desired),
        control=np.zeros_like(desired),
        desired_z=desired,
        selected_rows=np.arange(rows, dtype=np.int64),
        context={
            "holdout_groups": np.asarray(
                ["g{}".format(index) for index in range(rows)]
            ),
            "target_standardizer": make_standardizer(),
            "holdout_condition_name": np.asarray(["a"] * rows),
            "objective_contract": make_objective_contract(),
            "upper_gate": object(),
            "stage_d_contract": object(),
            "historical_geometry": object(),
        },
        spec=stageb258.ReachabilitySpec(),
    )
    assert result["in_sample_upper_bound"]["explained_ratio"] > 0.999
    assert result["explained_ratio"] < 0.5


def test_iterative_global_tangent_lower_bound_restores_model():
    rows = 4
    inputs = torch.arange(1, rows + 1, dtype=torch.float32).reshape(rows, 1)
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.5)

    def forward_fn():
        return model(inputs).reshape(rows, 1, 1)

    desired = (
        3.0 * inputs.detach().cpu().numpy()
    ).reshape(rows, 1, 1).astype(np.float32)
    before = stageb258.stageb.tensor_state_sha256(model)
    result = stageb258.iterative_global_tangent_reachability(
        model=model,
        named_parameters=list(model.named_parameters()),
        forward_fn=forward_fn,
        desired_z=desired,
        selected_rows=np.arange(rows, dtype=np.int64),
        active_flat=np.asarray([True]),
        spec=stageb258.ReachabilitySpec(
            tangent_krylov_iterations=3,
        ),
    )
    after = stageb258.stageb.tensor_state_sha256(model)
    assert result["final_explained_ratio"] > 0.999
    assert result["model_restored_exact"]
    assert before == after


def test_target_line_uses_first_cable_valid_upper_point(monkeypatch):
    rows = 1
    control = np.zeros((rows, 4, 48), dtype=np.float32)
    target = np.ones_like(control)

    def fake_evaluate(candidate, **kwargs):
        alpha = float(candidate[0, 0, 0])
        return {
            "upper_row_pass_rate": float(alpha >= 0.5),
            "lower_row_pass_rate": 1.0,
            "historical_physical_row_any_rate": float(alpha >= 1.0),
            "normalized_mse": float((1.0 - alpha) ** 2),
            "normalized_mse_ratio": float((1.0 - alpha) ** 2),
            "topk_upper": {"total": float(1.0 - alpha)},
            "segment_length": {"collapse_fraction": 0.0},
            "output_sha256": "a" * 64,
        }

    def fake_scores(candidate, reference):
        alpha = float(candidate[0, 0, 0])
        return {
            "upper": np.asarray(
                [[0.0 if alpha >= 0.5 else 2.0]],
                dtype=np.float64,
            )
        }

    def fake_physical(candidate, contract):
        alpha = float(candidate[0, 0, 0, 0])
        return {
            "valid": np.asarray([[alpha >= 1.0]], dtype=np.bool_),
        }

    monkeypatch.setattr(stageb258, "evaluate_output_population", fake_evaluate)
    monkeypatch.setattr(stageb258.staged, "segment_scores", fake_scores)
    monkeypatch.setattr(stageb258.stageb, "physical_validity", fake_physical)
    result = stageb258.target_line_audit(
        control=control,
        target=target,
        timestep=50,
        context={
            "holdout_groups": np.asarray(["g"]),
            "holdout_condition_name": np.asarray(["a"]),
            "target_standardizer": object(),
            "objective_contract": object(),
            "upper_gate": type("Gate", (), {"upper_threshold": 1.0})(),
            "stage_d_contract": type("Gate", (), {"reference": object()})(),
            "historical_geometry": object(),
        },
        spec=stageb258.ReachabilitySpec(),
    )
    assert result["row_first_upper_alpha"]["mean"] == pytest.approx(0.5)
    assert result["row_first_valid_alpha"]["mean"] == pytest.approx(1.0)
    assert result["_row_first_alpha"][0] == pytest.approx(1.0)
