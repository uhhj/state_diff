from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ccda_phase3 import phase314b_r258_stagec_balanced_geometry as stagec258


def standardizer():
    return stagec258.stageb.ArrayStandardizer(
        mean=np.zeros((4, 48), dtype=np.float32),
        scale=np.ones((4, 48), dtype=np.float32),
        active=np.ones((4, 48), dtype=np.bool_),
    )


def cable(length: float, rows: int = 2):
    points = np.zeros(
        (rows, 4, 24, 2),
        dtype=np.float32,
    )
    points[..., 0] = (
        np.arange(24, dtype=np.float32)
        * float(length)
    )
    return points.reshape(rows, 4, 48)


def candidate(
    *,
    candidate_id="band_q05_w1",
    lower=-0.5,
    upper=0.5,
    anchor_lower=-0.25,
    anchor_upper=0.25,
    lower_weight=1.0,
    anchor_weight=0.0,
    role="selectable",
):
    shape = (
        stagec258.stageb.FUTURE_STEPS,
        stagec258.stageb.BEADS - 1,
    )
    return {
        "candidate": {
            "candidate_id": candidate_id,
            "lower_quantile": 0.05,
            "lower_weight": lower_weight,
            "anchor_weight": anchor_weight,
            "role": role,
        },
        "lower_log_length":
            np.full(shape, lower, dtype=np.float32),
        "upper_log_length":
            np.full(shape, upper, dtype=np.float32),
        "anchor_lower_log_length":
            np.full(shape, anchor_lower, dtype=np.float32),
        "anchor_upper_log_length":
            np.full(shape, anchor_upper, dtype=np.float32),
        "contract_sha256": "a" * 64,
    }


def fake_evaluation(
    *,
    upper=1.0,
    nmse=1.0,
    lower=1.0,
    physical=1.0,
    collapse=0.0,
    stretch=0.0,
    cosine=0.8,
    reduction=0.1,
):
    return {
        "upper_row_pass_rate": upper,
        "normalized_mse_ratio": nmse,
        "lower_row_pass_rate": lower,
        "historical_physical_row_any_rate":
            physical,
        "segment_length": {
            "collapse_fraction": collapse,
            "stretch_fraction": stretch,
        },
        "target_direction_cosine": {
            "mean": cosine,
        },
        "target_distance_reduction_fraction": {
            "mean": reduction,
        },
    }


def fake_oracle_record(**kwargs):
    return {
        "loss_history": {
            "finite": True,
        },
        "final": {
            "evaluation":
                fake_evaluation(**kwargs)
        },
    }


def fake_selectable_record(
    *,
    candidate_id="band_q05_w1",
    eligible=True,
    anchor_weight=0.0,
    lower_weight=1.0,
    lower_quantile=0.05,
    direction=True,
    scientific=True,
    upper_fidelity=True,
    lower_physical=True,
):
    timestep_records = {}
    for timestep in (10, 25, 50):
        witness = (
            {
                "radius": 0.05,
                "final": {
                    "evaluation":
                        fake_evaluation()
                },
            }
            if scientific
            else None
        )
        timestep_records[str(timestep)] = {
            "gradient_alignment": {
                "direction_pass": direction,
                "descent_target_cosine": {
                    "mean": 0.5,
                },
            },
            "oracle": {
                "scientific_reachable":
                    scientific,
                "upper_fidelity_reachable":
                    upper_fidelity,
                "lower_physical_reachable":
                    lower_physical,
                "scientific_witness":
                    witness,
            },
        }
    return {
        "candidate_id": candidate_id,
        "definition": {
            "candidate_id": candidate_id,
            "lower_quantile": lower_quantile,
            "lower_weight": lower_weight,
            "anchor_weight": anchor_weight,
            "role": "selectable",
        },
        "candidate_contract": {
            "contract_sha256": "b" * 64,
        },
        "timestep_records":
            timestep_records,
        "direction_all_pass": direction,
        "scientific_witness_all_timesteps":
            scientific,
        "witness_radii": {
            "10": 0.05 if scientific else None,
            "25": 0.05 if scientific else None,
            "50": 0.05 if scientific else None,
        },
        "eligible": eligible,
        "negative_control": False,
    }


def negative_control():
    return {
        "candidate_id":
            "upper_only_reference",
        "definition": {
            "candidate_id":
                "upper_only_reference",
            "lower_quantile": 0.05,
            "lower_weight": 0.0,
            "anchor_weight": 0.0,
            "role": "negative_control",
        },
        "eligible": False,
        "negative_control": True,
    }


def test_phase_constant():
    assert stagec258.PHASE == (
        "Phase3.14b-r2.5.8 Stage C"
    )


def test_base_commit_constant():
    assert stagec258.BASE_EVIDENCE_COMMIT == (
        "174d4428f835bb7fa76d9bdd497c9ff63452122f"
    )


def test_base_worker_constant():
    assert stagec258.EXPECTED_BASE_WORKER_SHA256 == (
        "f61f2cb9226904d360a1e9199b6d9434"
        "f35b499d4d15cec7048d62e6d4b45d96"
    )


def test_base_contract_constant():
    assert stagec258.EXPECTED_BASE_CONTRACT_SHA256 == (
        "2322d276e57deae59135e0170920d6c9b"
        "2df3aeab040dbe58daf73db2adde684"
    )


def test_base_selection_constant():
    assert stagec258.EXPECTED_BASE_SELECTION_SHA256 == (
        "ed716f7fca55c2b222f6363908f50f89"
        "a981a821751f2bddbb7f24ea9df6a443"
    )


def test_candidate_population_is_seven():
    assert len(
        stagec258.CANDIDATE_DEFINITIONS
    ) == 7


def test_candidate_order():
    assert tuple(
        value.candidate_id
        for value
        in stagec258.CANDIDATE_DEFINITIONS
    ) == (
        "upper_only_reference",
        "band_q01_w1",
        "band_q05_w1",
        "band_q05_w2",
        "band_q10_w1",
        "band_q05_w1_a0p25",
        "band_q05_w1_a1p00",
    )


def test_negative_control_validates():
    stagec258.CANDIDATE_DEFINITIONS[
        0
    ].validate()


def test_negative_control_rejects_lower_weight():
    value = (
        stagec258
        .BalancedCandidateDefinition(
            "bad",
            0.05,
            1.0,
            0.0,
            "negative_control",
        )
    )
    with pytest.raises(ValueError):
        value.validate()


def test_selectable_requires_lower_term():
    value = (
        stagec258
        .BalancedCandidateDefinition(
            "bad",
            0.05,
            0.0,
            0.0,
            "selectable",
        )
    )
    with pytest.raises(ValueError):
        value.validate()


def test_candidate_rejects_quantile():
    value = (
        stagec258
        .BalancedCandidateDefinition(
            "bad",
            0.75,
            1.0,
            0.0,
            "selectable",
        )
    )
    with pytest.raises(ValueError):
        value.validate()


def test_spec_validates():
    stagec258.BalancedGeometrySpec().validate()


def test_spec_rejects_topk():
    with pytest.raises(ValueError):
        stagec258.BalancedGeometrySpec(
            top_k=8
        ).validate()


def test_spec_rejects_quantiles():
    with pytest.raises(ValueError):
        stagec258.BalancedGeometrySpec(
            lower_quantiles=(0.05,)
        ).validate()


def test_spec_rejects_unordered_radii():
    with pytest.raises(ValueError):
        stagec258.BalancedGeometrySpec(
            oracle_radii=(0.1, 0.05)
        ).validate()


def test_spec_rejects_checkpoint_end():
    with pytest.raises(ValueError):
        stagec258.BalancedGeometrySpec(
            oracle_checkpoints=(0, 1, 255)
        ).validate()


def test_stable_json_deterministic():
    assert stagec258.stable_json_bytes(
        {"b": 2, "a": 1}
    ) == stagec258.stable_json_bytes(
        {"a": 1, "b": 2}
    )


def test_atomic_write_once(tmp_path):
    path = tmp_path / "x.json"
    stagec258.atomic_write_once(
        path,
        b"{}\n",
    )
    with pytest.raises(FileExistsError):
        stagec258.atomic_write_once(
            path,
            b"{}\n",
        )


def test_linear_quantile_endpoints():
    value = np.asarray(
        [[0.0], [10.0]],
        dtype=np.float64,
    )
    assert stagec258.linear_quantile_axis0(
        value,
        0.0,
    )[0] == pytest.approx(0.0)
    assert stagec258.linear_quantile_axis0(
        value,
        1.0,
    )[0] == pytest.approx(10.0)


def test_linear_quantile_midpoint():
    value = np.asarray(
        [[0.0], [10.0]],
        dtype=np.float64,
    )
    assert stagec258.linear_quantile_axis0(
        value,
        0.5,
    )[0] == pytest.approx(5.0)


def test_linear_quantile_axis_shape():
    value = np.arange(
        24,
        dtype=np.float64,
    ).reshape(3, 2, 4)
    result = (
        stagec258.linear_quantile_axis0(
            value,
            0.5,
        )
    )
    assert result.shape == (2, 4)


def test_linear_quantile_rejects_empty():
    with pytest.raises(ValueError):
        stagec258.linear_quantile_axis0(
            np.zeros((0, 2)),
            0.5,
        )


@pytest.mark.parametrize(
    "value,key",
    [
        (0.01, "q01"),
        (0.05, "q05"),
        (0.10, "q10"),
    ],
)
def test_lower_key(value, key):
    assert stagec258.lower_key(value) == key


def test_lower_key_rejects_unknown():
    with pytest.raises(ValueError):
        stagec258.lower_key(0.2)


def test_torch_upper_penalty():
    value = torch.as_tensor(
        cable(2.0),
        dtype=torch.float32,
    )
    terms = (
        stagec258
        .balanced_objective_terms_torch(
            value,
            target_standardizer=
                standardizer(),
            candidate=candidate(
                lower=-2.0,
                upper=0.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert float(terms["upper_total"]) > 0.0


def test_torch_lower_penalty():
    value = torch.as_tensor(
        cable(0.1),
        dtype=torch.float32,
    )
    terms = (
        stagec258
        .balanced_objective_terms_torch(
            value,
            target_standardizer=
                standardizer(),
            candidate=candidate(
                lower=-1.0,
                upper=1.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert float(terms["lower_total"]) > 0.0


def test_torch_anchor_penalty():
    value = torch.as_tensor(
        cable(1.0),
        dtype=torch.float32,
    )
    terms = (
        stagec258
        .balanced_objective_terms_torch(
            value,
            target_standardizer=
                standardizer(),
            candidate=candidate(
                lower=-2.0,
                upper=2.0,
                anchor_lower=0.5,
                anchor_upper=1.0,
                anchor_weight=1.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert float(terms["anchor_total"]) > 0.0


def test_torch_zero_inside_band():
    value = torch.as_tensor(
        cable(1.0),
        dtype=torch.float32,
    )
    terms = (
        stagec258
        .balanced_objective_terms_torch(
            value,
            target_standardizer=
                standardizer(),
            candidate=candidate(
                lower=-0.1,
                upper=0.1,
                anchor_lower=-0.1,
                anchor_upper=0.1,
                anchor_weight=1.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert float(terms["total"]) == pytest.approx(
        0.0
    )


def test_numpy_profile_upper():
    result = (
        stagec258
        .balanced_objective_profile_numpy(
            cable(2.0),
            candidate=candidate(
                lower=-2.0,
                upper=0.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert result["upper_total"] > 0.0


def test_numpy_profile_lower():
    result = (
        stagec258
        .balanced_objective_profile_numpy(
            cable(0.1),
            candidate=candidate(
                lower=-1.0,
                upper=1.0,
            ),
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert result["lower_total"] > 0.0


def test_numpy_and_torch_totals_match():
    raw = cable(1.5)
    config = candidate(
        lower=-1.0,
        upper=0.2,
        anchor_lower=-0.2,
        anchor_upper=0.1,
        anchor_weight=0.25,
    )
    numpy_result = (
        stagec258
        .balanced_objective_profile_numpy(
            raw,
            candidate=config,
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    torch_result = (
        stagec258
        .balanced_objective_terms_torch(
            torch.as_tensor(raw),
            target_standardizer=
                standardizer(),
            candidate=config,
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert numpy_result["total"] == pytest.approx(
        float(torch_result["total"]),
        rel=1.0e-5,
        abs=1.0e-6,
    )


def test_fit_reference_uses_objective_train():
    target = np.concatenate(
        [
            cable(0.8, 4),
            cable(1.0, 4),
            cable(1.2, 4),
        ],
        axis=0,
    )
    mask = np.zeros(12, dtype=np.bool_)
    mask[:8] = True
    context = {
        "target": target,
        "groups": np.asarray(
            ["g{}".format(i // 2)
             for i in range(12)]
        ),
        "objective_train_mask": mask,
        "objective_contract":
            SimpleNamespace(
                allowed_upper_log_length=
                    np.full(
                        (4, 23),
                        1.0,
                        dtype=np.float32,
                    )
            ),
    }
    result = (
        stagec258.fit_balanced_reference(
            context=context,
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert result["fit_rows"] == 8
    assert result[
        "uses_selection_holdout_target"
    ] is False


def test_fit_reference_lower_below_upper():
    target = cable(1.0, 10)
    context = {
        "target": target,
        "groups": np.asarray(
            ["g{}".format(i)
             for i in range(10)]
        ),
        "objective_train_mask":
            np.ones(10, dtype=np.bool_),
        "objective_contract":
            SimpleNamespace(
                allowed_upper_log_length=
                    np.full(
                        (4, 23),
                        0.00005,
                        dtype=np.float32,
                    )
            ),
    }
    result = (
        stagec258.fit_balanced_reference(
            context=context,
            spec=
                stagec258
                .BalancedGeometrySpec(),
        )
    )
    assert np.all(
        result["clipped_lower"]["q05"]
        < result["frozen_upper"]
    )


def test_candidate_contract_sha():
    shape = (4, 23)
    reference = {
        "clipped_lower": {
            "q01":
                np.full(shape, -1.0, np.float32),
            "q05":
                np.full(shape, -0.5, np.float32),
            "q10":
                np.full(shape, -0.2, np.float32),
        },
        "frozen_upper":
            np.full(shape, 1.0, np.float32),
        "anchor_lower":
            np.full(shape, -0.1, np.float32),
        "anchor_upper":
            np.full(shape, 0.1, np.float32),
        "fit_target_sha256": "a" * 64,
        "fit_group_sha256": "b" * 64,
    }
    result = stagec258.candidate_contract(
        definition=
            stagec258
            .CANDIDATE_DEFINITIONS[2],
        reference=reference,
        spec=
            stagec258
            .BalancedGeometrySpec(),
    )
    assert len(
        result["contract_sha256"]
    ) == 64


def test_scientific_gates_pass():
    gates = stagec258.scientific_gates(
        timestep=10,
        record=fake_oracle_record(),
        spec=
            stagec258
            .BalancedGeometrySpec(),
    )
    assert gates["all"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("upper", 0.5),
        ("nmse", 1.3),
        ("lower", 0.8),
        ("physical", 0.8),
        ("collapse", 0.1),
        ("stretch", 0.1),
        ("cosine", 0.1),
        ("reduction", 0.0),
    ],
)
def test_scientific_gates_fail(field, value):
    gates = stagec258.scientific_gates(
        timestep=10,
        record=fake_oracle_record(
            **{field: value}
        ),
        spec=
            stagec258
            .BalancedGeometrySpec(),
    )
    assert not gates["all"]


def test_negative_control_not_selectable():
    selected = stagec258.select_objective(
        [negative_control()]
    )
    assert selected is None


def test_select_objective():
    record = fake_selectable_record()
    selected = stagec258.select_objective(
        [
            negative_control(),
            record,
        ]
    )
    assert selected["candidate_id"] == (
        "band_q05_w1"
    )


def test_selection_prefers_no_anchor_on_tie():
    plain = fake_selectable_record(
        candidate_id="plain",
        anchor_weight=0.0,
    )
    anchor = fake_selectable_record(
        candidate_id="anchor",
        anchor_weight=0.25,
    )
    selected = stagec258.select_objective(
        [plain, anchor]
    )
    assert selected["candidate_id"] == "plain"



def test_initial_direction_is_not_eligibility_gate():
    record = fake_selectable_record(
        eligible=True,
        direction=False,
        scientific=True,
    )
    selected = stagec258.select_objective(
        [record]
    )
    assert selected["candidate_id"] == (
        "band_q05_w1"
    )

def test_classify_plain_selected():
    record = fake_selectable_record()
    selected = stagec258.select_objective(
        [record]
    )
    result = stagec258.classify_objectives(
        records=[
            negative_control(),
            record,
        ],
        selected=selected,
    )
    assert result[
        "primary_failure_locus"
    ] == "balanced_objective_selected"


def test_classify_anchor_required():
    record = fake_selectable_record(
        candidate_id="anchor",
        anchor_weight=0.25,
    )
    selected = stagec258.select_objective(
        [record]
    )
    result = stagec258.classify_objectives(
        records=[record],
        selected=selected,
    )
    assert result[
        "primary_failure_locus"
    ] == "distribution_anchor_required"


def test_classify_nonphysical_shortcut():
    record = fake_selectable_record(
        eligible=False,
        scientific=False,
        upper_fidelity=True,
        lower_physical=False,
    )
    result = stagec258.classify_objectives(
        records=[record],
        selected=None,
    )
    assert result[
        "primary_failure_locus"
    ] == (
        "balanced_objective_nonphysical_shortcut"
    )


def test_classify_lower_overconstraint():
    record = fake_selectable_record(
        eligible=False,
        scientific=False,
        upper_fidelity=False,
        lower_physical=True,
    )
    result = stagec258.classify_objectives(
        records=[record],
        selected=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "lower_band_overconstraint"


def test_classify_timestep_conditional():
    record = fake_selectable_record(
        eligible=False,
        scientific=False,
        upper_fidelity=False,
        lower_physical=False,
    )
    record[
        "timestep_records"
    ]["10"]["oracle"][
        "scientific_reachable"
    ] = True
    result = stagec258.classify_objectives(
        records=[record],
        selected=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "timestep_conditional"


def test_classify_local_misalignment():
    record = fake_selectable_record(
        eligible=False,
        scientific=False,
        upper_fidelity=True,
        lower_physical=True,
        direction=False,
    )
    result = stagec258.classify_objectives(
        records=[record],
        selected=None,
    )
    assert result[
        "primary_failure_locus"
    ] in (
        "lower_band_overconstraint",
        "local_descent_misalignment",
    )


def test_classify_no_joint_solution():
    record = fake_selectable_record(
        eligible=False,
        scientific=False,
        upper_fidelity=False,
        lower_physical=False,
        direction=False,
    )
    result = stagec258.classify_objectives(
        records=[record],
        selected=None,
    )
    assert result[
        "primary_failure_locus"
    ] == "no_joint_solution"


def fake_result():
    return {
        "root_cause": "root",
        "required_next_path": "next",
        "scientific_status": "BLOCKED",
        "environment": {"x": 1},
        "cold_main_worker_context": {
            "x": 1
        },
        "control_capture": {"x": 1},
        "split": {"x": 1},
        "balanced_reference": {"x": 1},
        "candidate_records": [{"x": 1}],
        "classification": {"x": 1},
        "balanced_geometry_contract": {
            "x": 1
        },
        "selection": {"x": 1},
    }


def test_identity_projection():
    result = stagec258.identity_projection(
        fake_result()
    )
    assert result[
        "balanced_reference"
    ] == {"x": 1}


def test_compare_workers_exact():
    left = fake_result()
    right = copy.deepcopy(left)
    result = (
        stagec258.compare_worker_results(
            left,
            right,
        )
    )
    assert result["exact"]
    assert result[
        "candidate_records_exact"
    ]


def test_compare_workers_detects_candidate():
    left = fake_result()
    right = copy.deepcopy(left)
    right["candidate_records"] = [
        {"x": 2}
    ]
    result = (
        stagec258.compare_worker_results(
            left,
            right,
        )
    )
    assert not result["exact"]
    assert not result[
        "candidate_records_exact"
    ]


def test_historical_control_not_run():
    base = {
        "worker_result": {
            "root_cause":
                stagec258
                .EXPECTED_BASE_ROOT_CAUSE,
            "timestep_records": {
                str(timestep): {
                    "geometry_gradient_alignment":
                        {"x": timestep},
                    "direct_x0_oracle":
                        {
                            "upper_fidelity_reachable":
                                True,
                            "cable_valid_reachable":
                                False,
                        },
                }
                for timestep in (10, 25, 50)
            },
        }
    }
    result = (
        stagec258.historical_negative_control(
            base
        )
    )
    assert result["candidate_run"] is False
    assert result["negative_control"] is True


def test_topk_helper_shape():
    excess = torch.arange(
        2 * 4 * 23,
        dtype=torch.float32,
    ).reshape(2, 4, 23)
    row, indices = (
        stagec258._topk_mean_square(
            excess,
            top_k=16,
        )
    )
    assert row.shape == (2,)
    assert indices.shape == (2, 16)


def test_numpy_topk_shape():
    excess = np.arange(
        2 * 4 * 23,
        dtype=np.float64,
    ).reshape(2, 4, 23)
    result = (
        stagec258._numpy_topk_square(
            excess,
            top_k=16,
        )
    )
    assert result.shape == (2,)


def test_upper_thresholds():
    assert stagec258.TIMESTEP_UPPER_MIN == {
        10: 0.75,
        25: 0.60,
        50: 0.50,
    }


def test_no_candidate_uses_holdout_fit_flag():
    for definition in (
        stagec258.CANDIDATE_DEFINITIONS
    ):
        definition.validate()
