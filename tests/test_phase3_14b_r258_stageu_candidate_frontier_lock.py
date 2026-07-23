from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ccda_phase3 import phase314b_r258_stageu_candidate_frontier_lock as stageu


def add_false_boundaries(payload: dict) -> dict:
    payload.update({key: False for key in stageu.FALSE_BOUNDARIES})
    return payload


def make_frontier_cell(timestep: int) -> dict:
    expected = stageu.EXPECTED_CELL_LOCKS[timestep]
    return {
        "base_direction_id": stageu.FRONTIER_BACKBONE,
        "timestep": timestep,
        "feature_mode": expected["feature_mode"],
        "global_rank": expected["global_rank"],
        "aligned_acceptance_rate": expected["aligned_acceptance_rate"],
        "aligned_candidate_sha256": expected["aligned_candidate_sha256"],
        "aligned_selected_scale_sha256": expected[
            "aligned_selected_scale_sha256"
        ],
        "aligned_selected_scale_histogram": copy.deepcopy(
            expected["aligned_selected_scale_histogram"]
        ),
        "matrix_reconstruction_exact": True,
        "legacy_functional_identity_exact": True,
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "matrix_reconstruction_attempt_count": 7,
        "candidate_fidelity": {
            "mechanism_eligible": True,
            "fidelity_eligible": True,
            "row_count": expected["row_count"],
            "selected_row_count": expected["selected_row_count"],
            "overall_mse_ratio": expected["overall_mse_ratio"],
            "accepted_rows": {
                "mse_ratio": expected["accepted_row_mse_ratio"],
                "positive_distance_reduction_rate": expected[
                    "positive_distance_reduction_rate"
                ],
                "relative_distance_reduction": {
                    "mean": expected["relative_distance_reduction_mean"]
                },
            },
        },
    }


def make_gate() -> dict:
    inner = add_false_boundaries(
        {
            "schema": "phase314b_r258_staget_tolerance_aligned_gate_contract_v1",
            "contract_role": "shadow_frozen_not_written_to_stagee",
            "contract_sha256": stageu.EXPECTED_INNER_GATE_CONTRACT_SHA256,
            "contract_payload_sha256": stageu.EXPECTED_INNER_GATE_PAYLOAD_SHA256,
            "tolerance_factor": 5.0,
            "segment_tolerance": 2.5e-06,
            "length_tolerance": 1.25e-05,
            "position_shape": [4, 23],
            "upper_bound_sha256": (
                "afff5ceab890fed4cf83c015a0bfab6071080b562d717453abf51887c643bb43"
            ),
            "length_upper_sha256": (
                "5159d02ef16ef772e6a75b60ac179adbc6907541f56f167162ca2f88d986a1fc"
            ),
            "position_z_threshold_sha256": (
                "02175da1e304ed1d3a220fe665eafe7805aedaab556597cdfd4d45ad0affbabc"
            ),
            "objective_train_only": True,
            "worker_contract_byte_exact": True,
            "aligned_gate_written_to_stagee": False,
            "legacy_upper_gate_changed": False,
            "stagee_modified": False,
            "deployment_authorized": False,
            "selected_configuration": None,
            "train_only_recommendation": None,
        }
    )
    return add_false_boundaries(
        {
            "execution_verdict": "PASS",
            "scientific_status": "BLOCKED",
            "schema_recovery_only": True,
            "recovered_stage_t_gate_contract_sha256": (
                stageu.EXPECTED_RECOVERED_GATE_SHA256
            ),
            "recovered_stage_t_gate_contract": inner,
            "selected_configuration": None,
            "train_only_recommendation": None,
        }
    )


def make_summary() -> dict:
    frontier_cells = [make_frontier_cell(t) for t in stageu.LOCKED_TIMESTEPS]
    other_cells = [
        {
            "base_direction_id": f"other_{index}",
            "timestep": 10 + index,
        }
        for index in range(24)
    ]
    matrix = {
        "backbone_count": 9,
        "timestep_count": 3,
        "cell_count": 27,
        "mechanism_eligible_cell_count": 27,
        "fidelity_eligible_cell_count": 13,
        "matrix_eligible_backbone_count": 1,
        "matrix_eligible_backbones": [stageu.FRONTIER_BACKBONE],
        "pareto_frontier_backbones": [stageu.FRONTIER_BACKBONE],
        "selected_configuration": None,
        "train_only_recommendation": None,
        "backbone_ranking": [copy.deepcopy(dict(stageu.EXPECTED_BACKBONE_METRICS))],
        "cell_records": frontier_cells + other_cells,
        "per_timestep_ranking": {
            str(t): [
                {
                    "base_direction_id": stageu.FRONTIER_BACKBONE,
                    "rank": 1,
                    "fidelity_eligible": True,
                }
            ]
            for t in stageu.LOCKED_TIMESTEPS
        },
    }
    result = add_false_boundaries(
        {
            "execution_verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": stageu.EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": stageu.EXPECTED_BASE_NEXT_PATH,
            "selected_configuration": None,
            "train_only_recommendation": None,
            "candidate_matrix_frontier": [stageu.FRONTIER_BACKBONE],
            "candidate_matrix": matrix,
            "confirmation_execution": {
                "environment_probe_count": 1,
                "cold_science_worker_count": 2,
                "processes_distinct": True,
                "workers_sequential": True,
                "total_oof_fit_count": 108,
                "total_callback_pair_count": 54,
                "total_shadow_internal_scale_attempt_count": 378,
                "total_matrix_reconstruction_attempt_count": 378,
                "oracle_callback_rerun_count": 0,
                "worker_outputs_persisted": False,
                "candidate_matrix_sha256": stageu.EXPECTED_CANDIDATE_MATRIX_SHA256,
                "functional_projection_sha256": (
                    stageu.EXPECTED_FUNCTIONAL_PROJECTION_SHA256
                ),
                "current_fit_projection_sha256": (
                    stageu.EXPECTED_CURRENT_FIT_PROJECTION_SHA256
                ),
                "gate_contract_sha256": stageu.EXPECTED_INNER_GATE_CONTRACT_SHA256,
                "worker_comparison": {"all_exact": True},
            },
        }
    )
    return add_false_boundaries(
        {
            "execution_verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": stageu.EXPECTED_BASE_ROOT_CAUSE,
            "required_next_path": stageu.EXPECTED_BASE_NEXT_PATH,
            "scientific_result_sha256": stageu.EXPECTED_BASE_SCIENTIFIC_SHA256,
            "recovered_stage_t_result_sha256": (
                stageu.EXPECTED_RECOVERED_RESULT_SHA256
            ),
            "recovered_stage_t_gate_contract_sha256": (
                stageu.EXPECTED_RECOVERED_GATE_SHA256
            ),
            "selected_configuration": None,
            "train_only_recommendation": None,
            "recovered_stage_t_result": result,
        }
    )


def test_stable_json_is_deterministic() -> None:
    assert stageu.stable_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_sha256_bytes_is_stable() -> None:
    assert stageu.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_validate_gate_passes() -> None:
    inner = stageu.validate_gate_wrapper(make_gate())
    assert inner["contract_sha256"] == stageu.EXPECTED_INNER_GATE_CONTRACT_SHA256


def test_validate_summary_passes() -> None:
    validated = stageu.validate_summary(make_summary())
    assert len(validated["locked_cells"]) == 3


def test_build_contract_locks_joint_timestep_policy() -> None:
    validated = stageu.validate_summary(make_summary())
    contract = stageu.build_frontier_contract(
        validated_summary=validated,
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    assert contract["frontier_lock"]["timestep_policy"] == (
        "joint_all_timesteps_no_cherry_pick"
    )
    assert contract["frontier_lock"]["timesteps"] == [10, 25, 50]


def test_build_contract_has_zero_new_execution() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    assert set(contract["stageu_execution"].values()) >= {0, False, True}
    assert contract["stageu_execution"]["oof_fit_count"] == 0
    assert contract["stageu_execution"]["callback_pair_count"] == 0


def test_build_summary_remains_blocked() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    summary = stageu.build_summary(repository={"head": "abc"}, contract=contract)
    assert summary["scientific_status"] == "BLOCKED"
    assert summary["selected_configuration"] is None
    assert summary["train_only_recommendation"] is None


def test_contract_preregisters_one_holdout_evaluation() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    policy = contract["next_stage_holdout_policy"]
    assert policy["evaluation_count"] == 1
    assert policy["all_timesteps_must_pass"] is True
    assert policy["post_holdout_timestep_selection_forbidden"] is True


def test_summary_input_not_mutated() -> None:
    summary = make_summary()
    frozen = copy.deepcopy(summary)
    stageu.validate_summary(summary)
    assert summary == frozen


def test_gate_input_not_mutated() -> None:
    gate = make_gate()
    frozen = copy.deepcopy(gate)
    stageu.validate_gate_wrapper(gate)
    assert gate == frozen


@pytest.mark.parametrize(
    "key,bad",
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "READY"),
        ("root_cause", "wrong"),
        ("required_next_path", "wrong"),
        ("scientific_result_sha256", "0" * 64),
        ("recovered_stage_t_result_sha256", "1" * 64),
        ("recovered_stage_t_gate_contract_sha256", "2" * 64),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_summary_top_level_mutations_fail(key: str, bad: object) -> None:
    summary = make_summary()
    summary[key] = bad
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("execution_verdict", "BLOCKED"),
        ("scientific_status", "READY"),
        ("schema_recovery_only", False),
        ("recovered_stage_t_gate_contract_sha256", "3" * 64),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_gate_wrapper_mutations_fail(key: str, bad: object) -> None:
    gate = make_gate()
    gate[key] = bad
    with pytest.raises(stageu.StageUError):
        stageu.validate_gate_wrapper(gate)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("tolerance_factor", 4.0),
        ("segment_tolerance", 1e-6),
        ("length_tolerance", 2e-5),
        ("position_shape", [4, 22]),
        ("contract_role", "deployed"),
        ("aligned_gate_written_to_stagee", True),
        ("deployment_authorized", True),
    ],
)
def test_inner_gate_mutations_fail(key: str, bad: object) -> None:
    gate = make_gate()
    gate["recovered_stage_t_gate_contract"][key] = bad
    with pytest.raises(stageu.StageUError):
        stageu.validate_gate_wrapper(gate)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("environment_probe_count", 2),
        ("cold_science_worker_count", 1),
        ("processes_distinct", False),
        ("workers_sequential", False),
        ("total_oof_fit_count", 107),
        ("total_callback_pair_count", 53),
        ("total_shadow_internal_scale_attempt_count", 377),
        ("total_matrix_reconstruction_attempt_count", 377),
        ("oracle_callback_rerun_count", 1),
        ("worker_outputs_persisted", True),
        ("candidate_matrix_sha256", "4" * 64),
    ],
)
def test_execution_mutations_fail(key: str, bad: object) -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["confirmation_execution"][key] = bad
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("mechanism_eligible_cell_count", 26),
        ("fidelity_eligible_cell_count", 12),
        ("matrix_eligible_backbone_count", 2),
        ("matrix_eligible_backbones", [stageu.FRONTIER_BACKBONE, "other"]),
        ("pareto_frontier_backbones", []),
        ("selected_configuration", {}),
        ("train_only_recommendation", {}),
    ],
)
def test_matrix_mutations_fail(key: str, bad: object) -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"][key] = bad
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_second_frontier_backbone_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix_frontier"].append("other")
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_duplicate_frontier_timestep_fails() -> None:
    summary = make_summary()
    matrix = summary["recovered_stage_t_result"]["candidate_matrix"]
    matrix["cell_records"].append(copy.deepcopy(make_frontier_cell(10)))
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_missing_frontier_timestep_fails() -> None:
    summary = make_summary()
    matrix = summary["recovered_stage_t_result"]["candidate_matrix"]
    matrix["cell_records"] = [
        row
        for row in matrix["cell_records"]
        if not (
            row.get("base_direction_id") == stageu.FRONTIER_BACKBONE
            and row.get("timestep") == 50
        )
    ]
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_candidate_sha_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"]["cell_records"][0][
        "aligned_candidate_sha256"
    ] = "5" * 64
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_selected_scale_sha_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"]["cell_records"][1][
        "aligned_selected_scale_sha256"
    ] = "6" * 64
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_fidelity_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"]["cell_records"][2][
        "candidate_fidelity"
    ]["fidelity_eligible"] = False
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_per_timestep_rank_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"][
        "per_timestep_ranking"
    ]["25"][0]["rank"] = 2
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_backbone_metric_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"]["backbone_ranking"][0][
        "worst_overall_mse_ratio"
    ] = 1.0
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_frontier_histogram_population_mutation_fails() -> None:
    summary = make_summary()
    summary["recovered_stage_t_result"]["candidate_matrix"]["cell_records"][0][
        "aligned_selected_scale_histogram"
    ]["0"] = 24
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_false_boundary_mutation_fails() -> None:
    summary = make_summary()
    summary["selection_holdout_evaluated"] = True
    with pytest.raises(stageu.StageUError):
        stageu.validate_summary(summary)


def test_blocked_report_claims_zero_stageu_execution() -> None:
    report = stageu.blocked_report(repository=None, error=RuntimeError("x"))
    assert report["stageu_oof_fit_count"] == 0
    assert report["selection_holdout_evaluation_count"] == 0
    assert report["frontier_contract_written"] is False


def test_contract_hash_is_deterministic() -> None:
    validated = stageu.validate_summary(make_summary())
    inner = stageu.validate_gate_wrapper(make_gate())
    left = stageu.build_frontier_contract(
        validated_summary=validated,
        gate_inner=inner,
    )
    right = stageu.build_frontier_contract(
        validated_summary=validated,
        gate_inner=inner,
    )
    assert stageu.stable_json_bytes(left) == stageu.stable_json_bytes(right)


def test_lock_cells_are_sorted_by_timestep() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    assert [cell["timestep"] for cell in contract["frontier_lock"]["locked_cells"]] == [
        10,
        25,
        50,
    ]


def test_no_deployment_authorized() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    assert contract["deployment_authorized"] is False
    assert contract["aligned_gate_lock"]["deployment_authorized"] is False


def test_next_stage_has_no_fallback() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    lock = contract["frontier_lock"]
    assert lock["fallback_backbone_allowed"] is False
    assert lock["fallback_timestep_allowed"] is False


def test_summary_scientific_sha_is_present() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    result = stageu.build_summary(repository={}, contract=contract)
    assert len(result["scientific_result_sha256"]) == 64


def test_json_roundtrip_contract() -> None:
    contract = stageu.build_frontier_contract(
        validated_summary=stageu.validate_summary(make_summary()),
        gate_inner=stageu.validate_gate_wrapper(make_gate()),
    )
    assert json.loads(stageu.stable_json_bytes(contract)) == contract


def test_output_paths_are_distinct() -> None:
    assert len({stageu.SUCCESS_REPORT, stageu.CONTRACT_REPORT, stageu.BLOCKED_REPORT}) == 3


def test_source_contains_no_torch_or_numpy_import() -> None:
    source = Path(stageu.__file__).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "import numpy" not in source


def test_source_contains_no_worker_launch() -> None:
    source = Path(stageu.__file__).read_text(encoding="utf-8")
    assert "science_worker" not in source or "science_worker_count" in source
    assert "Popen(" not in source
