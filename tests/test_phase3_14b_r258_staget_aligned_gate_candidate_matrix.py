from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_staget_aligned_gate_candidate_matrix as staget


def _array_sha(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(repr(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _histogram(scale):
    values = np.asarray(scale, dtype=np.float64)
    return {str(v): int(np.count_nonzero(values == v)) for v in sorted(set(values.tolist()))}


def fake_runtime(rows=2):
    order = (2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.25)

    class StageE:
        stageb = None
        staged = None

        @staticmethod
        def segment_bounds(*, definition, context):
            return {
                "lower": np.full((4, 23), 0.01, dtype=np.float64),
                "upper": np.full((4, 23), 2.0, dtype=np.float64),
            }

        @staticmethod
        def reconstruct_segment_vectors(**kwargs):
            candidate = np.asarray(kwargs["proposed"], dtype=np.float32)
            return {
                "candidate": candidate,
                "bound_pass": np.ones(candidate.shape[0], dtype=np.bool_),
                "coordinate_possible": np.ones(candidate.shape[0], dtype=np.bool_),
            }

        @staticmethod
        def observable_row_metrics(**kwargs):
            count = np.asarray(kwargs["candidate"]).shape[0]
            return {
                "upper_pass": np.ones(count, dtype=np.bool_),
                "finite": np.ones(count, dtype=np.bool_),
            }

    stageb = SimpleNamespace(
        segment_lengths=lambda value: np.ones((np.asarray(value).shape[0], 4, 23), dtype=np.float64),
        physical_validity=lambda value, contract: {
            "topology": np.ones((np.asarray(value).shape[0], 1), dtype=np.bool_)
        },
    )
    staged = SimpleNamespace(
        segment_scores=lambda value, reference: {
            "z": np.zeros((np.asarray(value).shape[0], 1, 4, 23), dtype=np.float64),
            "upper": np.zeros((np.asarray(value).shape[0], 1), dtype=np.float64),
        }
    )
    StageE.stageb = stageb
    StageE.staged = staged

    class Definition:
        def validate(self):
            return None

    stagef = SimpleNamespace(fixed_integrator_definition=lambda: Definition())

    def tolerant_upper_contract(**kwargs):
        length = np.asarray(kwargs["upper_bound"], dtype=np.float64) + 5e-6
        z = np.ones_like(length)
        return {
            "length_upper": length,
            "position_z_threshold": z,
            "length_upper_sha256": _array_sha(length),
            "position_z_threshold_sha256": _array_sha(z),
            "length_tolerance": 5e-6,
        }

    def aligned_upper_masks(**kwargs):
        length = np.asarray(kwargs["lengths"])
        element = np.ones_like(length, dtype=np.bool_)
        row = np.ones(length.shape[0], dtype=np.bool_)
        return {
            "length_element_pass": element,
            "length_row_pass": row,
            "element_mismatch_count": 0,
        }

    def predicate_masks(**kwargs):
        count = np.asarray(kwargs["aligned_upper_pass"]).shape[0]
        return {"finite": np.ones(count, dtype=np.bool_), "upper": np.ones(count, dtype=np.bool_)}

    def sequential_attempt_summary(**kwargs):
        return {"accepted_mask": np.asarray(kwargs["active"], dtype=np.bool_)}

    stageq = SimpleNamespace(
        _expected_internal_order=lambda runtime: order,
        tolerant_upper_contract=tolerant_upper_contract,
        aligned_upper_masks=aligned_upper_masks,
        _predicate_masks=predicate_masks,
        sequential_attempt_summary=sequential_attempt_summary,
    )
    stager_fake = SimpleNamespace(
        PREDICATE_ORDER=("finite", "upper", "topology"),
        sha256_array=_array_sha,
        selected_scale_histogram=_histogram,
    )
    stagel = SimpleNamespace(stagee258=StageE)
    runtime = {
        "stageq": stageq,
        "stagel": stagel,
        "stager": stager_fake,
        "stagef": stagef,
        "integrator_spec": SimpleNamespace(segment_tolerance=1e-6),
    }
    context = {
        "stage_d_contract": SimpleNamespace(
            reference=SimpleNamespace(
                center_log=np.zeros((4, 23)),
                scale_log=np.ones((4, 23)),
            )
        ),
        "historical_geometry": SimpleNamespace(coordinate_abs_max=10.0),
    }
    return runtime, context, order


def make_base_cell(rows=2):
    runtime, _, order = fake_runtime(rows)
    control = np.zeros((rows, 4, 48), dtype=np.float32)
    direction = np.ones_like(control, dtype=np.float64)
    # external 0.25 and first internal scale 2.0 => candidate 0.5
    candidate = np.full_like(control, 0.5)
    selected_scale = np.full(rows, 2.0, dtype=np.float64)
    return {
        "aligned_candidate_sha256": runtime["stager"].sha256_array(candidate),
        "aligned_selected_scale_sha256": runtime["stager"].sha256_array(selected_scale),
        "aligned_acceptance_rate": 1.0,
        "internal_scale_attempt_order": list(order),
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
    }, control, direction, np.ones_like(control), runtime


def fake_fidelity(eligible=True, acceptance=0.8, mse=0.7, positive=0.8, relative=0.2):
    return {
        "row_count": 638,
        "selected_row_count": int(638 * acceptance),
        "acceptance_rate": acceptance,
        "overall_mse_ratio": mse,
        "mechanism_eligible": True,
        "fidelity_eligible": eligible,
        "accepted_rows": {
            "positive_distance_reduction_rate": positive,
            "relative_distance_reduction": {"mean": relative},
        },
    }


def fake_cell(backbone, timestep, eligible=True, mse=0.7, acceptance=0.8, positive=0.8):
    return {
        "base_direction_id": backbone,
        "timestep": timestep,
        "feature_mode": "full_centered_constraint",
        "aligned_acceptance_rate": acceptance,
        "aligned_candidate_sha256": f"candidate-{backbone}-{timestep}",
        "aligned_selected_scale_sha256": f"scale-{backbone}-{timestep}",
        "aligned_selected_scale_histogram": {"0.0": 10, "1.0": 90},
        "length_log_z_element_mismatch_count": 0,
        "aligned_upper_element_failure_count": 0,
        "strict_pass_aligned_fail_row_count": 0,
        "legacy_identity": {"all_functional_exact": True},
        "candidate_matrix_metrics": {
            "candidate_fidelity": fake_fidelity(eligible, acceptance, mse, positive),
            "reconstruction_exact": True,
            "matrix_reconstruction_attempt_count": 7,
        },
    }


def fake_cells(all_eligible=True):
    values = []
    for b in range(9):
        for t in staget.EXPECTED_TIMESTEPS:
            values.append(fake_cell(f"b{b}", t, eligible=all_eligible, mse=0.70 + b * 0.01))
    return values


def fake_base_report():
    return {
        "execution_verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": staget.EXPECTED_BASE_ROOT_CAUSE,
        "required_next_path": staget.EXPECTED_BASE_NEXT_PATH,
        "scientific_result_sha256": staget.EXPECTED_BASE_SCIENTIFIC_SHA256,
        "selected_configuration": None,
        "train_only_recommendation": None,
        "confirmation_execution": {
            "worker_count": 2,
            "total_oof_fit_count": 108,
            "total_callback_pair_count": 54,
            "total_internal_scale_attempt_count": 378,
            "processes_distinct": True,
            "workers_sequential": True,
        },
        "confirmation_summary": {
            "cell_count": 27,
            "nonzero_support_cell_count": 27,
            "zero_acceptance_cell_count": 0,
            "oracle_like_cell_count": 7,
            "dominant_discriminator": "direction_retention",
            "dominant_support_count": 2,
        },
        **{key: False for key in staget.FALSE_BOUNDARIES},
    }


def fake_worker(matrix=None):
    matrix = staget.build_candidate_matrix(fake_cells()) if matrix is None else matrix
    gate = {
        "contract_role": "shadow_frozen_not_written_to_stagee",
        "contract_sha256": "gate-sha",
    }
    return {
        "schema": staget.WORKER_SCHEMA,
        "execution_verdict": "PASS",
        "environment_sha256": "env",
        "cold_cuda_precheck": {"torch_cuda_is_initialized": False},
        "environment_probe_rerun_in_science_process": False,
        "functional_projection_sha256": staget.EXPECTED_BASE_FUNCTIONAL_SHA256,
        "functional_projection_matches_stage_s_resume3": True,
        "current_fit_projection_sha256": staget.EXPECTED_BASE_CURRENT_FIT_SHA256,
        "candidate_matrix": matrix,
        "candidate_matrix_sha256": staget.sha256_bytes(staget.stable_json_bytes(matrix)),
        "gate_contract": gate,
        "gate_contract_sha256": "gate-sha",
        "scientific_oof_fit_count": 27,
        "repeat_identity_fit_count": 27,
        "total_oof_fit_count": 54,
        "callback_pair_count": 27,
        "shadow_internal_scale_attempt_count": 189,
        "matrix_reconstruction_attempt_count": 189,
        "oracle_callback_rerun_count": 0,
        **{key: False for key in staget.FALSE_BOUNDARIES},
    }


def test_phase_schema_and_counts():
    assert staget.PHASE.endswith("Stage T")
    assert staget.EXPECTED_CELL_COUNT == 27
    assert staget.EXPECTED_TOTAL_FITS == 108


def test_candidate_policy_valid():
    staget.CandidatePolicy().validate()


@pytest.mark.parametrize("field,value", [
    ("relative_epsilon", 0.0),
    ("relative_epsilon", float("inf")),
    ("motion_epsilon", 0.0),
    ("minimum_positive_reduction_rate", 0.0),
    ("minimum_positive_reduction_rate", 1.1),
])
def test_candidate_policy_rejects_invalid(field, value):
    kwargs = {"relative_epsilon": 1e-12, "motion_epsilon": 1e-12, "minimum_positive_reduction_rate": 0.5}
    kwargs[field] = value
    with pytest.raises(staget.StageTError):
        staget.CandidatePolicy(**kwargs).validate()


def test_fidelity_metrics_improving_candidate_is_eligible():
    control = np.zeros((4, 2), dtype=np.float32)
    target = np.ones((4, 2), dtype=np.float32)
    candidate = np.full((4, 2), 0.5, dtype=np.float32)
    scale = np.ones(4)
    result = staget.compute_candidate_fidelity_metrics(
        control=control, candidate=candidate, target=target, selected_scale=scale
    )
    assert result["mechanism_eligible"] is True
    assert result["fidelity_eligible"] is True
    assert result["overall_mse_ratio"] == pytest.approx(0.25)
    assert result["accepted_rows"]["positive_distance_reduction_rate"] == 1.0


def test_fidelity_metrics_worsening_candidate_is_not_fidelity_eligible():
    control = np.zeros((4, 2), dtype=np.float32)
    target = np.ones((4, 2), dtype=np.float32)
    candidate = np.full((4, 2), -1.0, dtype=np.float32)
    scale = np.ones(4)
    result = staget.compute_candidate_fidelity_metrics(
        control=control, candidate=candidate, target=target, selected_scale=scale
    )
    assert result["mechanism_eligible"] is True
    assert result["fidelity_eligible"] is False
    assert result["overall_mse_ratio"] > 1.0


def test_fidelity_metrics_no_selection():
    control = np.zeros((3, 2), dtype=np.float32)
    result = staget.compute_candidate_fidelity_metrics(
        control=control,
        candidate=control.copy(),
        target=np.ones_like(control),
        selected_scale=np.zeros(3),
    )
    assert result["mechanism_eligible"] is False
    assert result["fidelity_eligible"] is False
    assert result["selected_row_count"] == 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fidelity_metrics_rejects_nonfinite(bad):
    control = np.zeros((2, 2), dtype=np.float32)
    candidate = np.ones((2, 2), dtype=np.float32)
    candidate[0, 0] = bad
    with pytest.raises(staget.StageTError):
        staget.compute_candidate_fidelity_metrics(
            control=control,
            candidate=candidate,
            target=np.ones_like(control),
            selected_scale=np.ones(2),
        )


def test_fidelity_metrics_rejects_shape_change():
    with pytest.raises(staget.StageTError):
        staget.compute_candidate_fidelity_metrics(
            control=np.zeros((2, 2)),
            candidate=np.zeros((2, 3)),
            target=np.zeros((2, 2)),
            selected_scale=np.ones(2),
        )


def test_fidelity_metrics_rejects_scale_motion_disagreement():
    with pytest.raises(staget.StageTError):
        staget.compute_candidate_fidelity_metrics(
            control=np.zeros((2, 2)),
            candidate=np.ones((2, 2)),
            target=np.ones((2, 2)),
            selected_scale=np.zeros(2),
        )


def test_reconstruct_aligned_candidate_metrics_exact():
    base, control, direction, target, runtime = make_base_cell()
    context = fake_runtime()[1]
    result = staget.reconstruct_aligned_candidate_metrics(
        base_cell=base,
        control=control,
        base_direction=direction,
        target=target,
        context=context,
        runtime=runtime,
        spec=SimpleNamespace(external_multiplier=0.25, reconstruction_tolerance_factor=5.0),
    )
    assert result["reconstruction_exact"] is True
    assert result["matrix_reconstruction_attempt_count"] == 7
    assert result["candidate_fidelity"]["fidelity_eligible"] is True
    assert result["gate_contract_observation"]["contract_role"] == "shadow_frozen_not_written_to_stagee"


@pytest.mark.parametrize("field", [
    "aligned_candidate_sha256",
    "aligned_selected_scale_sha256",
    "aligned_acceptance_rate",
    "internal_scale_attempt_order",
    "length_log_z_element_mismatch_count",
    "aligned_upper_element_failure_count",
    "strict_pass_aligned_fail_row_count",
])
def test_reconstruct_rejects_identity_change(field):
    base, control, direction, target, runtime = make_base_cell()
    context = fake_runtime()[1]
    base[field] = "bad" if "sha" in field else (99 if "count" in field else [1.0])
    with pytest.raises(staget.StageTError):
        staget.reconstruct_aligned_candidate_metrics(
            base_cell=base,
            control=control,
            base_direction=direction,
            target=target,
            context=context,
            runtime=runtime,
            spec=SimpleNamespace(external_multiplier=0.25, reconstruction_tolerance_factor=5.0),
        )


def test_build_candidate_matrix_counts_and_frontier():
    matrix = staget.build_candidate_matrix(fake_cells())
    assert matrix["cell_count"] == 27
    assert matrix["mechanism_eligible_cell_count"] == 27
    assert matrix["fidelity_eligible_cell_count"] == 27
    assert matrix["matrix_eligible_backbone_count"] == 9
    assert matrix["backbone_ranking"][0]["base_direction_id"] == "b0"
    assert "b0" in matrix["pareto_frontier_backbones"]


def test_build_candidate_matrix_is_order_stable():
    cells = fake_cells()
    first = staget.build_candidate_matrix(cells)
    second = staget.build_candidate_matrix(list(reversed(cells)))
    assert staget.stable_json_bytes(first) == staget.stable_json_bytes(second)


def test_build_candidate_matrix_partial_eligibility():
    cells = fake_cells()
    cells[0]["candidate_matrix_metrics"]["candidate_fidelity"]["fidelity_eligible"] = False
    matrix = staget.build_candidate_matrix(cells)
    assert matrix["fidelity_eligible_cell_count"] == 26
    assert matrix["matrix_eligible_backbone_count"] == 8


@pytest.mark.parametrize("mutation", ["cell_count", "backbone_count", "timestep_count"])
def test_build_candidate_matrix_rejects_population_change(mutation):
    cells = fake_cells()
    if mutation == "cell_count":
        cells.pop()
    elif mutation == "backbone_count":
        cells[-1]["base_direction_id"] = "b0"
    else:
        cells[-1]["timestep"] = 99
    with pytest.raises(staget.StageTError):
        staget.build_candidate_matrix(cells)


def test_classify_confirmed_frontier():
    result = staget.classify_candidate_matrix(staget.build_candidate_matrix(fake_cells()))
    assert result["primary_failure_locus"] == "confirmed_candidate_frontier"


def test_classify_no_fidelity():
    matrix = staget.build_candidate_matrix(fake_cells(all_eligible=False))
    result = staget.classify_candidate_matrix(matrix)
    assert result["primary_failure_locus"] == "candidate_fidelity"


def test_classify_stratified():
    cells = fake_cells()
    for index in range(0, len(cells), 3):
        cells[index]["candidate_matrix_metrics"]["candidate_fidelity"]["fidelity_eligible"] = False
    matrix = staget.build_candidate_matrix(cells)
    result = staget.classify_candidate_matrix(matrix)
    assert result["primary_failure_locus"] == "candidate_matrix_stratification"


def test_classify_mechanism_regression():
    matrix = staget.build_candidate_matrix(fake_cells())
    matrix["mechanism_eligible_cell_count"] = 26
    result = staget.classify_candidate_matrix(matrix)
    assert result["primary_failure_locus"] == "candidate_mechanism_regression"


def test_validate_base_report_accepts_real_schema_shape():
    assert staget.validate_base_report(fake_base_report())["execution_verdict"] == "PASS"


@pytest.mark.parametrize("field,bad", [
    ("execution_verdict", "BLOCKED"),
    ("scientific_status", "READY"),
    ("root_cause", "bad"),
    ("required_next_path", "bad"),
    ("scientific_result_sha256", "bad"),
    ("selected_configuration", "x"),
    ("train_only_recommendation", "x"),
])
def test_validate_base_report_rejects_top_level_change(field, bad):
    report = fake_base_report()
    report[field] = bad
    with pytest.raises(staget.StageTError):
        staget.validate_base_report(report)


@pytest.mark.parametrize("field,bad", [
    ("worker_count", 1),
    ("total_oof_fit_count", 107),
    ("total_callback_pair_count", 53),
    ("total_internal_scale_attempt_count", 377),
    ("processes_distinct", False),
    ("workers_sequential", False),
])
def test_validate_base_report_rejects_execution_change(field, bad):
    report = fake_base_report()
    report["confirmation_execution"][field] = bad
    with pytest.raises(staget.StageTError):
        staget.validate_base_report(report)


def test_validate_worker_payload_accepts_exact():
    worker = fake_worker()
    assert staget.validate_worker_payload(worker, expected_environment_sha256="env")["execution_verdict"] == "PASS"


@pytest.mark.parametrize("field,bad", [
    ("environment_sha256", "bad"),
    ("functional_projection_sha256", "bad"),
    ("functional_projection_matches_stage_s_resume3", False),
    ("scientific_oof_fit_count", 26),
    ("repeat_identity_fit_count", 26),
    ("total_oof_fit_count", 53),
    ("callback_pair_count", 26),
    ("shadow_internal_scale_attempt_count", 188),
    ("matrix_reconstruction_attempt_count", 188),
    ("oracle_callback_rerun_count", 1),
])
def test_validate_worker_payload_rejects_change(field, bad):
    worker = fake_worker()
    worker[field] = bad
    with pytest.raises(staget.StageTError):
        staget.validate_worker_payload(worker, expected_environment_sha256="env")


def test_compare_workers_accepts_exact():
    assert staget.compare_workers(fake_worker(), fake_worker())["all_exact"] is True


@pytest.mark.parametrize("field", [
    "environment_sha256",
    "current_fit_projection_sha256",
    "functional_projection_sha256",
    "candidate_matrix_sha256",
    "gate_contract_sha256",
    "candidate_matrix",
    "gate_contract",
])
def test_compare_workers_rejects_difference(field):
    first = fake_worker()
    second = copy.deepcopy(first)
    second[field] = "different"
    with pytest.raises(staget.StageTError):
        staget.compare_workers(first, second)


def test_blocked_report_does_not_claim_counts():
    result = staget.blocked_report(repository={"head": "x"}, error=RuntimeError("x"))
    assert result["execution_verdict"] == "BLOCKED"
    assert result["full_fit_count_claimed"] is False
    assert result["selected_configuration"] is None


def test_worker_source_has_no_environment_probe_call():
    source = Path(__file__).resolve().parents[1] / "scripts/phase3_14b_r258_staget_worker.py"
    text = source.read_text(encoding="utf-8")
    assert "probe_environment(" not in text
    assert "--probe" in text


def test_controller_order_is_probe_then_worker_loop():
    source = Path(staget.__file__).read_text(encoding="utf-8")
    probe = source.index("environment probe")
    worker = source.index("cold science worker {index + 1}")
    assert probe < worker


def test_gate_contract_rule_is_explicit():
    base, control, direction, target, runtime = make_base_cell()
    result = staget.reconstruct_aligned_candidate_metrics(
        base_cell=base,
        control=control,
        base_direction=direction,
        target=target,
        context=fake_runtime()[1],
        runtime=runtime,
        spec=SimpleNamespace(external_multiplier=0.25, reconstruction_tolerance_factor=5.0),
    )
    gate = result["gate_contract_observation"]
    assert "frozen_upper + tolerance_factor" in gate["length_space_rule"]
    assert gate["aligned_gate_written_to_stagee"] is False


def test_stable_json_rejects_nan():
    with pytest.raises(ValueError):
        staget.stable_json_bytes({"x": float("nan")})
