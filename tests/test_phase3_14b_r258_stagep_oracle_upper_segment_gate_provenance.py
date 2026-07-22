from __future__ import annotations

import copy

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r258_stagep_oracle_upper_segment_gate_provenance as stagep,
)


def score_payload(z: np.ndarray):
    value = np.asarray(z, dtype=np.float64)
    if value.ndim == 3:
        value = value[:, None]
    upper_element = np.maximum(value, 0.0)
    upper = np.max(upper_element, axis=(2, 3))
    return {
        "z": value,
        "upper_element": upper_element,
        "upper": upper,
    }


def fake_attempt(
    *,
    total: int = 10,
    within: int = 10,
    beyond: int = 0,
    not_above: int = 0,
    raw_pass_candidate_fail: int = 0,
    raw_fail_candidate_fail: int = 10,
    control_overlap: int = 0,
    scale: float = 1.0,
):
    assert within + beyond + not_above == total
    return {
        "internal_scale": scale,
        "callback_upper_first_failed_count": 1,
        "reconstruction_boundary": {
            "candidate_gate_violation_count": total,
            "candidate_gate_violation_within_reconstruction_tolerance_count": within,
            "candidate_gate_violation_beyond_reconstruction_tolerance_count": beyond,
            "candidate_gate_fail_while_length_not_above_bound_count": not_above,
        },
        "control_raw_candidate_transition": {
            "raw_pass_candidate_fail_count": raw_pass_candidate_fail,
            "control_fail_raw_fail_candidate_fail_count": control_overlap,
            "control_pass_raw_fail_candidate_fail_count": (
                raw_fail_candidate_fail - control_overlap
            ),
            "candidate_fail_already_in_control_count": control_overlap,
            "candidate_fail_not_in_control_count": total - control_overlap,
        },
    }


def fake_cells(attempt=None):
    template = fake_attempt() if attempt is None else attempt
    cells = []
    for timestep in stagep.EXPECTED_TIMESTEPS:
        for source in stagep.ORACLE_SOURCES:
            cells.append(
                {
                    "source_id": source,
                    "timestep": timestep,
                    "control_upper": {"row_failure_count": 0},
                    "attempt_records": [copy.deepcopy(template)],
                }
            )
    return cells


def test_phase_schema_and_population():
    assert stagep.PHASE.endswith("Stage P")
    assert stagep.SCHEMA.endswith("_v1")
    assert stagep.EXPECTED_PAIR_COUNT == 6
    assert stagep.EXPECTED_POSITION_SHAPE == (4, 23)
    assert stagep.EXPECTED_POSITION_COUNT == 92


def test_base_evidence_hashes_are_frozen():
    assert stagep.BASE_EVIDENCE_COMMIT == (
        "c165dbb7882de4789612747fa4e396b01bfc5d4b"
    )
    assert stagep.EXPECTED_BASE_REPORT_SHA256 == (
        "34c9cd1250b80af9311e8242bce4da1f583a335305ea34154db68b4694e100bd"
    )
    assert stagep.EXPECTED_RECOVERED_STAGEO_PAYLOAD_SHA256 == (
        "e6c062aa4fb4d03794cf5b9e54cd5e375a09b4bd7af6a25c98d3032e4a4e579e"
    )
    assert stagep.EXPECTED_STAGEO_SCIENTIFIC_RESULT_SHA256 == (
        "7f1306197a68291f4927c203b38219638855720c1c579e6908ad4b3abb529a2d"
    )


def test_implementation_population_is_add_only():
    assert len(stagep.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in stagep.IMPLEMENTATION_PATHS)


def test_source_binding_population():
    assert len(stagep.FROZEN_SOURCE_SHA256) == 4
    assert all(len(value) == 64 for value in stagep.FROZEN_SOURCE_SHA256.values())


def test_spec_default_is_frozen():
    spec = stagep.StagePSpec()
    spec.validate()
    assert spec.external_multiplier == 0.25
    assert spec.top_position_count == 12


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, -0.25])
def test_spec_rejects_changed_external_multiplier(value):
    if value == 0.25:
        pytest.skip("registered value")
    with pytest.raises(stagep.StagePError):
        stagep.StagePSpec(external_multiplier=value).validate()


def test_spec_rejects_changed_top_position_count():
    with pytest.raises(stagep.StagePError):
        stagep.StagePSpec(top_position_count=10).validate()


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_spec_rejects_invalid_binary_tolerance(value):
    with pytest.raises(stagep.StagePError):
        stagep.StagePSpec(binary_tolerance=value).validate()


def test_scalar_stats_exact_simple_values():
    result = stagep.scalar_stats(np.asarray([1.0, 2.0, 3.0]))
    assert result["count"] == 3
    assert result["mean"] == 2.0
    assert result["min"] == 1.0
    assert result["max"] == 3.0


@pytest.mark.parametrize("value", [np.asarray([]), np.asarray([np.nan])])
def test_scalar_stats_rejects_invalid_population(value):
    with pytest.raises(stagep.StagePError):
        stagep.scalar_stats(value)


def test_integer_matrix_accepts_4x23():
    value = np.zeros((4, 23), dtype=np.int64)
    result = stagep.integer_matrix(value, "x")
    assert len(result) == 4
    assert all(len(row) == 23 for row in result)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((23, 4), dtype=np.int64),
        np.zeros((4, 22), dtype=np.int64),
        np.full((4, 23), -1, dtype=np.int64),
        np.full((4, 23), 0.5, dtype=np.float64),
    ],
)
def test_integer_matrix_rejects_invalid(value):
    with pytest.raises(stagep.StagePError):
        stagep.integer_matrix(value, "x")


def test_float_matrix_accepts_4x23():
    value = np.zeros((4, 23), dtype=np.float64)
    result = stagep.float_matrix(value, "x")
    assert result[0][0] == 0.0


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((3, 23), dtype=np.float64),
        np.full((4, 23), np.nan, dtype=np.float64),
    ],
)
def test_float_matrix_rejects_invalid(value):
    with pytest.raises(stagep.StagePError):
        stagep.float_matrix(value, "x")


def test_upper_surface_summary_no_failures():
    z = np.zeros((2, 4, 23), dtype=np.float64)
    result = stagep.upper_surface_summary(
        scores=score_payload(z), threshold=4.0, top_position_count=12
    )
    assert result["row_pass_rate"] == 1.0
    assert result["element_failure_count"] == 0
    assert result["row_failure_count"] == 0


def test_upper_surface_summary_single_failure():
    z = np.zeros((2, 4, 23), dtype=np.float64)
    z[0, 1, 2] = 4.5
    result = stagep.upper_surface_summary(
        scores=score_payload(z), threshold=4.0, top_position_count=12
    )
    assert result["row_pass_rate"] == 0.5
    assert result["row_failure_count"] == 1
    assert result["element_failure_count"] == 1
    assert result["failure_count_by_position"][1][2] == 1
    assert result["worst_positions"][0]["horizon_index"] == 1
    assert result["worst_positions"][0]["segment_index"] == 2


def test_upper_surface_summary_boundary_is_pass():
    z = np.full((1, 4, 23), 4.0, dtype=np.float64)
    result = stagep.upper_surface_summary(
        scores=score_payload(z), threshold=4.0, top_position_count=12
    )
    assert result["row_pass_rate"] == 1.0
    assert result["element_failure_count"] == 0


@pytest.mark.parametrize(
    "shape",
    [(2, 4, 23), (2, 1, 4, 22), (2, 2, 4, 23)],
)
def test_upper_surface_summary_rejects_bad_z_shape(shape):
    z = np.zeros(shape, dtype=np.float64)
    payload = {
        "z": z,
        "upper_element": np.maximum(z, 0.0),
        "upper": np.zeros((2, 1), dtype=np.float64),
    }
    with pytest.raises(stagep.StagePError):
        stagep.upper_surface_summary(
            scores=payload, threshold=4.0, top_position_count=12
        )


def test_reconstruction_boundary_within_tolerance():
    raw = np.full((1, 4, 23), 2.0)
    upper = np.ones((4, 23))
    candidate = np.ones((1, 4, 23))
    candidate[0, 0, 0] += 1.0e-6
    z = np.zeros((1, 4, 23))
    z[0, 0, 0] = 4.1
    result = stagep.reconstruction_boundary_summary(
        raw_lengths=raw,
        candidate_lengths=candidate,
        upper_bound=upper,
        candidate_scores=score_payload(z),
        threshold=4.0,
        segment_tolerance=2.5e-6,
    )
    assert result["candidate_gate_violation_count"] == 1
    assert (
        result[
            "candidate_gate_violation_within_reconstruction_tolerance_count"
        ]
        == 1
    )
    assert result["candidate_gate_violation_beyond_reconstruction_tolerance_count"] == 0


def test_reconstruction_boundary_beyond_tolerance():
    upper = np.ones((4, 23))
    candidate = np.ones((1, 4, 23))
    candidate[0, 0, 0] += 1.0e-3
    z = np.zeros((1, 4, 23))
    z[0, 0, 0] = 4.1
    result = stagep.reconstruction_boundary_summary(
        raw_lengths=candidate,
        candidate_lengths=candidate,
        upper_bound=upper,
        candidate_scores=score_payload(z),
        threshold=4.0,
        segment_tolerance=2.5e-6,
    )
    assert result["candidate_gate_violation_beyond_reconstruction_tolerance_count"] == 1


def test_reconstruction_boundary_log_roundtrip_case():
    upper = np.ones((4, 23))
    candidate = np.ones((1, 4, 23))
    z = np.zeros((1, 4, 23))
    z[0, 0, 0] = 4.0000000001
    result = stagep.reconstruction_boundary_summary(
        raw_lengths=candidate,
        candidate_lengths=candidate,
        upper_bound=upper,
        candidate_scores=score_payload(z),
        threshold=4.0,
        segment_tolerance=2.5e-6,
    )
    assert result["candidate_gate_fail_while_length_not_above_bound_count"] == 1


def test_transition_summary_counts_cohorts():
    control = np.zeros((1, 4, 23))
    raw = np.zeros((1, 4, 23))
    candidate = np.zeros((1, 4, 23))
    raw[0, 0, 0] = 4.1
    candidate[0, 0, 0] = 4.2
    candidate[0, 0, 1] = 4.2
    result = stagep.transition_summary(
        control_scores=score_payload(control),
        raw_scores=score_payload(raw),
        candidate_scores=score_payload(candidate),
        threshold=4.0,
    )
    assert result["control_pass_raw_fail_candidate_fail_count"] == 1
    assert result["raw_pass_candidate_fail_count"] == 1
    assert result["candidate_fail_not_in_control_count"] == 2


def test_classify_tolerance_contract_mismatch():
    result = stagep.classify_upper_gate_provenance(fake_cells())
    assert result["root_cause"] == (
        "phase314b_r258_stagep_upper_gate_is_stricter_than_"
        "reconstruction_bound_tolerance"
    )
    assert result["required_next_path"] == (
        "ALIGN_RECONSTRUCTION_BOUND_TOLERANCE_WITH_UPPER_SEGMENT_GATE"
    )


def test_classify_material_bound_violation():
    attempt = fake_attempt(total=10, within=9, beyond=1, not_above=0)
    result = stagep.classify_upper_gate_provenance(fake_cells(attempt))
    assert "materially_exceeds" in result["root_cause"]


def test_classify_log_exp_roundoff():
    attempt = fake_attempt(total=10, within=0, beyond=0, not_above=10)
    result = stagep.classify_upper_gate_provenance(fake_cells(attempt))
    assert result["root_cause"].endswith("log_exp_upper_boundary_roundoff")


def test_classify_mixed_numerics():
    attempt = fake_attempt(total=10, within=5, beyond=0, not_above=5)
    result = stagep.classify_upper_gate_provenance(fake_cells(attempt))
    assert result["root_cause"].endswith("upper_boundary_numerical_contract_is_mixed")


def test_classification_aggregates_six_cells():
    result = stagep.classify_upper_gate_provenance(fake_cells())
    assert result["cell_count"] == 6
    assert result["internal_scale_attempt_count"] == 6
    assert result["candidate_gate_violation_count"] == 60
    assert result["callback_upper_failure_cell_count"] == 6


def test_classification_groups_sources_timesteps_and_scales():
    result = stagep.classify_upper_gate_provenance(fake_cells())
    assert set(result["candidate_violation_counts_by_source"]) == set(
        stagep.ORACLE_SOURCES
    )
    assert set(result["candidate_violation_counts_by_timestep"]) == {
        "10",
        "25",
        "50",
    }
    assert set(result["candidate_violation_counts_by_internal_scale"]) == {"1"}


def test_classification_rejects_wrong_cell_count():
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(fake_cells()[:-1])


def test_classification_rejects_unknown_source():
    cells = fake_cells()
    cells[0]["source_id"] = "x"
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(cells)


def test_classification_rejects_empty_attempts():
    cells = fake_cells()
    cells[0]["attempt_records"] = []
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(cells)


def test_classification_rejects_nonclosing_categories():
    cells = fake_cells()
    cells[0]["attempt_records"][0]["reconstruction_boundary"][
        "candidate_gate_violation_within_reconstruction_tolerance_count"
    ] = 9
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(cells)


def test_classification_rejects_missing_callback_upper_failure():
    cells = fake_cells()
    cells[0]["attempt_records"][0]["callback_upper_first_failed_count"] = 0
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(cells)


def test_classification_rejects_zero_direct_violations():
    attempt = fake_attempt(total=0, within=0, beyond=0, not_above=0, raw_fail_candidate_fail=0)
    with pytest.raises(stagep.StagePError):
        stagep.classify_upper_gate_provenance(fake_cells(attempt))


def test_blocked_report_preserves_base_provenance():
    result = stagep.blocked_report(
        repository={"head": "x"}, error=RuntimeError("synthetic")
    )
    assert result["execution_verdict"] == "BLOCKED"
    assert result["stageo_resume1_report_preserved"] is True
    assert result["original_stageo_blocked_report_preserved"] is True
    assert result["selected_configuration"] is None
    assert result["train_only_recommendation"] is None


def test_false_boundaries_are_unique():
    assert len(stagep.FALSE_BOUNDARIES) == len(set(stagep.FALSE_BOUNDARIES))


def test_expected_environment_contract_is_complete():
    assert stagep.EXPECTED_ENV == {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_scale_key_is_roundtrip_stable():
    for value in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        assert float(stagep._scale_key(value)) == value
