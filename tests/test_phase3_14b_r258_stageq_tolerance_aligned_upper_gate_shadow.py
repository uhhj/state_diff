from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pytest

from ccda_phase3 import (
    phase314b_r258_stageq_tolerance_aligned_upper_gate_shadow as stageq,
)


def reference_arrays():
    upper = np.full(stageq.EXPECTED_POSITION_SHAPE, 0.2, dtype=np.float64)
    center = np.full(stageq.EXPECTED_POSITION_SHAPE, np.log(0.1), dtype=np.float64)
    scale = np.full(stageq.EXPECTED_POSITION_SHAPE, 0.25, dtype=np.float64)
    return upper, center, scale


def contract():
    upper, center, scale = reference_arrays()
    return stageq.tolerant_upper_contract(
        upper_bound=upper,
        center_log=center,
        scale_log=scale,
        segment_tolerance=2.5e-6,
        tolerance_factor=5.0,
    )


def synthetic_cell(
    *,
    source: str,
    timestep: int,
    strict_count: int,
    admitted: bool = False,
    aligned_failures: int = 0,
    mismatch: int = 0,
    regression: int = 0,
    cleared: int = 1,
    predicate: Optional[str] = "segment_geometry",
):
    return {
        "source_id": source,
        "timestep": timestep,
        "strict_upper_element_failure_count": strict_count,
        "aligned_upper_element_failure_count": aligned_failures,
        "length_log_z_element_mismatch_count": mismatch,
        "strict_pass_aligned_fail_row_count": regression,
        "strict_fail_aligned_pass_row_count": cleared,
        "aligned_oracle_admitted": admitted,
        "dominant_post_upper_failure_predicate": predicate,
    }


def synthetic_cells(**kwargs):
    counts = [100000, 100000, 100000, 100000, 100000, 119626]
    output = []
    index = 0
    for timestep in stageq.EXPECTED_TIMESTEPS:
        for source in stageq.ORACLE_SOURCES:
            values = dict(kwargs)
            output.append(
                synthetic_cell(
                    source=source,
                    timestep=timestep,
                    strict_count=counts[index],
                    **values,
                )
            )
            index += 1
    return output


def test_phase_schema_and_next_base_are_frozen():
    assert stageq.PHASE.endswith("Stage Q")
    assert stageq.SCHEMA.endswith("_v1")
    assert stageq.EXPECTED_BASE_NEXT_PATH == (
        "ALIGN_RECONSTRUCTION_BOUND_TOLERANCE_WITH_UPPER_SEGMENT_GATE"
    )


def test_base_commits_are_frozen():
    assert stageq.BASE_EVIDENCE_COMMIT == (
        "d0a06346e33e4638e98e129a052b8f42f4da92bb"
    )
    assert stageq.BASE_IMPLEMENTATION_COMMIT == (
        "901654fb2b72f61a681568274f4fffb69595563d"
    )


def test_base_report_hashes_are_frozen():
    assert stageq.EXPECTED_BASE_REPORT_SHA256 == (
        "e4a7aba0715a0bb6b8da72654f0bd55d9bda0c67e52f532620f26d6f4f68898c"
    )
    assert stageq.EXPECTED_BASE_SCIENTIFIC_RESULT_SHA256 == (
        "1663c33b2430c97ef3f408a3c5de1244e15ee435f02bcd47480e9b919f646031"
    )


def test_implementation_population_is_add_only_three_paths():
    assert len(stageq.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in stageq.IMPLEMENTATION_PATHS)


def test_frozen_source_population_is_complete():
    assert len(stageq.FROZEN_SOURCE_SHA256) == 5
    assert all(len(value) == 64 for value in stageq.FROZEN_SOURCE_SHA256.values())


def test_spec_default_is_frozen():
    spec = stageq.StageQSpec()
    spec.validate()
    assert spec.external_multiplier == 0.25
    assert spec.reconstruction_tolerance_factor == 5.0
    assert spec.oracle_admission_rate_min == 0.95


@pytest.mark.parametrize("value", [0.1, 0.5, 1.0])
def test_spec_rejects_external_multiplier_change(value):
    with pytest.raises(stageq.StageQError, match="external multiplier"):
        stageq.StageQSpec(external_multiplier=value).validate()


@pytest.mark.parametrize("value", [0.0, 1.0, 4.0, 6.0])
def test_spec_rejects_tolerance_factor_change(value):
    with pytest.raises(stageq.StageQError, match="tolerance factor"):
        stageq.StageQSpec(reconstruction_tolerance_factor=value).validate()


@pytest.mark.parametrize("value", [0.0, -0.1, 1.1, float("nan")])
def test_spec_rejects_invalid_admission_threshold(value):
    with pytest.raises(stageq.StageQError):
        stageq.StageQSpec(oracle_admission_rate_min=value).validate()


def test_tolerant_contract_uses_exact_length_tolerance():
    upper, center, scale = reference_arrays()
    value = contract()
    np.testing.assert_array_equal(
        value["length_upper"], upper + 5.0 * 2.5e-6
    )
    expected_z = (np.log(upper + 5.0 * 2.5e-6) - center) / scale
    np.testing.assert_array_equal(value["position_z_threshold"], expected_z)


def test_tolerant_contract_has_stable_shas():
    value = contract()
    assert len(value["length_upper_sha256"]) == 64
    assert len(value["position_z_threshold_sha256"]) == 64


@pytest.mark.parametrize(
    "field,bad",
    [
        ("upper", np.zeros(stageq.EXPECTED_POSITION_SHAPE)),
        ("upper", np.full(stageq.EXPECTED_POSITION_SHAPE, np.nan)),
        ("center", np.full((2, 2), 0.0)),
        ("scale", np.zeros(stageq.EXPECTED_POSITION_SHAPE)),
        ("scale", np.full(stageq.EXPECTED_POSITION_SHAPE, np.inf)),
    ],
)
def test_tolerant_contract_rejects_invalid_arrays(field, bad):
    upper, center, scale = reference_arrays()
    values = {"upper": upper, "center": center, "scale": scale}
    values[field] = bad
    with pytest.raises(stageq.StageQError):
        stageq.tolerant_upper_contract(
            upper_bound=values["upper"],
            center_log=values["center"],
            scale_log=values["scale"],
            segment_tolerance=2.5e-6,
            tolerance_factor=5.0,
        )


@pytest.mark.parametrize("tolerance", [0.0, -1.0, float("nan"), float("inf")])
def test_tolerant_contract_rejects_invalid_tolerance(tolerance):
    upper, center, scale = reference_arrays()
    with pytest.raises(stageq.StageQError):
        stageq.tolerant_upper_contract(
            upper_bound=upper,
            center_log=center,
            scale_log=scale,
            segment_tolerance=tolerance,
            tolerance_factor=5.0,
        )


def test_aligned_masks_are_exact_for_values_below_and_at_boundary():
    value = contract()
    upper = np.asarray(value["length_upper"])
    lengths = np.stack([upper - 1e-8, upper], axis=0)
    center = reference_arrays()[1]
    scale = reference_arrays()[2]
    z = (np.log(lengths) - center[None]) / scale[None]
    result = stageq.aligned_upper_masks(
        lengths=lengths,
        z_scores=z,
        contract=value,
    )
    assert result["element_equivalence"] is True
    assert result["row_equivalence"] is True
    assert np.all(result["length_row_pass"])


def test_aligned_masks_reject_value_above_boundary():
    value = contract()
    upper = np.asarray(value["length_upper"])
    lengths = np.stack([upper + 1e-7], axis=0)
    center = reference_arrays()[1]
    scale = reference_arrays()[2]
    z = (np.log(lengths) - center[None]) / scale[None]
    result = stageq.aligned_upper_masks(
        lengths=lengths,
        z_scores=z,
        contract=value,
    )
    assert result["element_equivalence"] is True
    assert not result["length_row_pass"][0]


def test_aligned_masks_detect_explicit_z_mismatch():
    value = contract()
    upper = np.asarray(value["length_upper"])
    lengths = np.stack([upper], axis=0)
    z = np.asarray(value["position_z_threshold"])[None] + 1.0
    result = stageq.aligned_upper_masks(
        lengths=lengths,
        z_scores=z,
        contract=value,
    )
    assert result["element_equivalence"] is False
    assert result["element_mismatch_count"] == 92


@pytest.mark.parametrize(
    "length_shape,z_shape",
    [
        ((2, 4, 22), (2, 4, 22)),
        ((2, 4, 23), (2, 4, 22)),
        ((2, 1, 4, 23), (2, 1, 4, 23)),
    ],
)
def test_aligned_masks_reject_invalid_shapes(length_shape, z_shape):
    with pytest.raises(stageq.StageQError):
        stageq.aligned_upper_masks(
            lengths=np.ones(length_shape),
            z_scores=np.ones(z_shape),
            contract=contract(),
        )


def all_true_masks(rows=3):
    return {
        name: np.ones(rows, dtype=np.bool_)
        for name in stageq.PREDICATE_ORDER[:-1]
    }


def test_sequential_summary_all_accept():
    result = stageq.sequential_attempt_summary(
        active=np.ones(3, dtype=np.bool_),
        predicate_masks=all_true_masks(),
        topology_pass=np.ones(3, dtype=np.bool_),
    )
    assert result["accepted_row_count"] == 3
    assert sum(result["first_failed_counts"].values()) == 0
    assert result["population_closed"] is True


def test_sequential_summary_assigns_first_failure_once():
    masks = all_true_masks()
    masks["upper_segment_geometry"] = np.asarray([False, True, True])
    masks["segment_geometry"] = np.asarray([False, False, True])
    result = stageq.sequential_attempt_summary(
        active=np.ones(3, dtype=np.bool_),
        predicate_masks=masks,
        topology_pass=np.ones(3, dtype=np.bool_),
    )
    assert result["first_failed_counts"]["upper_segment_geometry"] == 1
    assert result["first_failed_counts"]["segment_geometry"] == 1
    assert result["accepted_row_count"] == 1


def test_sequential_summary_respects_inactive_rows():
    masks = all_true_masks()
    result = stageq.sequential_attempt_summary(
        active=np.asarray([False, True, False]),
        predicate_masks=masks,
        topology_pass=np.ones(3, dtype=np.bool_),
    )
    assert result["active_row_count"] == 1
    assert result["accepted_row_count"] == 1


def test_sequential_summary_topology_failure():
    result = stageq.sequential_attempt_summary(
        active=np.ones(2, dtype=np.bool_),
        predicate_masks=all_true_masks(2),
        topology_pass=np.asarray([True, False]),
    )
    assert result["first_failed_counts"]["topology"] == 1
    assert result["accepted_row_count"] == 1


def test_sequential_summary_rejects_missing_predicate():
    masks = all_true_masks()
    del masks["displacement"]
    with pytest.raises(stageq.StageQError):
        stageq.sequential_attempt_summary(
            active=np.ones(3, dtype=np.bool_),
            predicate_masks=masks,
            topology_pass=np.ones(3, dtype=np.bool_),
        )


def test_classify_length_z_mismatch_precedes_other_outcomes():
    cells = synthetic_cells(admitted=True)
    cells[0]["length_log_z_element_mismatch_count"] = 1
    result = stageq.classify_shadow_alignment(cells)
    assert result["root_cause"].endswith("masks_not_equivalent")


def test_classify_regression_precedes_admission():
    cells = synthetic_cells(admitted=True)
    cells[0]["strict_pass_aligned_fail_row_count"] = 1
    result = stageq.classify_shadow_alignment(cells)
    assert result["primary_failure_locus"] == "strict_pass_aligned_fail_regression"


def test_classify_remaining_aligned_upper_failure():
    cells = synthetic_cells()
    cells[0]["aligned_upper_element_failure_count"] = 1
    result = stageq.classify_shadow_alignment(cells)
    assert result["primary_failure_locus"] == "aligned_upper_violation_remains"


def test_classify_full_dual_oracle_admission():
    result = stageq.classify_shadow_alignment(
        synthetic_cells(admitted=True, predicate=None)
    )
    assert result["root_cause"].endswith("restores_dual_oracle_admission")
    assert result["aligned_oracle_admitted_cell_count"] == 6


def test_classify_partial_admission():
    cells = synthetic_cells(predicate="segment_geometry")
    cells[0]["aligned_oracle_admitted"] = True
    result = stageq.classify_shadow_alignment(cells)
    assert result["primary_failure_locus"] == "heterogeneous_aligned_oracle_admission"


@pytest.mark.parametrize(
    "predicate,next_path",
    [
        ("finite_state", "AUDIT_ORACLE_FINITE_STATE_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("lower_segment_geometry", "AUDIT_LOWER_SEGMENT_GATE_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("coordinate_recenter", "AUDIT_ORACLE_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("coordinate_geometry", "AUDIT_ORACLE_LOCAL_FRAME_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("reconstruction_bounds", "AUDIT_RECONSTRUCTION_BOUND_CLOSURE_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("segment_geometry", "ALIGN_HISTORICAL_SEGMENT_BOUND_TOLERANCE_WITH_RECONSTRUCTION_CONTRACT"),
        ("direction_retention", "AUDIT_ORACLE_DIRECTION_RETENTION_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("displacement", "AUDIT_ORACLE_DISPLACEMENT_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
        ("topology", "AUDIT_ORACLE_TOPOLOGY_AFTER_UPPER_TOLERANCE_ALIGNMENT"),
    ],
)
def test_classify_stable_post_upper_predicate(predicate, next_path):
    result = stageq.classify_shadow_alignment(
        synthetic_cells(predicate=predicate)
    )
    assert result["dominant_post_upper_failure_predicate"] == predicate
    assert result["required_next_path"] == next_path


def test_classify_heterogeneous_post_upper_predicates():
    cells = synthetic_cells(predicate="segment_geometry")
    cells[0]["dominant_post_upper_failure_predicate"] = "topology"
    cells[1]["dominant_post_upper_failure_predicate"] = "topology"
    result = stageq.classify_shadow_alignment(cells)
    assert result["primary_failure_locus"] == "heterogeneous_post_upper_rejection"


def test_classify_insufficient_surface():
    result = stageq.classify_shadow_alignment(
        synthetic_cells(predicate=None)
    )
    assert result["primary_failure_locus"] == "aligned_scalar_surface_insufficient"


def test_classify_rejects_wrong_cell_count():
    with pytest.raises(stageq.StageQError, match="cell population"):
        stageq.classify_shadow_alignment(synthetic_cells()[:-1])


def test_classify_rejects_wrong_strict_violation_total():
    cells = synthetic_cells()
    cells[0]["strict_upper_element_failure_count"] -= 1
    with pytest.raises(stageq.StageQError, match="strict violation"):
        stageq.classify_shadow_alignment(cells)


def test_blocked_report_preserves_stagep_and_boundaries():
    result = stageq.blocked_report(
        repository={"head": "x"},
        error=RuntimeError("synthetic"),
    )
    assert result["execution_verdict"] == "BLOCKED"
    assert result["stagep_report_preserved"] is True
    assert result["selected_configuration"] is None
    assert result["train_only_recommendation"] is None
    assert all(result[key] is False for key in stageq.FALSE_BOUNDARIES)


def test_expected_environment_contract_is_complete():
    assert stageq.EXPECTED_ENV == {
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


def test_false_boundaries_are_unique():
    assert len(stageq.FALSE_BOUNDARIES) == len(set(stageq.FALSE_BOUNDARIES))


def test_predicate_order_is_frozen():
    assert stageq.PREDICATE_ORDER[1] == "upper_segment_geometry"
    assert stageq.PREDICATE_ORDER[-1] == "topology"
    assert len(stageq.PREDICATE_ORDER) == 10


def test_audit_shadow_cell_synthetic_runtime_exercises_shadow_selection():
    from types import SimpleNamespace

    rows = 1
    control = np.zeros((rows, 4, 48), dtype=np.float32)
    direction = np.zeros_like(control, dtype=np.float64)
    scale_grid = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    tolerance = 2.5e-6
    upper = np.full(stageq.EXPECTED_POSITION_SHAPE, 0.2, dtype=np.float64)
    lower = np.full(stageq.EXPECTED_POSITION_SHAPE, 0.05, dtype=np.float64)
    reference = SimpleNamespace(
        center_log=np.full(
            stageq.EXPECTED_POSITION_SHAPE,
            np.log(0.2) - 4.0 * 0.25,
            dtype=np.float64,
        ),
        scale_log=np.full(stageq.EXPECTED_POSITION_SHAPE, 0.25, dtype=np.float64),
    )
    lengths = np.full(
        (rows, *stageq.EXPECTED_POSITION_SHAPE),
        0.2 + 5.0e-6,
        dtype=np.float64,
    )

    class FakeStageB:
        @staticmethod
        def segment_lengths(value):
            return lengths.copy()

        @staticmethod
        def physical_validity(value, historical):
            return {"topology": np.ones((value.shape[0], 1), dtype=np.bool_)}

    class FakeStaged:
        @staticmethod
        def segment_scores(value, ref):
            z = (
                np.log(lengths)
                - np.asarray(ref.center_log, dtype=np.float64)[None]
            ) / np.asarray(ref.scale_log, dtype=np.float64)[None]
            z = z[:, None]
            upper_element = np.maximum(z, 0.0)
            upper_score = np.max(upper_element, axis=(2, 3))
            return {"z": z, "upper": upper_score}

    class FakeStageE:
        stageb = FakeStageB
        staged = FakeStaged
        sha256_array = staticmethod(stageq.sha256_array)

        @staticmethod
        def segment_bounds(definition, context):
            return {
                "upper": upper.copy(),
                "lower": lower.copy(),
                "upper_sha256": stageq.sha256_array(upper),
                "lower_sha256": stageq.sha256_array(lower),
            }

        @staticmethod
        def reconstruct_segment_vectors(**kwargs):
            return {
                "candidate": control.copy(),
                "bound_pass": np.ones(rows, dtype=np.bool_),
                "coordinate_possible": np.ones(rows, dtype=np.bool_),
                "clipped_segment_rate": 0.0,
                "degenerate_segment_rate": 0.0,
            }

        @staticmethod
        def observable_row_metrics(**kwargs):
            z = FakeStaged.segment_scores(control, reference)["z"]
            upper_pass = np.max(z[:, 0], axis=(1, 2)) <= 4.0
            ones = np.ones(rows, dtype=np.bool_)
            return {
                "upper_pass": upper_pass,
                "finite": ones,
                "lower_pass": ones,
                "coordinate_pass": ones,
                "segment_pass": ones,
                "retention_pass": ones,
                "nonzero_move": ones,
            }

    definition = SimpleNamespace(
        maximum_scale=2.0,
        integration_mode="segment_reconstruct",
        validate=lambda: None,
    )
    integrator_spec = SimpleNamespace(
        scale_grid=scale_grid,
        standardizer_epsilon=1.0e-12,
        segment_tolerance=tolerance,
    )
    candidate_sha = stageq.sha256_array(control)
    selected = np.zeros(rows, dtype=np.float64)
    events = []
    for scale in sorted(scale_grid, reverse=True):
        events.append(
            {
                "event_type": "scale_attempt",
                "attempted_scale": scale,
                "first_failed_counts": {"upper_segment_geometry": rows},
            }
        )

    class FakeStageK:
        @staticmethod
        def callback_integrate_rowwise(**kwargs):
            return (
                {
                    "selected_scale": selected.copy(),
                    "candidate": control.copy(),
                    "candidate_sha256": candidate_sha,
                    "scale_candidates": [
                        {"scale": scale, "candidate_sha256": candidate_sha}
                        for scale in sorted(scale_grid, reverse=True)
                    ],
                },
                {
                    "events": events,
                    "events_sha256": "capture",
                    "returned_result_bit_exact": True,
                },
            )

    fake_stageo = SimpleNamespace(sha256_array=stageq.sha256_array)
    fake_stagef = SimpleNamespace(fixed_integrator_definition=lambda: definition)
    runtime = {
        "stageo": fake_stageo,
        "stagek": FakeStageK,
        "stagel": SimpleNamespace(stagee258=FakeStageE),
        "stagep": SimpleNamespace(),
        "stagef": fake_stagef,
        "integrator_spec": integrator_spec,
        "callback_spec": SimpleNamespace(),
    }
    context = {
        "stage_d_contract": SimpleNamespace(reference=reference),
        "upper_gate": SimpleNamespace(upper_threshold=4.0),
        "historical_geometry": SimpleNamespace(coordinate_abs_max=10.0),
    }
    base_attempts = []
    for scale in sorted(scale_grid, reverse=True):
        raw = (control.astype(np.float64) + scale * direction * 0.25).astype(
            np.float32
        )
        base_attempts.append(
            {
                "internal_scale": scale,
                "raw_proposal_sha256": stageq.sha256_array(raw),
                "reconstructed_candidate_sha256": candidate_sha,
            }
        )
    base_cell = {
        "source_id": "raw_oracle",
        "timestep": 10,
        "final_selected_scale_sha256": stageq.sha256_array(selected),
        "final_candidate_sha256": candidate_sha,
        "callback_capture_sha256": "capture",
        "attempt_records": base_attempts,
    }
    result = stageq.audit_shadow_cell(
        source_id="raw_oracle",
        timestep=10,
        control=control,
        base_direction=direction,
        context=context,
        runtime=runtime,
        base_cell=base_cell,
        spec=stageq.StageQSpec(),
    )
    assert result["legacy_oracle_acceptance_rate"] == 0.0
    assert result["aligned_oracle_acceptance_rate"] == 1.0
    assert result["aligned_oracle_admitted"] is True
    assert result["aligned_upper_element_failure_count"] == 0
    assert result["length_log_z_element_mismatch_count"] == 0
    assert result["strict_upper_element_failure_count"] == 7 * 92
