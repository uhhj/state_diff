from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef


class FakeReference:
    def __init__(self) -> None:
        self.center_log = np.full((4, 23), np.log(0.02), dtype=np.float64)
        self.scale_log = np.full((4, 23), 0.25, dtype=np.float64)

    def validate(self) -> None:
        assert self.center_log.shape == (4, 23)
        assert self.scale_log.shape == (4, 23)
        assert np.all(self.scale_log > 0.0)


def context() -> dict:
    return {"stage_d_contract": SimpleNamespace(reference=FakeReference())}


def control(rows: int = 4, base: float = 0.8) -> np.ndarray:
    value = np.zeros((rows, 4, 48), dtype=np.float32)
    for row in range(rows):
        for horizon in range(4):
            points = np.zeros((24, 2), dtype=np.float32)
            points[0] = [base + 0.01 * horizon, 0.4 + 0.001 * row]
            for index in range(1, 24):
                if index in (4, 9):
                    points[index] = points[index - 1]
                else:
                    points[index] = points[index - 1] + [0.02, 2.0e-6]
            value[row, horizon] = points.reshape(-1)
    return value


def test_clipping_aware_ulp_formula_covers_observed_roundtrip():
    result = stagef._float32_constraint_z_bound(
        control=control(),
        offset_xy=(0.375, -0.625),
        context=context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    diagnosis = result["formula_diagnostic"]
    assert result["formula_covers_observed"] is True
    assert diagnosis["formula_result"]["formula_covers_observed"] is True
    assert (
        diagnosis["formula_result"]["maximum_clipping_aware_z_bound"]
        >= diagnosis["observed_roundtrip"]["maximum_resolvable_z_difference"]
    )
    assert result["applied_bound"] <= result["maximum_allowed_bound"]
    assert diagnosis["observed_roundtrip"][
        "round_to_nearest_half_ulp_model_covers_length"
    ] is True


def test_formula_preserves_frozen_factor_and_maximum():
    spec = stagef.ConstraintAwareSpec()
    result = stagef._float32_constraint_z_bound(
        control=control(),
        offset_xy=spec.translation_offset_xy,
        context=context(),
        spec=spec,
    )
    frozen = result["formula_diagnostic"]["frozen_parameters"]
    assert frozen["ulp_factor"] == 8.0
    assert frozen["maximum_allowed_bound"] == 1.0e-2
    assert frozen["ordinary_translation_tolerance"] == 2.0e-5


def test_formula_is_segmentwise_and_clipping_aware():
    result = stagef._float32_constraint_z_bound(
        control=control(),
        offset_xy=(0.375, -0.625),
        context=context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    diagnosis = result["formula_diagnostic"]
    assert "segmentwise" in diagnosis["admission_formula"]
    assert "clipped_log_z_interval" in diagnosis["admission_formula"]
    assert diagnosis["worst_segments"]
    record = diagnosis["worst_segments"][0]
    assert set(record) == {
        "row",
        "horizon_index",
        "segment_index",
        "source_length",
        "rounded_length",
        "scale_log",
        "source_z",
        "rounded_z",
        "observed_z_difference",
        "observed_length_error",
        "full_endpoint_ulp_norm",
        "half_endpoint_ulp_norm",
        "safety_length_error",
        "safety_relative_error",
        "clipping_aware_z_bound",
        "source_lower_clipped",
        "source_upper_clipped",
    }


def test_formula_error_carries_json_safe_diagnosis_without_relaxation():
    with pytest.raises(stagef.ConstraintZULPAdmissionError) as captured:
        stagef._float32_constraint_z_bound(
            control=control(base=1000.0),
            offset_xy=(0.375, -0.625),
            context=context(),
            spec=stagef.ConstraintAwareSpec(
                translation_float32_z_bound_max=1.0e-8,
            ),
        )
    error = captured.value
    assert isinstance(error.diagnostic, dict)
    assert error.required_next_path
    policy = error.diagnostic["policy"]
    assert policy["exact_gate_relaxed"] is False
    assert policy["ulp_factor_changed"] is False
    assert policy["factor_selected_from_predeclared_population"] is True
    assert policy["maximum_allowed_bound_changed"] is False
    assert policy["holdout_used"] is False
    assert policy["frozen_probe_used"] is False
    assert policy["automatic_tolerance_inferred"] is False
    assert policy["admission_enforced"] is True


def test_v5_empirical_multiplier_is_comparison_only_not_admission():
    source = inspect.getsource(stagef._float32_constraint_z_bound)
    assert "v5_empirical_bound" in source
    assert "applied_bound = max(" in source
    assert "float(maximum_formula_bound)" in source
    assert (
        "applied_bound = max(\n"
        "        float(spec.translation_tolerance),\n"
        "        float(maximum_formula_bound),\n"
        "    )"
    ) in source


def test_structural_zeros_are_excluded_from_continuous_formula_population():
    result = stagef._float32_constraint_z_bound(
        control=control(rows=3),
        offset_xy=(0.375, -0.625),
        context=context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["structural_zero_count"] == 3 * 4 * 2
    assert result["resolvable_segment_count"] == 3 * 4 * 21
    assert result["float32_cast_collapse_count"] == 0


def test_formula_helper_matches_frozen_z_mapping():
    reference = FakeReference()
    lengths = np.asarray([[[0.0, 0.02, 1.0e6] + [0.02] * 20] * 4])
    direct = stagef._constraint_z_from_lengths(
        lengths=lengths,
        center=reference.center_log,
        scale=reference.scale_log,
        clip=stagef.CONSTRAINT_Z_CLIP,
    )
    assert direct.shape == (1, 4, 23)
    assert direct[0, 0, 0] == -stagef.CONSTRAINT_Z_CLIP
    assert abs(direct[0, 0, 1]) < 1.0e-12
    assert direct[0, 0, 2] == stagef.CONSTRAINT_Z_CLIP
