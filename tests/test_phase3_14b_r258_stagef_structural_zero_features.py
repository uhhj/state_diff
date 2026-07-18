from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np

from ccda_phase3 import phase314b_r258_stagef_constraint_aware_surrogate as stagef


class FakeReference:
    def __init__(self) -> None:
        self.center_log = np.full((4, 23), np.log(0.02), dtype=np.float64)
        self.scale_log = np.full((4, 23), 0.25, dtype=np.float64)

    def validate(self) -> None:
        assert self.center_log.shape == (4, 23)
        assert self.scale_log.shape == (4, 23)


def context() -> dict:
    return {"stage_d_contract": SimpleNamespace(reference=FakeReference())}


def control(rows: int = 3) -> np.ndarray:
    value = np.zeros((rows, 4, 48), dtype=np.float32)
    for row in range(rows):
        for horizon in range(4):
            points = np.zeros((24, 2), dtype=np.float32)
            points[0] = [0.8 + 0.01 * horizon, 0.4 + 0.001 * row]
            for index in range(1, 24):
                if index in (4, 9):
                    points[index] = points[index - 1]
                else:
                    points[index] = points[index - 1] + [0.02, 2.0e-6]
            value[row, horizon] = points.reshape(-1)
    return value


def test_structural_zero_mask_is_exact_and_binary():
    mask = stagef.structural_zero_mask(control())
    assert mask.dtype == np.bool_
    assert mask.shape == (3, 4, 23)
    assert np.all(mask[:, :, 3])
    assert np.all(mask[:, :, 8])
    assert int(mask.sum()) == 3 * 4 * 2


def test_resolvable_z_neutralizes_only_structural_zero_entries():
    raw = stagef.constraint_z_features(control(), context())
    state = stagef.constraint_state_features(control(), context())
    mask = state["structural_zero_mask"].astype(bool)
    assert np.all(raw[mask] == -stagef.CONSTRAINT_Z_CLIP)
    assert np.all(state["constraint_z_resolvable"][mask] == 0.0)
    assert np.array_equal(
        state["constraint_z_resolvable"][~mask], raw[~mask]
    )


def test_exact_translation_preserves_structural_zero_mask():
    source = control()
    translated = source.astype(np.float64).reshape(3, 4, 24, 2)
    translated += np.asarray([0.375, -0.625])[None, None, None, :]
    translated = translated.reshape(3, 4, 48)
    assert np.array_equal(
        stagef.structural_zero_mask(source),
        stagef.structural_zero_mask(translated),
    )


def test_float32_translation_preserves_structural_zero_mask():
    source = control()
    exact = source.astype(np.float64).reshape(3, 4, 24, 2)
    exact += np.asarray([0.375, -0.625])[None, None, None, :]
    rounded = exact.astype(np.float32).reshape(3, 4, 48)
    assert np.array_equal(
        stagef.structural_zero_mask(source),
        stagef.structural_zero_mask(rounded),
    )


def test_quantization_bound_excludes_structural_zero_segments():
    result = stagef._float32_constraint_z_bound(
        control=control(),
        offset_xy=(0.375, -0.625),
        context=context(),
        spec=stagef.ConstraintAwareSpec(),
    )
    assert result["mask_roundtrip_exact"] is True
    assert result["float32_cast_collapse_count"] == 0
    assert result["structural_zero_count"] == 3 * 4 * 2
    assert result["resolvable_segment_count"] == 3 * 4 * 21
    assert result["applied_bound"] <= result["maximum_allowed_bound"]


def test_feature_dimensions_include_explicit_mask():
    assert stagef.FULL_CENTERED_CONSTRAINT_DIMENSION == 619
    assert stagef.FULL_SEGMENT_CONSTRAINT_DIMENSION == 617
    centered = stagef._feature_block_slices("full_centered_constraint")
    segmented = stagef._feature_block_slices("full_segment_constraint")
    assert centered[-1] == (
        "structural_zero_mask",
        slice(527, 619),
        "structural_zero_mask",
    )
    assert segmented[5] == (
        "structural_zero_mask",
        slice(513, 605),
        "structural_zero_mask",
    )


def test_no_threshold_or_holdout_dependency_in_mask_api():
    signature = inspect.signature(stagef.structural_zero_mask)
    assert tuple(signature.parameters) == ("control",)
    source = inspect.getsource(stagef.structural_zero_mask)
    assert "threshold" not in source
    assert "holdout" not in source
    assert "frozen_probe" not in source
