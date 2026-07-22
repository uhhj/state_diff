from __future__ import annotations

import pytest

from ccda_phase3 import (
    phase314b_r258_stageo_resume1_descending_attempt_order_recovery as resume1,
)

def event(scale: float, marker: str = ""):
    return {"attempted_scale": scale, "marker": marker}

def expected_default():
    return resume1.expected_internal_scale_attempt_order(
        scale_grid=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
        maximum_scale=1.0,
        standardizer_epsilon=1.0e-8,
    )

def test_phase_and_schema_are_resume1():
    assert "Resume1" in resume1.PHASE
    assert resume1.SCHEMA.endswith("_v1")

def test_base_commit_chain_is_frozen():
    assert resume1.BASE_STAGEO_IMPLEMENTATION_COMMIT == (
        "ff4b4e1b94a095bf84e914af3515a3fd66eff8a1"
    )
    assert resume1.BASE_STAGEO_BLOCKED_EVIDENCE_COMMIT == (
        "8dd4a129708db0141d3d328e39270f9b4b8d8ebc"
    )

def test_blocked_report_sha_is_frozen():
    assert resume1.EXPECTED_STAGEO_BLOCKED_REPORT_SHA256 == (
        "9d3368304d3217e420b17564cc7aa103884ef970ba20916937b4f397c29167ad"
    )

def test_implementation_population_is_add_only_three_paths():
    assert len(resume1.IMPLEMENTATION_PATHS) == 3
    assert all(status == "A" for status, _ in resume1.IMPLEMENTATION_PATHS)

def test_expected_order_uses_descending_stagee_semantics():
    assert expected_default() == (1.0, 0.75, 0.5, 0.25)

def test_expected_order_filters_above_maximum():
    assert resume1.expected_internal_scale_attempt_order(
        scale_grid=(0.25, 0.5, 1.0, 2.0),
        maximum_scale=0.5,
        standardizer_epsilon=0.0,
    ) == (0.5, 0.25)

def test_expected_order_includes_epsilon_boundary():
    assert resume1.expected_internal_scale_attempt_order(
        scale_grid=(0.25, 0.5000000001),
        maximum_scale=0.5,
        standardizer_epsilon=1.0e-9,
    ) == (0.5000000001, 0.25)

@pytest.mark.parametrize("maximum", [0.0, -1.0, float("nan"), float("inf")])
def test_expected_order_rejects_invalid_maximum(maximum):
    with pytest.raises(resume1.StageOResume1Error):
        resume1.expected_internal_scale_attempt_order(
            scale_grid=(0.25,),
            maximum_scale=maximum,
            standardizer_epsilon=0.0,
        )

@pytest.mark.parametrize("epsilon", [-1.0, float("nan"), float("inf")])
def test_expected_order_rejects_invalid_epsilon(epsilon):
    with pytest.raises(resume1.StageOResume1Error):
        resume1.expected_internal_scale_attempt_order(
            scale_grid=(0.25,),
            maximum_scale=1.0,
            standardizer_epsilon=epsilon,
        )

@pytest.mark.parametrize("bad", [0.0, -0.25, float("nan"), float("inf")])
def test_expected_order_rejects_invalid_grid_value(bad):
    with pytest.raises(resume1.StageOResume1Error):
        resume1.expected_internal_scale_attempt_order(
            scale_grid=(0.25, bad),
            maximum_scale=1.0,
            standardizer_epsilon=0.0,
        )

def test_expected_order_rejects_duplicate_grid_value():
    with pytest.raises(resume1.StageOResume1Error):
        resume1.expected_internal_scale_attempt_order(
            scale_grid=(0.25, 0.5, 0.5),
            maximum_scale=1.0,
            standardizer_epsilon=0.0,
        )

def test_expected_order_rejects_empty_eligible_population():
    with pytest.raises(resume1.StageOResume1Error):
        resume1.expected_internal_scale_attempt_order(
            scale_grid=(1.0,),
            maximum_scale=0.5,
            standardizer_epsilon=0.0,
        )

def test_grouping_accepts_exact_descending_sequence():
    values = [
        event(1.0, "a"), event(0.75, "b"),
        event(0.5, "c"), event(0.25, "d"),
    ]
    result = resume1.group_attempts_in_exact_descending_order(
        values,
        expected_order=expected_default(),
        scale_key=lambda value: format(value, ".17g"),
    )
    assert tuple(float(key) for key in result) == expected_default()

def test_grouping_preserves_event_object_identity():
    values = [event(value, str(index)) for index, value in enumerate(expected_default())]
    result = resume1.group_attempts_in_exact_descending_order(
        values,
        expected_order=expected_default(),
        scale_key=lambda value: format(value, ".17g"),
    )
    flattened = [items[0] for items in result.values()]
    assert all(left is right for left, right in zip(flattened, values))

@pytest.mark.parametrize(
    "observed",
    [
        (0.25, 0.5, 0.75, 1.0),
        (1.0, 0.5, 0.75, 0.25),
        (1.0, 0.75, 0.5),
        (1.0, 0.75, 0.5, 0.25, 0.125),
        (1.0, 0.75, 0.75, 0.25),
        (1.0, 0.75, 0.25, 0.5),
        (0.75, 1.0, 0.5, 0.25),
    ],
)
def test_grouping_rejects_any_nonexact_sequence(observed):
    with pytest.raises(
        resume1.StageOResume1Error,
        match="differs from frozen Stage-E execution order",
    ):
        resume1.group_attempts_in_exact_descending_order(
            [event(value) for value in observed],
            expected_order=expected_default(),
            scale_key=lambda value: format(value, ".17g"),
        )

@pytest.mark.parametrize("bad", [0.0, -0.25, float("nan"), float("inf")])
def test_grouping_rejects_invalid_attempt_scale(bad):
    values = [event(1.0), event(0.75), event(0.5), event(bad)]
    with pytest.raises(resume1.StageOResume1Error, match="invalid attempted"):
        resume1.group_attempts_in_exact_descending_order(
            values,
            expected_order=expected_default(),
            scale_key=lambda value: format(value, ".17g"),
        )

def test_grouping_does_not_sort_events():
    values = [event(0.25), event(0.5), event(0.75), event(1.0)]
    original = list(values)
    with pytest.raises(resume1.StageOResume1Error):
        resume1.group_attempts_in_exact_descending_order(
            values,
            expected_order=expected_default(),
            scale_key=lambda value: format(value, ".17g"),
        )
    assert values == original

def test_patched_grouping_restores_original_function():
    original = object()
    class FakeStageO:
        group_attempts_by_internal_scale = original
        @staticmethod
        def _scale_key(value):
            return format(value, ".17g")
    fake = FakeStageO()
    with resume1.patched_descending_grouping(fake, expected_default()):
        assert fake.group_attempts_by_internal_scale is not original
        result = fake.group_attempts_by_internal_scale(
            [event(value) for value in expected_default()]
        )
        assert tuple(float(key) for key in result) == expected_default()
    assert fake.group_attempts_by_internal_scale is original

def test_patched_grouping_restores_after_exception():
    original = object()
    class FakeStageO:
        group_attempts_by_internal_scale = original
        @staticmethod
        def _scale_key(value):
            return format(value, ".17g")
    fake = FakeStageO()
    with pytest.raises(RuntimeError):
        with resume1.patched_descending_grouping(fake, expected_default()):
            raise RuntimeError("synthetic")
    assert fake.group_attempts_by_internal_scale is original

def test_blocked_payload_preserves_original_provenance():
    payload = resume1.blocked_report(
        repository={"head": "x"}, error=RuntimeError("synthetic")
    )
    assert payload["execution_verdict"] == "BLOCKED"
    assert payload["original_stage_o_blocked_report_preserved"] is True
    assert payload["original_stage_o_source_modified"] is False
    assert payload["selected_configuration"] is None
    assert payload["train_only_recommendation"] is None

def test_false_boundaries_population_is_unique():
    assert len(resume1.FALSE_BOUNDARIES) == len(set(resume1.FALSE_BOUNDARIES))

def test_expected_environment_contract_is_complete():
    assert resume1.EXPECTED_ENV == {
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
