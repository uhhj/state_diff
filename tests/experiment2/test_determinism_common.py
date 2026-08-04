import numpy as np

from scripts.experiment2.phase0.determinism_common import (
    compare_repeat_metadata,
    compare_traces,
)


def test_compare_traces_accepts_exact_payloads():
    first = {
        "phase": np.asarray(["no_action", "preload"]),
        "x": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
    }
    second = {key: value.copy() for key, value in first.items()}
    result = compare_traces(first, second, numeric_atol=0.0)
    assert result["shape_match"]
    assert result["phase_exact"]
    assert result["byte_exact"]
    assert result["passed"]


def test_compare_traces_reports_shape_mismatch():
    first = {
        "phase": np.asarray(["no_action"]),
        "x": np.zeros((1, 2)),
    }
    second = {
        "phase": np.asarray(["no_action", "preload"]),
        "x": np.zeros((2, 2)),
    }
    result = compare_traces(first, second, numeric_atol=1e-9)
    assert not result["shape_match"]
    assert not result["passed"]


def test_compare_traces_uses_numeric_tolerance():
    first = {
        "phase": np.asarray(["main_pull"]),
        "x": np.asarray([1.0]),
    }
    second = {
        "phase": np.asarray(["main_pull"]),
        "x": np.asarray([1.0 + 5e-10]),
    }
    result = compare_traces(first, second, numeric_atol=1e-9)
    assert not result["byte_exact"]
    assert result["passed"]


def test_compare_repeat_metadata_checks_hashes_lengths_and_events():
    template = {
        "base_state_hash": "base",
        "action_hash": "action",
        "free_trace_length": 10,
        "hidden_trace_length": 11,
        "free_events": [{"name": "end", "physics_step": 20}],
        "hidden_events": [{"name": "end", "physics_step": 21}],
        "action_hash_match": True,
        "free_base_state_hash_match": True,
        "hidden_base_state_hash_match": True,
        "free_hidden_initial_hash_match": True,
        "max_initial_state_difference": 0.0,
    }
    result = compare_repeat_metadata([dict(template), dict(template)])
    assert result["passed"]
