import numpy as np
import pytest

from scripts.experiment2.phase0.exact_counterfactual import (
    array_payload_sha256,
    max_state_difference,
)


def test_array_payload_hash_is_deterministic_and_shape_sensitive():
    first = {"x": np.arange(6, dtype=np.float64).reshape(2, 3)}
    second = {"x": np.arange(6, dtype=np.float64).reshape(2, 3)}
    reshaped = {"x": np.arange(6, dtype=np.float64).reshape(3, 2)}
    assert array_payload_sha256(first) == array_payload_sha256(second)
    assert array_payload_sha256(first) != array_payload_sha256(
        reshaped
    )


def test_max_state_difference():
    first = {
        "x": np.array([1.0, 2.0]),
        "y": np.array([[3.0]]),
    }
    second = {
        "x": np.array([1.0, 2.25]),
        "y": np.array([[3.0]]),
    }
    assert max_state_difference(first, second) == pytest.approx(
        0.25
    )


def test_max_state_difference_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        max_state_difference(
            {"x": np.zeros((2, 2))},
            {"x": np.zeros((4,))},
        )


def test_array_payload_hash_changes_when_phase_length_changes():
    first = {
        "phase": np.asarray(["no_action", "preload"]),
        "x": np.asarray([1.0, 2.0]),
    }
    second = {
        "phase": np.asarray(["no_action"]),
        "x": np.asarray([1.0]),
    }
    assert array_payload_sha256(first) != array_payload_sha256(second)
