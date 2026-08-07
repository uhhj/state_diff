import numpy as np
import pytest

from scripts.experiment3.phase0c_hidden_dynamics.highrate_sampling import (
    build_policy_sensor_windows)


def test_sensor_windows_are_exact_causal_outer_slices():
    sensors = np.arange(48 * 45, dtype=np.float64).reshape(48, 45)
    phases = np.asarray(["probe"] * 24 + ["post_probe"] * 24)
    windows, labels = build_policy_sensor_windows(sensors, phases, 24)
    assert windows.shape == (2, 24, 45)
    np.testing.assert_array_equal(windows[0], sensors[:24])
    np.testing.assert_array_equal(windows[1], sensors[24:])
    np.testing.assert_array_equal(labels, ["probe", "post_probe"])


def test_sensor_window_rejects_phase_crossing():
    with pytest.raises(ValueError, match="crosses phase boundary"):
        build_policy_sensor_windows(
            np.zeros((24, 45)), np.asarray(["a"] * 12 + ["b"] * 12), 24)
