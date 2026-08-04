from __future__ import annotations

import unittest

import numpy as np

from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
    compute_pair_metrics,
    no_action_drift,
    ordered_bead_distance,
    phase_slice,
)


def synthetic_trace(offset=0.0):
    phases = np.array(["no_action", "no_action", "preload", "preload", "main_pull", "main_pull", "post_main"])
    beads = np.zeros((phases.size, 2, 3), dtype=np.float64)
    beads[:, :, 0] = np.arange(phases.size)[:, None] * 0.01
    beads[4:, :, 1] += offset
    return {
        "physics_step": np.arange(phases.size),
        "phase": phases,
        "bead_positions": beads,
        "contact_force_norm": np.array([0, 0, 0.1, 0.2, 0, 0, 0], dtype=np.float64),
    }


class Phase0MetricTest(unittest.TestCase):
    def test_canonical_hash_ignores_key_order(self):
        self.assertEqual(canonical_json_sha256({"a": 1, "b": 2}), canonical_json_sha256({"b": 2, "a": 1}))

    def test_action_change_changes_hash(self):
        self.assertNotEqual(canonical_json_sha256({"distance": 1}), canonical_json_sha256({"distance": 2}))

    def test_phase_slice(self):
        sliced = phase_slice(synthetic_trace(), "preload")
        self.assertEqual(sliced["bead_positions"].shape[0], 2)
        self.assertTrue(np.all(sliced["phase"] == "preload"))

    def test_ordered_distance_zero(self):
        beads = np.zeros((3, 2, 3))
        np.testing.assert_array_equal(ordered_bead_distance(beads, beads), np.zeros(3))

    def test_known_translation_ade_fde(self):
        free = synthetic_trace(0.0)
        hidden = synthetic_trace(0.02)
        meta = {"action_hash": "same", "privileged_state": {"arm_max_abs_jump": 0.0}}
        metrics = compute_pair_metrics(free, hidden, meta, meta, hz=10.0, trace_stride=1)
        self.assertAlmostEqual(metrics["main_branch_ade"], 0.02)
        self.assertAlmostEqual(metrics["main_branch_fde"], 0.02)

    def test_no_action_drift(self):
        beads = np.zeros((2, 1, 3))
        beads[1, 0, 0] = 0.03
        self.assertAlmostEqual(no_action_drift(beads), 0.03)

    def test_action_hash_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compute_pair_metrics(
                synthetic_trace(),
                synthetic_trace(),
                {"action_hash": "a"},
                {"action_hash": "b"},
                hz=480,
                trace_stride=4,
            )


if __name__ == "__main__":
    unittest.main()
