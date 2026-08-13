import numpy as np

from scripts.experiment3.dlolab_wrapping.preregister_alignment_pb3r3 import (
    coordinate_rmse_m,
    derive_state_level_envelope,
    live_pair_revalidation,
    targeted_replay_alignment,
)


def _rows():
    rows = []
    for state_index in range(20):
        cid = f"s{state_index:02d}"
        if state_index == 19:
            values = [50e-6, 60e-6, 70e-6]
        elif state_index == 18:
            values = [1e-6, 1e-6, 500e-6]
        else:
            base = (state_index + 1) * 1e-6
            values = [base, base, base]
        for repeat, value in enumerate(values):
            rows.append(
                {
                    "calibration_id": cid,
                    "rollout_id": state_index,
                    "batch_index": state_index // 5,
                    "time_index": 13 if state_index % 2 == 0 else 20,
                    "repeat_index": repeat,
                    "coordinate_rmse_m": value,
                }
            )
    return rows


def test_state_level_envelope_uses_median_then_max():
    state_rows, governing, threshold = derive_state_level_envelope(_rows())

    assert len(state_rows) == 20
    assert governing["calibration_id"] == "s19"
    assert np.isclose(threshold, 60e-6)

    # The single 500 um outlier in s18 must not become the threshold because
    # each independent state contributes one median, not three pseudo-repeats.
    assert threshold < 500e-6


def test_coordinate_rmse_is_coordinatewise_not_vertex_norm_rmse():
    live = np.zeros((50, 3), dtype=np.float64)
    frozen = np.zeros_like(live)
    live[0, 0] = 300e-6

    expected = 300e-6 / np.sqrt(150.0)
    assert np.isclose(
        coordinate_rmse_m(live, frozen),
        expected,
    )


def _alignment_rule(threshold=30e-6):
    return {
        "targeted_replay_engineering_alignment": {
            "rope": {
                "threshold_m": threshold,
            },
            "ee_max_abs_m": 50e-6,
            "motor_qpos_max_abs_rad": 50e-6,
            "require_live_winding_index_equal_frozen": True,
        }
    }


def _branch_state():
    return {
        "rope_xyz": np.zeros((50, 3), dtype=np.float64),
        "ee1_pos": np.zeros(3, dtype=np.float64),
        "ee2_pos": np.zeros(3, dtype=np.float64),
        "motor_qpos_1": np.zeros(7, dtype=np.float64),
        "motor_qpos_2": np.zeros(7, dtype=np.float64),
        "signed_winding_turns": np.zeros(3, dtype=np.float64),
    }


def test_targeted_alignment_gates_on_rmse_not_rope_max_abs():
    frozen = _branch_state()
    live = _branch_state()
    live["rope_xyz"][0, 0] = 300e-6

    row = targeted_replay_alignment(
        live,
        frozen,
        _alignment_rule(threshold=30e-6),
    )

    assert row["rope_max_abs_coordinate_m_diagnostic"] > 50e-6
    assert row["rope_coordinate_rmse_m"] < 30e-6
    assert row["valid"] is True


def test_targeted_alignment_retains_robot_and_winding_requirements():
    frozen = _branch_state()
    live = _branch_state()
    live["ee1_pos"][0] = 60e-6

    row = targeted_replay_alignment(
        live,
        frozen,
        _alignment_rule(threshold=30e-6),
    )
    assert row["valid"] is False


def _pair_rule():
    return {
        "live_pair_revalidation": {
            "history_samples": 3,
            "partial_rope": {
                "occlusion_radius_m": 0.05,
                "max_visible_history_chamfer_m": 0.01,
                "require_nonempty_visible_rope_each_frame": True,
            },
            "robot": {
                "max_dual_ee_position_history_mean_m": 0.01,
                "max_dual_ee_quaternion_geodesic_history_mean_rad":
                    0.08726646259971647,
                "max_dual_motor_qpos_history_rms_rad": 0.05,
            },
        }
    }


def _history(winding):
    rope = np.stack(
        [
            np.linspace(0.0, 0.2, 50),
            np.zeros(50),
            np.zeros(50),
        ],
        axis=1,
    )
    posts = np.array(
        [
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
            [12.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    result = []
    for _ in range(3):
        result.append(
            {
                "rope_xyz": rope.copy(),
                "post_xyz": posts.copy(),
                "ee1_pos": np.zeros(3),
                "ee1_quat": quat.copy(),
                "ee2_pos": np.zeros(3),
                "ee2_quat": quat.copy(),
                "motor_qpos_1": np.zeros(7),
                "motor_qpos_2": np.zeros(7),
                "signed_winding_turns": np.asarray(winding, dtype=np.float64),
            }
        )
    return result


def test_live_pair_revalidation_mirrors_pb2c_semantics():
    a = _history([0.0, 0.0, 0.0])
    b = _history([0.0, 1.0, 0.0])

    row = live_pair_revalidation(
        a,
        b,
        _pair_rule(),
        same_time=True,
        common_action_history_equal=True,
    )

    assert row["valid"] is True
    assert row["live_winding_index_different"] is True
    assert row["visible_rope_history_chamfer_m"] == 0.0


def test_live_pair_revalidation_requires_hidden_index_difference():
    a = _history([0.0, 0.0, 0.0])
    b = _history([0.0, 0.0, 0.0])

    row = live_pair_revalidation(
        a,
        b,
        _pair_rule(),
        same_time=True,
        common_action_history_equal=True,
    )

    assert row["valid"] is False


def test_live_pair_revalidation_requires_common_action_history():
    a = _history([0.0, 0.0, 0.0])
    b = _history([0.0, 1.0, 0.0])

    row = live_pair_revalidation(
        a,
        b,
        _pair_rule(),
        same_time=True,
        common_action_history_equal=False,
    )

    assert row["valid"] is False
