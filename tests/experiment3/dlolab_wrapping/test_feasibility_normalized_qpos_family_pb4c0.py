import numpy as np
import pytest

import scripts.experiment3.dlolab_wrapping.feasibility_normalized_qpos_family_pb4c0 as pb4c0
from scripts.experiment3.dlolab_wrapping.feasibility_normalized_qpos_family_pb4c0 import (
    count_grid_violations,
    derive_alpha_max,
    exact_action_grid,
    load_pinned_panda_limits,
    symmetric_alpha_limit,
)


def test_symmetric_alpha_limit_centered():
    assert np.isclose(
        symmetric_alpha_limit(
            base=0.0,
            delta=0.5,
            lower=-1.0,
            upper=1.0,
        ),
        2.0,
    )


def test_symmetric_alpha_limit_uses_smaller_slack():
    # q=0.8 in [-1,1], |delta|=0.4:
    # upper slack=0.2 is limiting => alpha=0.5.
    assert np.isclose(
        symmetric_alpha_limit(
            base=0.8,
            delta=0.4,
            lower=-1.0,
            upper=1.0,
        ),
        0.5,
    )


def test_symmetric_alpha_limit_is_invariant_to_delta_sign():
    positive = symmetric_alpha_limit(
        base=0.4,
        delta=0.3,
        lower=-1.0,
        upper=1.0,
    )
    negative = symmetric_alpha_limit(
        base=0.4,
        delta=-0.3,
        lower=-1.0,
        upper=1.0,
    )
    assert np.isclose(positive, negative)


def test_zero_delta_is_not_a_constraint():
    assert np.isinf(
        symmetric_alpha_limit(
            base=0.5,
            delta=0.0,
            lower=-1.0,
            upper=1.0,
        )
    )


def test_base_outside_limits_is_rejected():
    with pytest.raises(ValueError, match="outside hard limit"):
        symmetric_alpha_limit(
            base=1.1,
            delta=0.1,
            lower=-1.0,
            upper=1.0,
        )


def _fake_limits(lower=-1.0, upper=1.0):
    return {
        "urdf_path": "/fake/panda.urdf",
        "joint_names": [row[0] for row in pb4c0.PANDA_QPOS_JOINTS],
        "lower": np.full(9, lower, dtype=np.float64),
        "upper": np.full(9, upper, dtype=np.float64),
        "metadata": [
            {
                "joint_name": name,
                "joint_type": joint_type,
                "unit": unit,
                "lower": lower,
                "upper": upper,
            }
            for name, joint_type, unit in pb4c0.PANDA_QPOS_JOINTS
        ],
    }


def _short_qpos():
    qpos = np.zeros((40, 18), dtype=np.float64)
    return qpos


def test_global_alpha_max_uses_most_constrained_branch_arm_joint():
    qpos = _short_qpos()

    # At t=10, arm1 joint1 base=0.8 and nominal at r=1 is 0.4.
    # delta=-0.4, smaller slack to upper=0.2 -> alpha_max=0.5.
    qpos[10, 0] = 0.8
    qpos[11:31, 0] = 0.4

    # Other joints have smaller deltas around center.
    qpos[11:31, 1:] = 0.1

    result = derive_alpha_max(
        qpos,
        times=[10],
        horizon=20,
        limits=_fake_limits(),
    )

    assert np.isclose(
        result["alpha_max_hard_limits"],
        0.5,
    )
    limiting = result["limiting_constraint"]
    assert limiting["time_index"] == 10
    assert limiting["arm_index"] == 1
    assert limiting["joint_index"] == 0


def test_formal_alpha_rule_is_deterministic():
    alpha_max = 0.5
    cap = 1.0
    safety = 0.95
    alpha_formal = safety * min(alpha_max, cap)
    assert np.isclose(alpha_formal, 0.475)


def test_exact_action_grid_is_symmetric_3x3():
    alpha = 0.475
    actions = exact_action_grid(alpha)

    assert len(actions) == 9
    masks = {
        (
            float(row["arm1_mask"]),
            float(row["arm2_mask"]),
        )
        for row in actions
    }
    assert masks == {
        (a, b)
        for a in (-alpha, 0.0, alpha)
        for b in (-alpha, 0.0, alpha)
    }


def test_unit_grid_can_be_infeasible_but_normalized_grid_feasible():
    qpos = _short_qpos()
    qpos[10, 0] = 0.8
    qpos[11:31, 0] = 0.4

    limits = _fake_limits()
    unit = count_grid_violations(
        qpos,
        times=[10],
        horizon=20,
        mask_values=[-1.0, 0.0, 1.0],
        limits=limits,
        tolerance=1e-12,
    )
    assert unit["out_of_limit_command_count"] > 0

    alpha = derive_alpha_max(
        qpos,
        times=[10],
        horizon=20,
        limits=limits,
    )["alpha_max_hard_limits"]
    formal_alpha = 0.95 * min(alpha, 1.0)

    formal = count_grid_violations(
        qpos,
        times=[10],
        horizon=20,
        mask_values=[-formal_alpha, 0.0, formal_alpha],
        limits=limits,
        tolerance=1e-12,
    )
    assert formal["out_of_limit_command_count"] == 0


def test_alpha_is_global_not_per_arm_or_per_pair():
    qpos = _short_qpos()

    # arm1 permits alpha=0.5
    qpos[10, 0] = 0.8
    qpos[11:31, 0] = 0.4

    # arm2 permits only alpha=0.25
    qpos[10, 9] = 0.9
    qpos[11:31, 9] = 0.5

    result = derive_alpha_max(
        qpos,
        times=[10],
        horizon=20,
        limits=_fake_limits(),
    )
    assert np.isclose(
        result["alpha_max_hard_limits"],
        0.25,
    )
    assert result["limiting_constraint"]["arm_index"] == 2


def test_pinned_panda_limits_match_wrapping_asset():
    limits = load_pinned_panda_limits()

    assert limits["joint_names"] == [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    assert np.allclose(
        limits["lower"],
        np.asarray(
            [
                -2.9671,
                -1.8326,
                -2.9671,
                -3.1416,
                -2.9671,
                -0.0873,
                -2.9671,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        ),
    )
    assert np.allclose(
        limits["upper"],
        np.asarray(
            [
                2.9671,
                1.8326,
                2.9671,
                0.0,
                2.9671,
                3.8223,
                2.9671,
                0.04,
                0.04,
            ],
            dtype=np.float64,
        ),
    )


def test_family_rule_forbids_adaptive_scales_and_clipping():
    config = {
        "classification": "CPU_ONLY_ACTION_FAMILY_DERIVATION",
        "family_rule": {
            "intervention_space": "per_arm_joint_position_target_qpos",
            "cartesian_motion_reversal_claim": False,
            "primary_horizon_microsteps": 20,
            "arm_dof_block": 9,
            "unit_mask_template": [-1.0, 0.0, 1.0],
            "symmetric_scale_cap": 1.0,
            "feasibility_safety_factor": 0.95,
            "position_limit_tolerance_native_units": 1e-12,
            "allow_clipping": False,
            "allow_per_joint_scale": False,
            "allow_per_arm_scale": False,
            "allow_per_pair_scale": False,
            "allow_horizon_change": False,
            "uses_pb4_reward_or_regret": False,
            "uses_pb3_future_divergence": False,
            "uses_winding_labels": False,
            "require_unit_scale_infeasible": True,
            "require_formal_grid_zero_limit_violations": True,
        },
        "next_phase_boundary": {
            "gpu_allowed_in_this_phase": False,
            "condition5_confirmation_allowed": False,
        },
    }

    pb4c0.validate_static_rule(config)

    changed = {
        **config,
        "family_rule": {
            **config["family_rule"],
            "allow_per_joint_scale": True,
        },
    }
    with pytest.raises(RuntimeError, match="allow_per_joint_scale"):
        pb4c0.validate_static_rule(changed)
