import numpy as np

from scripts.experiment3.phase0c_hidden_dynamics.commands import (
    build_probe_test_plan)
from scripts.experiment3.phase0c_hidden_dynamics.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv

CONFIG = "configs/experiment3/hlf_sbp_phase0c_dynamics.json"


def test_snapshot_and_fixed_commands_are_branch_independent():
    config = load_config(CONFIG)
    config["execution"]["pre_snapshot_settle_outer_steps"] = 2
    env = SoftBlockPushEnv(config)
    try:
        plan = build_probe_test_plan(env, config)
        base = env.capture_explicit_state()
        env.restore_saved_state(env.saved_state_id); env.arm_condition("uniform_low")
        low = env.capture_explicit_state()
        env.restore_saved_state(env.saved_state_id); env.arm_condition("right_local_high")
        high = env.capture_explicit_state()
        for key in base:
            assert np.array_equal(base[key], low[key])
            assert np.array_equal(low[key], high[key])
        assert np.array_equal(plan.joint_target, plan.joint_target.copy())
        assert len(plan.phase) == 912 and len(plan.action_xy) == 38
    finally:
        env.close()
