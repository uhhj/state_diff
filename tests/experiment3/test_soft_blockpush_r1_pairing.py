from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush.fixed_commands import (
    build_fixed_command_script, command_arrays, copy_fixed_command_script)
from scripts.experiment3.phase0_soft_blockpush.snapshot import (
    capture_explicit_state, max_state_difference)
from scripts.experiment3.phase0_soft_blockpush_r1.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment3/soft_blockpush_phase0b_r1.json"


def _short_config():
    config = load_config(str(CONFIG))
    config["execution"]["pre_snapshot_settle_steps"] = 2
    return config


def test_r1_snapshot_commands_telemetry_and_render():
    env = SoftBlockPushEnv(_short_config())
    try:
        base = capture_explicit_state(env)
        script = build_fixed_command_script(env, env.config, base)
        copied = copy_fixed_command_script(script)
        assert all(np.array_equal(command_arrays(script)[key], command_arrays(copied)[key])
                   for key in command_arrays(script))
        env.restore_saved_state(env.saved_state_id)
        assert max_state_difference(base, capture_explicit_state(env)) == 0
        env.step_one_physics(); row = env.trace()[-1]
        assert row["spring_force_evaluation_count"] == 512
        assert np.isfinite(row["spring_energy_total_j"])
        assert env.get_lowdim_state().shape == (98,)
        assert env.render().shape == (240, 320, 3)
    finally:
        env.close()


def test_same_seed_short_trace_is_numerically_identical():
    first, second = SoftBlockPushEnv(_short_config()), SoftBlockPushEnv(_short_config())
    try:
        for _ in range(2):
            first.step_one_physics(); second.step_one_physics()
        assert np.array_equal(first.soft_block.positions(), second.soft_block.positions())
        assert first.trace() == second.trace()
    finally:
        first.close(); second.close()
