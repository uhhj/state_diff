import copy
from pathlib import Path

import numpy as np

from scripts.experiment3.phase0_soft_blockpush.analyze_pair import classify_verdict
from scripts.experiment3.phase0_soft_blockpush.common import load_config
from scripts.experiment3.phase0_soft_blockpush.fixed_commands import (
    build_fixed_command_script, command_arrays, copy_fixed_command_script)
from scripts.experiment3.phase0_soft_blockpush.run_single_pair import run_pair
from scripts.experiment3.phase0_soft_blockpush.snapshot import (
    capture_explicit_state, max_state_difference)
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv


CONFIG = Path(__file__).resolve().parents[2] / "configs/experiment3/soft_blockpush_phase0b.json"


def test_snapshot_and_fixed_commands_are_branch_independent():
    config = load_config(str(CONFIG)); config["execution"]["pre_snapshot_settle_steps"] = 2
    env = SoftBlockPushEnv(config)
    try:
        base = capture_explicit_state(env)
        script = build_fixed_command_script(env, config, base)
        copied = copy_fixed_command_script(script)
        assert all(np.array_equal(command_arrays(script)[key], command_arrays(copied)[key])
                   for key in command_arrays(script))
        env.restore_saved_state(env.saved_state_id)
        free = capture_explicit_state(env); env.arm_condition("uniform_low")
        env.restore_saved_state(env.saved_state_id)
        high = capture_explicit_state(env); env.arm_condition("right_local_high")
        assert max_state_difference(base, free) == 0
        assert max_state_difference(free, high) == 0
    finally:
        env.close()


def test_probe_only_pair_has_equal_steps_and_does_not_execute_test(tmp_path):
    config = load_config(str(CONFIG))
    for key in config["execution"]:
        config["execution"][key] = 2
    config["camera"]["frame_stride"] = 1000
    config["output_root"] = str(tmp_path)
    _, metadata = run_pair(config, stop_after_probe=True)
    assert metadata["fixed_command_arrays_equal"]
    assert metadata["physics_step_arrays_equal"] and metadata["phase_arrays_equal"]
    assert all(not value["test_executed"] for value in metadata["branches"].values())


def test_mismatched_arrays_can_trigger_engineering_block():
    assert classify_verdict(False, True, True, False)[0] == "PHASE0B_ENGINEERING_BLOCKED"
