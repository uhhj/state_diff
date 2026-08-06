import inspect
import json
import types

import numpy as np

from scripts.experiment3.phase0.occp_commands import (
    build_fixed_command_script, command_arrays, copy_fixed_command_script,
    execute_fixed_command_phase)
from scripts.experiment3.phase0.run_occp_single_pair import run_branch


class FakeEnv:
    ur5 = 1
    ee_tip_link = 12
    joints = list(range(6))

    def __init__(self):
        self.calls = 0
        self.steps = 0
        self.ee = types.SimpleNamespace(activated=True, contact_constraint=7)

    def solve_IK(self, pose):
        self.calls += 1
        return np.full(6, 2 * np.pi * self.calls + pose[0])

    def step_physics(self, count):
        self.steps += count


def _script(monkeypatch):
    monkeypatch.setattr(
        'scripts.experiment3.phase0.occp_commands.p.getJointState',
        lambda *args: (0., 0., (), 0.))
    env = FakeEnv()
    layout = {'probe_delta': np.array([.01, 0., 0.]),
              'test_delta': np.array([.03, .01, 0.])}
    return env, build_fixed_command_script(
        env, start_ee_position=[.5, 0., .1], orientation=[0., 0., 0., 1.],
        layout=layout, probe_command_steps=4, test_command_steps=6,
        control_substeps=1)


def test_fixed_command_script_is_byte_equal_after_copy(monkeypatch):
    _, script = _script(monkeypatch)
    copied = copy_fixed_command_script(script)
    assert copied is not script
    assert json.dumps(copied, sort_keys=True) == json.dumps(script, sort_keys=True)
    assert all(np.array_equal(command_arrays(script)[key],
                              command_arrays(copied)[key])
               for key in command_arrays(script))


def test_probe_and_test_command_counts_are_fixed(monkeypatch):
    _, script = _script(monkeypatch)
    assert len(script['phases'][0]['joint_targets']) == 4
    assert len(script['phases'][1]['joint_targets']) == 6


def test_test_phase_starts_from_probe_target(monkeypatch):
    _, script = _script(monkeypatch)
    probe_end = np.asarray(script['phases'][0]['ee_targets'][-1])
    first_test = np.asarray(script['phases'][1]['ee_targets'][0])
    expected = probe_end + np.array([.03, .01, 0.]) / 6
    assert np.allclose(first_test, expected)


def test_joint_targets_are_unwrapped(monkeypatch):
    _, script = _script(monkeypatch)
    targets = np.concatenate([
        np.asarray(phase['joint_targets']) for phase in script['phases']], axis=0)
    assert np.max(np.abs(np.diff(targets, axis=0))) < np.pi


def test_branch_execution_uses_no_movep():
    source = inspect.getsource(run_branch)
    assert '.movep(' not in source
    assert '.movej(' not in source


def test_expected_and_actual_command_physics_steps_match(monkeypatch):
    env, script = _script(monkeypatch)
    monkeypatch.setattr(
        'scripts.experiment3.phase0.occp_commands.p.setJointMotorControlArray',
        lambda **kwargs: None)
    monkeypatch.setattr(
        'scripts.experiment3.phase0.occp_commands.p.getLinkState',
        lambda *args, **kwargs: (script['phases'][0]['ee_targets'][-1],))
    task = types.SimpleNamespace(
        physics_step_count=lambda: env.steps,
        set_ccda_phase=lambda value: None,
        set_ee_target_position=lambda value: None)
    record = execute_fixed_command_phase(
        env, task, script['phases'][0], position_gains=1.0)
    assert record['expected_physics_steps'] == record['actual_physics_steps'] == 4
