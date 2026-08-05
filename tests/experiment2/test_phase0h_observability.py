import inspect
import json
from pathlib import Path

import numpy as np

from ravens.environment import Environment
from ravens.tasks.ccda_hidden_friction_cable import CCDAHiddenFrictionCable
from scripts.experiment2.phase0.run_hidden_friction_pairs import TRACE_KEYS
from scripts.experiment2.phase0.run_hidden_hook_observability import probe_phase_index


ROOT = Path(__file__).resolve().parents[2]


def test_phase0h_is_one_fixed_topology_and_three_required_seeds():
    config = json.loads(
        (ROOT / "configs/experiment2/phase0/hidden_hook_phase0h.json").read_text()
    )
    assert config["topology_policy"] == "single_fixed_topology_no_grid_search"
    assert config["seeds"] == [71001, 71002, 71003]
    assert "candidates" not in config
    assert config["hook"]["min_initial_clearance"] >= 0.002


def test_formal_trace_includes_joint_reaction_and_constraint_signals():
    required = {
        "sensor_joint_reaction_force_torque",
        "sensor_joint_reaction_force_torque_norm",
        "sensor_suction_force_xyz",
        "sensor_suction_torque_xyz",
        "sensor_grasp_active",
        "sensor_constraint_available",
    }
    assert required.issubset(TRACE_KEYS)
    source = inspect.getsource(Environment.ccda_sensor_observation)
    for forbidden in ("hidden_condition", "hook_body_ids", "hook_layout"):
        assert forbidden not in source
    assert "enableJointForceTorqueSensor" in inspect.getsource(
        CCDAHiddenFrictionCable.reset
    )


def test_probe_phase_index_uses_motion_step_intervals():
    steps = np.asarray([10, 20, 30, 40, 50])
    events = [
        {"stage": "hook_probe_out", "physics_step_start": 18, "physics_step_end": 31},
        {"stage": "hook_probe_return", "physics_step_start": 39, "physics_step_end": 45},
    ]
    assert probe_phase_index(steps, events).tolist() == [0, 2, 2, 3, 0]


def test_precise_motion_debug_keeps_timeout_but_uses_cartesian_endpoint():
    source = inspect.getsource(Environment.movep_precise)
    for required in (
        "physics_step_count", "timeout_reason", "final_joint_error",
        "cartesian_endpoint_error", "joint_timeout_recovered_by_cartesian_endpoint",
    ):
        assert required in source
    assert "'success': endpoint_reached" in source
