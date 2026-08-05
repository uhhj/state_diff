import inspect

from ravens.environment import Environment
from scripts.experiment2.phase0.run_hidden_friction_pairs import TRACE_KEYS


def test_oracle_and_formal_sensor_fields_are_both_present_and_separate():
    oracle = {"contact_force_norm", "contact_active_beads"}
    sensor = {
        "sensor_joint_motor_torque",
        "sensor_joint_motor_torque_norm",
        "sensor_suction_force_xyz",
        "sensor_suction_force_norm",
        "sensor_suction_torque_xyz",
        "sensor_suction_torque_norm",
        "sensor_grasp_active",
        "sensor_constraint_available",
    }
    assert oracle.issubset(TRACE_KEYS)
    assert sensor.issubset(TRACE_KEYS)
    assert oracle.isdisjoint(sensor)


def test_formal_sensor_does_not_read_hidden_hook_state():
    source = inspect.getsource(Environment.ccda_sensor_observation)
    for forbidden in ("hook_body_ids", "hook_layout", "hidden_condition"):
        assert forbidden not in source
