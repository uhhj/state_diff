from scripts.experiment2.phase0.run_hidden_friction_pairs import TRACE_KEYS


def test_sensor_fields_present():
    required = {
        "sensor_joint_motor_torque",
        "sensor_joint_motor_torque_norm",
        "sensor_suction_force_xyz",
        "sensor_suction_force_norm",
        "sensor_suction_torque_xyz",
        "sensor_suction_torque_norm",
        "sensor_grasp_active",
        "sensor_constraint_available",
    }
    assert required.issubset(set(TRACE_KEYS))


def test_oracle_and_sensor_are_separate():
    assert "contact_force_norm" in TRACE_KEYS
    assert "sensor_suction_force_norm" in TRACE_KEYS
