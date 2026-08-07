"""Minimal config, path, trace, and JSON helpers for OHJ Phase 0D."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CONDITIONS = ("free", "jam_right")


def load_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config):
    if config.get("task_name") != "ccda-ohj-cable-phase0":
        raise ValueError("unexpected OHJ task")
    if tuple(config.get("conditions", ())) != CONDITIONS:
        raise ValueError("Phase 0D requires FREE and JAM-R only")
    if config["state"]["state_dim"] != 51:
        raise ValueError("OHJ State Diff state must be 51D")
    trace_field = config["sensor"].get("trace_field", "formal_wrench")
    expected_sensor_dims = {
        "formal_wrench": 6,
        "formal_sensor": 9,
        "formal_sensor_spatial": 18,
    }
    if trace_field not in expected_sensor_dims:
        raise ValueError("unsupported OHJ sensor trace field")
    if config["sensor"]["sensor_dim"] != expected_sensor_dims[trace_field]:
        raise ValueError("OHJ sensor dimension does not match trace field")
    if not config["execution"]["deterministic"]:
        raise ValueError("Phase 0D must be deterministic")
    if config["execution"]["hz"] != config["sensor"]["sampling_hz"]:
        raise ValueError("physics and sensor rates must match")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False, default=_json_default) + "\n", encoding="utf-8")


def trace_to_arrays(trace):
    if not trace:
        raise ValueError("trace is empty")
    fields = (
        "physics_step", "phase", "statediff_state", "visible_keypoints",
        "all_bead_positions", "ee_position", "formal_wrench",
        "gripper_surface_tactile_force", "gripper_surface_contact_count",
        "formal_sensor", "gripper_surface_tactile_patch_force",
        "gripper_surface_tactile_patch_contact_count",
        "formal_sensor_spatial",
        "joint_motor_torque", "joint_reaction_wrench",
        "extraction_progress_m", "oracle_latch_contact_force",
        "oracle_latch_contact_count")
    return {name: np.asarray([row[name] for row in trace]) for name in fields}


def pair_dir(config):
    return Path(config["output_root"]) / config["pair_id"]


def report_dir(config):
    return Path(config["report_root"]) / config["pair_id"]
