"""Analyze causal outer-rate sensor observability for Phase 0C-R1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0c_hidden_dynamics.common import (  # noqa: E402
    external_pair_dir, load_config, report_pair_dir)
from scripts.experiment3.phase0c_hidden_dynamics.io_utils import write_json  # noqa: E402
from state_diff.env.block_pushing.policy_rate_metrics import (  # noqa: E402
    formal_feature_scale_floors)


def rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b), axis=-1))


def first_sustained(values, mask, threshold, consecutive, steps):
    hit = np.asarray(values) >= float(threshold)
    mask = np.asarray(mask, dtype=bool)
    for start in range(len(hit) - consecutive + 1):
        stop = start + consecutive
        if np.all(mask[start:stop]) and np.all(hit[start:stop]):
            return int(steps[start])
    return None


def sensor_group(index):
    if index < 6:
        return "joint_motor_torque"
    if index < 42:
        component = (index - 6) % 6
        return ("joint_reaction_force" if component < 3
                else "joint_reaction_torque")
    return "ee_tracking_error"


def _load(directory: Path) -> tuple[dict, dict]:
    with np.load(directory / "trajectory.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads((directory / "metadata.json").read_text(
        encoding="utf-8"))
    return arrays, metadata


def evaluate_highrate_arrays(low: dict, high: dict, repeat: dict, config: dict,
                             metadata=None) -> dict:
    """Evaluate aligned measured or synthetic outer-rate trajectories."""
    metadata = metadata or {}
    analysis = config["analysis"]
    command_keys = ("joint_target", "ee_target", "command_action", "command_phase")
    commands = {key: (np.array_equal(low[key], high[key])
                      and np.array_equal(low[key], repeat[key]))
                for key in command_keys}
    initial_difference = float(max(
        np.max(np.abs(low["state"][0] - high["state"][0])),
        np.max(np.abs(low["state"][0] - repeat["state"][0]))))
    phase = low["phase_hr"].astype(str)
    no_action = phase == "no_action"
    future = ~no_action
    if not np.any(no_action):
        raise ValueError("no no_action outer sample")
    steps = low["outer_step_hr"].astype(np.int64)
    if int(steps[0]) != 1:
        raise ValueError("high-rate sampling must start after outer step 1")

    numeric_keys = (
        "state", "goal_distance", "state_hr", "contact_sensor_hr",
        "robot_proprio_extended_hr", "goal_distance_hr",
        "spring_cap_count_hr", "spring_evaluation_count_hr")
    finite = all(np.all(np.isfinite(branch[key]))
                 for branch in (low, high, repeat) for key in numeric_keys)
    end = int(np.flatnonzero(no_action)[-1])
    drifts = {
        name: float(rmse(branch["state_hr"][end], branch["state"][0]))
        for name, branch in (("uniform_low", low),
                             ("right_local_high", high))}
    cap_fraction = {
        name: float(np.sum(branch["spring_cap_count_hr"])
                    / max(np.sum(branch["spring_evaluation_count_hr"]), 1))
        for name, branch in (("uniform_low", low),
                             ("right_local_high", high),
                             ("uniform_low_repeat", repeat))}

    state_rmse = rmse(low["state_hr"], high["state_hr"])
    deformable_rmse = rmse(low["state_hr"][:, :72], high["state_hr"][:, :72])
    ee_xy_rmse = rmse(low["state_hr"][:, 72:74], high["state_hr"][:, 72:74])
    repeat_rmse = rmse(low["state_hr"], repeat["state_hr"])
    repeat_peak = float(np.max(repeat_rmse))
    baseline_displacements = np.concatenate([
        (branch["state_hr"][no_action] - branch["state"][0]).reshape(-1)
        for branch in (low, high, repeat)])
    baseline_state_std = float(np.std(baseline_displacements))
    state_threshold = max(
        float(analysis["state_sigma_multiplier"]) * baseline_state_std,
        float(analysis["state_divergence_floor_m"]),
        float(analysis["future_effect_vs_repeat_multiplier"]) * repeat_peak)
    state_onset = first_sustained(
        state_rmse, future, state_threshold,
        int(analysis["state_consecutive_outer_samples"]), steps)

    low_sensor = low["contact_sensor_hr"]
    high_sensor = high["contact_sensor_hr"]
    pooled = np.concatenate([low_sensor[no_action], high_sensor[no_action]], axis=0)
    scales = np.maximum(
        float(analysis["sensor_sigma_multiplier"]) * np.std(pooled, axis=0),
        formal_feature_scale_floors(config))
    normalized = np.abs(low_sensor - high_sensor) / scales
    fused = np.max(normalized, axis=1)
    sensor_onset = first_sustained(
        fused, future, 1.0,
        int(analysis["sensor_consecutive_outer_samples"]), steps)
    trigger = None
    if sensor_onset is not None:
        row = int(np.flatnonzero(steps == sensor_onset)[0])
        channel = int(np.argmax(normalized[row]))
        trigger = {"channel_index": channel,
                   "channel_group": sensor_group(channel),
                   "normalized_gap": float(normalized[row, channel])}

    low_mu = metadata.get("low_mu", .2)
    high_mu = metadata.get("high_mu", 1.6)
    oracle_load_gap = float(np.max(np.abs(
        low["oracle_patch_tangential_force_hr"]
        - high["oracle_patch_tangential_force_hr"])))
    oracle_slip_gap = float(np.max(np.abs(
        low["oracle_patch_mean_slip_speed_hr"]
        - high["oracle_patch_mean_slip_speed_hr"])))
    oracle_intervention = bool(
        low_mu != high_mu and (oracle_load_gap > 1e-12 or oracle_slip_gap > 1e-12))
    engineering_gate = {
        "finite": bool(finite),
        "initial_state": initial_difference <= analysis["initial_state_max_abs_m"],
        "joint_commands_equal": commands["joint_target"],
        "ee_targets_equal": commands["ee_target"],
        "actions_equal": commands["command_action"],
        "phases_equal": commands["command_phase"],
        "no_action_stable": max(drifts.values()) <=
                            analysis["no_action_state_drift_max_m"],
        "spring_force_cap": max(cap_fraction.values()) <=
                            analysis["spring_force_cap_fraction_max"],
        "oracle_intervention": oracle_intervention,
    }
    if not all(engineering_gate.values()):
        verdict = "PHASE0C_R1_ENGINEERING_BLOCKED"
    elif state_onset is None:
        verdict = "PHASE0C_R1_FUTURE_DYNAMICS_LOST"
    elif sensor_onset is None:
        verdict = "PHASE0C_R1_SENSOR_NOT_SEPARABLE"
    elif sensor_onset > state_onset:
        verdict = "PHASE0C_R1_SENSOR_STILL_LATE"
    else:
        verdict = "PHASE0C_R1_SENSOR_OBSERVABILITY_COMPLETE"

    dt_ms = float(config["physics"]["outer_timestep_s"]) * 1000.0
    lead = (None if state_onset is None or sensor_onset is None
            else state_onset - sensor_onset)
    return {
        "verdict": verdict,
        "engineering_gate": engineering_gate,
        "initial_state_max_abs_m": initial_difference,
        "command_arrays_equal": commands,
        "no_action_state_drift_m": drifts,
        "spring_force_cap_fraction": cap_fraction,
        "peak_joint_state_rmse_m": float(np.max(state_rmse[future])),
        "peak_deformable_rmse_m": float(np.max(deformable_rmse[future])),
        "peak_ee_xy_rmse_m": float(np.max(ee_xy_rmse[future])),
        "repeat_peak_state_rmse_m": repeat_peak,
        "baseline_state_std_m": baseline_state_std,
        "state_threshold_m": state_threshold,
        "state_onset_outer_step": state_onset,
        "sensor_onset_outer_step": sensor_onset,
        "state_onset_ms": None if state_onset is None else state_onset * dt_ms,
        "sensor_onset_ms": None if sensor_onset is None else sensor_onset * dt_ms,
        "sensor_lead_outer_steps": lead,
        "sensor_lead_ms": None if lead is None else lead * dt_ms,
        "sensor_trigger": trigger,
        "peak_fused_sensor_gap": float(np.max(fused[future])),
        "oracle": {"low_patch_mu": low_mu, "high_patch_mu": high_mu,
                   "intervention": oracle_intervention,
                   "peak_tangential_load_gap_n": oracle_load_gap,
                   "peak_mean_slip_speed_gap_mps": oracle_slip_gap},
        "sampling": {"first_outer_step": int(steps[0]),
                     "formal_pre_step_sensor": False,
                     "outer_samples_per_policy": int(
                         config["sensor"]["outer_samples_per_policy"])},
    }


def analyze(config_path: str) -> dict:
    config = load_config(config_path)
    low, low_meta = _load(external_pair_dir(config, "uniform_low"))
    high, high_meta = _load(external_pair_dir(config, "right_local_high"))
    repeat, _ = _load(external_pair_dir(config, "uniform_low_repeat"))
    metrics = evaluate_highrate_arrays(low, high, repeat, config, {
        "low_mu": low_meta["arm"]["patch_lateral_friction"],
        "high_mu": high_meta["arm"]["patch_lateral_friction"]})
    report = report_pair_dir(config)
    write_json(report / "highrate_pair_metrics.json", metrics)
    (report / "highrate_pair_summary.md").write_text(
        "# Phase 0C-R1 high-rate pair\n\n"
        "- Verdict: `{}`\n- Sensor onset: `{}` outer steps\n"
        "- State onset: `{}` outer steps\n".format(
            metrics["verdict"], metrics["sensor_onset_outer_step"],
            metrics["state_onset_outer_step"]), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
