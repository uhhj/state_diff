"""Analyze formal-sensor and future-state evidence for the strict pair."""
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


def _load(directory: Path) -> tuple[dict, dict]:
    with np.load(directory / "trajectory.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return arrays, metadata


def _rmse(left, right):
    return np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)), axis=-1))


def _first_sustained(values, mask, threshold, consecutive, policy_steps):
    hit = np.asarray(values) >= float(threshold)
    allowed = np.asarray(mask, dtype=bool)
    for index in range(len(hit) - consecutive + 1):
        if np.all(allowed[index:index + consecutive] & hit[index:index + consecutive]):
            return int(policy_steps[index])
    return None


def evaluate_pair_arrays(low: dict, high: dict, repeat: dict, config: dict,
                         metadata=None) -> dict:
    """Evaluate one aligned synthetic or measured Phase 0C triplet."""
    metadata = metadata or {}
    command_keys = ("joint_target", "ee_target", "command_action", "command_phase")
    commands = {key: (np.array_equal(low[key], high[key])
                      and np.array_equal(low[key], repeat[key]))
                for key in command_keys}
    initial_difference = float(max(
        np.max(np.abs(low["state"][0] - high["state"][0])),
        np.max(np.abs(low["state"][0] - repeat["state"][0]))))
    no_action = low["phase"].astype(str) == "no_action"
    future = ~np.isin(low["phase"].astype(str), ["initial", "no_action"])
    no_action_indices = np.flatnonzero(no_action)
    if len(no_action_indices) == 0:
        raise ValueError("no no_action policy sample")
    end = int(no_action_indices[-1])
    drifts = {
        "uniform_low": float(_rmse(low["state"][end], low["state"][0])),
        "right_local_high": float(_rmse(high["state"][end], high["state"][0]))}
    numeric_keys = ("state", "contact_sensor", "robot_proprio_extended",
                    "goal_distance", "spring_cap_count", "spring_evaluation_count")
    finite = all(np.all(np.isfinite(branch[key]))
                 for branch in (low, high, repeat) for key in numeric_keys)
    cap_fraction = {name: float(np.sum(branch["spring_cap_count"])
                         / max(np.sum(branch["spring_evaluation_count"]), 1))
                    for name, branch in (("uniform_low", low),
                                         ("right_local_high", high),
                                         ("uniform_low_repeat", repeat))}
    repeat_state_rmse = _rmse(low["state"], repeat["state"])
    repeat_sensor_difference = _rmse(
        low["contact_sensor"], repeat["contact_sensor"])
    repeat_peak = float(np.max(repeat_state_rmse))
    baseline_displacements = np.concatenate([
        (branch["state"][no_action] - branch["state"][0]).reshape(-1)
        for branch in (low, high, repeat)])
    baseline_state_std = float(np.std(baseline_displacements))
    state_rmse = _rmse(low["state"], high["state"])
    analysis = config["analysis"]
    state_threshold = max(
        float(analysis["state_sigma_multiplier"]) * baseline_state_std,
        float(analysis["state_divergence_floor_m"]),
        float(analysis["future_effect_vs_repeat_multiplier"]) * repeat_peak)
    state_onset = _first_sustained(
        state_rmse, future, state_threshold, 1, low["policy_step"])
    pooled = np.concatenate([
        low["contact_sensor"][no_action], high["contact_sensor"][no_action]], axis=0)
    scales = np.maximum(
        float(analysis["sensor_sigma_multiplier"]) * np.std(pooled, axis=0),
        formal_feature_scale_floors(config))
    normalized_gap = np.abs(low["contact_sensor"] - high["contact_sensor"]) / scales
    fused_sensor_gap = np.max(normalized_gap, axis=1)
    sensor_onset = _first_sustained(
        fused_sensor_gap, future, 1.0,
        int(analysis["sensor_consecutive_policy_samples"]), low["policy_step"])
    low_mu = metadata.get("low_mu", .2)
    high_mu = metadata.get("high_mu", 1.6)
    oracle_load_gap = float(np.max(np.abs(
        low["oracle_patch_tangential_force"]
        - high["oracle_patch_tangential_force"])))
    oracle_slip_gap = float(np.max(np.abs(
        low["oracle_patch_mean_slip_speed"]
        - high["oracle_patch_mean_slip_speed"])))
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
                            analysis["spring_force_cap_fraction_max"]}
    sensor_leads = (sensor_onset is not None and state_onset is not None
                    and (not analysis["require_sensor_not_later_than_state"]
                         or sensor_onset <= state_onset))
    scientific_gate = {
        "oracle_intervention": oracle_intervention,
        "formal_sensor_separable": sensor_onset is not None,
        "future_state_divergence": state_onset is not None,
        "state_exceeds_repeat_floor": float(np.max(state_rmse)) >= state_threshold,
        "sensor_not_later_than_state": sensor_leads}
    if not all(engineering_gate.values()):
        verdict = "PHASE0C_ENGINEERING_BLOCKED"
    elif not all(scientific_gate.values()):
        verdict = "PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED"
    else:
        verdict = "PHASE0C_PAIR_COMPLETE"
    return {
        "verdict": verdict, "engineering_gate": engineering_gate,
        "scientific_gate": scientific_gate,
        "initial_state_max_abs_m": initial_difference,
        "command_arrays_equal": commands,
        "no_action_state_drift_m": drifts,
        "spring_force_cap_fraction": cap_fraction,
        "repeat_peak_state_rmse_m": repeat_peak,
        "repeat_state_rmse": repeat_state_rmse,
        "repeat_sensor_difference": repeat_sensor_difference,
        "baseline_state_std_m": baseline_state_std,
        "state_threshold_m": state_threshold,
        "state_rmse": state_rmse,
        "peak_future_state_rmse_m": float(np.max(state_rmse[future])),
        "fused_sensor_gap": fused_sensor_gap,
        "t_physics_separable": sensor_onset,
        "t_state_divergence": state_onset,
        "sensor_lead_policy_steps": (None if sensor_onset is None or state_onset is None
                                     else state_onset - sensor_onset),
        "oracle": {"low_patch_mu": low_mu, "high_patch_mu": high_mu,
                   "peak_tangential_load_gap_n": oracle_load_gap,
                   "peak_mean_slip_speed_gap_mps": oracle_slip_gap},
        "deformation_diagnostics": {
            "optional_only": True, "used_as_hard_gate": False}}


def analyze(config_path: str) -> dict:
    config = load_config(config_path)
    low, low_meta = _load(external_pair_dir(config, "uniform_low"))
    high, high_meta = _load(external_pair_dir(config, "right_local_high"))
    repeat, _ = _load(external_pair_dir(config, "uniform_low_repeat"))
    metrics = evaluate_pair_arrays(low, high, repeat, config, {
        "low_mu": low_meta["arm"]["patch_lateral_friction"],
        "high_mu": high_meta["arm"]["patch_lateral_friction"]})
    report = report_pair_dir(config)
    write_json(report / "pair_metrics.json", metrics)
    (report / "pair_summary.md").write_text(
        "# Phase 0C strict pair\n\n- Verdict: `{}`\n"
        "- Formal sensor onset: `{}`\n- State onset: `{}`\n".format(
            metrics["verdict"], metrics["t_physics_separable"],
            metrics["t_state_divergence"]), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
