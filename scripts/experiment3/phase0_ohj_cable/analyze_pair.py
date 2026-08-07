"""Evaluate the four pre-control OHJ Phase 0D scientific gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, pair_dir, report_dir, write_json)


def rmse(left, right):
    return np.sqrt(np.mean(np.square(np.asarray(left) - np.asarray(right)),
                           axis=-1))


def first_sustained(values, mask, threshold, consecutive, steps):
    hit = np.asarray(values) >= float(threshold)
    mask = np.asarray(mask, dtype=bool)
    for start in range(len(hit) - int(consecutive) + 1):
        stop = start + int(consecutive)
        if np.all(mask[start:stop]) and np.all(hit[start:stop]):
            return int(steps[start]), int(start)
    return None, None


def evaluate_pair(free, jam, repeat, branch_metadata, config):
    analysis, sensor_cfg = config["analysis"], config["sensor"]
    command_keys = ("command_phase", "command_ee_target",
                    "command_joint_target")
    commands_equal = all(
        np.array_equal(free[key], jam[key])
        and np.array_equal(free[key], repeat[key]) for key in command_keys)
    aligned = (np.array_equal(free["physics_step"], jam["physics_step"])
               and np.array_equal(free["phase"], jam["phase"])
               and np.array_equal(free["physics_step"], repeat["physics_step"])
               and np.array_equal(free["phase"], repeat["phase"]))
    finite = all(np.all(np.isfinite(branch[key]))
                 for branch in (free, jam, repeat)
                 for key in ("statediff_state", "formal_wrench",
                             "extraction_progress_m"))
    initial_free = np.asarray(
        branch_metadata["free"]["initial_statediff_state"])
    initial_jam = np.asarray(
        branch_metadata["jam_right"]["initial_statediff_state"])
    post_free = np.asarray(
        branch_metadata["free"]["post_probe_statediff_state"])
    post_jam = np.asarray(
        branch_metadata["jam_right"]["post_probe_statediff_state"])
    initial_rmse = float(rmse(initial_free, initial_jam))
    post_rmse = float(rmse(post_free, post_jam))
    post_keypoint_rmse = float(rmse(post_free[:48], post_jam[:48]))
    post_ee_rmse = float(rmse(post_free[48:], post_jam[48:]))

    phase = free["phase"].astype(str)
    steps = free["physics_step"].astype(np.int64)
    no_action = phase == "no_action"
    probe = np.isin(phase, ["probe_forward", "probe_hold", "probe_return"])
    future = np.isin(phase, ["test_pull", "post_test"])
    pooled = np.concatenate([
        free["formal_wrench"][no_action], jam["formal_wrench"][no_action]])
    std = np.std(pooled, axis=0)
    scale = np.concatenate([
        np.maximum(5.0 * std[:3], float(sensor_cfg["force_floor_n"])),
        np.maximum(5.0 * std[3:], float(sensor_cfg["torque_floor_nm"])),
    ])
    gap = np.abs(free["formal_wrench"] - jam["formal_wrench"])
    normalized = gap / scale
    fused = np.max(normalized, axis=1)
    sensor_step, sensor_index = first_sustained(
        fused, probe, analysis["sensor_normalized_gap_min"],
        sensor_cfg["consecutive_samples"], steps)
    channel_names = sensor_cfg["channels"]
    trigger = None
    if sensor_index is not None:
        channel = int(np.argmax(normalized[sensor_index]))
        trigger = {"channel_index": channel,
                   "channel": channel_names[channel],
                   "normalized_gap": float(normalized[sensor_index, channel])}

    visible_gap = rmse(
        free["statediff_state"][:, :48],
        jam["statediff_state"][:, :48])
    repeat_gap = rmse(
        free["statediff_state"][:, :48],
        repeat["statediff_state"][:, :48])
    repeat_peak = float(np.max(repeat_gap[future]))
    future_threshold = max(
        float(analysis["future_visible_rmse_min_m"]),
        float(analysis["future_vs_repeat_multiplier"]) * repeat_peak)
    future_peak = float(np.max(visible_gap[future]))
    gate1 = initial_rmse <= analysis["initial_visible_rmse_max_m"]
    gate2 = post_rmse <= analysis["post_probe_visible_rmse_max_m"]
    gate3 = sensor_step is not None
    gate4 = future_peak >= future_threshold
    engineering = bool(commands_equal and aligned and finite)
    if not engineering:
        verdict = "PHASE0D_ENGINEERING_BLOCKED"
    elif not (gate1 and gate2):
        verdict = "PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL"
    elif not gate3:
        verdict = "PHASE0D_SENSOR_NOT_OBSERVABLE"
    elif not gate4:
        verdict = "PHASE0D_FUTURE_BRANCH_NOT_ESTABLISHED"
    else:
        verdict = "PHASE0D_PAIR_COMPLETE"
    return {
        "verdict": verdict,
        "engineering": {"commands_equal": commands_equal,
                        "trace_alignment": aligned, "finite": finite},
        "gates": {"initial_observable_equivalence": bool(gate1),
                  "post_probe_observable_equivalence": bool(gate2),
                  "sensor_observability": bool(gate3),
                  "same_action_future_divergence": bool(gate4)},
        "initial_51d_rmse_m": initial_rmse,
        "post_probe_51d_rmse_m": post_rmse,
        "post_probe_keypoint_rmse_m": post_keypoint_rmse,
        "post_probe_ee_rmse_m": post_ee_rmse,
        "sensor_onset_step": sensor_step,
        "sensor_onset_phase": (None if sensor_index is None
                               else str(phase[sensor_index])),
        "sensor_trigger": trigger,
        "peak_fused_sensor_gap": float(np.max(fused[probe])),
        "peak_force_norm_gap_n": float(np.max(np.linalg.norm(gap[probe, :3], axis=1))),
        "peak_torque_norm_gap_nm": float(np.max(np.linalg.norm(gap[probe, 3:], axis=1))),
        "future_peak_visible_rmse_m": future_peak,
        "repeat_peak_visible_rmse_m": repeat_peak,
        "future_visible_threshold_m": future_threshold,
        "final_extraction_progress_m": {
            "free": float(free["extraction_progress_m"][-1]),
            "jam_right": float(jam["extraction_progress_m"][-1]),
            "free_repeat": float(repeat["extraction_progress_m"][-1])},
        "oracle": {
            "free_peak_latch_force_n": float(np.max(
                free["oracle_latch_contact_force"])),
            "jam_peak_latch_force_n": float(np.max(
                jam["oracle_latch_contact_force"])),
            "jam_contact_samples": int(np.count_nonzero(
                jam["oracle_latch_contact_count"]))},
    }


def _load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def analyze(config_path):
    config = load_config(config_path)
    root = pair_dir(config)
    branches = {name: _load(root / name / "trajectory.npz")
                for name in ("free", "jam_right", "free_repeat")}
    metadata = {name: json.loads((root / name / "metadata.json").read_text(
        encoding="utf-8")) for name in branches}
    metrics = evaluate_pair(
        branches["free"], branches["jam_right"], branches["free_repeat"],
        metadata, config)
    output = report_dir(config)
    write_json(output / "pair_metrics.json", metrics)
    (output / "pair_summary.md").write_text(
        "# OHJ Phase 0D pair\n\n- Verdict: `{}`\n"
        "- Sensor onset: `{}`\n- Future peak: `{:.6f} m\n".format(
            metrics["verdict"], metrics["sensor_onset_step"],
            metrics["future_peak_visible_rmse_m"]), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
