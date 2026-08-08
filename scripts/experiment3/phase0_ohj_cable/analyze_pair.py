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


def sensor_floor_vector(sensor_cfg):
    if "channel_floor" in sensor_cfg:
        return np.asarray(sensor_cfg["channel_floor"], dtype=np.float64)
    return np.asarray(
        [sensor_cfg["force_floor_n"]] * 3
        + [sensor_cfg["torque_floor_nm"]] * 3,
        dtype=np.float64,
    )


def formal_sensor_array(branch, sensor_cfg):
    field = sensor_cfg.get("trace_field", "formal_wrench")
    return np.asarray(branch[field], dtype=np.float64)


def _scalar_rmse(left, right):
    return float(np.sqrt(np.mean(np.square(
        np.asarray(left, dtype=np.float64)
        - np.asarray(right, dtype=np.float64)))))


def _state_gap_payload(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return {
        "rmse_51d_m": _scalar_rmse(left, right),
        "keypoint_rmse_m": _scalar_rmse(left[:48], right[:48]),
        "ee_rmse_m": _scalar_rmse(left[48:], right[48:]),
    }


def _phase_indices(arrays, phase_name):
    phase = arrays["phase"].astype(str)
    return np.flatnonzero(phase == str(phase_name))


REPEAT_PHASES = (
    "no_action",
    "probe_forward",
    "probe_hold",
    "probe_return",
    "post_probe",
    "test_pull",
    "post_test",
)

PROBE_PHASES = (
    "probe_forward",
    "probe_hold",
    "probe_return",
)

LOAD_PATH_NUMERIC_EPS_N = 1e-12


def sustained_window_peak(values, consecutive):
    values = np.asarray(values, dtype=np.float64)
    width = int(consecutive)
    if len(values) < width:
        return None
    return float(max(
        np.min(values[start:start + width])
        for start in range(len(values) - width + 1)))


def repeat_corrected_signal_stats(
        branch_gap, repeat_gap, mask, *, consecutive,
        repeat_multiplier, mechanical_floor):
    branch_values = np.asarray(branch_gap[mask], dtype=np.float64)
    repeat_values = np.asarray(repeat_gap[mask], dtype=np.float64)
    branch_sustained = sustained_window_peak(branch_values, consecutive)
    repeat_sustained = sustained_window_peak(repeat_values, consecutive)
    if branch_sustained is None or repeat_sustained is None:
        raise ValueError("load-path diagnostic window is too short")
    excess = max(0.0, branch_sustained - repeat_sustained)
    repeat_near_zero = bool(
        repeat_sustained <= LOAD_PATH_NUMERIC_EPS_N)
    ratio = (None if repeat_near_zero else float(
        branch_sustained / repeat_sustained))
    branch_specific = bool(
        branch_sustained > LOAD_PATH_NUMERIC_EPS_N
        and branch_sustained >= float(repeat_multiplier) * max(
            repeat_sustained, LOAD_PATH_NUMERIC_EPS_N))
    return {
        "branch_peak_gap_n": float(np.max(branch_values)),
        "repeat_peak_gap_n": float(np.max(repeat_values)),
        "branch_median_gap_n": float(np.median(branch_values)),
        "repeat_median_gap_n": float(np.median(repeat_values)),
        "branch_sustained_gap_n": branch_sustained,
        "repeat_sustained_gap_n": repeat_sustained,
        "repeat_corrected_excess_n": float(excess),
        "branch_over_repeat_ratio": ratio,
        "repeat_near_zero": repeat_near_zero,
        "branch_specific_vs_repeat": branch_specific,
        "mechanically_large_excess": bool(
            excess >= float(mechanical_floor)),
    }


def contact_adjacent_constraint_indices(contact_bead_indices, num_constraints):
    indices = set()
    for bead in np.asarray(contact_bead_indices, dtype=np.int64):
        bead = int(bead)
        if bead > 0:
            indices.add(bead - 1)
        if bead < int(num_constraints):
            indices.add(bead)
    return sorted(indices)


def load_path_diagnostics(free, jam, repeat, config):
    diagnostic = config.get("diagnostic")
    if (not diagnostic or diagnostic.get("mode")
            != "spatial_repeat_corrected_load_path_information"):
        return None
    free_force = np.asarray(
        free["oracle_internal_cable_constraint_force_xyz"],
        dtype=np.float64)
    jam_force = np.asarray(
        jam["oracle_internal_cable_constraint_force_xyz"],
        dtype=np.float64)
    repeat_force = np.asarray(
        repeat["oracle_internal_cable_constraint_force_xyz"],
        dtype=np.float64)
    phase = free["phase"].astype(str)
    no_action = phase == "no_action"
    probe = np.isin(phase, PROBE_PHASES)
    branch_gap = np.linalg.norm(free_force - jam_force, axis=2)
    repeat_gap = np.linalg.norm(free_force - repeat_force, axis=2)
    num_constraints = int(branch_gap.shape[1])
    consecutive = int(diagnostic["consecutive_samples"])
    repeat_multiplier = float(diagnostic["branch_vs_repeat_multiplier"])
    mechanical_floor = float(diagnostic["mechanical_reference_floor_n"])
    rows = []
    for segment in range(num_constraints):
        no_action_stats = repeat_corrected_signal_stats(
            branch_gap[:, segment], repeat_gap[:, segment], no_action,
            consecutive=consecutive,
            repeat_multiplier=repeat_multiplier,
            mechanical_floor=mechanical_floor)
        probe_stats = repeat_corrected_signal_stats(
            branch_gap[:, segment], repeat_gap[:, segment], probe,
            consecutive=consecutive,
            repeat_multiplier=repeat_multiplier,
            mechanical_floor=mechanical_floor)
        rows.append({
            "constraint_index": int(segment),
            "beads": [int(segment), int(segment + 1)],
            "no_action": no_action_stats,
            "probe": probe_stats,
            "probe_emergence_excess_n": float(
                probe_stats["repeat_corrected_excess_n"]
                - no_action_stats["repeat_corrected_excess_n"]),
        })
    jam_contact_mask = np.asarray(
        jam["oracle_latch_contact_bead_mask"], dtype=np.int8)
    probe_contact_bead_indices = np.flatnonzero(np.any(
        jam_contact_mask[probe] > 0, axis=0)).astype(int).tolist()
    contact_reference_constraints = contact_adjacent_constraint_indices(
        probe_contact_bead_indices, num_constraints)
    if not contact_reference_constraints:
        raise ValueError("R7S requires actual JAM latch contact during probe")
    branch_specific_segments = [
        int(row["constraint_index"]) for row in rows
        if row["probe"]["branch_specific_vs_repeat"]]
    contact_reference_specific = [
        int(index) for index in contact_reference_constraints
        if rows[index]["probe"]["branch_specific_vs_repeat"]]
    reference_candidates = (contact_reference_specific
                            if contact_reference_specific
                            else contact_reference_constraints)
    contact_reference_peak = max(
        reference_candidates,
        key=lambda index: rows[index]["probe"][
            "repeat_corrected_excess_n"])
    contact_reference_peak_excess = float(
        rows[contact_reference_peak]["probe"][
            "repeat_corrected_excess_n"])
    global_peak_segment = max(
        range(num_constraints),
        key=lambda index: rows[index]["probe"][
            "repeat_corrected_excess_n"])
    global_peak_excess = float(
        rows[global_peak_segment]["probe"]["repeat_corrected_excess_n"])
    proximal_index = num_constraints - 1
    proximal = rows[proximal_index]["probe"]
    proximal_excess = float(proximal["repeat_corrected_excess_n"])
    retention_ratio = (
        None if contact_reference_peak_excess <= LOAD_PATH_NUMERIC_EPS_N
        else float(proximal_excess / contact_reference_peak_excess))
    contact_reference_has_signal = bool(contact_reference_specific)
    proximal_specific = bool(proximal["branch_specific_vs_repeat"])
    proximal_large = bool(proximal["mechanically_large_excess"])
    if not contact_reference_has_signal:
        route_hint = "repair_hidden_interaction_or_probe_source_mechanics"
    elif not proximal_specific:
        route_hint = "repair_mechanical_load_transmission"
    elif proximal_large:
        route_hint = "repair_physical_grasp_sensing_coupling"
    else:
        route_hint = "amplify_probe_or_mechanical_signal"
    return {
        "method": "spatial_repeat_corrected_pair_level_load_profile",
        "interpretation_limit": (
            "pair-level oracle mechanism diagnostic; not a mutual-"
            "information or held-out prediction claim"),
        "branch_vs_repeat_multiplier": repeat_multiplier,
        "mechanical_reference_floor_n": mechanical_floor,
        "num_constraints": num_constraints,
        "probe_contact_bead_indices": probe_contact_bead_indices,
        "contact_reference_constraint_indices": contact_reference_constraints,
        "contact_reference_branch_specific_segment_indices": (
            contact_reference_specific),
        "contact_reference_has_branch_specific_signal": (
            contact_reference_has_signal),
        "contact_reference_peak_segment_index": int(contact_reference_peak),
        "contact_reference_peak_excess_n": contact_reference_peak_excess,
        "global_peak_segment_index": int(global_peak_segment),
        "global_peak_excess_n": global_peak_excess,
        "global_peak_is_contact_adjacent": bool(
            global_peak_segment in contact_reference_constraints),
        "branch_specific_segment_indices": branch_specific_segments,
        "furthest_branch_specific_segment_toward_gripper": (
            None if not branch_specific_segments
            else int(max(branch_specific_segments))),
        "gripper_proximal_constraint_index": int(proximal_index),
        "proximal_branch_specific_vs_repeat": proximal_specific,
        "proximal_repeat_corrected_excess_n": proximal_excess,
        "proximal_mechanically_large_excess": proximal_large,
        "proximal_retention_ratio": retention_ratio,
        "segments": rows,
        "route_hint": route_hint,
    }


def _ratio_or_none(current, baseline):
    baseline = float(baseline)
    if abs(baseline) <= LOAD_PATH_NUMERIC_EPS_N:
        return None
    return float(float(current) / baseline)


def _least_squares_slope(values, hz):
    values = np.asarray(values, dtype=np.float64)
    time = np.arange(len(values), dtype=np.float64) / float(hz)
    time = time - np.mean(time)
    centered = values - np.mean(values)
    return float(np.dot(time, centered) / np.dot(time, time))


def future_horizon_diagnostics(
        visible_gap, repeat_gap, future_mask, physics_steps, phases, config):
    audit = config.get("future_horizon_audit")
    if (not audit
            or audit.get("mode") != "final_fixed_horizon_sufficiency"):
        return None

    values = np.asarray(visible_gap[future_mask], dtype=np.float64)
    repeat_values = np.asarray(repeat_gap[future_mask], dtype=np.float64)
    future_steps = np.asarray(physics_steps[future_mask], dtype=np.int64)
    future_phases = np.asarray(phases[future_mask]).astype(str)
    hz = float(config["execution"]["hz"])
    repeat_peak = float(np.max(repeat_values))
    threshold = max(
        float(config["analysis"]["future_visible_rmse_min_m"]),
        float(config["analysis"]["future_vs_repeat_multiplier"])
        * repeat_peak)
    peak_index = int(np.argmax(values))
    peak = float(values[peak_index])
    endpoint = float(values[-1])
    crossing = np.flatnonzero(values >= threshold)
    if len(crossing):
        first_crossing = int(crossing[0])
        crossing_payload = {
            "future_index_zero_based": first_crossing,
            "time_from_future_start_s": float(first_crossing / hz),
            "physics_step": int(future_steps[first_crossing]),
            "phase": str(future_phases[first_crossing]),
            "visible_rmse_m": float(values[first_crossing]),
        }
    else:
        crossing_payload = None

    tails = []
    for window_s in audit["tail_window_seconds"]:
        width = int(round(float(window_s) * hz))
        tail = values[-width:]
        difference = np.diff(tail)
        tails.append({
            "window_seconds": float(window_s),
            "samples": int(width),
            "delta_m": float(tail[-1] - tail[0]),
            "least_squares_slope_m_s": _least_squares_slope(tail, hz),
            "positive_step_differences": int(np.count_nonzero(
                difference > 0.0)),
            "total_step_differences": int(len(difference)),
        })

    threshold_reached = bool(peak >= threshold)
    endpoint_is_global_max = bool(peak_index == len(values) - 1)
    all_tail_slopes_positive = all(
        row["least_squares_slope_m_s"] > 0.0 for row in tails)
    tail_still_rising = bool(
        (not threshold_reached)
        and endpoint_is_global_max
        and all_tail_slopes_positive)
    if threshold_reached:
        interpretation = "gate4_threshold_reached_with_frozen_benchmark"
    elif tail_still_rising:
        interpretation = (
            "strong_right_censoring_signature_remains_at_final_horizon")
    else:
        interpretation = (
            "threshold_not_reached_without_strong_endpoint_right_censoring")

    return {
        "mode": audit["mode"],
        "diagnostic_only": True,
        "benchmark_gates_unchanged": True,
        "stop_after_this_trial": bool(audit["stop_after_this_trial"]),
        "baseline": dict(audit["baseline"]),
        "future_samples": int(len(values)),
        "future_span_s": float((len(values) - 1) / hz),
        "post_test_steps": int(config["execution"]["post_test_steps"]),
        "post_test_seconds": float(
            config["execution"]["post_test_steps"] / hz),
        "future_threshold_m": threshold,
        "threshold_reached": threshold_reached,
        "first_threshold_crossing": crossing_payload,
        "endpoint_visible_rmse_m": endpoint,
        "peak_visible_rmse_m": peak,
        "peak_index_zero_based": peak_index,
        "peak_time_from_future_start_s": float(peak_index / hz),
        "endpoint_is_global_max": endpoint_is_global_max,
        "repeat_peak_visible_rmse_m": repeat_peak,
        "future_to_repeat_ratio": (
            None if repeat_peak <= 1e-12 else float(peak / repeat_peak)),
        "remaining_margin_to_threshold_m": float(max(0.0, threshold - peak)),
        "tail_windows": tails,
        "all_tail_slopes_positive": bool(all_tail_slopes_positive),
        "tail_still_rising_at_horizon_end": tail_still_rising,
        "interpretation": interpretation,
    }


def _unit_probe_direction(delta):
    delta = np.asarray(delta, dtype=np.float64)
    amplitude = float(np.linalg.norm(delta))
    return delta / amplitude


def probe_amplification_diagnostics(
        config, *, post_probe_rmse_m, formal_sensor_peak, load_path):
    repair = config.get("repair")
    if (not repair or repair.get("mode")
            != "single_fixed_speed_probe_excursion_amplification"):
        return None
    baseline = repair["baseline"]
    motion = config["motion"]
    hz = float(config["execution"]["hz"])
    amplitude = float(np.linalg.norm(np.asarray(
        motion["probe_delta_xyz_m"], dtype=np.float64)))
    forward_steps = int(motion["probe_forward_steps"])
    speed = float(amplitude / (forward_steps / hz))
    current = {
        "probe_amplitude_m": amplitude,
        "probe_forward_steps": forward_steps,
        "probe_return_steps": int(motion["probe_return_steps"]),
        "probe_speed_m_s": speed,
        "post_probe_rmse_m": float(post_probe_rmse_m),
        "formal_sensor_peak_fused_gap": float(formal_sensor_peak),
        "contact_reference_peak_excess_n": float(
            load_path["contact_reference_peak_excess_n"]),
        "proximal_excess_n": float(
            load_path["proximal_repeat_corrected_excess_n"]),
        "proximal_retention_ratio": (
            None if load_path["proximal_retention_ratio"] is None
            else float(load_path["proximal_retention_ratio"])),
    }
    return {
        "mode": repair["mode"],
        "report_only": True,
        "stop_after_this_trial": bool(repair["stop_after_this_trial"]),
        "baseline": dict(baseline),
        "current": current,
        "gain_ratio": {
            "probe_amplitude": _ratio_or_none(
                current["probe_amplitude_m"],
                baseline["probe_amplitude_m"]),
            "probe_speed": _ratio_or_none(
                current["probe_speed_m_s"], baseline["probe_speed_m_s"]),
            "post_probe_rmse": _ratio_or_none(
                current["post_probe_rmse_m"],
                baseline["post_probe_rmse_m"]),
            "formal_sensor_peak": _ratio_or_none(
                current["formal_sensor_peak_fused_gap"],
                baseline["formal_sensor_peak_fused_gap"]),
            "contact_reference_peak_excess": _ratio_or_none(
                current["contact_reference_peak_excess_n"],
                baseline["contact_reference_peak_excess_n"]),
            "proximal_excess": _ratio_or_none(
                current["proximal_excess_n"],
                baseline["proximal_excess_n"]),
            "proximal_retention": (
                None if current["proximal_retention_ratio"] is None
                else _ratio_or_none(
                    current["proximal_retention_ratio"],
                    baseline["proximal_retention_ratio"])),
        },
    }


def probe_direction_diagnostics(
        config, *, post_probe_rmse_m, formal_sensor_peak, load_path,
        recovery, future_peak):
    repair = config.get("repair")
    if (not repair or repair.get("mode")
            != "single_orthogonal_contact_loading_probe"):
        return None
    baseline = repair["baseline"]
    motion = config["motion"]
    hz = float(config["execution"]["hz"])
    delta = np.asarray(motion["probe_delta_xyz_m"], dtype=np.float64)
    amplitude = float(np.linalg.norm(delta))
    forward_steps = int(motion["probe_forward_steps"])
    speed = float(amplitude / (forward_steps / hz))
    direction = _unit_probe_direction(delta)
    baseline_delta = np.asarray(
        baseline["probe_delta_xyz_m"], dtype=np.float64)
    baseline_direction = _unit_probe_direction(baseline_delta)
    direction_dot = float(np.dot(direction, baseline_direction))
    current = {
        "probe_delta_xyz_m": delta.astype(float).tolist(),
        "probe_direction_unit": direction.astype(float).tolist(),
        "probe_amplitude_m": amplitude,
        "probe_forward_steps": forward_steps,
        "probe_return_steps": int(motion["probe_return_steps"]),
        "probe_speed_m_s": speed,
        "post_probe_rmse_m": float(post_probe_rmse_m),
        "formal_sensor_peak_fused_gap": float(formal_sensor_peak),
        "contact_reference_peak_excess_n": float(
            load_path["contact_reference_peak_excess_n"]),
        "proximal_excess_n": float(
            load_path["proximal_repeat_corrected_excess_n"]),
        "proximal_retention_ratio": (
            None if load_path["proximal_retention_ratio"] is None
            else float(load_path["proximal_retention_ratio"])),
        "post_probe_jam_latch_contact_fraction": recovery[
            "post_probe_jam_latch_contact_fraction"],
        "future_peak_visible_rmse_m": float(future_peak),
    }
    return {
        "mode": repair["mode"],
        "report_only": True,
        "stop_after_this_trial": bool(repair["stop_after_this_trial"]),
        "baseline": dict(baseline),
        "current": current,
        "direction_dot_baseline": direction_dot,
        "gain_ratio": {
            "probe_amplitude": _ratio_or_none(
                current["probe_amplitude_m"],
                baseline["probe_amplitude_m"]),
            "probe_speed": _ratio_or_none(
                current["probe_speed_m_s"], baseline["probe_speed_m_s"]),
            "post_probe_rmse": _ratio_or_none(
                current["post_probe_rmse_m"],
                baseline["post_probe_rmse_m"]),
            "formal_sensor_peak": _ratio_or_none(
                current["formal_sensor_peak_fused_gap"],
                baseline["formal_sensor_peak_fused_gap"]),
            "contact_reference_peak_excess": _ratio_or_none(
                current["contact_reference_peak_excess_n"],
                baseline["contact_reference_peak_excess_n"]),
            "proximal_excess": _ratio_or_none(
                current["proximal_excess_n"],
                baseline["proximal_excess_n"]),
            "proximal_retention": (
                None if current["proximal_retention_ratio"] is None
                else _ratio_or_none(
                    current["proximal_retention_ratio"],
                    baseline["proximal_retention_ratio"])),
            "post_probe_jam_latch_contact_fraction": (
                None if current[
                    "post_probe_jam_latch_contact_fraction"] is None
                else _ratio_or_none(
                    current["post_probe_jam_latch_contact_fraction"],
                    baseline["post_probe_jam_latch_contact_fraction"])),
            "future_peak": _ratio_or_none(
                current["future_peak_visible_rmse_m"],
                baseline["future_peak_visible_rmse_m"]),
        },
    }


def repeatability_by_phase(free, repeat, config):
    threshold = float(
        config["analysis"]["post_probe_visible_rmse_max_m"])
    rows = []
    first_phase_above = None
    for phase_name in REPEAT_PHASES:
        mask = free["phase"].astype(str) == phase_name
        if not np.any(mask):
            continue
        free_state = np.asarray(
            free["statediff_state"][mask], dtype=np.float64)
        repeat_state = np.asarray(
            repeat["statediff_state"][mask], dtype=np.float64)
        gap_51d = rmse(free_state, repeat_state)
        gap_keypoint = rmse(free_state[:, :48], repeat_state[:, :48])
        gap_ee = rmse(free_state[:, 48:], repeat_state[:, 48:])
        row = {
            "phase": phase_name,
            "start_51d_rmse_m": float(gap_51d[0]),
            "peak_51d_rmse_m": float(np.max(gap_51d)),
            "end_51d_rmse_m": float(gap_51d[-1]),
            "peak_keypoint_rmse_m": float(np.max(gap_keypoint)),
            "peak_ee_rmse_m": float(np.max(gap_ee)),
        }
        if (first_phase_above is None
                and row["peak_51d_rmse_m"] > threshold):
            first_phase_above = phase_name
        rows.append(row)
    return {
        "phases": rows,
        "first_phase_above_equivalence_threshold": first_phase_above,
        "equivalence_threshold_m": threshold,
    }


def probe_contact_diagnostics(free, jam):
    free_phase = free["phase"].astype(str)
    jam_phase = jam["phase"].astype(str)
    free_mask = np.isin(free_phase, PROBE_PHASES)
    jam_mask = np.isin(jam_phase, PROBE_PHASES)
    free_contact_samples = int(np.count_nonzero(
        free["oracle_latch_contact_count"][free_mask]))
    jam_contact_samples = int(np.count_nonzero(
        jam["oracle_latch_contact_count"][jam_mask]))
    jam_probe_samples = int(np.count_nonzero(jam_mask))
    return {
        "free_probe_contact_samples": free_contact_samples,
        "jam_probe_contact_samples": jam_contact_samples,
        "jam_probe_contact_fraction": (
            None if jam_probe_samples == 0 else float(
                jam_contact_samples / jam_probe_samples)),
        "jam_probe_peak_latch_force_n": float(np.max(
            jam["oracle_latch_contact_force"][jam_mask])
            if np.any(jam_mask) else 0.0),
    }


def recovery_curve(free, jam, repeat, config):
    hz = float(config["execution"]["hz"])
    checkpoints = [int(value) for value in config["analysis"].get(
        "recovery_checkpoints_steps", [0, 24, 48, 120, 240])]
    return_indices = _phase_indices(free, "probe_return")
    if not len(return_indices):
        raise ValueError("probe_return phase is missing")
    return_index = int(return_indices[-1])
    free_post = _phase_indices(free, "post_probe")
    jam_post = _phase_indices(jam, "post_probe")
    repeat_post = _phase_indices(repeat, "post_probe")
    if not (len(free_post) == len(jam_post) == len(repeat_post)):
        raise ValueError("post_probe trace length mismatch")
    rows = []
    for steps_after_return in checkpoints:
        if steps_after_return == 0:
            free_index = jam_index = repeat_index = return_index
        else:
            offset = min(steps_after_return, len(free_post)) - 1
            free_index = int(free_post[offset])
            jam_index = int(jam_post[offset])
            repeat_index = int(repeat_post[offset])
        free_jam = _state_gap_payload(
            free["statediff_state"][free_index],
            jam["statediff_state"][jam_index])
        free_repeat = _state_gap_payload(
            free["statediff_state"][free_index],
            repeat["statediff_state"][repeat_index])
        rows.append({
            "post_probe_steps": int(steps_after_return),
            "time_s": float(steps_after_return / hz),
            "free_jam": free_jam,
            "free_repeat": free_repeat,
            "branch_excess_over_repeat_m": max(
                0.0, free_jam["rmse_51d_m"] - free_repeat["rmse_51d_m"]),
        })
    immediate = float(rows[0]["free_jam"]["rmse_51d_m"])
    final = float(rows[-1]["free_jam"]["rmse_51d_m"])
    recovery_fraction = (None if immediate <= 1e-12
                         else float(1.0 - final / immediate))
    post_mask = jam["phase"].astype(str) == "post_probe"
    final_repeat = float(rows[-1]["free_repeat"]["rmse_51d_m"])
    final_excess = float(rows[-1]["branch_excess_over_repeat_m"])
    final_repeat_fraction = (
        None if final <= 1e-12 else float(final_repeat / final))
    final_excess_fraction = (
        None if final <= 1e-12 else float(final_excess / final))
    post_probe_samples = int(np.count_nonzero(post_mask))
    post_probe_contact_samples = int(np.count_nonzero(
        jam["oracle_latch_contact_count"][post_mask]))
    post_probe_contact_fraction = (
        None if post_probe_samples == 0 else float(
            post_probe_contact_samples / post_probe_samples))
    return {
        "checkpoints": rows,
        "immediate_free_jam_rmse_m": immediate,
        "final_free_jam_rmse_m": final,
        "final_free_repeat_rmse_m": final_repeat,
        "final_branch_excess_over_repeat_m": final_excess,
        "final_repeat_fraction_of_free_jam": final_repeat_fraction,
        "final_branch_excess_fraction_of_free_jam": final_excess_fraction,
        "recovery_fraction": recovery_fraction,
        "post_probe_jam_latch_contact_samples": post_probe_contact_samples,
        "post_probe_jam_latch_contact_fraction": post_probe_contact_fraction,
        "post_probe_jam_peak_latch_force_n": float(np.max(
            jam["oracle_latch_contact_force"][post_mask])
            if np.any(post_mask) else 0.0),
    }


def evaluate_pair(free, jam, repeat, branch_metadata, config):
    analysis, sensor_cfg = config["analysis"], config["sensor"]
    sensor_field = sensor_cfg.get("trace_field", "formal_wrench")
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
                 for key in ("statediff_state", sensor_field,
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
    recovery = recovery_curve(free, jam, repeat, config)
    repeatability = repeatability_by_phase(free, repeat, config)
    probe_contact = probe_contact_diagnostics(free, jam)
    load_path = load_path_diagnostics(free, jam, repeat, config)

    phase = free["phase"].astype(str)
    steps = free["physics_step"].astype(np.int64)
    no_action = phase == "no_action"
    probe = np.isin(phase, PROBE_PHASES)
    future = np.isin(phase, ["test_pull", "post_test"])
    free_sensor = formal_sensor_array(free, sensor_cfg)
    jam_sensor = formal_sensor_array(jam, sensor_cfg)
    pooled = np.concatenate([
        free_sensor[no_action], jam_sensor[no_action]])
    std = np.std(pooled, axis=0)
    floor = sensor_floor_vector(sensor_cfg)
    scale = np.maximum(5.0 * std, floor)
    gap = np.abs(free_sensor - jam_sensor)
    normalized = gap / scale
    fused = np.max(normalized, axis=1)
    grasp_fused = np.max(normalized[:, :6], axis=1)

    aggregate_tactile_peak = None
    spatial_patch_metrics = None

    if sensor_field == "formal_sensor_spatial":
        tactile_normalized = normalized[:, 6:18]
        tactile_fused = np.max(tactile_normalized, axis=1)
        tactile_peak = float(np.max(tactile_fused[probe]))
        patch_normalized = tactile_normalized.reshape(
            len(tactile_normalized), 4, 3)
        patch_gap = gap[:, 6:18].reshape(len(gap), 4, 3)
        free_patch_count = np.asarray(
            free["gripper_surface_tactile_patch_contact_count"],
            dtype=np.int64)
        jam_patch_count = np.asarray(
            jam["gripper_surface_tactile_patch_contact_count"],
            dtype=np.int64)
        spatial_patch_metrics = []
        for patch in range(4):
            patch_fused = np.max(patch_normalized[:, patch, :], axis=1)
            patch_raw_norm = np.linalg.norm(patch_gap[:, patch, :], axis=1)
            spatial_patch_metrics.append({
                "patch": patch,
                "peak_fused_gap": float(np.max(patch_fused[probe])),
                "raw_force_gap_peak_n": float(
                    np.max(patch_raw_norm[probe])),
                "probe_contact_samples": {
                    "free": int(np.count_nonzero(
                        free_patch_count[probe, patch])),
                    "jam_right": int(np.count_nonzero(
                        jam_patch_count[probe, patch])),
                },
                "contact_count_peak_gap": float(np.max(np.abs(
                    free_patch_count[probe, patch]
                    - jam_patch_count[probe, patch]))),
            })
        tactile_raw_peak = float(max(
            row["raw_force_gap_peak_n"] for row in spatial_patch_metrics))
        free_tactile_contacts = free[
            "gripper_surface_contact_count"][probe]
        jam_tactile_contacts = jam[
            "gripper_surface_contact_count"][probe]
        tactile_contact_samples = {
            "free": int(np.count_nonzero(free_tactile_contacts)),
            "jam_right": int(np.count_nonzero(jam_tactile_contacts)),
        }
        tactile_contact_count_peak_gap = float(np.max(np.abs(
            free_tactile_contacts - jam_tactile_contacts)))
        aggregate_free = np.asarray(
            free["gripper_surface_tactile_force"], dtype=np.float64)
        aggregate_jam = np.asarray(
            jam["gripper_surface_tactile_force"], dtype=np.float64)
        aggregate_pooled = np.concatenate([
            aggregate_free[no_action], aggregate_jam[no_action]])
        aggregate_scale = np.maximum(
            5.0 * np.std(aggregate_pooled, axis=0),
            float(sensor_cfg["force_floor_n"]))
        aggregate_gap = np.abs(aggregate_free - aggregate_jam)
        aggregate_normalized = aggregate_gap / aggregate_scale
        aggregate_tactile_peak = float(np.max(
            np.max(aggregate_normalized, axis=1)[probe]))
    elif normalized.shape[1] >= 9:
        tactile_fused = np.max(normalized[:, 6:9], axis=1)
        tactile_raw_gap = gap[:, 6:9]
        tactile_peak = float(np.max(tactile_fused[probe]))
        tactile_raw_peak = float(np.max(np.linalg.norm(
            tactile_raw_gap[probe], axis=1)))
        free_tactile_contacts = free[
            "gripper_surface_contact_count"][probe]
        jam_tactile_contacts = jam[
            "gripper_surface_contact_count"][probe]
        tactile_contact_samples = {
            "free": int(np.count_nonzero(free_tactile_contacts)),
            "jam_right": int(np.count_nonzero(jam_tactile_contacts)),
        }
        tactile_contact_count_peak_gap = float(np.max(np.abs(
            free_tactile_contacts - jam_tactile_contacts)))
    else:
        tactile_peak = None
        tactile_raw_peak = None
        tactile_contact_samples = None
        tactile_contact_count_peak_gap = None
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
    horizon_diagnostic = future_horizon_diagnostics(
        visible_gap, repeat_gap, future, steps, phase, config)
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
    formal_sensor_peak = float(np.max(fused[probe]))
    amplification = probe_amplification_diagnostics(
        config,
        post_probe_rmse_m=post_rmse,
        formal_sensor_peak=formal_sensor_peak,
        load_path=load_path,
    )
    direction_diagnostic = probe_direction_diagnostics(
        config,
        post_probe_rmse_m=post_rmse,
        formal_sensor_peak=formal_sensor_peak,
        load_path=load_path,
        recovery=recovery,
        future_peak=future_peak,
    )
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
        "recovery": recovery,
        "repeatability_by_phase": repeatability,
        "probe_contact": probe_contact,
        "load_path_diagnostic": load_path,
        "sensor_trace_field": sensor_field,
        "grasp_only_peak_fused_gap": float(np.max(grasp_fused[probe])),
        "tactile_only_peak_fused_gap": tactile_peak,
        "aggregate_tactile_peak_fused_gap": aggregate_tactile_peak,
        "spatial_tactile_patches": spatial_patch_metrics,
        "tactile_raw_force_gap_peak_n": tactile_raw_peak,
        "tactile_probe_contact_samples": tactile_contact_samples,
        "tactile_contact_count_peak_gap": tactile_contact_count_peak_gap,
        "sensor_onset_step": sensor_step,
        "sensor_onset_phase": (None if sensor_index is None
                               else str(phase[sensor_index])),
        "sensor_trigger": trigger,
        "peak_fused_sensor_gap": formal_sensor_peak,
        "probe_amplification_diagnostic": amplification,
        "probe_direction_diagnostic": direction_diagnostic,
        "future_horizon_diagnostic": horizon_diagnostic,
        "peak_force_norm_gap_n": float(np.max(np.linalg.norm(gap[probe, :3], axis=1))),
        "peak_torque_norm_gap_nm": float(np.max(np.linalg.norm(gap[probe, 3:], axis=1))),
        "future_peak_visible_rmse_m": future_peak,
        "repeat_peak_visible_rmse_m": repeat_peak,
        "future_visible_threshold_m": future_threshold,
        "future_branch_to_repeat_ratio": (
            None if repeat_peak <= 1e-12 else float(future_peak / repeat_peak)),
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
    horizon_diagnostic = metrics["future_horizon_diagnostic"]
    if horizon_diagnostic is not None:
        write_json(
            output / "FUTURE_HORIZON_SUFFICIENCY.json",
            horizon_diagnostic)
    recovery = metrics["recovery"]
    repeatability = metrics["repeatability_by_phase"]
    probe_contact = metrics["probe_contact"]
    load_path = metrics["load_path_diagnostic"]
    amplification = metrics["probe_amplification_diagnostic"]
    direction_diagnostic = metrics["probe_direction_diagnostic"]
    probe_delta = np.asarray(
        config["motion"]["probe_delta_xyz_m"], dtype=np.float64)
    probe_amplitude_mm = float(1000.0 * np.linalg.norm(probe_delta))
    probe_direction = _unit_probe_direction(
        probe_delta).astype(float).tolist()
    summary = (
        "# OHJ Phase 0D pair\n\n"
        "- Verdict: `{}`\n"
        "- Probe amplitude: `{:.3f} mm`\n"
        "- Probe delta XYZ: `{}`\n"
        "- Probe direction unit: `{}`\n"
        "- Final post-probe RMSE: `{:.6f} m`\n"
        "- Immediate-return RMSE: `{:.6f} m`\n"
        "- Recovery fraction: `{}`\n"
        "- Final FREE-repeat RMSE: `{:.6f} m`\n"
        "- Final repeat/FREE-JAM fraction: `{}`\n"
        "- Final branch-excess/FREE-JAM fraction: `{}`\n"
        "- Post-probe JAM contact fraction: `{}`\n"
        "- FREE probe latch contacts: `{}`\n"
        "- JAM probe latch contacts: `{}`\n"
        "- JAM probe latch contact fraction: `{}`\n"
        "- JAM probe latch peak force: `{:.6f} N`\n"
        "- Sensor trace field: `{}`\n"
        "- Grasp-only peak fused gap: `{:.6f}`\n"
        "- Tactile-only peak fused gap: `{}`\n"
        "- R5-equivalent aggregate tactile peak: `{}`\n"
        "- Spatial tactile patch metrics: `{}`\n"
        "- Tactile raw force-gap peak: `{}`\n"
        "- Tactile probe contact samples: `{}`\n"
        "- Tactile contact-count peak gap: `{}`\n"
        "- Combined formal peak fused gap: `{:.6f}`\n"
        "- Future peak: `{:.6f} m`\n"
        "- Repeat future floor: `{:.6f} m`\n"
        "- Repeat first phase above equivalence threshold: `{}`\n"
        "- Repeat phase diagnostics: `{}`\n"
    ).format(
        metrics["verdict"],
        probe_amplitude_mm,
        probe_delta.astype(float).tolist(),
        probe_direction,
        metrics["post_probe_51d_rmse_m"],
        recovery["immediate_free_jam_rmse_m"],
        recovery["recovery_fraction"],
        recovery["final_free_repeat_rmse_m"],
        recovery["final_repeat_fraction_of_free_jam"],
        recovery["final_branch_excess_fraction_of_free_jam"],
        recovery["post_probe_jam_latch_contact_fraction"],
        probe_contact["free_probe_contact_samples"],
        probe_contact["jam_probe_contact_samples"],
        probe_contact["jam_probe_contact_fraction"],
        probe_contact["jam_probe_peak_latch_force_n"],
        metrics["sensor_trace_field"],
        metrics["grasp_only_peak_fused_gap"],
        metrics["tactile_only_peak_fused_gap"],
        metrics["aggregate_tactile_peak_fused_gap"],
        metrics["spatial_tactile_patches"],
        metrics["tactile_raw_force_gap_peak_n"],
        metrics["tactile_probe_contact_samples"],
        metrics["tactile_contact_count_peak_gap"],
        metrics["peak_fused_sensor_gap"],
        metrics["future_peak_visible_rmse_m"],
        metrics["repeat_peak_visible_rmse_m"],
        repeatability["first_phase_above_equivalence_threshold"],
        repeatability["phases"])
    if load_path is not None:
        summary += (
            "\n## Spatial repeat-corrected load-path audit\n\n"
            "- Method: `{}`\n"
            "- Interpretation limit: `{}`\n"
            "- Probe contact bead indices: `{}`\n"
            "- Contact reference constraint indices: `{}`\n"
            "- Contact reference branch-specific indices: `{}`\n"
            "- Contact reference peak segment/excess: `{}` / `{} N`\n"
            "- Global peak segment/excess: `{}` / `{} N`\n"
            "- Global peak contact-adjacent: `{}`\n"
            "- Branch-specific segment indices: `{}`\n"
            "- Furthest branch-specific segment toward gripper: `{}`\n"
            "- Proximal constraint index: `{}`\n"
            "- Proximal branch-specific: `{}`\n"
            "- Proximal excess: `{} N`\n"
            "- Proximal mechanically large: `{}`\n"
            "- Proximal retention ratio: `{}`\n"
            "- Route hint: `{}`\n"
        ).format(
            load_path["method"], load_path["interpretation_limit"],
            load_path["probe_contact_bead_indices"],
            load_path["contact_reference_constraint_indices"],
            load_path[
                "contact_reference_branch_specific_segment_indices"],
            load_path["contact_reference_peak_segment_index"],
            load_path["contact_reference_peak_excess_n"],
            load_path["global_peak_segment_index"],
            load_path["global_peak_excess_n"],
            load_path["global_peak_is_contact_adjacent"],
            load_path["branch_specific_segment_indices"],
            load_path["furthest_branch_specific_segment_toward_gripper"],
            load_path["gripper_proximal_constraint_index"],
            load_path["proximal_branch_specific_vs_repeat"],
            load_path["proximal_repeat_corrected_excess_n"],
            load_path["proximal_mechanically_large_excess"],
            load_path["proximal_retention_ratio"], load_path["route_hint"])
    if amplification is not None:
        summary += (
            "\n## Fixed-speed probe amplification comparison\n\n"
            "- Report only: `{}`\n"
            "- Stop after this trial: `{}`\n"
            "- Baseline: `{}`\n"
            "- Current: `{}`\n"
            "- Gain ratios: `{}`\n"
        ).format(
            amplification["report_only"],
            amplification["stop_after_this_trial"],
            amplification["baseline"],
            amplification["current"],
            amplification["gain_ratio"],
        )
    if direction_diagnostic is not None:
        summary += (
            "\n## Orthogonal contact-loading probe comparison\n\n"
            "- Report only: `{}`\n"
            "- Stop after this trial: `{}`\n"
            "- Direction dot baseline: `{}`\n"
            "- Baseline: `{}`\n"
            "- Current: `{}`\n"
            "- Gain ratios: `{}`\n"
        ).format(
            direction_diagnostic["report_only"],
            direction_diagnostic["stop_after_this_trial"],
            direction_diagnostic["direction_dot_baseline"],
            direction_diagnostic["baseline"],
            direction_diagnostic["current"],
            direction_diagnostic["gain_ratio"],
        )
    (output / "pair_summary.md").write_text(summary, encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(args.config)["verdict"]))


if __name__ == "__main__":
    main()
