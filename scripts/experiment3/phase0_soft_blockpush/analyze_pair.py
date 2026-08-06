"""Analyze one strict Soft BlockPush counterfactual pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from state_diff.env.block_pushing.soft_block_metrics import (  # noqa: E402
    edge_strain, first_sustained_onset, onset_from_baseline, paired_rmse,
    rigid_aligned_series, standardized_branch_gap)
from scripts.experiment3.phase0_soft_blockpush.common import write_json  # noqa: E402


FORMAL_CHANNELS = (
    "joint_motor_torque", "joint_reaction_wrench",
    "ee_tracking_error_xyz", "ee_contact_wrench")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _finite(*arrays: np.ndarray) -> bool:
    return all(bool(np.all(np.isfinite(array))) for array in arrays)


def _phase_onset_name(phases: np.ndarray, onset: Optional[int]) -> Optional[str]:
    return None if onset is None else str(phases[onset])


def _oracle_gap(free: Dict[str, np.ndarray], high: Dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for key in ("oracle_patch_tangential_force", "oracle_patch_mean_slip_speed",
                "oracle_patch_stick_ratio"):
        left = np.nan_to_num(free[key].astype(float), nan=0.0)
        right = np.nan_to_num(high[key].astype(float), nan=0.0)
        values.append(np.abs(right - left))
    return np.sqrt(np.mean(np.square(np.stack(values, axis=1)), axis=1))


def classify_verdict(engineering_pass: bool, no_action_pass: bool,
                     mechanism_pass: bool, full: bool,
                     full_gates: Optional[Dict[str, bool]] = None) -> tuple[str, Optional[str]]:
    """Classify probe/full result; public to support synthetic gate tests."""
    if not engineering_pass or not no_action_pass:
        return "PHASE0B_ENGINEERING_BLOCKED", "engineering_or_no_action_gate"
    if not mechanism_pass:
        return "PHASE0B_PROBE_SCIENTIFIC_FAIL", "probe_mechanism_gate"
    if not full:
        return "PHASE0B_PROBE_MECHANISM_COMPLETE", None
    gates = full_gates or {}
    if all(gates.values()):
        return "PHASE0B_SINGLE_PAIR_COMPLETE", None
    if (gates.get("visible", False) and gates.get("full", False)
            and gates.get("com", True) and not gates.get("rigid", False)):
        return ("PHASE0B_SINGLE_PAIR_SCIENTIFIC_FAIL",
                "translation_only_no_deformation_branch")
    return "PHASE0B_SINGLE_PAIR_SCIENTIFIC_FAIL", "full_pair_gate"


def analyze(pair_dir: Path, report_dir: Path) -> Dict[str, Any]:
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
    free = _load_npz(pair_dir / "uniform_low" / "trajectory.npz")
    high = _load_npz(pair_dir / "right_local_high" / "trajectory.npz")
    config = metadata["config"]
    analysis_cfg = config["analysis"]
    phases = free["phase"].astype(str)
    baseline = phases == "no_action"
    post_baseline = ~baseline
    visible_gap = paired_rmse(free["visible_keypoints"], high["visible_keypoints"])
    full_gap = paired_rmse(free["node_positions"], high["node_positions"])
    rigid_gap = rigid_aligned_series(
        free["visible_keypoints"], high["visible_keypoints"])
    free_com = np.mean(free["node_positions"], axis=1)
    high_com = np.mean(high["node_positions"], axis=1)
    com_gap = np.linalg.norm(free_com - high_com, axis=1)
    goal = np.asarray(config["goal"]["center_xy"])
    free_progress = (np.linalg.norm(free_com[0, :2] - goal)
                     - np.linalg.norm(free_com[:, :2] - goal, axis=1))
    high_progress = (np.linalg.norm(high_com[0, :2] - goal)
                     - np.linalg.norm(high_com[:, :2] - goal, axis=1))
    progress_gap = high_progress - free_progress

    formal_gap, formal_meta = {}, {}
    for key in FORMAL_CHANNELS:
        gap = standardized_branch_gap(
            free[key], high[key], baseline, analysis_cfg["std_floor"])
        threshold, onset = onset_from_baseline(
            gap, baseline, analysis_cfg["sigma_multiplier"],
            analysis_cfg["consecutive_samples"])
        formal_gap[key] = gap
        formal_meta[key] = {"threshold": threshold, "onset_index": onset,
                            "onset_phase": _phase_onset_name(phases, onset)}
    fused_free = np.concatenate(
        [free[key].reshape(len(phases), -1) for key in FORMAL_CHANNELS], axis=1)
    fused_high = np.concatenate(
        [high[key].reshape(len(phases), -1) for key in FORMAL_CHANNELS], axis=1)
    fused_gap = standardized_branch_gap(
        fused_free, fused_high, baseline, analysis_cfg["std_floor"])
    fused_threshold, fused_onset = onset_from_baseline(
        fused_gap, baseline, analysis_cfg["sigma_multiplier"],
        analysis_cfg["consecutive_samples"])
    visible_threshold = float(np.mean(visible_gap[baseline])
                              + analysis_cfg["sigma_multiplier"]
                              * np.std(visible_gap[baseline]))
    visible_threshold = max(visible_threshold, analysis_cfg["std_floor"])
    visible_onset = first_sustained_onset(
        visible_gap, visible_threshold, post_baseline,
        analysis_cfg["consecutive_samples"])
    oracle_gap = _oracle_gap(free, high)
    oracle_threshold = float(np.mean(oracle_gap[baseline])
                             + analysis_cfg["sigma_multiplier"]
                             * np.std(oracle_gap[baseline]))
    oracle_threshold = max(oracle_threshold, analysis_cfg["std_floor"])
    oracle_onset = first_sustained_onset(
        oracle_gap, oracle_threshold, post_baseline,
        analysis_cfg["consecutive_samples"])

    edges = np.asarray(metadata["structural_edges"], dtype=np.int64)
    rest = np.asarray(metadata["structural_rest_lengths"], dtype=float)
    free_strain = np.asarray([edge_strain(row, edges, rest)
                              for row in free["node_positions"]])
    high_strain = np.asarray([edge_strain(row, edges, rest)
                              for row in high["node_positions"]])
    all_ratios = np.concatenate([free_strain + 1.0, high_strain + 1.0])
    ratio_min, ratio_max = float(np.min(all_ratios)), float(np.max(all_ratios))
    no_action_visible = visible_gap[baseline]
    floor_min = float(config["floor"]["top_z_m"] - 0.001)
    formal_nonzero = sum(bool(np.any(np.abs(np.concatenate([
        free[key].reshape(-1), high[key].reshape(-1)])) > 0))
        for key in ("joint_motor_torque", "joint_reaction_wrench",
                    "ee_tracking_error_xyz"))
    engineering = {
        "initial_state": metadata["initial_state_max_abs"] <= analysis_cfg["initial_state_max_abs"],
        "fixed_commands": bool(metadata["fixed_command_arrays_equal"]),
        "physics_steps": bool(metadata["physics_step_arrays_equal"]),
        "phases": bool(metadata["phase_arrays_equal"]),
        "formal_finite": _finite(*[free[k] for k in FORMAL_CHANNELS],
                                  *[high[k] for k in FORMAL_CHANNELS]),
        "formal_nonzero_channels": formal_nonzero >= 2,
        "all_finite": _finite(*[v for v in free.values() if v.dtype.kind not in "USO"],
                              *[v for v in high.values() if v.dtype.kind not in "USO"]),
        "node_count": metadata["node_count"] == 72,
        "visible_count": metadata["visible_count"] == 24,
        "execution": not any(metadata["branches"][name]["execution_failures"]
                             for name in metadata["conditions"]),
    }
    no_action = {
        "final_visible": float(no_action_visible[-1]) <= analysis_cfg["no_action_visible_final_max_m"],
        "peak_visible": float(np.max(no_action_visible)) <= analysis_cfg["no_action_visible_peak_max_m"],
        "nodes_finite": _finite(free["node_positions"][baseline], high["node_positions"][baseline]),
        "floor_penetration": (float(np.min(free["node_positions"][:, :, 2])) >= floor_min
                              and float(np.min(high["node_positions"][:, :, 2])) >= floor_min),
        "edge_ratios": (ratio_min >= analysis_cfg["min_structural_edge_ratio"]
                        and ratio_max <= analysis_cfg["max_structural_edge_ratio"]),
    }
    probe_phases = {"probe", "post_probe"}
    oracle_phase = _phase_onset_name(phases, oracle_onset)
    high_probe = np.isin(phases, list(probe_phases))
    high_tangent = float(np.nanmean(high["oracle_patch_tangential_force"][high_probe]))
    free_tangent = float(np.nanmean(free["oracle_patch_tangential_force"][high_probe]))
    high_stick = float(np.nanmean(high["oracle_patch_stick_ratio"][high_probe]))
    free_stick = float(np.nanmean(free["oracle_patch_stick_ratio"][high_probe]))
    mechanism = {
        "oracle_onset_in_probe": oracle_onset is not None and oracle_phase in probe_phases,
        "formal_fused_onset": fused_onset is not None,
        "sensor_leads_visible": (fused_onset is not None
                                 and (visible_onset is None or fused_onset < visible_onset)),
        "high_friction_response": high_tangent > free_tangent or high_stick > free_stick,
    }
    full = not bool(metadata["stop_after_probe"])
    post_probe_indices = np.flatnonzero(phases == "post_probe")
    end_post_probe = int(post_probe_indices[-1])
    amplification = float(visible_gap[-1] / max(visible_gap[end_post_probe], 1e-4))
    full_gates = {
        "visible": float(visible_gap[-1]) >= analysis_cfg["final_visible_rmse_min_m"],
        "full": float(full_gap[-1]) >= analysis_cfg["final_full_rmse_min_m"],
        "rigid": float(rigid_gap[-1]) >= analysis_cfg["final_rigid_aligned_rmse_min_m"],
        "progress": abs(float(progress_gap[-1])) >= analysis_cfg["target_progress_gap_min_m"],
        "com": float(com_gap[-1]) >= analysis_cfg["final_full_rmse_min_m"],
        "amplification": amplification >= analysis_cfg["branch_amplification_min"],
        "edge_ratios": no_action["edge_ratios"],
    }
    verdict, failure_cause = classify_verdict(
        all(engineering.values()), all(no_action.values()), all(mechanism.values()),
        full, full_gates)
    metrics = {
        "verdict": verdict, "failure_cause": failure_cause,
        "setup_diagnostics": {
            "initial_pusher_node_signed_distance_m": metadata[
                "initial_pusher_node_signed_distance"],
            "initial_pusher_clearance_within_2mm": (
                0.0 <= metadata["initial_pusher_node_signed_distance"] <= 0.002),
        },
        "engineering_gate": engineering, "no_action_gate": no_action,
        "mechanism_gate": mechanism, "full_gate": full_gates if full else None,
        "no_action_final_visible_rmse_m": float(no_action_visible[-1]),
        "no_action_peak_visible_rmse_m": float(np.max(no_action_visible)),
        "oracle_friction_load_onset_index": oracle_onset,
        "oracle_friction_load_onset_phase": oracle_phase,
        "fused_formal_sensor_onset_index": fused_onset,
        "fused_formal_sensor_onset_phase": _phase_onset_name(phases, fused_onset),
        "visible_divergence_onset_index": visible_onset,
        "visible_divergence_onset_phase": _phase_onset_name(phases, visible_onset),
        "sensor_to_visual_lead_steps": (None if fused_onset is None or visible_onset is None
                                        else int(visible_onset - fused_onset)),
        "high_patch_tangential_force_probe_mean_n": high_tangent,
        "free_patch_tangential_force_probe_mean_n": free_tangent,
        "high_patch_stick_ratio_probe_mean": high_stick,
        "free_patch_stick_ratio_probe_mean": free_stick,
        "final_visible_rmse_m": float(visible_gap[-1]),
        "final_full_rmse_m": float(full_gap[-1]),
        "final_rigid_aligned_rmse_m": float(rigid_gap[-1]),
        "final_com_gap_m": float(com_gap[-1]),
        "final_target_progress_gap_m": float(progress_gap[-1]),
        "branch_amplification": amplification,
        "structural_edge_ratio_min": ratio_min,
        "structural_edge_ratio_max": ratio_max,
        "formal_channels": formal_meta,
    }
    write_json(report_dir / "metrics.json", metrics)
    np.savez_compressed(
        report_dir / "timeseries.npz", physics_step=free["physics_step"],
        phase=phases, visible_gap=visible_gap, full_gap=full_gap,
        rigid_gap=rigid_gap, com_gap=com_gap, free_com=free_com, high_com=high_com,
        free_progress=free_progress, high_progress=high_progress,
        progress_gap=progress_gap, fused_gap=fused_gap,
        fused_threshold=np.asarray(fused_threshold), oracle_gap=oracle_gap,
        oracle_threshold=np.asarray(oracle_threshold), free_strain=free_strain,
        high_strain=high_strain, **{"formal_" + k: v for k, v in formal_gap.items()})
    summary = ["# Phase 0B Soft BlockPush result", "", "- Verdict: `{}`".format(verdict),
               "- Failure cause: `{}`".format(failure_cause),
               "- Engineering gate: `{}`".format(all(engineering.values())),
               "- No-action gate: `{}`".format(all(no_action.values())),
               "- Mechanism gate: `{}`".format(all(mechanism.values())),
               "- Final visible RMSE: `{:.6f} m`".format(visible_gap[-1]),
               "- Final rigid-aligned RMSE: `{:.6f} m`".format(rigid_gap[-1]), ""]
    (report_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    metrics = analyze(Path(args.pair_dir), Path(args.report_dir))
    print("verdict={}".format(metrics["verdict"]))


if __name__ == "__main__":
    main()
