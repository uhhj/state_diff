"""Analyze R1 local anchoring, policy lead, and test-end deformation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r1.common import (  # noqa: E402
    sustained_max, write_json)
from state_diff.env.block_pushing.policy_rate_metrics import (  # noqa: E402
    formal_feature_scale_floors, onset_with_floor,
    resample_phase_aligned_last, resample_phase_aligned_mean,
    scaled_fused_gap)
from state_diff.env.block_pushing.soft_block_metrics import (  # noqa: E402
    edge_strain, paired_rmse, rigid_aligned_series)


FORMAL_CHANNELS = ("joint_motor_torque", "joint_reaction_wrench",
                   "ee_tracking_error_xyz", "ee_contact_wrench")


def displacement_from_reference(positions: np.ndarray, reference: np.ndarray,
                                indices: np.ndarray) -> np.ndarray:
    """Return mean indexed-node displacement from one common reference."""
    return np.mean(positions[:, indices] - reference[None, indices], axis=1)


def classify_pair(engineering: bool, no_action: bool, hidden_load: bool,
                  local_anchor: bool, policy_lead: bool, full: bool,
                  gates: Optional[Dict[str, bool]] = None) -> tuple[str, Optional[str]]:
    """Classify R1 probe or full result without weakening any gate."""
    if not engineering or not no_action:
        return "PHASE0B_R1_ENGINEERING_BLOCKED", "engineering_or_no_action_gate"
    if not hidden_load or not local_anchor:
        return "PHASE0B_R1_PROBE_LOCAL_ANCHOR_FAIL", "local_anchor_gate"
    if not policy_lead:
        return "PHASE0B_R1_PROBE_POLICY_LEAD_FAIL", "policy_rate_lead_gate"
    if not full:
        return "PHASE0B_R1_PROBE_MECHANISM_COMPLETE", None
    gates = gates or {}
    if all(value for key, value in gates.items() if key != "com"):
        return "PHASE0B_R1_SINGLE_PAIR_COMPLETE", None
    if (gates.get("visible", False) and gates.get("full", False)
            and gates.get("com", False)
            and (not gates.get("rigid", False)
                 or not gates.get("sustained_rigid", False))):
        return ("PHASE0B_R1_SINGLE_PAIR_SCIENTIFIC_FAIL",
                "translation_only_under_compliant_material")
    if (gates.get("rigid", False) and gates.get("sustained_rigid", False)
            and not gates.get("progress", False)):
        return ("PHASE0B_R1_SINGLE_PAIR_SCIENTIFIC_FAIL",
                "deformation_branch_without_task_progress")
    return "PHASE0B_R1_SINGLE_PAIR_SCIENTIFIC_FAIL", "full_pair_gate"


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _formal(data: dict) -> np.ndarray:
    return np.concatenate([data[key].reshape(len(data[key]), -1)
                           for key in FORMAL_CHANNELS], axis=1)


def _phase_name(phases: np.ndarray, onset: Optional[int]) -> Optional[str]:
    return None if onset is None else str(phases[onset])


def analyze_pair(pair_dir: Path, report_dir: Path) -> dict:
    """Compute all prescribed R1 probe/full gates and save small evidence."""
    pair_dir, report_dir = Path(pair_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((pair_dir / "metadata.json").read_text(encoding="utf-8"))
    free = _load(pair_dir / "uniform_low" / "trajectory.npz")
    high = _load(pair_dir / "right_local_high" / "trajectory.npz")
    base = _load(pair_dir / "base_state.npz")
    config, analysis = metadata["config"], metadata["config"]["analysis"]
    phases, steps = free["phase"].astype(str), free["physics_step"]
    baseline = phases == "no_action"
    probe_window = np.isin(phases, ["probe", "post_probe"])
    floors = formal_feature_scale_floors(config)
    free_formal, high_formal = _formal(free), _formal(high)
    physics_gap = scaled_fused_gap(free_formal, high_formal, baseline, floors)
    physics_threshold, physics_onset = onset_with_floor(
        physics_gap, baseline, analysis["physics_sigma_multiplier"],
        analysis["policy_fused_effect_floor"],
        analysis["physics_consecutive_samples"])
    visible_gap = paired_rmse(free["visible_keypoints"], high["visible_keypoints"])
    full_gap = paired_rmse(free["node_positions"], high["node_positions"])
    rigid_gap = rigid_aligned_series(
        free["visible_keypoints"], high["visible_keypoints"])
    visible_threshold, physics_visible_onset = onset_with_floor(
        visible_gap, baseline, analysis["physics_sigma_multiplier"],
        analysis["visible_macro_floor_m"], analysis["physics_consecutive_samples"])

    spp = metadata["steps_per_policy"]
    free_policy = resample_phase_aligned_mean(free_formal, phases, steps, spp)
    high_policy = resample_phase_aligned_mean(high_formal, phases, steps, spp)
    policy_baseline = free_policy.phase == "no_action"
    policy_gap = scaled_fused_gap(
        free_policy.values, high_policy.values, policy_baseline, floors)
    policy_threshold, policy_onset = onset_with_floor(
        policy_gap, policy_baseline, analysis["physics_sigma_multiplier"],
        analysis["policy_fused_effect_floor"],
        analysis["policy_consecutive_samples"])
    free_visual_policy = resample_phase_aligned_last(
        free["visible_keypoints"], phases, steps, spp)
    high_visual_policy = resample_phase_aligned_last(
        high["visible_keypoints"], phases, steps, spp)
    policy_visible_gap = paired_rmse(
        free_visual_policy.values, high_visual_policy.values)
    policy_visible_threshold, policy_visible_onset = onset_with_floor(
        policy_visible_gap, policy_baseline,
        analysis["physics_sigma_multiplier"], analysis["visible_macro_floor_m"],
        analysis["policy_consecutive_samples"])
    policy_lead_samples = (None if policy_onset is None or policy_visible_onset is None
                           else int(policy_visible_onset - policy_onset))

    inside = np.asarray(metadata["initial_patch_bottom_indices"], dtype=np.int64)
    outside = np.asarray(metadata["initial_outside_bottom_indices"], dtype=np.int64)
    reference = base["node_positions"]
    free_inside = displacement_from_reference(
        free["node_positions"], reference, inside)
    high_inside = displacement_from_reference(
        high["node_positions"], reference, inside)
    free_outside = displacement_from_reference(
        free["node_positions"], reference, outside)
    high_outside = displacement_from_reference(
        high["node_positions"], reference, outside)
    free_inside_slip = np.linalg.norm(free_inside[:, :2], axis=1)
    high_inside_slip = np.linalg.norm(high_inside[:, :2], axis=1)
    patch_slip_reduction = free_inside_slip - high_inside_slip
    free_gradient = np.linalg.norm(free_outside - free_inside, axis=1)
    high_gradient = np.linalg.norm(high_outside - high_inside, axis=1)
    local_gradient_advantage = high_gradient - free_gradient
    sustained_slip = sustained_max(
        patch_slip_reduction, probe_window, analysis["physics_consecutive_samples"])
    sustained_gradient = sustained_max(
        local_gradient_advantage, probe_window,
        analysis["physics_consecutive_samples"])
    high_tangent = float(np.mean(
        high["oracle_patch_tangential_force"][probe_window]))
    free_tangent = float(np.mean(
        free["oracle_patch_tangential_force"][probe_window]))

    edges = np.asarray(metadata["structural_edges"], dtype=np.int64)
    rest = np.asarray(metadata["structural_rest_lengths"], dtype=np.float64)
    free_strain = np.asarray([edge_strain(row, edges, rest)
                              for row in free["node_positions"]])
    high_strain = np.asarray([edge_strain(row, edges, rest)
                              for row in high["node_positions"]])
    ratios = np.concatenate([free_strain + 1.0, high_strain + 1.0])
    numeric_free = [value for value in free.values() if value.dtype.kind not in "USO"]
    numeric_high = [value for value in high.values() if value.dtype.kind not in "USO"]
    evaluations = float(np.sum(free["spring_force_evaluation_count"])
                        + np.sum(high["spring_force_evaluation_count"]))
    cap_fraction = float((np.sum(free["spring_capped_force_count"])
                          + np.sum(high["spring_capped_force_count"]))
                         / max(evaluations, 1.0))
    branch_failures = [failure for branch in metadata["branches"].values()
                       for failure in branch["execution_failures"]]
    formal_nonzero = sum(bool(np.any(np.abs(np.concatenate([
        free[key].reshape(-1), high[key].reshape(-1)])) > 0))
        for key in FORMAL_CHANNELS)
    engineering_gate = {
        "initial_state": metadata["initial_state_max_abs"] <=
                         analysis["initial_state_max_abs"],
        "fixed_commands": metadata["fixed_command_arrays_equal"],
        "physics_steps": metadata["physics_step_arrays_equal"],
        "phases": metadata["phase_arrays_equal"],
        "node_count": metadata["node_count"] == 72,
        "visible_count": metadata["visible_count"] == 24,
        "patch_node_count": 2 <= len(inside) <= 6,
        "finite": all(np.all(np.isfinite(value))
                      for value in numeric_free + numeric_high),
        "formal_nonzero": formal_nonzero >= 2,
        "spring_evaluations": evaluations > 0,
        "force_cap": cap_fraction <= analysis["coupon_force_cap_fraction_max"],
        "execution": not branch_failures,
    }
    no_action_visible = visible_gap[baseline]
    no_action_gate = {
        "final_visible": float(no_action_visible[-1]) <=
                         analysis["no_action_visible_final_max_m"],
        "peak_visible": float(np.max(no_action_visible)) <=
                        analysis["no_action_visible_peak_max_m"],
        "edge_ratios": (float(np.min(ratios)) >= analysis["min_structural_edge_ratio"]
                        and float(np.max(ratios)) <= analysis["max_structural_edge_ratio"]),
        "floor": (float(np.min(free["node_positions"][:, :, 2])) >=
                  config["floor"]["top_z_m"] - .001 and
                  float(np.min(high["node_positions"][:, :, 2])) >=
                  config["floor"]["top_z_m"] - .001),
        "no_collapsed_edge": float(np.min(ratios)) > 1e-8,
    }
    hidden_load = high_tangent > free_tangent
    local_anchor = (sustained_slip >= analysis["probe_patch_slip_reduction_min_m"]
                    and sustained_gradient >=
                    analysis["probe_local_gradient_advantage_min_m"])
    policy_lead = (policy_onset is not None
                   and (policy_visible_onset is None
                        or policy_onset + analysis["required_policy_lead_samples"]
                        <= policy_visible_onset))
    full = not metadata["stop_after_probe"]
    test_end = int(np.flatnonzero(phases == "test")[-1]) if full else -1
    post_probe_end = int(np.flatnonzero(phases == "post_probe")[-1])
    test_policy_indices = np.flatnonzero(free_visual_policy.phase == "test")
    sustained_rigid = (float(np.median(rigid_gap[
        [int(np.flatnonzero(steps == free_visual_policy.physics_step_end[i])[-1])
         for i in test_policy_indices[-3:]]])) if full else 0.0)
    free_com = np.mean(free["node_positions"], axis=1)
    high_com = np.mean(high["node_positions"], axis=1)
    com_gap = np.linalg.norm(free_com - high_com, axis=1)
    goal = np.asarray(config["goal"]["center_xy"])
    free_progress = np.linalg.norm(free_com[0, :2] - goal) - np.linalg.norm(
        free_com[:, :2] - goal, axis=1)
    high_progress = np.linalg.norm(high_com[0, :2] - goal) - np.linalg.norm(
        high_com[:, :2] - goal, axis=1)
    progress_gap = high_progress - free_progress
    amplification = (float(visible_gap[test_end] /
                           max(visible_gap[post_probe_end], 1e-4)) if full else 0.0)
    deformation_fraction = (float(rigid_gap[test_end] /
                                  max(visible_gap[test_end], 1e-9)) if full else 0.0)
    full_gates = {
        "visible": full and visible_gap[test_end] >= analysis["test_end_visible_rmse_min_m"],
        "full": full and full_gap[test_end] >= analysis["test_end_full_rmse_min_m"],
        "rigid": full and rigid_gap[test_end] >= analysis["test_end_rigid_aligned_rmse_min_m"],
        "sustained_rigid": full and sustained_rigid >= analysis["sustained_rigid_aligned_rmse_min_m"],
        "deformation_fraction": full and deformation_fraction >= analysis["deformation_fraction_min"],
        "progress": full and abs(progress_gap[test_end]) >= analysis["test_end_target_progress_gap_min_m"],
        "amplification": full and amplification >= analysis["branch_amplification_min"],
        "edge_ratios": no_action_gate["edge_ratios"],
        "force_cap": engineering_gate["force_cap"],
        "probe_mechanism": hidden_load and local_anchor and policy_lead,
        "com": full and com_gap[test_end] >= analysis["test_end_full_rmse_min_m"],
    }
    verdict, cause = classify_pair(
        all(engineering_gate.values()), all(no_action_gate.values()), hidden_load,
        local_anchor, policy_lead, full, full_gates)
    post_test_end = int(np.flatnonzero(phases == "post_test")[-1]) if full else -1
    metrics = {
        "verdict": verdict, "failure_cause": cause,
        "engineering_gate": engineering_gate, "no_action_gate": no_action_gate,
        "hidden_friction_load_gate": hidden_load,
        "local_anchor_gate": local_anchor, "policy_lead_gate": policy_lead,
        "full_gate": full_gates if full else None,
        "initial_patch_node_count": len(inside),
        "no_action_final_visible_rmse_m": float(no_action_visible[-1]),
        "no_action_peak_visible_rmse_m": float(np.max(no_action_visible)),
        "high_patch_tangential_force_probe_mean_n": high_tangent,
        "free_patch_tangential_force_probe_mean_n": free_tangent,
        "sustained_patch_slip_reduction_m": sustained_slip,
        "sustained_local_gradient_advantage_m": sustained_gradient,
        "physics_formal_onset_index": physics_onset,
        "physics_formal_onset_phase": _phase_name(phases, physics_onset),
        "physics_visible_onset_index": physics_visible_onset,
        "physics_visible_onset_phase": _phase_name(phases, physics_visible_onset),
        "policy_formal_onset_index": policy_onset,
        "policy_formal_onset_phase": _phase_name(free_policy.phase, policy_onset),
        "policy_macro_visible_onset_index": policy_visible_onset,
        "policy_macro_visible_onset_phase": _phase_name(
            free_policy.phase, policy_visible_onset),
        "policy_lead_samples": policy_lead_samples,
        "physics_formal_threshold": physics_threshold,
        "physics_visible_threshold_m": visible_threshold,
        "policy_formal_threshold": policy_threshold,
        "policy_visible_threshold_m": policy_visible_threshold,
        "test_end_visible_rmse_m": float(visible_gap[test_end]) if full else None,
        "test_end_full_rmse_m": float(full_gap[test_end]) if full else None,
        "test_end_rigid_aligned_rmse_m": float(rigid_gap[test_end]) if full else None,
        "sustained_rigid_aligned_median_m": sustained_rigid if full else None,
        "peak_test_rigid_aligned_rmse_m": (float(np.max(rigid_gap[phases == "test"]))
                                             if full else None),
        "post_test_end_rigid_aligned_rmse_m": (float(rigid_gap[post_test_end])
                                                 if full else None),
        "deformation_recovery_ratio": (float(rigid_gap[post_test_end] /
            max(np.max(rigid_gap[phases == "test"]), 1e-9)) if full else None),
        "deformation_fraction": deformation_fraction if full else None,
        "test_end_com_gap_m": float(com_gap[test_end]) if full else None,
        "test_end_target_progress_gap_m": float(progress_gap[test_end]) if full else None,
        "branch_amplification": amplification if full else None,
        "structural_edge_ratio_min": float(np.min(ratios)),
        "structural_edge_ratio_max": float(np.max(ratios)),
        "force_cap_fraction": cap_fraction,
    }
    write_json(report_dir / "metrics.json", metrics)
    (report_dir / "summary.md").write_text(
        "# Phase 0B-R1 Pair\n\n- Verdict: `{}`\n- Failure cause: `{}`\n"
        "- Local anchor: `{}`\n- Policy lead: `{}`\n".format(
            verdict, cause, local_anchor, policy_lead), encoding="utf-8")
    np.savez_compressed(
        report_dir / "timeseries.npz", physics_step=steps, phase=phases,
        physics_formal_gap=physics_gap, visible_gap=visible_gap,
        full_gap=full_gap, rigid_gap=rigid_gap,
        policy_step=free_policy.physics_step_end, policy_phase=free_policy.phase,
        policy_formal_gap=policy_gap, policy_visible_gap=policy_visible_gap,
        patch_slip_reduction=patch_slip_reduction,
        local_gradient_advantage=local_gradient_advantage,
        free_progress=free_progress, high_progress=high_progress,
        com_gap=com_gap, free_strain=free_strain, high_strain=high_strain,
        free_energy=free["spring_energy_total_j"],
        high_energy=high["spring_energy_total_j"])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze_pair(
        Path(args.pair_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__":
    main()
