"""Analyze separated R2 axial/shear coupon gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.common import write_json
from state_diff.env.block_pushing.soft_block_metrics import edge_strain, paired_rmse


def classify_coupon(mode: str, engineering: bool, common: bool,
                    peak_primary: float, peak_rigid: float,
                    peak_lateral: float, config: dict) -> str:
    """Classify one R2 coupon without using capped/runaway data as material."""
    prefix = "PHASE0B_R2_" + mode.upper() + "_"
    if not engineering:
        return prefix + "ENGINEERING_BLOCKED"
    if not common:
        return prefix + "UNSTABLE"
    analysis = config["analysis"]
    low = analysis[mode + "_primary_displacement_min_m"]
    high = analysis[mode + "_primary_displacement_max_m"]
    if peak_primary < low or (mode == "shear" and peak_rigid <
                              analysis["shear_peak_rigid_aligned_min_m"]):
        return prefix + "TOO_STIFF"
    if peak_primary > high or (mode == "shear" and peak_rigid >
                               analysis["shear_peak_rigid_aligned_max_m"]):
        return prefix + "TOO_SOFT"
    if mode == "axial" and peak_lateral > analysis["axial_lateral_leakage_max_m"]:
        return prefix + "UNSTABLE"
    return prefix + "COMPLETE"


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _plot(path: Path, x: np.ndarray, values: dict, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, series in values.items(): ax.plot(x, series, label=label)
    ax.set(xlabel="outer step", ylabel=ylabel, title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def analyze(coupon_dir: Path, report_dir: Path) -> dict:
    """Compute R2 coupon engineering/scientific metrics and artifacts."""
    coupon_dir, report_dir = Path(coupon_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((coupon_dir / "metadata.json").read_text(encoding="utf-8"))
    data = _load(coupon_dir / "trajectory.npz")
    config, mode = metadata["config"], metadata["mode"]
    phases, positions = data["phase"].astype(str), data["node_positions"]
    no_action = phases == "no_action"
    loaded = np.isin(phases, ["load_ramp", "load_hold"])
    reference_index = int(np.flatnonzero(no_action)[-1])
    reference = positions[reference_index]
    anchor, load = np.asarray(metadata["anchor_indices"]), np.asarray(metadata["load_indices"])
    direction = np.asarray(config["coupon"][mode + "_load_direction_xyz"], dtype=float)
    direction /= np.linalg.norm(direction)
    relative = (np.mean(positions[:, load] - reference[None, load], axis=1)
                - np.mean(positions[:, anchor] - reference[None, anchor], axis=1))
    primary = relative @ direction
    lateral = np.linalg.norm(relative - primary[:, None] * direction, axis=1)
    rigid = data["rigid_aligned_rmse_m"]
    peak_primary = float(np.max(primary[loaded])); peak_lateral = float(np.max(lateral[loaded]))
    peak_rigid = float(np.max(rigid[loaded]))
    primary_recovery = float(abs(primary[-1]) / max(peak_primary, 1e-9))
    rigid_recovery = float(rigid[-1] / max(peak_rigid, 1e-9))
    cap_fraction = float(np.sum(data["spring_capped_force_count"])
                         / np.sum(data["spring_force_evaluation_count"]))
    edges = np.asarray(metadata["structural_edges"])
    rest = np.asarray(metadata["structural_rest_lengths"])
    strain = np.asarray([edge_strain(row, edges, rest) for row in positions])
    ratios = strain + 1
    visible = data["visible_keypoints"]
    no_action_drift = paired_rmse(
        np.repeat(visible[0:1], len(visible), axis=0), visible)
    anchor_drift = float(np.max(np.linalg.norm(
        positions[:, anchor] - positions[0:1, anchor], axis=2)))
    expected_evals = metadata.get(
        "internal_evaluations_per_microstep", 512
    ) * metadata["microstep_selection"]["microsteps_per_outer"]
    coupon = config["coupon"]
    expected_lengths = {"no_action": coupon["no_action_outer_steps"],
                        "load_ramp": coupon["load_ramp_outer_steps"],
                        "load_hold": coupon["load_hold_outer_steps"],
                        "recovery": coupon["recovery_outer_steps"]}
    engineering_gate = {
        "finite": all(np.all(np.isfinite(value)) for value in data.values()
                      if value.dtype.kind not in "USO"),
        "node_count": metadata["node_count"] == 72,
        "anchor_count": len(anchor) == 12, "load_count": len(load) == 12,
        "static_anchor_mass": all(mass == 0 for mass in metadata["static_anchor_masses"]),
        "dynamic_mass": abs(metadata["dynamic_node_mass"] - .03 / 72) <= 1e-15,
        "no_fixture": metadata["fixture_constraint_count"] == 0,
        "phase_lengths": all(np.sum(phases == key) == value
                             for key, value in expected_lengths.items()),
        "selected_microsteps": np.all(data["microsteps_per_outer"] ==
                                      metadata["microstep_selection"]["microsteps_per_outer"]),
        "no_collapse": float(np.min(data["spring_min_edge_length_m"])) > 1e-12,
        "net_residual": float(np.max(data["spring_net_internal_force_residual_n"])) <=
                        config["analysis"]["net_internal_force_residual_max_n"],
        "force_evaluations": bool(np.all(
            data["spring_force_evaluation_count"] == expected_evals)),
    }
    recovery = max(primary_recovery, rigid_recovery) if mode == "shear" else primary_recovery
    common_gate = {
        "no_action": float(np.max(no_action_drift[no_action])) <=
                     config["analysis"]["free_space_no_action_visible_peak_max_m"],
        "anchor_drift": anchor_drift <= config["analysis"]["anchor_drift_max_m"],
        "force_cap": cap_fraction < config["analysis"]["force_cap_fraction_max"],
        "edge_ratios": (float(np.min(ratios)) >=
                        config["analysis"]["coupon_structural_edge_ratio_min"] and
                        float(np.max(ratios)) <=
                        config["analysis"]["coupon_structural_edge_ratio_max"]),
        "recovery": recovery <= config["analysis"][mode + "_recovery_ratio_max"],
    }
    verdict = classify_coupon(mode, all(engineering_gate.values()),
                              all(common_gate.values()), peak_primary,
                              peak_rigid, peak_lateral, config)
    metrics = {"verdict": verdict, "mode": mode,
               "profile_name": metadata["material_profile"]["profile_name"],
               "engineering_gate": engineering_gate, "common_gate": common_gate,
               "peak_primary_displacement_m": peak_primary,
               "peak_lateral_leakage_m": peak_lateral,
               "peak_rigid_aligned_rmse_m": peak_rigid,
               "primary_recovery_ratio": primary_recovery,
               "rigid_recovery_ratio": rigid_recovery,
               "structural_edge_ratio_min": float(np.min(ratios)),
               "structural_edge_ratio_max": float(np.max(ratios)),
               "force_cap_fraction": cap_fraction,
               "anchor_drift_m": anchor_drift,
               "no_action_visible_peak_drift_m": float(np.max(no_action_drift[no_action])),
               "max_net_internal_force_residual_n": float(np.max(
                   data["spring_net_internal_force_residual_n"])),
               "microsteps_per_outer": metadata["microstep_selection"]["microsteps_per_outer"]}
    write_json(report_dir / "metrics.json", metrics)
    (report_dir / "summary.md").write_text(
        "# R2 {} coupon\n\n- Profile: `{}`\n- Verdict: `{}`\n".format(
            mode, metrics["profile_name"], verdict), encoding="utf-8")
    steps = data["outer_step"]
    _plot(report_dir / "face_displacement.png", steps,
          {"primary": primary, "lateral": lateral}, "displacement (m)", "Face response")
    _plot(report_dir / "rigid_deformation.png", steps,
          {"rigid-aligned": rigid}, "RMSE (m)", "Non-rigid response")
    _plot(report_dir / "strain.png", steps,
          {"strain RMS": np.sqrt(np.mean(strain ** 2, axis=1))}, "strain", "Structural strain")
    _plot(report_dir / "energy.png", steps,
          {"energy": data["spring_energy_total_j"]}, "J", "Spring energy")
    _plot(report_dir / "recovery.png", steps,
          {"primary": primary, "load": data["external_load_scale"]}, "response", "Recovery")
    np.savez_compressed(report_dir / "trajectory_timeseries.npz",
                        primary=primary, lateral=lateral, rigid=rigid,
                        strain=strain, no_action_drift=no_action_drift)
    shutil.copy2(coupon_dir / "coupon.mp4", report_dir / "coupon.mp4")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupon-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(Path(args.coupon_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__":
    main()
