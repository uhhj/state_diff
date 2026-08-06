"""Analyze material-coupon compliance, stability, cap use, and recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r1.common import write_json  # noqa: E402
from state_diff.env.block_pushing.soft_block_metrics import (  # noqa: E402
    edge_strain, paired_rmse, rigid_aligned_rmse)


def classify_coupon(engineering_pass: bool, peak_rigid: float,
                    peak_face: float, metrics: Dict[str, float],
                    config: dict) -> str:
    """Classify coupon outcome using only the frozen material gates."""
    if not engineering_pass:
        return "PHASE0B_R1_COUPON_ENGINEERING_BLOCKED"
    analysis = config["analysis"]
    if (peak_rigid < analysis["coupon_peak_rigid_min_m"]
            or peak_face < analysis["coupon_face_relative_min_m"]):
        return "PHASE0B_R1_COUPON_TOO_RIGID"
    unstable = (
        peak_rigid > analysis["coupon_peak_rigid_max_m"]
        or peak_face > analysis["coupon_face_relative_max_m"]
        or metrics["recovery_ratio"] > analysis["coupon_recovery_ratio_max"]
        or metrics["force_cap_fraction"] > analysis["coupon_force_cap_fraction_max"]
        or metrics["edge_ratio_min"] < analysis["min_structural_edge_ratio"]
        or metrics["edge_ratio_max"] > analysis["max_structural_edge_ratio"]
        or metrics["no_action_visible_peak_m"] > .0005)
    return ("PHASE0B_R1_COUPON_UNSTABLE" if unstable
            else "PHASE0B_R1_COUPON_COMPLETE")


def _plot(path: Path, x: np.ndarray, values: dict, ylabel: str,
          title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, series in values.items():
        ax.plot(x, series, label=label)
    ax.set(xlabel="physics step", ylabel=ylabel, title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def analyze_coupon(coupon_dir: Path, report_dir: Path) -> dict:
    """Compute coupon gates, write evidence, and return metrics."""
    coupon_dir, report_dir = Path(coupon_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((coupon_dir / "metadata.json").read_text(encoding="utf-8"))
    with np.load(coupon_dir / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    config, phases = metadata["config"], data["phase"].astype(str)
    positions, visible = data["node_positions"], data["visible_keypoints"]
    no_action = phases == "no_action"
    loaded = np.isin(phases, ["load_ramp", "load_hold"])
    reference_index = int(np.flatnonzero(no_action)[-1])
    reference = positions[reference_index]
    rigid = np.asarray([rigid_aligned_rmse(reference, row) for row in positions])
    anchor = np.asarray(metadata["anchor_indices"], dtype=np.int64)
    load = np.asarray(metadata["load_indices"], dtype=np.int64)
    loaded_delta = np.mean(positions[:, load] - reference[None, load], axis=1)
    anchor_delta = np.mean(positions[:, anchor] - reference[None, anchor], axis=1)
    face_relative = np.linalg.norm(loaded_delta - anchor_delta, axis=1)
    structural_edges = np.asarray(metadata["structural_edges"], dtype=np.int64)
    structural_rest = np.asarray(metadata["structural_rest_lengths"])
    shear_edges = np.asarray(metadata["shear_edges"], dtype=np.int64)
    shear_rest = np.asarray(metadata["shear_rest_lengths"])
    structural = np.asarray([edge_strain(row, structural_edges, structural_rest)
                             for row in positions])
    shear = np.asarray([edge_strain(row, shear_edges, shear_rest)
                        for row in positions])
    structural_rms = np.sqrt(np.mean(np.square(structural), axis=1))
    shear_rms = np.sqrt(np.mean(np.square(shear), axis=1))
    ratios = structural + 1.0
    visible_drift = paired_rmse(
        np.repeat(visible[0:1], len(visible), axis=0), visible)
    evaluations = float(np.sum(data["spring_force_evaluation_count"]))
    cap_fraction = float(np.sum(data["spring_capped_force_count"]) /
                         max(evaluations, 1.0))
    peak_rigid = float(np.max(rigid[loaded]))
    peak_face = float(np.max(face_relative[loaded]))
    metrics_core = {
        "recovery_ratio": float(rigid[-1] / max(peak_rigid, 1e-9)),
        "force_cap_fraction": cap_fraction,
        "edge_ratio_min": float(np.min(ratios)),
        "edge_ratio_max": float(np.max(ratios)),
        "no_action_visible_peak_m": float(np.max(visible_drift[no_action])),
    }
    expected = config["coupon"]
    expected_lengths = {
        "no_action": expected["no_action_steps"],
        "load_ramp": expected["load_ramp_steps"],
        "load_hold": expected["load_hold_steps"],
        "recovery": expected["recovery_steps"]}
    engineering = {
        "finite": all(np.all(np.isfinite(value)) for value in data.values()
                      if value.dtype.kind not in "USO"),
        "node_count": metadata["node_count"] == 72,
        "anchor_count": len(anchor) == 12,
        "load_count": len(load) == 12,
        "phase_lengths": all(int(np.sum(phases == name)) == count
                             for name, count in expected_lengths.items()),
        "no_collapsed_edge": float(np.min(ratios)) > 1e-8,
        "floor": float(np.min(positions[:, :, 2])) >=
                 config["floor"]["top_z_m"] - .001,
        "force_evaluations": evaluations > 0,
        "no_internal_constraints": metadata["internal_constraint_count"] == 0,
    }
    verdict = classify_coupon(
        all(engineering.values()), peak_rigid, peak_face, metrics_core, config)
    metrics = {
        "verdict": verdict, "engineering_gate": engineering,
        "profile_name": metadata["material_profile"]["profile_name"],
        "material_profile": metadata["material_profile"],
        "material_profile_path": metadata["material_profile_path"],
        "peak_rigid_aligned_rmse_m": peak_rigid,
        "peak_face_relative_displacement_m": peak_face,
        "peak_structural_strain_rms": float(np.max(structural_rms[loaded])),
        "peak_shear_strain_rms": float(np.max(shear_rms[loaded])),
        "com_drift_peak_m": float(np.max(np.linalg.norm(
            np.mean(positions, axis=1) - np.mean(reference, axis=0), axis=1))),
        **metrics_core,
    }
    write_json(report_dir / "metrics.json", metrics)
    (report_dir / "summary.md").write_text(
        "# R1 material coupon\n\n- Profile: `{}`\n- Verdict: `{}`\n"
        "- Peak rigid-aligned RMSE: `{:.6f} m`\n"
        "- Peak face-relative displacement: `{:.6f} m`\n"
        "- Recovery ratio: `{:.6f}`\n".format(
            metrics["profile_name"], verdict, peak_rigid, peak_face,
            metrics["recovery_ratio"]), encoding="utf-8")
    steps = data["physics_step"]
    _plot(report_dir / "rigid_deformation.png", steps,
          {"rigid-aligned": rigid}, "RMSE (m)", "Coupon deformation")
    _plot(report_dir / "face_displacement.png", steps,
          {"face relative": face_relative}, "displacement (m)",
          "Loaded versus anchored face")
    _plot(report_dir / "strain_energy.png", steps,
          {"structural strain RMS": structural_rms,
           "shear strain RMS": shear_rms,
           "spring energy": data["spring_energy_total_j"]},
          "strain / energy (J)", "Strain and spring energy")
    _plot(report_dir / "recovery.png", steps,
          {"rigid-aligned": rigid, "load scale": data["external_load_scale"]},
          "deformation / load scale", "Loading and recovery")
    np.savez_compressed(report_dir / "timeseries.npz", rigid=rigid,
                        face_relative=face_relative,
                        structural_rms=structural_rms, shear_rms=shear_rms,
                        visible_drift=visible_drift)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupon-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    metrics = analyze_coupon(Path(args.coupon_dir), Path(args.report_dir))
    print("verdict={}".format(metrics["verdict"]))


if __name__ == "__main__":
    main()
