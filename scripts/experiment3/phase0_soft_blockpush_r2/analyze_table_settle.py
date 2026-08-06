"""Analyze table-settle gates and freeze an independently passed R2 profile."""
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


COMPLETE = "PHASE0B_R2_TABLE_SETTLE_COMPLETE"


def classify_table(engineering: bool, scientific: bool) -> str:
    """Keep malformed telemetry distinct from a finite unstable settle."""
    if not engineering:
        return "PHASE0B_R2_TABLE_SETTLE_ENGINEERING_BLOCKED"
    if not scientific:
        return "PHASE0B_R2_TABLE_SETTLE_UNSTABLE"
    return COMPLETE


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _plot(path: Path, x: np.ndarray, values: dict, ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, series in values.items():
        ax.plot(x, series, label=label)
    ax.set(xlabel="outer step", ylabel=ylabel, title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def freeze_outputs(report_dir: Path, metadata: dict, metrics: dict) -> dict:
    """Write freeze artifacts only after all four independent verdicts pass."""
    if metrics["verdict"] != COMPLETE:
        return {}
    config = metadata["config"]
    root = Path(config["report_root"])
    profile = metadata["material_profile"]
    profile_name = profile["profile_name"]
    mechanics_path = root / "mechanics_validation" / "selected_microsteps.json"
    axial_path = root / "coupon" / profile_name / "axial" / "metrics.json"
    shear_path = root / "coupon" / profile_name / "shear" / "metrics.json"
    required = (mechanics_path, axial_path, shear_path)
    if not all(path.is_file() for path in required):
        return {}
    mechanics, axial, shear = [json.loads(path.read_text(encoding="utf-8"))
                               for path in required]
    verdicts_ok = (
        mechanics.get("verdict") == "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE" and
        axial.get("verdict") == "PHASE0B_R2_AXIAL_COMPLETE" and
        shear.get("verdict") == "PHASE0B_R2_SHEAR_COMPLETE")
    if not verdicts_ok:
        return {}
    frozen_dir = root / "frozen_material"
    frozen_dir.mkdir(parents=True, exist_ok=True)
    material_path = frozen_dir / "frozen_material.json"
    microsteps_path = frozen_dir / "frozen_microsteps.json"
    manifest_path = frozen_dir / "calibration_manifest.json"
    write_json(material_path, profile)
    write_json(microsteps_path, {
        "outer_timestep_s": config["physics"]["outer_timestep_s"],
        "microsteps_per_outer": mechanics["microsteps_per_outer"],
        "micro_timestep_s": (config["physics"]["outer_timestep_s"] /
                              mechanics["microsteps_per_outer"]),
        "selection_evidence": "mechanics_validation/selected_microsteps.json"})
    write_json(manifest_path, {
        "verdict": "PHASE0B_R2_MATERIAL_CALIBRATION_COMPLETE",
        "profile_name": profile_name,
        "mechanics_validation": str(mechanics_path),
        "axial_coupon": str(axial_path), "shear_coupon": str(shear_path),
        "table_settle": str(report_dir / "metrics.json"),
        "pair_executed": False, "training_executed": False})
    return {"material": str(material_path), "microsteps": str(microsteps_path),
            "manifest": str(manifest_path)}


def analyze(settle_dir: Path, report_dir: Path) -> dict:
    """Compute audit-only drift, contact, strain, and stability gates."""
    settle_dir, report_dir = Path(settle_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((settle_dir / "metadata.json").read_text(encoding="utf-8"))
    data = _load(settle_dir / "trajectory.npz")
    config = metadata["config"]
    phases = data["phase"].astype(str)
    audit_indices = np.flatnonzero(phases == "audit")
    positions, velocities = data["node_positions"], data["node_velocities"]
    visible = data["visible_keypoints"]
    reference_index = int(audit_indices[0])
    audit_visible = visible[audit_indices]
    visible_drift = paired_rmse(
        np.repeat(visible[reference_index:reference_index + 1], len(audit_indices), axis=0),
        audit_visible)
    com = np.mean(positions, axis=1)
    com_drift = np.linalg.norm(com[audit_indices] - com[reference_index], axis=1)
    node_speed = np.linalg.norm(velocities, axis=2)
    edges = np.asarray(metadata["structural_edges"])
    rest = np.asarray(metadata["structural_rest_lengths"])
    ratios = np.asarray([edge_strain(row, edges, rest) + 1 for row in positions])
    radius = float(config["soft_block"]["node_radius_m"])
    floor_top = float(config["table_settle"]["floor_top_z_m"])
    penetration = np.maximum(0.0, floor_top - (np.min(positions[:, :, 2], axis=1) - radius))
    cap_fraction = float(np.sum(data["spring_capped_force_count"])
                         / np.sum(data["spring_force_evaluation_count"]))
    settle = config["table_settle"]
    expected_evals = 512 * metadata["microstep_selection"]["microsteps_per_outer"]
    engineering = {
        "finite": all(np.all(np.isfinite(value)) for value in data.values()
                      if value.dtype.kind not in "USO"),
        "node_count": metadata["node_count"] == 72,
        "all_nodes_dynamic": bool(metadata["all_nodes_dynamic"]),
        "one_floor": metadata["floor_body_count"] == 1,
        "no_fixture": metadata["fixture_constraint_count"] == 0,
        "no_external_load": not metadata["external_load"],
        "no_recenter": metadata["recenter_calls"] == 0,
        "no_velocity_reset": metadata["velocity_reset_calls"] == 0,
        "phase_lengths": (np.sum(phases == "warmup") == settle["warmup_outer_steps"] and
                          len(audit_indices) == settle["audit_outer_steps"]),
        "force_evaluations": bool(np.all(
            data["spring_force_evaluation_count"] == expected_evals)),
        "net_residual": float(np.max(data["spring_net_internal_force_residual_n"])) <=
                        config["analysis"]["net_internal_force_residual_max_n"]}
    analysis = config["analysis"]
    scientific = {
        "force_cap": cap_fraction < analysis["force_cap_fraction_max"],
        "audit_visible_drift": float(np.max(visible_drift)) <=
                               analysis["table_audit_visible_drift_max_m"],
        "final_speed": float(np.max(node_speed[-1])) <=
                       analysis["table_final_max_node_speed_mps"],
        "edge_ratios": (float(np.min(ratios[audit_indices])) >=
                        analysis["table_structural_edge_ratio_min"] and
                        float(np.max(ratios[audit_indices])) <=
                        analysis["table_structural_edge_ratio_max"]),
        "floor_penetration": float(np.max(penetration[audit_indices])) <=
                             analysis["table_floor_penetration_max_m"]}
    verdict = classify_table(all(engineering.values()), all(scientific.values()))
    metrics = {
        "verdict": verdict, "profile_name": metadata["material_profile"]["profile_name"],
        "engineering_gate": engineering, "scientific_gate": scientific,
        "audit_visible_peak_drift_m": float(np.max(visible_drift)),
        "audit_com_peak_drift_m": float(np.max(com_drift)),
        "audit_max_node_speed_mps": float(np.max(node_speed[audit_indices])),
        "audit_mean_node_speed_mps": float(np.mean(node_speed[audit_indices])),
        "final_max_node_speed_mps": float(np.max(node_speed[-1])),
        "structural_edge_ratio_min": float(np.min(ratios[audit_indices])),
        "structural_edge_ratio_max": float(np.max(ratios[audit_indices])),
        "floor_penetration_max_m": float(np.max(penetration[audit_indices])),
        "force_cap_fraction": cap_fraction,
        "max_net_internal_force_residual_n": float(np.max(
            data["spring_net_internal_force_residual_n"])),
        "audit_contact_count_min": int(np.min(data["contact_count"][audit_indices])),
        "audit_contact_count_max": int(np.max(data["contact_count"][audit_indices])),
        "microsteps_per_outer": metadata["microstep_selection"]["microsteps_per_outer"]}
    write_json(report_dir / "metrics.json", metrics)
    metrics["frozen_outputs"] = freeze_outputs(report_dir, metadata, metrics)
    write_json(report_dir / "metrics.json", metrics)
    (report_dir / "summary.md").write_text(
        "# R2 table settle\n\n- Profile: `{}`\n- Verdict: `{}`\n".format(
            metrics["profile_name"], verdict), encoding="utf-8")
    steps = data["outer_step"][audit_indices]
    _plot(report_dir / "drift.png", steps,
          {"visible": visible_drift, "COM": com_drift}, "drift (m)", "Audit drift")
    _plot(report_dir / "speed.png", steps,
          {"max": np.max(node_speed[audit_indices], axis=1),
           "mean": np.mean(node_speed[audit_indices], axis=1)}, "m/s", "Audit node speed")
    _plot(report_dir / "energy.png", data["outer_step"],
          {"energy": data["spring_energy_total_j"]}, "J", "Spring energy")
    _plot(report_dir / "contact.png", data["outer_step"],
          {"contacts": data["contact_count"]}, "count", "Floor contacts")
    np.savez_compressed(report_dir / "audit_timeseries.npz",
                        visible_drift=visible_drift, com_drift=com_drift,
                        speed=node_speed[audit_indices],
                        edge_ratios=ratios[audit_indices],
                        floor_penetration=penetration[audit_indices])
    shutil.copy2(settle_dir / "table_settle.mp4", report_dir / "table_settle.mp4")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settle-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(Path(args.settle_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__":
    main()
