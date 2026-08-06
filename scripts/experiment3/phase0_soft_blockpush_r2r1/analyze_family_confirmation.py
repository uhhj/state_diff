"""Gate A1.5 M8/M16 confirmation without reopening microstep selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_mechanics_validation import (  # noqa: E402
    summarize_cube, summarize_two_node)
from scripts.experiment3.phase0_soft_blockpush_r2r1.common import write_json  # noqa: E402


COMPLETE = "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE"
BLOCKED = "PHASE0B_R2R1_FAMILY_CONFIRMATION_BLOCKED"


def relative_error(a: float, b: float, floor: float) -> float:
    """Return a floor-stabilized relative difference against M16."""
    return float(abs(a - b) / max(abs(b), floor))


def evaluate_confirmation(two: dict, cube: dict, config: dict) -> dict:
    """Evaluate stability, caps, and six M8/M16 convergence observables."""
    errors = {
        "two_peak_extension": relative_error(
            two[8]["peak_extension_m"], two[16]["peak_extension_m"], 1e-6),
        "two_zero_crossing": relative_error(
            two[8]["first_zero_crossing_time_s"] or 0,
            two[16]["first_zero_crossing_time_s"] or 0,
            config["physics"]["outer_timestep_s"] / 16),
        "two_final_energy_ratio": relative_error(
            two[8]["final_energy_ratio"], two[16]["final_energy_ratio"], 1e-6),
        "cube_peak_primary": relative_error(
            cube[8]["peak_primary_face_displacement_m"],
            cube[16]["peak_primary_face_displacement_m"], 1e-6),
        "cube_peak_rigid": relative_error(
            cube[8]["peak_rigid_aligned_rmse_m"],
            cube[16]["peak_rigid_aligned_rmse_m"], 1e-6),
        "cube_recovery": relative_error(
            cube[8]["recovery_ratio"], cube[16]["recovery_ratio"], 1e-6)}
    zero_cap = all(two[m]["gate"]["zero_cap"] and cube[m]["gate"]["zero_cap"]
                   for m in (8, 16))
    aligned_energy = all(two[m].get("energy_measurement") == "post_step_aligned"
                         for m in (8, 16))
    passed = (aligned_energy
              and all(two[m]["stable"] and cube[m]["stable"] for m in (8, 16))
              and zero_cap and all(value <= .05 for value in errors.values()))
    return {"verdict": COMPLETE if passed else BLOCKED,
            "relative_errors": errors, "maximum_relative_error": max(errors.values()),
            "zero_cap": zero_cap, "aligned_energy": aligned_energy,
            "selection_reopened": False,
            "retained_microsteps": 8 if passed else None}


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def analyze(confirmation_dir: Path, report_dir: Path) -> dict:
    """Analyze M8/M16 traces and emit R2-compatible retained selection."""
    confirmation_dir, report_dir = Path(confirmation_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((confirmation_dir / "metadata.json").read_text(
        encoding="utf-8"))
    if metadata["confirmation_microsteps"] != [8, 16]:
        raise ValueError("confirmation traces must be M8/M16 only")
    config = metadata["config"]
    two, cube = {}, {}
    for microsteps in (8, 16):
        two[microsteps] = summarize_two_node(
            _load(confirmation_dir / "two_node_{}.npz".format(microsteps)),
            config, microsteps)
        cube[microsteps] = summarize_cube(
            _load(confirmation_dir / "cube_{}.npz".format(microsteps)),
            config, microsteps)
    decision = evaluate_confirmation(two, cube, config)
    diagnostics = {"m{}".format(m): {
        "increase_count": two[m]["energy_increase_count"],
        "increase_fraction": two[m]["energy_increase_fraction"],
        "cumulative_positive_injection_ratio":
            two[m]["cumulative_positive_energy_injection_ratio"],
        "max_positive_increment_ratio":
            two[m]["max_positive_energy_increment_ratio"],
        "max_total_energy_ratio": two[m]["max_total_energy_ratio"]}
        for m in (8, 16)}
    summary = dict(decision, profile=metadata["material_profile"],
                   two_node=two, cube=cube,
                   energy_diagnostics=diagnostics,
                   prior_r2_selection=metadata["prior_r2_selection"])
    write_json(report_dir / "summary.json", summary)
    selection = {
        "verdict": ("PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE"
                    if decision["verdict"] == COMPLETE
                    else "PHASE0B_R2_MICROSTEP_VALIDATION_BLOCKED"),
        "confirmation_verdict": decision["verdict"],
        "outer_timestep_s": config["physics"]["outer_timestep_s"],
        "microsteps_per_outer": 8 if decision["verdict"] == COMPLETE else None,
        "micro_timestep_s": (config["physics"]["outer_timestep_s"] / 8
                             if decision["verdict"] == COMPLETE else None),
        "selection_reopened": False,
        "prior_selection_evidence": config["prior_r2_selection_path"],
        "family_confirmation_evidence": "mechanics_validation/summary.json"}
    write_json(report_dir / "selected_microsteps.json", selection)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(decision["relative_errors"])
    ax.bar(range(len(labels)), [decision["relative_errors"][key] for key in labels])
    ax.axhline(.05, color="red", linestyle="--", label="5% gate")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set(ylabel="relative error", title="A1.5 M8/M16 family confirmation")
    ax.legend(); fig.tight_layout(); fig.savefig(report_dir / "convergence.png", dpi=150)
    plt.close(fig)
    return summary


def main() -> None:
    """Run the family confirmation analyzer CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(
        Path(args.confirmation_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__":
    main()
