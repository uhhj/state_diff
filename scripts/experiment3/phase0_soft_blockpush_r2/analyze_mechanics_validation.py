"""Gate microstep stability/convergence and select the smallest valid M."""
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

from scripts.experiment3.phase0_soft_blockpush_r2.common import write_json


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _aligned_energy_series(data: dict) -> np.ndarray:
    """Return post-step aligned total energy.

    New traces use total_energy with aligned semantics. Older
    synthetic/legacy fixtures remain readable.
    """
    if "aligned_total_energy" in data:
        return np.asarray(data["aligned_total_energy"], dtype=np.float64)
    return np.asarray(data["total_energy"], dtype=np.float64)


def summarize_two_node(data: dict, config: dict, microsteps: int) -> dict:
    """Summarize oscillator stability and convergence observables."""
    settings = config["mechanics_validation"]
    extension = np.asarray(data["extension"], dtype=np.float64)
    energy = _aligned_energy_series(data)
    energy_scale = max(float(energy[0]), 1e-30)
    increment_tolerance = max(float(energy[0]) * 1e-10, 1e-15)
    energy_delta = np.diff(energy)
    positive_mask = energy_delta > increment_tolerance
    positive_delta = np.where(positive_mask, energy_delta, 0.0)
    energy_increase_count = int(np.count_nonzero(positive_mask))
    energy_increase_fraction = float(
        energy_increase_count / max(len(energy_delta), 1))
    cumulative_positive_injection_ratio = float(
        np.sum(positive_delta) / energy_scale)
    max_positive_increment_ratio = float(
        np.max(positive_delta, initial=0.0) / energy_scale)
    max_total_energy_ratio = float(np.max(energy) / energy_scale)
    crossing = np.flatnonzero(np.signbit(extension) != np.signbit(extension[0]))
    crossing_time = (float((crossing[0] + 1) * config["physics"]["outer_timestep_s"])
                     if len(crossing) else None)
    ratio = data["edge_length"] / config["soft_block"]["spacing_m"][0]
    gate = {
        "finite": all(np.all(np.isfinite(value)) for value in data.values()),
        "zero_cap": int(np.sum(data["capped_force_count"])) == 0,
        "edge_ratio": (float(np.min(ratio)) >= settings["edge_ratio_min"] and
                       float(np.max(ratio)) <= settings["edge_ratio_max"]),
        "final_energy": float(energy[-1] / max(energy[0], 1e-30)) <=
                        settings["final_energy_ratio_max"],
        "energy_increase": energy_increase_fraction <=
                           settings["energy_increase_fraction_max"],
        "zero_crossing": crossing_time is not None,
    }
    return {"microsteps": microsteps, "gate": gate,
            "stable": all(gate.values()),
            "peak_extension_m": float(np.max(np.abs(extension))),
            "final_energy_ratio": float(energy[-1] / max(energy[0], 1e-30)),
            "first_zero_crossing_time_s": crossing_time,
            "edge_ratio_min": float(np.min(ratio)),
            "edge_ratio_max": float(np.max(ratio)),
            "energy_measurement": "post_step_aligned",
            "energy_increase_count": energy_increase_count,
            "energy_increase_fraction": energy_increase_fraction,
            "cumulative_positive_energy_injection_ratio":
                cumulative_positive_injection_ratio,
            "max_positive_energy_increment_ratio": max_positive_increment_ratio,
            "max_total_energy_ratio": max_total_energy_ratio}


def summarize_cube(data: dict, config: dict, microsteps: int) -> dict:
    """Summarize cube load response and bounded recovery."""
    settings = config["mechanics_validation"]
    phases = data["phase"].astype(str)
    loaded = np.isin(phases, ["load_ramp", "load_hold"])
    peak_primary = float(np.max(np.abs(data["primary_face_displacement"][loaded])))
    peak_rigid = float(np.max(data["rigid_rmse"][loaded]))
    recovery = float(abs(data["primary_face_displacement"][-1]) /
                     max(peak_primary, 1e-9))
    gate = {
        "finite": all(np.all(np.isfinite(value)) for value in data.values()
                      if value.dtype.kind not in "USO"),
        "zero_cap": int(np.sum(data["capped_force_count"])) == 0,
        "edge_ratio": (float(np.min(data["edge_ratio_min"])) >=
                       settings["edge_ratio_min"] and
                       float(np.max(data["edge_ratio_max"])) <=
                       settings["edge_ratio_max"]),
        "recovery_bounded": np.isfinite(recovery) and recovery <= 1.0,
        "no_collapse": float(np.min(data["edge_ratio_min"])) > 0,
        "net_residual": float(np.max(data["net_internal_force_residual"])) <=
                        config["analysis"]["net_internal_force_residual_max_n"],
    }
    return {"microsteps": microsteps, "gate": gate,
            "stable": all(gate.values()),
            "peak_primary_face_displacement_m": peak_primary,
            "peak_rigid_aligned_rmse_m": peak_rigid,
            "recovery_ratio": recovery,
            "edge_ratio_min": float(np.min(data["edge_ratio_min"])),
            "edge_ratio_max": float(np.max(data["edge_ratio_max"]))}


def _relative(a: float, b: float, floor: float) -> float:
    return float(abs(a - b) / max(abs(b), floor))


def select_smallest_converged(two: Dict[int, dict], cube: Dict[int, dict],
                              config: dict) -> tuple[Optional[int], dict]:
    """Select the smallest stable M whose M→2M metrics converge."""
    tolerance = config["mechanics_validation"]["convergence_relative_tolerance"]
    comparisons = {}
    for first, second in ((4, 8), (8, 16), (16, 32)):
        errors = {
            "two_peak_extension": _relative(
                two[first]["peak_extension_m"], two[second]["peak_extension_m"], 1e-6),
            "two_final_energy_ratio": _relative(
                two[first]["final_energy_ratio"], two[second]["final_energy_ratio"], 1e-6),
            "two_zero_crossing": _relative(
                two[first]["first_zero_crossing_time_s"] or 0,
                two[second]["first_zero_crossing_time_s"] or 0,
                config["physics"]["outer_timestep_s"] / second),
            "cube_peak_primary": _relative(
                cube[first]["peak_primary_face_displacement_m"],
                cube[second]["peak_primary_face_displacement_m"], 1e-6),
            "cube_peak_rigid": _relative(
                cube[first]["peak_rigid_aligned_rmse_m"],
                cube[second]["peak_rigid_aligned_rmse_m"], 1e-6),
            "cube_recovery": _relative(
                cube[first]["recovery_ratio"], cube[second]["recovery_ratio"], 1e-6),
        }
        passed = (two[first]["stable"] and two[second]["stable"]
                  and cube[first]["stable"] and cube[second]["stable"]
                  and all(value <= tolerance for value in errors.values()))
        comparisons["{}_vs_{}".format(first, second)] = {
            "relative_errors": errors, "passed": passed}
        if passed:
            return first, comparisons
    return None, comparisons


def analyze(validation_dir: Path, report_dir: Path) -> dict:
    """Analyze all candidate rollouts and write the sole selection artifact."""
    validation_dir, report_dir = Path(validation_dir), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((validation_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]; two, cube = {}, {}
    for microsteps in metadata["microstep_candidates"]:
        two[microsteps] = summarize_two_node(
            _load(validation_dir / "two_node_{}.npz".format(microsteps)),
            config, microsteps)
        cube[microsteps] = summarize_cube(
            _load(validation_dir / "cube_{}.npz".format(microsteps)),
            config, microsteps)
    selected, comparisons = select_smallest_converged(two, cube, config)
    verdict = ("PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE" if selected is not None
               else "PHASE0B_R2_MICROSTEP_VALIDATION_BLOCKED")
    summary = {"verdict": verdict, "profile": metadata["material_profile"],
               "two_node": two, "cube": cube, "comparisons": comparisons,
               "selected_microsteps": selected}
    write_json(report_dir / "summary.json", summary)
    selection = {"verdict": verdict,
                 "outer_timestep_s": config["physics"]["outer_timestep_s"],
                 "microsteps_per_outer": selected,
                 "micro_timestep_s": (None if selected is None else
                    config["physics"]["outer_timestep_s"] / selected),
                 "selection_evidence": "mechanics_validation/summary.json"}
    write_json(report_dir / "selected_microsteps.json", selection)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ms = metadata["microstep_candidates"]
    ax.plot(ms, [two[m]["final_energy_ratio"] for m in ms], "o-", label="energy ratio")
    ax.plot(ms, [cube[m]["recovery_ratio"] for m in ms], "o-", label="cube recovery")
    ax.set(xlabel="microsteps per outer", title="Microstep convergence")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(report_dir / "convergence.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for microsteps in ms:
        series = _load(validation_dir / "two_node_{}.npz".format(microsteps))
        ax.plot(series["outer_step"], series["total_energy"],
                label="M={}".format(microsteps))
    ax.set(xlabel="outer step", ylabel="J", title="Two-node total energy")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(report_dir / "energy.png", dpi=150); plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(
        Path(args.validation_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__":
    main()
