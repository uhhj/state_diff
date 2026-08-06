"""Analyze the fixed A070 M8/M16 angle-cell confirmation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r3.common import write_json  # noqa: E402

BLOCKED = "PHASE0B_R3_ANGLE_VALIDATION_BLOCKED"
COMPLETE = "PHASE0B_R3_ANGLE_VALIDATION_COMPLETE"


def first_zero_crossing(error: np.ndarray, timestep: float) -> float:
    indices = np.flatnonzero(error[:-1] * error[1:] <= 0)
    return float("inf") if len(indices) == 0 else float((indices[0] + 1) * timestep)


def summarize(path: Path) -> dict:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    with np.load(path / "trajectory.npz", allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    config = metadata["config"]
    initial = float(config["angle_validation"]["initial_cosine_error"])
    arm = float(config["soft_block"]["spacing_m"][0])
    return {
        "finite": all(np.all(np.isfinite(v)) for v in data.values()),
        "zero_cap": int(np.sum(data["cap_count"])) == 0,
        "arm_ratio_ok": (float(np.min(data["arm_lengths"] / arm)) >= .98 and
                         float(np.max(data["arm_lengths"] / arm)) <= 1.02),
        "final_error_ratio": float(abs(data["cosine_error"][-1]) / abs(initial)),
        "peak_force_n": float(np.max(data["max_force"])),
        "first_zero_crossing_s": first_zero_crossing(
            data["cosine_error"], float(config["physics"]["outer_timestep_s"])),
        "microsteps_per_outer": metadata["microsteps_per_outer"]}


def _relative(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def analyze(data_root: Path, report_root: Path) -> dict:
    report_root.mkdir(parents=True, exist_ok=True)
    results = {"M8": summarize(data_root / "M8"),
               "M16": summarize(data_root / "M16")}
    config = json.loads((data_root / "M8" / "metadata.json").read_text(
        encoding="utf-8"))["config"]
    limit = float(config["angle_validation"]["convergence_relative_tolerance"])
    convergence = {key: _relative(results["M8"][key], results["M16"][key]) <= limit
                   for key in ("peak_force_n", "first_zero_crossing_s",
                               "final_error_ratio")}
    per_run = all(all(row[key] for key in ("finite", "zero_cap", "arm_ratio_ok"))
                  and row["final_error_ratio"] <=
                  config["angle_validation"]["final_error_ratio_max"]
                  for row in results.values())
    verdict = COMPLETE if per_run and all(convergence.values()) else BLOCKED
    metrics = {"verdict": verdict, "runs": results, "convergence": convergence,
               "selected_microsteps": 8}
    write_json(report_root / "metrics.json", metrics)
    write_json(report_root / "selected_microsteps.json", {
        "verdict": "PHASE0B_R2_MICROSTEP_VALIDATION_COMPLETE",
        "confirmation_verdict": verdict, "microsteps_per_outer": 8,
        "selection_reopened": False,
        "source": "R3 A070 three-node angle validation"})
    (report_root / "summary.md").write_text(
        "# R3 angle validation\n\n- Verdict: `{}`\n- Selected: `M8`\n".format(verdict),
        encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--report-root", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(Path(args.data_root), Path(args.report_root))["verdict"]))


if __name__ == "__main__":
    main()
