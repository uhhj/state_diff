"""Reclassify R2 coupon metrics for decoupled shear calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_material_coupon import (  # noqa: E402
    analyze as analyze_r2_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2.common import write_json  # noqa: E402


def classify_r2r2(metrics: dict, config: dict) -> str:
    """Classify finite material softness before loaded edge overflow."""
    mode = metrics["mode"]
    prefix = "PHASE0B_R2_" + mode.upper() + "_"
    if not all(metrics["engineering_gate"].values()):
        return prefix + "ENGINEERING_BLOCKED"
    common = metrics["common_gate"]
    baseline_and_numerics = (common["no_action"] and common["anchor_drift"]
                             and common["force_cap"] and common["recovery"])
    if not baseline_and_numerics:
        return prefix + "UNSTABLE"
    analysis = config["analysis"]
    peak_primary = float(metrics["peak_primary_displacement_m"])
    low = float(analysis[mode + "_primary_displacement_min_m"])
    high = float(analysis[mode + "_primary_displacement_max_m"])
    if mode == "shear":
        peak_rigid = float(metrics["peak_rigid_aligned_rmse_m"])
        rigid_low = float(analysis["shear_peak_rigid_aligned_min_m"])
        rigid_high = float(analysis["shear_peak_rigid_aligned_max_m"])
        if peak_primary > high or peak_rigid > rigid_high:
            return prefix + "TOO_SOFT"
        if peak_primary < low or peak_rigid < rigid_low:
            return prefix + "TOO_STIFF"
    else:
        if peak_primary > high:
            return prefix + "TOO_SOFT"
        if peak_primary < low:
            return prefix + "TOO_STIFF"
        if metrics["peak_lateral_leakage_m"] > analysis["axial_lateral_leakage_max_m"]:
            return prefix + "UNSTABLE"
    if not common["edge_ratios"]:
        return prefix + "UNSTABLE"
    return prefix + "COMPLETE"


def analyze(coupon_dir: Path, report_dir: Path) -> dict:
    """Run existing analysis and update only the verdict semantics."""
    metrics = analyze_r2_coupon(coupon_dir, report_dir)
    config = json.loads((Path(coupon_dir) / "metadata.json").read_text(
        encoding="utf-8"))["config"]
    old_verdict = metrics["verdict"]
    new_verdict = classify_r2r2(metrics, config)
    metrics["legacy_r2_verdict"] = old_verdict
    metrics["verdict"] = new_verdict
    metrics["classification_revision"] = "r2r2_decoupled_shear"
    write_json(Path(report_dir) / "metrics.json", metrics)
    (Path(report_dir) / "summary.md").write_text(
        "# R2-R2 {} coupon\n\n- Profile: `{}`\n- Verdict: `{}`\n"
        "- Legacy R2 verdict: `{}`\n".format(
            metrics["mode"], metrics["profile_name"], new_verdict, old_verdict),
        encoding="utf-8")
    return metrics


def main() -> None:
    """Run the R2-R2 coupon analyzer CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupon-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    result = analyze(Path(args.coupon_dir), Path(args.report_dir))
    print("verdict={}".format(result["verdict"]))


if __name__ == "__main__":
    main()
