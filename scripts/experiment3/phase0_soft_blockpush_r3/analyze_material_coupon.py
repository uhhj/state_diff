"""Extend the unchanged R2 coupon gates with angle telemetry."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_material_coupon import (  # noqa: E402
    analyze as analyze_r2)
from scripts.experiment3.phase0_soft_blockpush_r3.common import write_json  # noqa: E402


def analyze(coupon_dir: Path, report_dir: Path) -> dict:
    metrics = analyze_r2(coupon_dir, report_dir)
    metadata = json.loads((Path(coupon_dir) / "metadata.json").read_text(encoding="utf-8"))
    with np.load(Path(coupon_dir) / "trajectory.npz", allow_pickle=False) as source:
        peak = float(np.max(source["spring_energy_angle_j"]))
    angle = metadata["material_profile"]["angle"]
    metrics.update({
        "angle_stiffness_n_m": angle["stiffness_n_m"],
        "angle_damping_n_m_s": angle["damping_n_m_s"],
        "peak_angle_energy_j": peak,
        "angle_constraint_count": metadata["angle_constraint_count"]})
    write_json(Path(report_dir) / "metrics.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coupon-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(analyze(Path(args.coupon_dir), Path(args.report_dir))["verdict"]))


if __name__ == "__main__": main()
