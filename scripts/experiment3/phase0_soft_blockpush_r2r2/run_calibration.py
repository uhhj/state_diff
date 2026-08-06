"""Run bounded decoupled-shear calibration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_table_settle import (  # noqa: E402
    analyze as analyze_table)
from scripts.experiment3.phase0_soft_blockpush_r2.run_material_coupon import (  # noqa: E402
    run_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2.run_table_settle import (  # noqa: E402
    run_settle)
from scripts.experiment3.phase0_soft_blockpush_r2r2.analyze_coupon import (  # noqa: E402
    analyze as analyze_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2r2.common import (  # noqa: E402
    SOFTER_PROFILE, START_PROFILE, STIFFER_PROFILE, choose_next_profile,
    load_config, load_profile, write_json)


def execution_summary(verdict: str, profiles_run: list[str], reason: str,
                      axial: Optional[dict] = None,
                      shear: Optional[dict] = None,
                      table: Optional[dict] = None,
                      selected: Optional[str] = None) -> dict:
    """Create one terminal stage summary with explicit safety fields."""
    return {"verdict": verdict, "profiles_run": profiles_run, "reason": reason,
            "selected_profile": selected, "axial": axial, "shear": shear,
            "table_settle": table, "pair_run": False, "training_run": False}


def execute_coupon(config_path: str, profile_path: str,
                   selection_path: str, mode: str) -> dict:
    """Run and analyze one coupon with R2-R2 verdict semantics."""
    config = load_config(config_path)
    _, profile = load_profile(profile_path, config)
    coupon_dir = run_coupon(config_path, profile_path, selection_path, mode)
    report_dir = (Path(config["report_root"]) / "coupon" /
                  profile["profile_name"] / mode)
    return analyze_coupon(coupon_dir, report_dir)


def write_summary(config: dict, summary: dict) -> dict:
    """Persist one execution summary and return it unchanged."""
    write_json(Path(config["report_root"]) / "execution_summary.json", summary)
    return summary


def run_profile(config_path: str, profile_path: str,
                selection_path: str) -> tuple[dict, Optional[dict]]:
    """Run axial first and shear only after axial completes."""
    axial = execute_coupon(config_path, profile_path, selection_path, "axial")
    if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
        return axial, None
    return axial, execute_coupon(config_path, profile_path, selection_path, "shear")


def run_calibration(config_path: str, profile_dir: str) -> dict:
    """Run S12.5 and at most one verdict-selected adjacent profile."""
    config = load_config(config_path)
    source_selection = Path(config["selected_microsteps_path"])
    selection = json.loads(source_selection.read_text(encoding="utf-8"))
    if (selection.get("microsteps_per_outer") != 8
            or selection.get("confirmation_verdict") !=
            "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE"):
        return write_summary(config, execution_summary(
            "PHASE0B_R2R2_ENGINEERING_BLOCKED", [], "M8_selection_unavailable"))
    local_selection = (Path(config["report_root"]) / "mechanics_validation" /
                       "selected_microsteps.json")
    local_selection.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_selection, local_selection)
    directory = Path(profile_dir)
    profile_paths = {
        START_PROFILE: directory / "soft_block_kv_r2r2_s12p5_z025.json",
        SOFTER_PROFILE: directory / "soft_block_kv_r2r2_s10p5_z025.json",
        STIFFER_PROFILE: directory / "soft_block_kv_r2r2_s15p0_z025.json"}
    profiles_run: list[str] = []
    current = START_PROFILE
    axial, shear = run_profile(config_path, str(profile_paths[current]),
                               str(local_selection))
    profiles_run.append(current)
    failure_verdict = axial["verdict"] if shear is None else shear["verdict"]
    if (axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE" or shear is None
            or shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE"):
        decision = choose_next_profile(current, failure_verdict, len(profiles_run))
        if decision.stop:
            return write_summary(config, execution_summary(
                "PHASE0B_R2R2_NO_PROFILE_PASSED", profiles_run,
                decision.reason, axial, shear))
        current = str(decision.next_profile_name)
        axial, shear = run_profile(config_path, str(profile_paths[current]),
                                   str(local_selection))
        profiles_run.append(current)
        if (axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE" or shear is None
                or shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE"):
            return write_summary(config, execution_summary(
                "PHASE0B_R2R2_NO_PROFILE_PASSED", profiles_run,
                "second_profile_failed", axial, shear))
    settle_dir = run_settle(config_path, str(profile_paths[current]),
                            str(local_selection))
    table_report = Path(config["report_root"]) / "table_settle" / current
    table = analyze_table(settle_dir, table_report)
    complete = (table["verdict"] == "PHASE0B_R2_TABLE_SETTLE_COMPLETE"
                and bool(table.get("frozen_outputs")))
    verdict = ("PHASE0B_R2R2_MATERIAL_CALIBRATION_COMPLETE" if complete
               else "PHASE0B_R2R2_TABLE_SETTLE_FAILED")
    return write_summary(config, execution_summary(
        verdict, profiles_run, "complete" if complete else "table_settle_failed",
        axial, shear, table, current))


def main() -> None:
    """Run the R2-R2 calibration CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(run_calibration(
        args.config, args.profile_dir)["verdict"]))


if __name__ == "__main__":
    main()
