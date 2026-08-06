"""Orchestrate bounded R2-R1 coupons and table settle using R2 runners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_material_coupon import (  # noqa: E402
    analyze as analyze_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2.analyze_table_settle import (  # noqa: E402
    analyze as analyze_table)
from scripts.experiment3.phase0_soft_blockpush_r2.run_material_coupon import (  # noqa: E402
    run_coupon)
from scripts.experiment3.phase0_soft_blockpush_r2.run_table_settle import (  # noqa: E402
    run_settle)
from scripts.experiment3.phase0_soft_blockpush_r2r1.common import (  # noqa: E402
    SOFTER_PROFILE, START_PROFILE, STIFFER_PROFILE, choose_next_profile,
    load_config, load_profile, write_json)


def execute_coupon(config_path: str, profile_path: str,
                   selection_path: str, mode: str) -> dict:
    """Run and analyze one R2 coupon under the stricter R2-R1 profile loader."""
    config = load_config(config_path)
    _, profile = load_profile(profile_path, config)
    coupon_dir = run_coupon(config_path, profile_path, selection_path, mode)
    report_dir = (Path(config["report_root"]) / "coupon" /
                  profile["profile_name"] / mode)
    return analyze_coupon(coupon_dir, report_dir)


def _summary(verdict: str, profiles_run: list[str], reason: str,
             axial: Optional[dict] = None, shear: Optional[dict] = None,
             table: Optional[dict] = None, selected: Optional[str] = None) -> dict:
    """Create one terminal execution summary with explicit safety fields."""
    return {"verdict": verdict, "selected_profile": selected,
            "profiles_run": profiles_run, "reason": reason,
            "axial": axial, "shear": shear, "table_settle": table,
            "pair_run": False, "training_run": False}


def blocked_summary(reason: str) -> dict:
    """Return an engineering-blocked summary before any coupon."""
    return _summary("PHASE0B_R2R1_ENGINEERING_BLOCKED", [], reason)


def no_profile_summary(profiles_run: list[str], reason: str,
                       axial: Optional[dict] = None,
                       shear: Optional[dict] = None) -> dict:
    """Return a bounded no-profile verdict without attempting another trial."""
    return _summary("PHASE0B_R2R1_NO_PROFILE_PASSED", profiles_run, reason,
                    axial=axial, shear=shear)


def _write(config: dict, summary: dict) -> dict:
    write_json(Path(config["report_root"]) / "execution_summary.json", summary)
    return summary


def run_calibration(config_path: str, profile_dir: str,
                    selection_path: str) -> dict:
    """Run A1.25 first and at most one rule-selected neighboring profile."""
    config = load_config(config_path)
    confirmation_path = (Path(config["report_root"]) /
                         "mechanics_validation" / "summary.json")
    if not confirmation_path.is_file():
        return _write(config, blocked_summary("family_confirmation_missing"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if confirmation.get("verdict") != "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE":
        return _write(config, blocked_summary("family_confirmation_failed"))
    selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    if (selection.get("microsteps_per_outer") != 8
            or selection.get("confirmation_verdict") !=
            "PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE"
            or selection.get("selection_reopened") is not False):
        return _write(config, blocked_summary("incompatible_microstep_selection"))
    directory = Path(profile_dir)
    profile_paths = {
        START_PROFILE: directory / "soft_block_kv_r2r1_a1p25_z025.json",
        SOFTER_PROFILE: directory / "soft_block_kv_r2r1_a1p00_z025.json",
        STIFFER_PROFILE: directory / "soft_block_kv_r2r1_a1p50_z025.json"}
    profiles_run: list[str] = []
    current = START_PROFILE
    axial = execute_coupon(config_path, str(profile_paths[current]),
                            selection_path, "axial")
    profiles_run.append(current)
    if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
        decision = choose_next_profile(current, axial["verdict"], len(profiles_run))
        if decision.stop:
            return _write(config, no_profile_summary(
                profiles_run, decision.reason, axial=axial))
        current = str(decision.next_profile_name)
        axial = execute_coupon(config_path, str(profile_paths[current]),
                                selection_path, "axial")
        profiles_run.append(current)
        if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
            return _write(config, no_profile_summary(
                profiles_run, "second_axial_failed", axial=axial))
    shear = execute_coupon(config_path, str(profile_paths[current]),
                            selection_path, "shear")
    if shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE":
        if len(profiles_run) >= 2:
            return _write(config, no_profile_summary(
                profiles_run, "second_profile_shear_failed", axial, shear))
        decision = choose_next_profile(current, shear["verdict"], len(profiles_run))
        if decision.stop:
            return _write(config, no_profile_summary(
                profiles_run, decision.reason, axial, shear))
        current = str(decision.next_profile_name)
        axial = execute_coupon(config_path, str(profile_paths[current]),
                                selection_path, "axial")
        profiles_run.append(current)
        if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
            return _write(config, no_profile_summary(
                profiles_run, "replacement_profile_axial_failed", axial, shear))
        shear = execute_coupon(config_path, str(profile_paths[current]),
                                selection_path, "shear")
        if shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE":
            return _write(config, no_profile_summary(
                profiles_run, "replacement_profile_shear_failed", axial, shear))
    settle_dir = run_settle(config_path, str(profile_paths[current]), selection_path)
    table_report = Path(config["report_root"]) / "table_settle" / current
    table = analyze_table(settle_dir, table_report)
    complete = (table["verdict"] == "PHASE0B_R2_TABLE_SETTLE_COMPLETE"
                and bool(table.get("frozen_outputs")))
    verdict = ("PHASE0B_R2R1_MATERIAL_CALIBRATION_COMPLETE" if complete
               else "PHASE0B_R2R1_TABLE_SETTLE_FAILED")
    return _write(config, _summary(
        verdict, profiles_run, "complete" if complete else "table_settle_failed",
        axial, shear, table, current))


def main() -> None:
    """Run the bounded calibration CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--selected-microsteps", required=True)
    args = parser.parse_args()
    print("verdict={}".format(run_calibration(
        args.config, args.profile_dir, args.selected_microsteps)["verdict"]))


if __name__ == "__main__":
    main()
