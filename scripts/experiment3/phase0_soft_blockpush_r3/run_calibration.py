"""Run the bounded R3 angle-stiffness calibration."""
import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2.analyze_table_settle import (  # noqa: E402
    analyze as analyze_table)
from scripts.experiment3.phase0_soft_blockpush_r3.analyze_material_coupon import (  # noqa: E402
    analyze as analyze_coupon)
from scripts.experiment3.phase0_soft_blockpush_r3.common import (  # noqa: E402
    SOFTER_PROFILE, START_PROFILE, STIFFER_PROFILE, choose_next_profile,
    load_config, write_json)
from scripts.experiment3.phase0_soft_blockpush_r3.run_material_coupon import (  # noqa: E402
    run_coupon)
from scripts.experiment3.phase0_soft_blockpush_r3.run_table_settle import (  # noqa: E402
    run_settle)


def _summary(verdict, profiles, reason, axial=None, shear=None, table=None,
             selected=None):
    return {"verdict": verdict, "profiles_run": profiles, "reason": reason,
            "selected_profile": selected, "axial": axial, "shear": shear,
            "table_settle": table, "pair_run": False, "training_run": False}


def _write(config, payload):
    write_json(Path(config["report_root"]) / "execution_summary.json", payload)
    return payload


def _coupon(config_path, profile_path, selection, mode):
    config = load_config(config_path)
    output = run_coupon(config_path, str(profile_path), str(selection), mode)
    report = (Path(config["report_root"]) / "coupon" /
              profile_path.stem.replace("soft_block_", "").replace("_z025", "_z025") / mode)
    # Use profile_name from metadata to avoid filename assumptions.
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    report = Path(config["report_root"]) / "coupon" / metadata["material_profile"]["profile_name"] / mode
    return analyze_coupon(output, report)


def _profile(config_path, path, selection):
    axial = _coupon(config_path, path, selection, "axial")
    if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
        return axial, None
    return axial, _coupon(config_path, path, selection, "shear")


def run_calibration(config_path: str, profile_dir: str) -> dict:
    config = load_config(config_path)
    validation = Path(config["report_root"]) / "angle_validation" / "metrics.json"
    selection = Path(config["report_root"]) / "mechanics_validation" / "selected_microsteps.json"
    if not validation.is_file() or not selection.is_file():
        return _write(config, _summary(
            "PHASE0B_R3_ENGINEERING_BLOCKED", [], "angle_validation_missing"))
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    selected_payload = json.loads(selection.read_text(encoding="utf-8"))
    if (validation_payload.get("verdict") != "PHASE0B_R3_ANGLE_VALIDATION_COMPLETE"
            or selected_payload.get("microsteps_per_outer") != 8):
        return _write(config, _summary(
            "PHASE0B_R3_ENGINEERING_BLOCKED", [], "angle_validation_incomplete"))
    root = Path(profile_dir)
    paths = {
        START_PROFILE: root / "soft_block_kv_angle_r3_a055_z025.json",
        SOFTER_PROFILE: root / "soft_block_kv_angle_r3_a040_z025.json",
        STIFFER_PROFILE: root / "soft_block_kv_angle_r3_a070_z025.json"}
    profiles = [START_PROFILE]
    current = START_PROFILE
    axial, shear = _profile(config_path, paths[current], selection)
    if axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE":
        return _write(config, _summary(
            "PHASE0B_R3_NO_PROFILE_PASSED", profiles,
            "start_profile_axial_failed", axial, shear))
    if shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE":
        next_profile = choose_next_profile(shear["verdict"])
        if next_profile is None:
            return _write(config, _summary(
                "PHASE0B_R3_NO_PROFILE_PASSED", profiles,
                "start_profile_shear_not_selectable", axial, shear))
        current = next_profile; profiles.append(current)
        axial, shear = _profile(config_path, paths[current], selection)
        if (axial["verdict"] != "PHASE0B_R2_AXIAL_COMPLETE" or shear is None
                or shear["verdict"] != "PHASE0B_R2_SHEAR_COMPLETE"):
            return _write(config, _summary(
                "PHASE0B_R3_NO_PROFILE_PASSED", profiles,
                "second_profile_failed", axial, shear))
    settle = run_settle(config_path, str(paths[current]), str(selection))
    table = analyze_table(
        settle, Path(config["report_root"]) / "table_settle" / current)
    complete = (table["verdict"] == "PHASE0B_R2_TABLE_SETTLE_COMPLETE"
                and bool(table.get("frozen_outputs")))
    verdict = ("PHASE0B_R3_MATERIAL_CALIBRATION_COMPLETE" if complete
               else "PHASE0B_R3_TABLE_SETTLE_FAILED")
    return _write(config, _summary(
        verdict, profiles, "complete" if complete else "table_settle_failed",
        axial, shear, table, current))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile-dir", required=True)
    args = parser.parse_args()
    print("verdict={}".format(run_calibration(args.config, args.profile_dir)["verdict"]))


if __name__ == "__main__": main()
