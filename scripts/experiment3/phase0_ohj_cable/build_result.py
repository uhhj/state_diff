"""Build the two small committed OHJ Phase 0D result files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_ohj_cable.common import (  # noqa: E402
    load_config, report_dir, write_json)


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def build_result(config_path):
    config = load_config(config_path)
    source = report_dir(config)
    pair = json.loads((source / "pair_metrics.json").read_text(encoding="utf-8"))
    control_path = source / "control_relevance_metrics.json"
    control = (json.loads(control_path.read_text(encoding="utf-8"))
               if control_path.is_file() else None)
    if pair["verdict"] != "PHASE0D_PAIR_COMPLETE":
        verdict = pair["verdict"]
    elif control is None:
        verdict = "PHASE0D_ENGINEERING_BLOCKED"
    elif control["verdict"] == "PHASE0D_CONTROL_RELEVANCE_COMPLETE":
        verdict = "PHASE0D_OHJ_CABLE_VALIDATED"
    else:
        verdict = "PHASE0D_CONTROL_NOT_RELEVANT"
    branch = _git("branch", "--show-current")
    ending_main = _git("rev-parse", "HEAD")
    remote = _git("ls-remote", "origin", "refs/heads/" + branch).split()[0]
    ending_submodule = _git("rev-parse", "HEAD:external/deformable-ravens")
    tests_passed = int(os.environ.get("PHASE0D_TESTS_PASSED", "0"))
    tests_failed = int(os.environ.get("PHASE0D_TESTS_FAILED", "0"))
    pip_check = os.environ.get("PHASE0D_PIP_CHECK", "not recorded")
    evidence = {
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": ending_main,
            "branch": branch,
            "clean_before_result": _git("status", "--porcelain") == "",
            "remote_tip": remote,
            "remote_tip_matches": remote == ending_main,
            "starting_submodule_sha": config["provenance"][
                "starting_submodule_sha"],
            "ending_submodule_sha": ending_submodule,
        },
        "gates": {
            "initial_observable_equivalence": pair["gates"][
                "initial_observable_equivalence"],
            "post_probe_observable_equivalence": pair["gates"][
                "post_probe_observable_equivalence"],
            "sensor_observability": pair["gates"]["sensor_observability"],
            "same_action_future_divergence": pair["gates"][
                "same_action_future_divergence"],
            "control_relevance": (None if control is None
                                  else control["control_relevant"]),
        },
        "pair": pair,
        "control_relevance": control,
        "tests": {"passed": tests_passed, "failed": tests_failed,
                  "pip_check": pip_check},
        "training": {"B0": False, "B1": False, "CFPM": False,
                     "IDM": False},
    }
    destination = REPO_ROOT / config.get(
        "committed_report_dir", "reports/experiment3/phase0d_ohj_cable")
    write_json(destination / "EVIDENCE.json", evidence)
    probe_mm = 1000.0 * float(config["motion"]["probe_delta_xyz_m"][0])
    clearance_mm = 1000.0 * float(
        config["geometry"]["jam_surface_clearance_m"])
    sensor_field = config["sensor"].get("trace_field", "formal_wrench")
    repeat_floor = float(pair["repeat_peak_visible_rmse_m"])
    equiv_threshold = float(
        config["analysis"]["post_probe_visible_rmse_max_m"])
    hold_repair = bool(config["execution"].get(
        "reassert_canonical_hold_after_restore", False))
    if verdict == "PHASE0D_OHJ_CABLE_VALIDATED":
        next_task = (
            "Generate physics_pairs, then train B0 StateDiff and B1 "
            "StateDiff-FT.")
    elif verdict == "PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL":
        if hold_repair and repeat_floor > equiv_threshold:
            next_task = (
                "Same-condition repeatability remains unresolved after "
                "canonical motor-hold repair. Stop probe/contact tuning and "
                "isolate same-world restore against fresh-world branch "
                "execution.")
        elif hold_repair:
            next_task = (
                "The repeat floor is now below the observable-equivalence "
                "scale, so the remaining post-probe mismatch is "
                "JAM-specific. Reduce the zero-net probe from 1.0 mm to "
                "0.5 mm while keeping 240-step settle and 0.5 mm clearance "
                "fixed.")
        elif probe_mm > 1.0 + 1e-9:
            next_task = (
                "Reduce zero-net probe amplitude to 1.0 mm.")
        else:
            next_task = (
                "Prepare the canonical-hold same-condition repeatability "
                "repair.")
    elif verdict == "PHASE0D_SENSOR_NOT_OBSERVABLE":
        if sensor_field == "formal_wrench":
            next_task = (
                "Add deployable-equivalent gripper-surface tactile without "
                "changing task physics.")
        elif sensor_field == "formal_sensor":
            next_task = (
                "Aggregate 3D tactile is not observable. Freeze task physics "
                "and evaluate a minimal four-patch spatial gripper-tactile "
                "representation.")
        else:
            next_task = (
                "The four-patch spatial tactile sensor is still not "
                "observable. Do not automatically increase tactile "
                "resolution and do not change task physics. Compare the "
                "spatial and aggregate tactile signals: if spatial sensing "
                "gives no clear gain, revisit physical grasp/sensing "
                "coupling; if it gives clear localized gain but remains "
                "below threshold, plan one separate small-resolution tactile "
                "study.")
    elif verdict == "PHASE0D_FUTURE_BRANCH_NOT_ESTABLISHED":
        next_task = (
            "Deployable contact sensing is now observable. Freeze sensor, "
            "probe, clearance, and canonical hold; prepare a single-purpose "
            "future-branch effect-size repair.")
    else:
        next_task = (
            "Stop at the current scientific gate and prepare the next "
            "single-purpose repair.")
    control_show = control or {}
    recovery = pair["recovery"]
    repeatability = pair["repeatability_by_phase"]
    probe_contact = pair["probe_contact"]
    text = """Verdict: {verdict}

Repository:
- Main start: {main_start}
- Main end: {main_end}
- Branch: {branch}
- Clean before result: {clean}
- Remote tip: {remote}
- Submodule start: {sub_start}
- Submodule end: {sub_end}

Repair:
- Canonical hold reassert after restore: {hold_repair}

Frozen:
- Probe amplitude: {probe_mm} mm
- Jam clearance: {clearance_mm} mm
- Post-probe settle: {settle_steps} steps / {settle_s} s
- State: {state_dim}D
- Sensor: {sensor_dim}D
- Sensor representation: {sensor_representation}
- Sensor trace field: {sensor_field}

Probe engagement:
- FREE probe latch contact samples: {probe_free_contacts}
- JAM probe latch contact samples: {probe_jam_contacts}
- JAM probe latch contact fraction: {probe_contact_fraction}
- JAM probe latch peak force: {probe_peak_force} N

Five gates:
- Initial observable equivalence: {gate1}
- Post-probe observable equivalence: {gate2}
- Sensor observability: {gate3}
- Same-action future divergence: {gate4}
- Control relevance: {gate5}

Key values:
- Initial 51D RMSE: {initial} m
- Post-probe 51D RMSE: {post} m
- Post-probe keypoint RMSE: {post_keypoint} m
- Post-probe EE RMSE: {post_ee} m
- Immediate-return 51D RMSE: {immediate} m
- Post-probe recovery curve: {recovery_curve}
- Final FREE-repeat post-probe RMSE: {recovery_repeat} m
- Final branch excess over repeat: {recovery_excess} m
- Final repeat/FREE-JAM fraction: {repeat_fraction}
- Final branch-excess/FREE-JAM fraction: {excess_fraction}
- Recovery fraction: {recovery_fraction}
- JAM post-probe latch contact samples: {recovery_contacts}
- JAM post-probe latch contact fraction: {contact_fraction}
- JAM post-probe latch peak force: {recovery_force} N
- Sensor onset: {sensor_step} / {sensor_phase}
- Sensor trigger: {trigger}
- Grasp-only peak fused gap: {grasp_sensor_peak}
- Tactile-only peak fused gap: {tactile_sensor_peak}
- Aggregate 3D tactile peak: {aggregate_tactile_peak}
- Spatial tactile patch metrics: {spatial_tactile_patches}
- Tactile raw force-gap peak: {tactile_raw_peak} N
- Tactile probe contact samples: {tactile_contact_samples}
- Tactile contact-count peak gap: {tactile_contact_count_gap}
- Combined formal peak fused gap: {sensor_peak}
- Future visible RMSE peak: {future} m
- Repeat floor: {repeat} m
- Future/repeat ratio: {future_repeat_ratio}
- FREE/JAM extraction progress: {progress}
- Repeat first phase above equivalence threshold: {repeat_first_phase}
- Repeat phase diagnostics: {repeat_phases}
- Control verdict: {control_verdict}

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Hidden latch/contact used as formal sensor: No

Tests:
- Passed: {passed}
- Failed: {failed}
- pip check: {pip_check}

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: {next_task}
""".format(
        verdict=verdict,
        main_start=config["provenance"]["starting_main_sha"],
        main_end=ending_main, branch=branch,
        clean=evidence["repository"]["clean_before_result"], remote=remote,
        sub_start=config["provenance"]["starting_submodule_sha"],
        sub_end=ending_submodule,
        hold_repair=hold_repair,
        probe_mm=probe_mm,
        clearance_mm=clearance_mm,
        settle_steps=config["execution"]["post_probe_steps"],
        settle_s=(config["execution"]["post_probe_steps"]
                  / config["execution"]["hz"]),
        state_dim=config["state"]["state_dim"],
        sensor_dim=config["sensor"]["sensor_dim"],
        sensor_representation=config["sensor"]["formal"],
        sensor_field=sensor_field,
        probe_free_contacts=probe_contact["free_probe_contact_samples"],
        probe_jam_contacts=probe_contact["jam_probe_contact_samples"],
        probe_contact_fraction=probe_contact["jam_probe_contact_fraction"],
        probe_peak_force=probe_contact["jam_probe_peak_latch_force_n"],
        gate1=evidence["gates"]["initial_observable_equivalence"],
        gate2=evidence["gates"]["post_probe_observable_equivalence"],
        gate3=evidence["gates"]["sensor_observability"],
        gate4=evidence["gates"]["same_action_future_divergence"],
        gate5=evidence["gates"]["control_relevance"],
        initial=pair["initial_51d_rmse_m"],
        post=pair["post_probe_51d_rmse_m"],
        post_keypoint=pair["post_probe_keypoint_rmse_m"],
        post_ee=pair["post_probe_ee_rmse_m"],
        immediate=recovery["immediate_free_jam_rmse_m"],
        recovery_curve=recovery["checkpoints"],
        recovery_repeat=recovery["final_free_repeat_rmse_m"],
        recovery_excess=recovery["final_branch_excess_over_repeat_m"],
        repeat_fraction=recovery["final_repeat_fraction_of_free_jam"],
        excess_fraction=recovery[
            "final_branch_excess_fraction_of_free_jam"],
        recovery_fraction=recovery["recovery_fraction"],
        recovery_contacts=recovery["post_probe_jam_latch_contact_samples"],
        contact_fraction=recovery[
            "post_probe_jam_latch_contact_fraction"],
        recovery_force=recovery["post_probe_jam_peak_latch_force_n"],
        sensor_step=pair["sensor_onset_step"],
        sensor_phase=pair["sensor_onset_phase"], trigger=pair["sensor_trigger"],
        grasp_sensor_peak=pair["grasp_only_peak_fused_gap"],
        tactile_sensor_peak=pair["tactile_only_peak_fused_gap"],
        aggregate_tactile_peak=pair["aggregate_tactile_peak_fused_gap"],
        spatial_tactile_patches=pair["spatial_tactile_patches"],
        tactile_raw_peak=pair["tactile_raw_force_gap_peak_n"],
        tactile_contact_samples=pair["tactile_probe_contact_samples"],
        tactile_contact_count_gap=pair["tactile_contact_count_peak_gap"],
        sensor_peak=pair["peak_fused_sensor_gap"],
        future=pair["future_peak_visible_rmse_m"],
        repeat=pair["repeat_peak_visible_rmse_m"],
        future_repeat_ratio=pair["future_branch_to_repeat_ratio"],
        progress=pair["final_extraction_progress_m"],
        repeat_first_phase=repeatability[
            "first_phase_above_equivalence_threshold"],
        repeat_phases=repeatability["phases"],
        control_verdict=control_show.get("verdict", "not run"),
        passed=tests_passed, failed=tests_failed, pip_check=pip_check,
        next_task=next_task)
    (destination / "RESULT.md").write_text(text, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(build_result(args.config)["verdict"]))


if __name__ == "__main__":
    main()
