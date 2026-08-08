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
    probe_delta = [
        float(value) for value in config["motion"]["probe_delta_xyz_m"]]
    probe_norm = sum(value * value for value in probe_delta) ** 0.5
    probe_mm = 1000.0 * probe_norm
    probe_direction = [value / probe_norm for value in probe_delta]
    probe_speed_m_s = probe_norm / (
        config["motion"]["probe_forward_steps"] / config["execution"]["hz"])
    clearance_mm = 1000.0 * float(
        config["geometry"]["jam_surface_clearance_m"])
    sensor_field = config["sensor"].get("trace_field", "formal_wrench")
    diagnostic_mode = config.get("diagnostic", {}).get("mode")
    load_path = pair.get("load_path_diagnostic")
    repair_mode = config.get("repair", {}).get("mode")
    latch_topology = config["geometry"].get("latch_topology", "single_post")
    amplification = pair.get("probe_amplification_diagnostic")
    direction_diagnostic = pair.get("probe_direction_diagnostic")
    horizon_diagnostic = pair.get("future_horizon_diagnostic")
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
        if repair_mode == "single_dual_post_future_consequence_repair":
            next_task = (
                "The dual-post future-consequence topology breaks "
                "observable equivalence. Reject this topology. Keep the R9 "
                "probe, 18D sensor, 5 s horizon, and Gate definitions "
                "frozen. Do not tune post radius/spacing/count; if another "
                "future repair is attempted, use a qualitatively different "
                "continuous slot/hook topology.")
        elif repair_mode == "single_orthogonal_contact_loading_probe":
            next_task = (
                "The one-shot 1 mm -Y contact-loading probe breaks post-"
                "probe observable equivalence. Reject this direction. Do "
                "not increase its amplitude, do not mirror into the +Y "
                "release direction, and stop Cartesian translation-probe "
                "tuning. Prepare a mechanical load-path / coupling redesign.")
        elif repair_mode == "single_fixed_speed_probe_excursion_amplification":
            next_task = (
                "The one-shot 2 mm fixed-speed zero-net probe does not "
                "preserve the required observable equivalence. Reject this "
                "excursion amplification. Do not continue a 3/4/5 mm "
                "amplitude sweep; return to the 1 mm R7S probe and plan a "
                "different mechanical information-gathering probe/coupling "
                "repair.")
        else:
            if hold_repair and repeat_floor > equiv_threshold:
                next_task = (
                    "Same-condition repeatability remains unresolved after "
                    "canonical motor-hold repair. Stop probe/contact tuning "
                    "and isolate same-world restore against fresh-world "
                    "branch execution.")
            elif hold_repair:
                next_task = (
                    "The repeat floor is now below the observable-equivalence "
                    "scale, so the remaining post-probe mismatch is "
                    "JAM-specific. Reduce the zero-net probe from 1.0 mm to "
                    "0.5 mm while keeping 240-step settle and 0.5 mm "
                    "clearance fixed.")
            elif probe_mm > 1.0 + 1e-9:
                next_task = "Reduce zero-net probe amplitude to 1.0 mm."
            else:
                next_task = (
                    "Prepare the canonical-hold same-condition repeatability "
                    "repair.")
    elif verdict == "PHASE0D_SENSOR_NOT_OBSERVABLE":
        if repair_mode == "single_dual_post_future_consequence_repair":
            next_task = (
                "The dual-post topology regresses the previously validated "
                "Gate 3 sensing chain. Reject this topology. Return to "
                "frozen R9 probe/sensor mechanics; do not reopen tactile "
                "resolution or probe tuning. Any next future repair must "
                "preserve Gate 3.")
        elif repair_mode == "single_orthogonal_contact_loading_probe":
            route = load_path["route_hint"]
            if route == "repair_physical_grasp_sensing_coupling":
                next_task = (
                    "The 1 mm -Y contact-loading probe preserves Gate 2 and "
                    "a mechanically-large repeat-corrected branch-specific "
                    "load reaches the gripper-proximal segment, but the "
                    "deployable 18D sensor still fails Gate 3. Stop probe "
                    "tuning and prepare a minimal physical grasp/sensing-"
                    "coupling repair.")
            elif route == "amplify_probe_or_mechanical_signal":
                next_task = (
                    "The one-shot orthogonal contact-loading probe still "
                    "leaves the gripper-proximal branch-specific excess "
                    "below the mechanical reference scale and Gate 3 "
                    "remains false. Axial amplitude and orthogonal "
                    "translation have both failed. Stop translational probe "
                    "tuning and prepare a mechanical load-path / grasp-cable "
                    "coupling redesign.")
            elif route == "repair_mechanical_load_transmission":
                next_task = (
                    "The actual hidden-contact region contains a repeat-"
                    "corrected branch-specific signal under the -Y probe, "
                    "but it is not preserved at the gripper-proximal "
                    "segment. Stop probe direction tuning and repair "
                    "mechanical load transmission.")
            else:
                next_task = (
                    "The actual hidden-contact reference region does not "
                    "show a stable repeat-corrected branch-specific internal "
                    "signal under the -Y probe. Stop tactile/gripper and "
                    "translation-probe tuning; redesign the hidden-"
                    "interaction / mechanical coupling.")
        elif repair_mode == "single_fixed_speed_probe_excursion_amplification":
            route = load_path["route_hint"]
            if route == "repair_physical_grasp_sensing_coupling":
                next_task = (
                    "The 2 mm fixed-speed probe preserves observable "
                    "equivalence and produces a mechanically-large repeat-"
                    "corrected branch-specific load at the gripper-proximal "
                    "cable segment, but the deployable 18D sensor still "
                    "fails Gate 3. Stop probe-amplitude tuning and prepare a "
                    "minimal physical grasp/sensing-coupling repair.")
            elif route == "amplify_probe_or_mechanical_signal":
                next_task = (
                    "The one-shot 2 mm fixed-speed probe still leaves the "
                    "gripper-proximal branch-specific excess below the "
                    "mechanical reference scale and Gate 3 remains false. "
                    "Stop scalar amplitude escalation; do not try 3/4/5 mm. "
                    "Prepare a different minimal mechanical information-"
                    "gathering probe/coupling repair while preserving Gate 2.")
            elif route == "repair_mechanical_load_transmission":
                next_task = (
                    "The actual hidden-contact region retains a repeat-"
                    "corrected branch-specific signal under the 2 mm probe, "
                    "but the gripper-proximal segment does not. Stop "
                    "amplitude tuning and repair mechanical load transmission "
                    "/ probe mechanics.")
            else:
                next_task = (
                    "The 2 mm fixed-speed probe does not establish a stable "
                    "repeat-corrected branch-specific signal in the actual "
                    "hidden-contact reference region. Stop tactile/gripper "
                    "work and redesign the hidden-interaction/probe mechanics.")
        elif diagnostic_mode == (
                "spatial_repeat_corrected_load_path_information"):
            route = load_path["route_hint"]
            if route == "repair_physical_grasp_sensing_coupling":
                next_task = (
                    "Actual JAM contact-adjacent internal cable constraints "
                    "show a repeat-corrected branch-specific signal, and "
                    "that signal remains branch-specific with a mechanically "
                    "large excess at the gripper-proximal constraint. The "
                    "deployable 18D sensor still fails Gate 3. Freeze "
                    "probe/task physics and prepare a minimal physical "
                    "pinch/tactile grasp repair.")
            elif route == "amplify_probe_or_mechanical_signal":
                next_task = (
                    "A repeat-corrected branch-specific internal signal "
                    "reaches the gripper-proximal segment, but its mechanical "
                    "excess is below the reference force scale. Preserve "
                    "Gate 2 and prepare one minimal probe/mechanical-signal "
                    "amplification repair.")
            elif route == "repair_mechanical_load_transmission":
                next_task = (
                    "The actual JAM contact-adjacent region contains a "
                    "repeat-corrected branch-specific internal signal, but "
                    "the gripper-proximal segment does not. Do not redesign "
                    "the gripper yet; repair mechanical load transmission / "
                    "information-gathering probe mechanics.")
            else:
                next_task = (
                    "The actual JAM contact-adjacent internal constraints do "
                    "not show a stable branch-specific signal above the "
                    "FREE-repeat floor. Stop tactile/gripper work and repair "
                    "hidden-interaction / probe source mechanics.")
        elif sensor_field == "formal_wrench":
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
        if repair_mode == "single_dual_post_future_consequence_repair":
            next_task = (
                "The one-shot dual-post directional guide preserves the "
                "frozen sensing benchmark but still does not reach the "
                "existing Gate 4 effect-size threshold. Stop post radius/"
                "spacing/count tuning and do not change the 5 s horizon or "
                "STRAIGHT amplitude. If the project continues task repair, "
                "prepare one qualitatively different continuous hidden "
                "slot/hook future-consequence topology.")
        elif (horizon_diagnostic is not None
                and horizon_diagnostic["mode"]
                == "final_fixed_horizon_sufficiency"):
            if horizon_diagnostic["tail_still_rising_at_horizon_end"]:
                next_task = (
                    "The final fixed 5 s post-action observation still "
                    "ends at the global future-RMSE maximum with positive "
                    "0.5/1.0/2.0 s tail slopes. Do not modify task physics "
                    "from this censored measurement and do not extend the "
                    "horizon again automatically. Keep the R9 probe and "
                    "18D sensor frozen; review the Gate 4/model prediction "
                    "horizon definition and task timescale before any "
                    "future-consequence topology repair.")
            else:
                next_task = (
                    "The final fixed 5 s post-action observation does not "
                    "reach the existing Gate 4 threshold and no longer "
                    "shows the strong endpoint right-censoring signature "
                    "used in R10. Keep the R9 probe and 18D sensor frozen. "
                    "Do not lower Gate 4 and do not extend the horizon again "
                    "automatically; prepare a single-purpose future-"
                    "consequence task/topology repair.")
        elif repair_mode == "single_orthogonal_contact_loading_probe":
            next_task = (
                "The orthogonal contact-loading probe makes the deployable "
                "sensor observable while preserving the observable-state "
                "gate. Freeze the R9 probe and 18D sensor. Do not continue "
                "probe-direction tuning; prepare the single-purpose same-"
                "action future-branch effect-size repair.")
        else:
            next_task = (
                "Deployable contact sensing is now observable. Freeze "
                "sensor, probe, clearance, and canonical hold; prepare a "
                "single-purpose future-branch effect-size repair.")
    elif (verdict == "PHASE0D_CONTROL_NOT_RELEVANT"
          and repair_mode == "single_dual_post_future_consequence_repair"):
        next_task = (
            "Gate 4 is established under the frozen R9 probe/sensor and "
            "R11 horizon, but the fixed action matrix does not establish "
            "Gate 5. Freeze the dual-post task physics, horizon, probe, "
            "sensor, and Gate 4 definition. Inspect the existing FREE/JAM "
            "matrix once to distinguish FREE STRAIGHT task success from "
            "JAM-R LEFT-RELEASE advantage, then prepare only the control-"
            "consequence repair.")
    elif (verdict == "PHASE0D_CONTROL_NOT_RELEVANT"
          and horizon_diagnostic is not None):
        next_task = (
            "The frozen R9 benchmark now passes Gate 4 within the final "
            "fixed future horizon, but the fixed action matrix does not "
            "establish Gate 5 control relevance. Freeze the horizon, probe, "
            "sensor, and Gate 4 definition. Inspect the existing FREE/JAM "
            "control matrix once and prepare only the control-consequence "
            "repair; do not reopen future-horizon or sensing tuning.")
    else:
        next_task = (
            "Stop at the current scientific gate and prepare the next "
            "single-purpose repair.")
    control_show = control or {}
    recovery = pair["recovery"]
    repeatability = pair["repeatability_by_phase"]
    probe_contact = pair["probe_contact"]
    if load_path is None:
        load_path_text = (
            "Spatial repeat-corrected load-path audit: not configured")
    else:
        load_path_lines = [
            "Spatial repeat-corrected load-path audit:",
            "- Method: {}".format(load_path["method"]),
            "- Interpretation limit: {}".format(
                load_path["interpretation_limit"]),
            "- Actual JAM probe contact bead indices: {}".format(
                load_path["probe_contact_bead_indices"]),
            "- Contact-adjacent constraint indices: {}".format(
                load_path["contact_reference_constraint_indices"]),
            "- Contact-region branch-specific constraint indices: {}".format(
                load_path[
                    "contact_reference_branch_specific_segment_indices"]),
            "- Contact-region peak segment/excess: {} / {} N".format(
                load_path["contact_reference_peak_segment_index"],
                load_path["contact_reference_peak_excess_n"]),
            "- Global peak segment/excess: {} / {} N".format(
                load_path["global_peak_segment_index"],
                load_path["global_peak_excess_n"]),
            "- Global peak is contact-adjacent: {}".format(
                load_path["global_peak_is_contact_adjacent"]),
            "- All branch-specific segment indices: {}".format(
                load_path["branch_specific_segment_indices"]),
            "- Furthest branch-specific segment toward gripper: {}".format(
                load_path[
                    "furthest_branch_specific_segment_toward_gripper"]),
            "- Gripper-proximal constraint index: {}".format(
                load_path["gripper_proximal_constraint_index"]),
            "- Proximal branch-specific: {}".format(
                load_path["proximal_branch_specific_vs_repeat"]),
            "- Proximal repeat-corrected excess: {} N".format(
                load_path["proximal_repeat_corrected_excess_n"]),
            "- Proximal mechanically large: {}".format(
                load_path["proximal_mechanically_large_excess"]),
            "- Proximal retention ratio: {}".format(
                load_path["proximal_retention_ratio"]),
            "- Route hint: {}".format(load_path["route_hint"]),
            "",
            "Segment rows:",
        ]
        for row in load_path["segments"]:
            no_action = row["no_action"]
            probe = row["probe"]
            load_path_lines.append(
                "- Segment {index}: no_action branch/repeat sustained "
                "{na_branch}/{na_repeat} N; no_action excess {na_excess} N; "
                "probe branch/repeat sustained {p_branch}/{p_repeat} N; "
                "probe excess {p_excess} N; branch/repeat ratio {ratio}; "
                "branch-specific {specific}; mechanically large {large}; "
                "probe emergence excess {emergence} N".format(
                    index=row["constraint_index"],
                    na_branch=no_action["branch_sustained_gap_n"],
                    na_repeat=no_action["repeat_sustained_gap_n"],
                    na_excess=no_action["repeat_corrected_excess_n"],
                    p_branch=probe["branch_sustained_gap_n"],
                    p_repeat=probe["repeat_sustained_gap_n"],
                    p_excess=probe["repeat_corrected_excess_n"],
                    ratio=probe["branch_over_repeat_ratio"],
                    specific=probe["branch_specific_vs_repeat"],
                    large=probe["mechanically_large_excess"],
                    emergence=row["probe_emergence_excess_n"]))
        load_path_text = "\n".join(load_path_lines)
    if amplification is None:
        amplification_text = (
            "Fixed-speed probe amplification comparison: not configured")
    else:
        amplification_text = (
            "Fixed-speed probe amplification comparison:\n"
            "- Report only: {}\n"
            "- Stop after this trial: {}\n"
            "- Baseline: {}\n"
            "- Current: {}\n"
            "- Gain ratios: {}"
        ).format(
            amplification["report_only"],
            amplification["stop_after_this_trial"],
            amplification["baseline"],
            amplification["current"],
            amplification["gain_ratio"],
        )
    if direction_diagnostic is None:
        direction_text = (
            "Orthogonal contact-loading probe comparison: not configured")
    else:
        direction_text = (
            "Orthogonal contact-loading probe comparison:\n"
            "- Report only: {}\n"
            "- Stop after this trial: {}\n"
            "- Direction dot baseline: {}\n"
            "- Baseline: {}\n"
            "- Current: {}\n"
            "- Gain ratios: {}"
        ).format(
            direction_diagnostic["report_only"],
            direction_diagnostic["stop_after_this_trial"],
            direction_diagnostic["direction_dot_baseline"],
            direction_diagnostic["baseline"],
            direction_diagnostic["current"],
            direction_diagnostic["gain_ratio"],
        )
    if horizon_diagnostic is None:
        horizon_text = (
            "Final future-horizon sufficiency audit: not configured")
    else:
        horizon_text = (
            "Final future-horizon sufficiency audit:\n"
            "- Diagnostic only: {}\n"
            "- Gates unchanged: {}\n"
            "- Stop after this trial: {}\n"
            "- Future samples/span: {} / {} s\n"
            "- Post-test observation: {} steps / {} s\n"
            "- Future threshold: {} m\n"
            "- Threshold reached: {}\n"
            "- First threshold crossing: {}\n"
            "- Endpoint / peak: {} / {} m\n"
            "- Peak index/time: {} / {} s\n"
            "- Endpoint is global max: {}\n"
            "- Repeat peak: {} m\n"
            "- Future/repeat: {}\n"
            "- Remaining margin: {} m\n"
            "- Tail windows: {}\n"
            "- All tail slopes positive: {}\n"
            "- Tail still rising at horizon end: {}\n"
            "- Interpretation: {}"
        ).format(
            horizon_diagnostic["diagnostic_only"],
            horizon_diagnostic["benchmark_gates_unchanged"],
            horizon_diagnostic["stop_after_this_trial"],
            horizon_diagnostic["future_samples"],
            horizon_diagnostic["future_span_s"],
            horizon_diagnostic["post_test_steps"],
            horizon_diagnostic["post_test_seconds"],
            horizon_diagnostic["future_threshold_m"],
            horizon_diagnostic["threshold_reached"],
            horizon_diagnostic["first_threshold_crossing"],
            horizon_diagnostic["endpoint_visible_rmse_m"],
            horizon_diagnostic["peak_visible_rmse_m"],
            horizon_diagnostic["peak_index_zero_based"],
            horizon_diagnostic["peak_time_from_future_start_s"],
            horizon_diagnostic["endpoint_is_global_max"],
            horizon_diagnostic["repeat_peak_visible_rmse_m"],
            horizon_diagnostic["future_to_repeat_ratio"],
            horizon_diagnostic["remaining_margin_to_threshold_m"],
            horizon_diagnostic["tail_windows"],
            horizon_diagnostic["all_tail_slopes_positive"],
            horizon_diagnostic["tail_still_rising_at_horizon_end"],
            horizon_diagnostic["interpretation"],
        )
    future_consequence_text = (
        "Dual-post future-consequence comparison: not configured")
    if repair_mode == "single_dual_post_future_consequence_repair":
        baseline = config["repair"]["baseline"]
        current = {
            "post_probe_rmse_m": float(pair["post_probe_51d_rmse_m"]),
            "formal_sensor_peak_fused_gap": float(
                pair["peak_fused_sensor_gap"]),
            "future_peak_visible_rmse_m": float(
                pair["future_peak_visible_rmse_m"]),
            "repeat_peak_visible_rmse_m": float(
                pair["repeat_peak_visible_rmse_m"]),
            "future_branch_to_repeat_ratio": pair[
                "future_branch_to_repeat_ratio"],
            "final_extraction_progress_m": dict(
                pair["final_extraction_progress_m"]),
        }
        future_consequence_text = (
            "Dual-post future-consequence comparison:\n"
            "- Report only: True\n"
            "- Stop after this trial: {}\n"
            "- Baseline topology: {}\n"
            "- Current topology: {}\n"
            "- Baseline future peak: {} m\n"
            "- Current future peak: {} m\n"
            "- Future-peak ratio: {}\n"
            "- Baseline repeat peak: {} m\n"
            "- Current repeat peak: {} m\n"
            "- Baseline future/repeat: {}\n"
            "- Current future/repeat: {}\n"
            "- Baseline formal sensor peak: {}\n"
            "- Current formal sensor peak: {}\n"
            "- Baseline post-probe RMSE: {} m\n"
            "- Current post-probe RMSE: {} m\n"
            "- Baseline extraction progress: {}\n"
            "- Current extraction progress: {}"
        ).format(
            config["repair"]["stop_after_this_trial"],
            baseline["latch_topology"],
            latch_topology,
            baseline["future_peak_visible_rmse_m"],
            current["future_peak_visible_rmse_m"],
            (current["future_peak_visible_rmse_m"]
             / baseline["future_peak_visible_rmse_m"]),
            baseline["repeat_peak_visible_rmse_m"],
            current["repeat_peak_visible_rmse_m"],
            baseline["future_branch_to_repeat_ratio"],
            current["future_branch_to_repeat_ratio"],
            baseline["formal_sensor_peak_fused_gap"],
            current["formal_sensor_peak_fused_gap"],
            baseline["post_probe_rmse_m"],
            current["post_probe_rmse_m"],
            baseline["final_extraction_progress_m"],
            current["final_extraction_progress_m"],
        )
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
- Probe delta XYZ: {probe_delta}
- Probe direction unit: {probe_direction}
- Probe forward/hold/return: {probe_forward_steps}/{probe_hold_steps}/{probe_return_steps}
- Probe speed: {probe_speed_m_s} m/s
- Jam clearance: {clearance_mm} mm
- Hidden latch topology: {latch_topology}
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

{load_path_text}

{amplification_text}

{direction_text}

{horizon_text}

{future_consequence_text}

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Internal cable constraint force used for Gate 3: No
- Internal cable constraint force used only as oracle mechanism diagnostic: {oracle_diagnostic}
- Latch-contact bead mask used as formal sensor: No
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
        probe_delta=probe_delta,
        probe_direction=probe_direction,
        probe_speed_m_s=probe_speed_m_s,
        probe_forward_steps=config["motion"]["probe_forward_steps"],
        probe_hold_steps=config["motion"]["probe_hold_steps"],
        probe_return_steps=config["motion"]["probe_return_steps"],
        clearance_mm=clearance_mm,
        latch_topology=latch_topology,
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
        load_path_text=load_path_text,
        amplification_text=amplification_text,
        direction_text=direction_text,
        horizon_text=horizon_text,
        future_consequence_text=future_consequence_text,
        oracle_diagnostic="Yes" if load_path is not None else "No",
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
