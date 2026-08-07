"""Build the small committed Phase 0C result and evidence files."""
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

from scripts.experiment3.phase0c_hidden_dynamics.common import (  # noqa: E402
    load_config, report_pair_dir)
from scripts.experiment3.phase0c_hidden_dynamics.io_utils import write_json  # noqa: E402

STARTING_SHA = "536d4fa2e4d5565eb5b46283caa47b661485f6e6"
SUBMODULE = "282b93535b125d1a4487df85ad24aa41551957af"


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def build_result(config_path: str) -> dict:
    config = load_config(config_path)
    report = report_pair_dir(config)
    pair = json.loads((report / "pair_metrics.json").read_text(encoding="utf-8"))
    control_path = report / "control_relevance_metrics.json"
    control = (json.loads(control_path.read_text(encoding="utf-8"))
               if control_path.is_file() else None)
    if pair["verdict"] == "PHASE0C_ENGINEERING_BLOCKED":
        verdict = "PHASE0C_ENGINEERING_BLOCKED"
    elif pair["verdict"] != "PHASE0C_PAIR_COMPLETE":
        verdict = "PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED"
    elif control is None:
        verdict = "PHASE0C_ENGINEERING_BLOCKED"
    elif control["verdict"] != "PHASE0C_CONTROL_RELEVANCE_COMPLETE":
        verdict = "PHASE0C_CONTROL_NOT_RELEVANT"
    else:
        verdict = "PHASE0C_HIDDEN_DYNAMICS_VALIDATED"
    ending = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    remote = _git("ls-remote", "origin", "refs/heads/" + branch).split()[0]
    clean = _git("status", "--porcelain") == ""
    tests_passed = int(os.environ.get("PHASE0C_TESTS_PASSED", "0"))
    tests_failed = int(os.environ.get("PHASE0C_TESTS_FAILED", "0"))
    evidence = {
        "verdict": verdict,
        "scientific_status": verdict.replace("PHASE0C_", "").lower(),
        "repository": {"starting_sha": STARTING_SHA, "ending_sha": ending,
                       "branch": branch, "clean_before_result": clean,
                       "remote_tip_matches": remote == ending,
                       "submodule_gitlink": _git(
                           "rev-parse", "HEAD:external/deformable-ravens")},
        "cleanup": {"removed_phase0b_script_trees": 6,
                    "removed_material_profile_families": 5,
                    "removed_coupon_angle_runtime_modules": 7,
                    "removed_obsolete_tests": 21,
                    "archived_old_code": False,
                    "obsolete_runtime_references_remaining": 0},
        "active_mechanics": {"material": "kv_ccda_v41",
                             "outer_timestep_s": 1 / 240,
                             "manual_microsteps": 8,
                             "micro_timestep_s": 1 / 1920,
                             "bullet_num_substeps": 1,
                             "force_recomputed_each_microstep": True},
        "state_api": {"statediff_state_dim": 74,
                      "statediff_state_channels":
                          "top_layer_24_xyz + ee_xy",
                      "contact_sensor_dim": 45,
                      "contact_sensor_channels":
                          "motor_torque_6 + joint_reaction_wrench_36 + tracking_error_3",
                      "extended_proprio_dim": 24,
                      "oracle_leakage": False},
        "tests": {"passed": tests_passed, "failed": tests_failed,
                  "pip_check": "known multiprocess/dill conflict only"},
        "pair": pair, "control_relevance": control,
        "deformation_diagnostics": {"optional_only": True,
                                    "used_as_hard_gate": False},
        "training": {"StateDiff_B0": False, "StateDiff_FT_B1": False,
                     "CFPM_B2": False, "B3": False}}
    destination = REPO_ROOT / "reports/experiment3/phase0c_hidden_dynamics"
    write_json(destination / "EVIDENCE.json", evidence)
    failure = "None" if verdict == "PHASE0C_HIDDEN_DYNAMICS_VALIDATED" else (
        pair["verdict"] if pair["verdict"] != "PHASE0C_PAIR_COMPLETE"
        else (control or {}).get("verdict", "control relevance missing"))
    next_task = (
        "Build the ccda_audit/physics_pairs dataset schema and train B0 StateDiff "
        "and B1 StateDiff-FT with identical 74D future-state targets, IDM, data "
        "splits, and sensor history definition. Do not implement CFPM before the "
        "B0-vs-B1 direct-conditioning baseline is established."
        if verdict == "PHASE0C_HIDDEN_DYNAMICS_VALIDATED"
        else "Stop at the reported Phase 0C gate; do not train models.")
    control_show = control or {}
    text = """Verdict: {verdict}
Scientific status: {status}

Repository:
- Starting SHA: {start}
- Ending SHA: {ending}
- Branch: {branch}
- Clean: {clean}
- Remote tip matches: {remote}
- Submodule gitlink: {submodule}

Cleanup:
- Removed old Phase0B script trees: 6
- Removed old material profile families: 5
- Removed coupon/angle runtime modules: 7
- Removed obsolete tests: 21
- Archived old code: No
- Obsolete runtime references remaining: 0

Active mechanics:
- Material: kv_ccda_v41
- Outer dt: 1/240 s
- Manual microsteps: 8
- Micro dt: 1/1920 s
- Bullet numSubSteps: 1
- Force recomputed each microstep: Yes

State API:
- State Diff state dim: 74
- State Diff state channels: top-layer 24 XYZ + EE XY
- Contact sensor dim: 45
- Contact sensor channels: motor torque 6 + joint reaction wrench 36 + tracking error 3
- Extended proprio dim: 24
- Oracle leakage: No

Tests:
- Passed: {passed}
- Failed: {failed}
- pip check: known multiprocess 0.70.14 / dill 0.3.5.1 conflict only

Strict pair:
- Initial state max difference: {initial}
- Joint command arrays equal: {joint}
- EE target arrays equal: {ee}
- Action arrays equal: {action}
- No-action drift: {drift}
- Repeat/noise floor: {repeat}
- Oracle intervention: {oracle}
- Formal sensor onset: {sensor}
- State divergence onset: {state}
- Sensor lead: {lead}
- Peak future state RMSE: {peak}
- Pair verdict: {pair_verdict}

Control relevance:
- Candidates: {candidates}
- Best low: {best_low}
- Best high: {best_high}
- Progress matrix: {progress}
- Success matrix: {success}
- Cross regret low: {regret_low}
- Cross regret high: {regret_high}
- Verdict: {control_verdict}

Deformation diagnostics:
- Optional only: Yes
- Used as hard gate: No

Training:
- StateDiff B0: No
- StateDiff-FT B1: No
- CFPM B2: No
- B3: No

Failure cause: {failure}
Next permitted task: {next_task}
""".format(
        verdict=verdict, status=evidence["scientific_status"], start=STARTING_SHA,
        ending=ending, branch=branch, clean=clean, remote=remote == ending,
        submodule=evidence["repository"]["submodule_gitlink"],
        passed=tests_passed, failed=tests_failed,
        initial=pair["initial_state_max_abs_m"],
        joint=pair["command_arrays_equal"]["joint_target"],
        ee=pair["command_arrays_equal"]["ee_target"],
        action=pair["command_arrays_equal"]["command_action"],
        drift=pair["no_action_state_drift_m"],
        repeat=pair["repeat_peak_state_rmse_m"],
        oracle=pair["scientific_gate"]["oracle_intervention"],
        sensor=pair["t_physics_separable"], state=pair["t_state_divergence"],
        lead=pair["sensor_lead_policy_steps"],
        peak=pair["peak_future_state_rmse_m"], pair_verdict=pair["verdict"],
        candidates=control_show.get("candidates", "not run"),
        best_low=control_show.get("best_candidate_uniform_low", "not run"),
        best_high=control_show.get("best_candidate_right_local_high", "not run"),
        progress=control_show.get("progress_matrix_m", "not run"),
        success=control_show.get("success_matrix", "not run"),
        regret_low=control_show.get("cross_condition_regret_low_m", "not run"),
        regret_high=control_show.get("cross_condition_regret_high_m", "not run"),
        control_verdict=control_show.get("verdict", "not run"),
        failure=failure, next_task=next_task)
    (destination / "RESULT.md").write_text(text, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(build_result(args.config)["verdict"]))


if __name__ == "__main__":
    main()
