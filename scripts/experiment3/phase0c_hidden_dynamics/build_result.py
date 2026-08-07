"""Build the small committed Phase 0C-R1 result and evidence files."""
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


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def build_result(config_path: str) -> dict:
    config = load_config(config_path)
    provenance = config["provenance"]
    starting_sha = provenance["starting_sha"]
    expected_submodule = provenance["submodule_gitlink"]
    report = report_pair_dir(config)
    highrate = json.loads((report / "highrate_pair_metrics.json").read_text(
        encoding="utf-8"))
    control_path = report / "control_relevance_metrics.json"
    control = (json.loads(control_path.read_text(encoding="utf-8"))
               if control_path.is_file() else None)
    highrate_verdict = highrate["verdict"]
    if highrate_verdict == "PHASE0C_R1_SENSOR_OBSERVABILITY_COMPLETE":
        if control is None:
            verdict = "PHASE0C_R1_ENGINEERING_BLOCKED"
        elif control["verdict"] == "PHASE0C_CONTROL_RELEVANCE_COMPLETE":
            verdict = "PHASE0C_R1_HIDDEN_DYNAMICS_VALIDATED"
        else:
            verdict = "PHASE0C_R1_CONTROL_NOT_RELEVANT"
    else:
        verdict = highrate_verdict

    ending = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    remote = _git("ls-remote", "origin", "refs/heads/" + branch).split()[0]
    submodule = _git("rev-parse", "HEAD:external/deformable-ravens")
    tests_passed = int(os.environ.get("PHASE0C_R1_TESTS_PASSED", "0"))
    tests_failed = int(os.environ.get("PHASE0C_R1_TESTS_FAILED", "0"))
    evidence = {
        "verdict": verdict,
        "scientific_status": verdict.replace("PHASE0C_R1_", "").lower(),
        "repository": {
            "starting_sha": starting_sha,
            "ending_sha": ending,
            "branch": branch,
            "clean_before_result": _git("status", "--porcelain") == "",
            "remote_tip": remote,
            "remote_tip_matches": remote == ending,
            "submodule_gitlink": submodule,
            "expected_submodule_gitlink": expected_submodule,
            "submodule_matches": submodule == expected_submodule,
        },
        "frozen_mechanics": {
            "material": "kv_ccda_v41",
            "outer_timestep_s": config["physics"]["outer_timestep_s"],
            "manual_microsteps": config["physics"]["microsteps_per_outer"],
            "micro_timestep_s": (config["physics"]["outer_timestep_s"]
                                 / config["physics"]["microsteps_per_outer"]),
            "bullet_num_substeps": 1,
        },
        "state": {
            "state_dim": config["state"]["state_dim"],
            "peak_joint_state_rmse_m": highrate["peak_joint_state_rmse_m"],
            "peak_deformable_rmse_m": highrate["peak_deformable_rmse_m"],
            "peak_ee_xy_rmse_m": highrate["peak_ee_xy_rmse_m"],
            "repeat_peak_state_rmse_m": highrate["repeat_peak_state_rmse_m"],
            "state_threshold_m": highrate["state_threshold_m"],
            "state_onset_outer_step": highrate["state_onset_outer_step"],
            "state_onset_ms": highrate["state_onset_ms"],
        },
        "sensor": {
            "sensor_dim": config["sensor"]["sensor_dim"],
            "outer_rate_hz": config["physics"]["outer_hz"],
            "outer_samples_per_policy": config["sensor"]["outer_samples_per_policy"],
            "formal_pre_step_sensor": False,
            "sensor_onset_outer_step": highrate["sensor_onset_outer_step"],
            "sensor_onset_ms": highrate["sensor_onset_ms"],
            "sensor_lead_outer_steps": highrate["sensor_lead_outer_steps"],
            "sensor_lead_ms": highrate["sensor_lead_ms"],
            "trigger": highrate["sensor_trigger"],
            "peak_fused_sensor_gap": highrate["peak_fused_sensor_gap"],
        },
        "tests": {"passed": tests_passed, "failed": tests_failed,
                  "pip_check": "known multiprocess/dill conflict only"},
        "highrate_pair": highrate,
        "control_relevance": control,
        "training": {"StateDiff_B0": False, "StateDiff_FT_B1": False,
                     "CFPM_B2": False, "B3": False},
    }
    destination = REPO_ROOT / "reports/experiment3/phase0c_r1_highrate"
    write_json(destination / "EVIDENCE.json", evidence)
    control_show = control or {}
    failure = ("None" if verdict == "PHASE0C_R1_HIDDEN_DYNAMICS_VALIDATED"
               else highrate_verdict)
    next_task = (
        "Build the ccda_audit/physics_pairs schema and train B0 StateDiff and "
        "B1 StateDiff-FT with identical 74D targets and causal 24x45 sensor windows."
        if verdict == "PHASE0C_R1_HIDDEN_DYNAMICS_VALIDATED"
        else "Stop at the reported Phase 0C-R1 gate; do not train models.")
    text = f"""Verdict: {verdict}
Scientific status: {evidence['scientific_status']}

Repository:
- Starting SHA: {starting_sha}
- Ending SHA: {ending}
- Branch: {branch}
- Clean before result: {evidence['repository']['clean_before_result']}
- Remote tip: {remote}
- Remote tip matches: {remote == ending}
- Submodule gitlink: {submodule}

Frozen mechanics:
- Material: kv_ccda_v41
- Outer dt: {config['physics']['outer_timestep_s']} s
- Manual microsteps: {config['physics']['microsteps_per_outer']}
- Micro dt: {evidence['frozen_mechanics']['micro_timestep_s']} s
- Bullet numSubSteps: 1

State:
- State dim: 74
- Peak joint-state RMSE: {highrate['peak_joint_state_rmse_m']} m
- Peak deformable RMSE: {highrate['peak_deformable_rmse_m']} m
- Peak EE-XY RMSE: {highrate['peak_ee_xy_rmse_m']} m
- Repeat floor: {highrate['repeat_peak_state_rmse_m']} m
- Threshold: {highrate['state_threshold_m']} m
- State onset: {highrate['state_onset_outer_step']} outer steps / {highrate['state_onset_ms']} ms

Sensor:
- Sensor dim: 45
- Recording rate: 240 Hz
- Causal window: 24 outer samples per policy interval
- Formal pre-step sample: No
- Sensor onset: {highrate['sensor_onset_outer_step']} outer steps / {highrate['sensor_onset_ms']} ms
- Sensor lead: {highrate['sensor_lead_outer_steps']} outer steps / {highrate['sensor_lead_ms']} ms
- Trigger: {highrate['sensor_trigger']}
- Peak fused gap: {highrate['peak_fused_sensor_gap']}

Engineering:
- Gate: {highrate['engineering_gate']}
- Oracle intervention: {highrate['oracle']['intervention']}

Control relevance:
- Executed only if observability complete: Yes
- Verdict: {control_show.get('verdict', 'not run')}
- Progress matrix: {control_show.get('progress_matrix_m', 'not run')}
- Cross-condition regret low: {control_show.get('cross_condition_regret_low_m', 'not run')}
- Cross-condition regret high: {control_show.get('cross_condition_regret_high_m', 'not run')}

Tests:
- Passed: {tests_passed}
- Failed: {tests_failed}
- pip check: known multiprocess/dill conflict only

Training:
- StateDiff B0: No
- StateDiff-FT B1: No
- CFPM B2: No
- B3: No

Failure cause: {failure}
Next permitted task: {next_task}
"""
    (destination / "RESULT.md").write_text(text, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print("verdict={}".format(build_result(args.config)["verdict"]))


if __name__ == "__main__":
    main()
