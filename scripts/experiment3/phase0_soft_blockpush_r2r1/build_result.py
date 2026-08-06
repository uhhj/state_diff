"""Build small committed Phase 0B-R2-R1 result artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r2r1.common import (  # noqa: E402
    load_config, write_json)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def build_result(config_path: str, starting_sha: str,
                 committed_report_dir: str, tests_passed: int = 0,
                 tests_failed: int = 0) -> dict:
    """Summarize server evidence and copy frozen JSON only on completion."""
    config = load_config(config_path)
    report_root = Path(config["report_root"])
    confirmation = json.loads((report_root / "mechanics_validation" /
                               "summary.json").read_text(encoding="utf-8"))
    execution = json.loads((report_root / "execution_summary.json").read_text(
        encoding="utf-8"))
    verdict = execution["verdict"]
    evidence = {
        "verdict": verdict, "starting_sha": starting_sha,
        "ending_code_sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "submodule_gitlink": _git("rev-parse", "HEAD:external/deformable-ravens"),
        "tests": {"passed": tests_passed, "failed": tests_failed},
        "pip_check": {"known_pre_existing_only": True,
                      "conflict": "multiprocess 0.70.14 requires dill>=0.3.6; installed dill is 0.3.5.1"},
        "prior_r2_microsteps": 8, "family_confirmation": confirmation,
        "execution": execution, "pair_run": False, "training_run": False}
    destination = Path(committed_report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if verdict == "PHASE0B_R2R1_MATERIAL_CALIBRATION_COMPLETE":
        frozen = report_root / "frozen_material"
        required = {"frozen_material.json": "FROZEN_MATERIAL.json",
                    "frozen_microsteps.json": "FROZEN_MICROSTEPS.json"}
        if not all((frozen / source).is_file() for source in required):
            raise FileNotFoundError("complete verdict lacks frozen outputs")
        for source, target in required.items():
            shutil.copy2(frozen / source, destination / target)
        evidence["frozen"] = {
            "material": "FROZEN_MATERIAL.json",
            "microsteps": "FROZEN_MICROSTEPS.json",
            "manifest": str(frozen / "calibration_manifest.json")}
    else:
        evidence["frozen"] = None
    write_json(destination / "EVIDENCE.json", evidence)
    axial, shear, table = (execution.get("axial"), execution.get("shear"),
                           execution.get("table_settle"))
    def value(section: object, key: str) -> object:
        return section.get(key) if isinstance(section, dict) else "not run"
    confirmation_failure = (
        "A1.5 M8 two-node energy increase fraction {} exceeded the fixed 0.02 gate"
        .format(confirmation.get("two_node", {}).get("8", {}).get(
            "energy_increase_fraction")))
    failure = {
        "PHASE0B_R2R1_ENGINEERING_BLOCKED": confirmation_failure,
        "PHASE0B_R2R1_NO_PROFILE_PASSED": execution.get("reason"),
        "PHASE0B_R2R1_TABLE_SETTLE_FAILED": "table-settle gate failed",
        "PHASE0B_R2R1_MATERIAL_CALIBRATION_COMPLETE": "none"}[verdict]
    next_task = ("define a separate strict-Pair integration phase" if verdict.endswith("COMPLETE")
                 else "stop and address only the reported calibration gate; do not run Pair")
    markdown = """# Phase 0B-R2-R1 restricted low-stiffness result

## Verdict

- Final verdict: `{verdict}`.
- Starting SHA: `{starting}`.
- Ending implementation SHA: `{ending}`.
- Frozen gitlink: `{gitlink}`.
- Tests: {passed} passed, {failed_tests} failed.
- Known pip conflict only: yes.

## Family confirmation

- Profile: `kv_r2r1_a1p50_z025`.
- M8/M16 verdict: `{confirmation}`.
- Maximum relative error: `{maximum_error}`.
- M retained for calibration: `{retained}`; prior M8 selection reopened: no.

## Calibration

- Profiles run: `{profiles}`.
- Axial verdict: `{axial_verdict}`; peak primary: `{axial_peak}` m.
- Shear verdict: `{shear_verdict}`; peak primary: `{shear_peak}` m.
- Table verdict: `{table_verdict}`.
- Material frozen: `{frozen}`.
- Pair run: no; training run: no.

Failure cause: `{failure}`.

Next permitted task: {next_task}.
""".format(
        verdict=verdict, starting=starting_sha, ending=evidence["ending_code_sha"],
        gitlink=evidence["submodule_gitlink"], passed=tests_passed,
        failed_tests=tests_failed, confirmation=confirmation["verdict"],
        maximum_error=confirmation.get("maximum_relative_error"),
        retained=confirmation.get("retained_microsteps"),
        profiles=", ".join(execution.get("profiles_run", [])),
        axial_verdict=value(axial, "verdict"),
        axial_peak=value(axial, "peak_primary_displacement_m"),
        shear_verdict=value(shear, "verdict"),
        shear_peak=value(shear, "peak_primary_displacement_m"),
        table_verdict=value(table, "verdict"), frozen=evidence["frozen"] is not None,
        failure=failure, next_task=next_task)
    (destination / "RESULT.md").write_text(markdown, encoding="utf-8")
    return evidence


def main() -> None:
    """Build the committed result artifacts CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--starting-sha", required=True)
    parser.add_argument("--committed-report-dir", required=True)
    parser.add_argument("--tests-passed", type=int, default=0)
    parser.add_argument("--tests-failed", type=int, default=0)
    args = parser.parse_args()
    print("verdict={}".format(build_result(
        args.config, args.starting_sha, args.committed_report_dir,
        args.tests_passed, args.tests_failed)["verdict"]))


if __name__ == "__main__":
    main()
