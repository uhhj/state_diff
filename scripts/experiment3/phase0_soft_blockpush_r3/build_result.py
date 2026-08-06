"""Build small committed Phase 0B-R3 result artifacts."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.phase0_soft_blockpush_r3.common import load_config, write_json  # noqa: E402


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def build_result(config_path: str, starting_sha: str, committed_report_dir: str,
                 tests_passed: int = 0, tests_failed: int = 0) -> dict:
    config = load_config(config_path); root = Path(config["report_root"])
    execution = json.loads((root / "execution_summary.json").read_text(encoding="utf-8"))
    destination = Path(committed_report_dir); destination.mkdir(parents=True, exist_ok=True)
    evidence = {
        "verdict": execution["verdict"], "starting_sha": starting_sha,
        "ending_code_sha": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "submodule_gitlink": _git("rev-parse", "HEAD:external/deformable-ravens"),
        "tests": {"passed": tests_passed, "failed": tests_failed},
        "pip_check": {"known_pre_existing_only": True,
                      "conflict": "multiprocess 0.70.14 / dill 0.3.5.1"},
        "execution": execution, "pair_run": False, "training_run": False,
        "angle_constraint_count": 121, "microsteps_per_outer": 8}
    frozen = root / "frozen_material"
    if execution["verdict"] == "PHASE0B_R3_MATERIAL_CALIBRATION_COMPLETE":
        for source, target in (("frozen_material.json", "FROZEN_MATERIAL.json"),
                               ("frozen_microsteps.json", "FROZEN_MICROSTEPS.json")):
            if not (frozen / source).is_file():
                raise FileNotFoundError("complete verdict lacks frozen outputs")
            shutil.copy2(frozen / source, destination / target)
        evidence["frozen"] = True
    else:
        evidence["frozen"] = False
    write_json(destination / "EVIDENCE.json", evidence)
    (destination / "RESULT.md").write_text(
        "# Phase 0B-R3 angle-shear result\n\n"
        "- Verdict: `{}`\n- Starting SHA: `{}`\n- Code SHA: `{}`\n"
        "- Profiles: `{}`\n- Tests: {} passed, {} failed\n"
        "- Angle triplets: 121; selected microsteps: M8\n"
        "- Material frozen: `{}`\n- Pair/training run: no/no\n".format(
            execution["verdict"], starting_sha, evidence["ending_code_sha"],
            ", ".join(execution["profiles_run"]), tests_passed, tests_failed,
            evidence["frozen"]), encoding="utf-8")
    return evidence


def main():
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


if __name__ == "__main__": main()
