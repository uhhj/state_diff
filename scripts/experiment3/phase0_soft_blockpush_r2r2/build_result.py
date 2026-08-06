"""Build small committed Phase 0B-R2-R2 result artifacts."""
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

from scripts.experiment3.phase0_soft_blockpush_r2r2.common import (  # noqa: E402
    load_config, write_json)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def failure_text(execution: dict) -> str:
    """Return the most specific terminal coupon failure available."""
    shear, axial = execution.get("shear"), execution.get("axial")
    if isinstance(shear, dict):
        return ("shear verdict={}; primary={:.6f} mm; rigid={:.6f} mm; "
                "edge_ratio=[{:.6f}, {:.6f}]").format(
            shear["verdict"], 1000 * shear["peak_primary_displacement_m"],
            1000 * shear["peak_rigid_aligned_rmse_m"],
            shear["structural_edge_ratio_min"],
            shear["structural_edge_ratio_max"])
    if isinstance(axial, dict):
        return "axial verdict={}; primary={:.6f} mm".format(
            axial["verdict"], 1000 * axial["peak_primary_displacement_m"])
    return execution.get("reason", "unknown")


def build_result(config_path: str, starting_sha: str,
                 committed_report_dir: str, tests_passed: int = 0,
                 tests_failed: int = 0) -> dict:
    """Summarize execution and copy frozen JSON only after completion."""
    config = load_config(config_path)
    report_root = Path(config["report_root"])
    execution = json.loads((report_root / "execution_summary.json").read_text(
        encoding="utf-8"))
    evidence = {"verdict": execution["verdict"], "starting_sha": starting_sha,
                "ending_code_sha": _git("rev-parse", "HEAD"),
                "branch": _git("branch", "--show-current"),
                "submodule_gitlink": _git(
                    "rev-parse", "HEAD:external/deformable-ravens"),
                "tests": {"passed": tests_passed, "failed": tests_failed},
                "pip_check": {"known_pre_existing_only": True,
                              "conflict": "multiprocess 0.70.14 / dill 0.3.5.1"},
                "execution": execution, "pair_run": False, "training_run": False}
    destination = Path(committed_report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frozen = report_root / "frozen_material"
    if execution["verdict"] == "PHASE0B_R2R2_MATERIAL_CALIBRATION_COMPLETE":
        required = {"frozen_material.json": "FROZEN_MATERIAL.json",
                    "frozen_microsteps.json": "FROZEN_MICROSTEPS.json"}
        if not all((frozen / source).is_file() for source in required):
            raise FileNotFoundError("complete verdict lacks frozen outputs")
        for source, target in required.items():
            shutil.copy2(frozen / source, destination / target)
        evidence["frozen"] = {"material": "FROZEN_MATERIAL.json",
                              "microsteps": "FROZEN_MICROSTEPS.json",
                              "manifest": str(frozen / "calibration_manifest.json")}
    else:
        evidence["frozen"] = None
    write_json(destination / "EVIDENCE.json", evidence)
    axial, shear, table = (execution.get("axial"), execution.get("shear"),
                           execution.get("table_settle"))
    def value(section: object, key: str) -> object:
        return section.get(key) if isinstance(section, dict) else "not run"
    next_task = ("integrate frozen material and M=8 into strict HLF-SBP Pair"
                 if execution["verdict"].endswith("MATERIAL_CALIBRATION_COMPLETE")
                 else "stop at the reported calibration gate; do not run Pair")
    markdown = """# Phase 0B-R2-R2 decoupled shear result

- Verdict: `{verdict}`.
- Starting SHA: `{starting}`.
- Ending implementation SHA: `{ending}`.
- Tests: {passed} passed, {failed} failed.
- Profiles run: `{profiles}`.
- Axial: `{axial_verdict}`; primary `{axial_primary}` m.
- Shear: `{shear_verdict}`; legacy `{legacy}`; primary `{shear_primary}` m.
- Table: `{table_verdict}`.
- Material frozen: `{frozen}`.
- Pair/training run: no/no.
- Failure cause: `{cause}`.
- Next permitted task: {next_task}.
""".format(
        verdict=execution["verdict"], starting=starting_sha,
        ending=evidence["ending_code_sha"], passed=tests_passed,
        failed=tests_failed, profiles=", ".join(execution["profiles_run"]),
        axial_verdict=value(axial, "verdict"),
        axial_primary=value(axial, "peak_primary_displacement_m"),
        shear_verdict=value(shear, "verdict"),
        legacy=value(shear, "legacy_r2_verdict"),
        shear_primary=value(shear, "peak_primary_displacement_m"),
        table_verdict=value(table, "verdict"), frozen=evidence["frozen"] is not None,
        cause="none" if evidence["frozen"] else failure_text(execution),
        next_task=next_task)
    (destination / "RESULT.md").write_text(markdown, encoding="utf-8")
    return evidence


def main() -> None:
    """Run the result builder CLI."""
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
