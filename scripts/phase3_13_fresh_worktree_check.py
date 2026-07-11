#!/usr/bin/env python3
"""Validate the pending Phase3.13 code in an isolated Git worktree."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


def main():
    root = Path("/data/state_diff2")
    target = Path("/tmp/state_diff_phase313_check")
    if target.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(target)], cwd=root, check=False)
        shutil.rmtree(target, ignore_errors=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(target), "HEAD"], cwd=root, check=True)
    try:
        for relative in ["ccda_phase3", "scripts", "tests"]:
            source = root / relative
            destination = target / relative
            destination.mkdir(parents=True, exist_ok=True)
            patterns = ["*.py"] if relative == "ccda_phase3" else (["phase3_13_*.py", "phase3_13_*.sh"] if relative == "scripts" else ["test_ccda_slack_breakaway.py", "test_ccda_state_v2.py", "test_ccda_dataset_windows.py", "test_phase3_13_legacy_purge.py"])
            for pattern in patterns:
                for path in source.glob(pattern):
                    shutil.copy2(path, destination / path.name)
        sub_source = root / "external/deformable-ravens"
        sub_target = target / "external/deformable-ravens"
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=target, check=True)
        for relative in ["ravens/environment.py", "ravens/tasks/__init__.py", "ravens/tasks/ccda_slack_breakaway.py", "ravens/tasks/ccda_slack_cable_v2.py"]:
            destination = sub_target / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sub_source / relative, destination)
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{target}:{target / 'scripts'}:{sub_target}"
        subprocess.run(["/miniforge3/envs/coord_bimanual/bin/python", "-m", "py_compile", *[str(path) for path in (target / "ccda_phase3").glob("*.py")], *[str(path) for path in (target / "scripts").glob("phase3_13_*.py")]], cwd=target, env=env, check=True)
        subprocess.run(["/miniforge3/envs/coord_bimanual/bin/pytest", "-q", "tests/test_ccda_slack_breakaway.py", "tests/test_ccda_state_v2.py", "tests/test_ccda_dataset_windows.py", "tests/test_phase3_13_legacy_purge.py"], cwd=target, env=env, check=True)
        payload = {"verdict": "PASS", "worktree": str(target), "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip(), "pending_phase313_files_overlaid": True}
        (root / "reports/phase3_13_fresh_worktree_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(target)], cwd=root, check=False)
        shutil.rmtree(target, ignore_errors=True)


if __name__ == "__main__": main()
