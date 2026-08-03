#!/usr/bin/env python3
"""Controller for Stage-I fresh untouched evaluation-set acquisition and seal."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _bootstrap() -> Path:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


ROOT_FROM_FILE = _bootstrap()
from ccda_phase3 import phase314b_r259_stagei_fresh_eval_seal as stagei  # noqa: E402


def _run(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd), env=dict(env), check=True)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    stagei.atomic_write_once(path, stagei.stable_json_bytes(value))


def execute(root: Path, python_bin: Path, implementation_commit: str, workers: int) -> Mapping[str, Any]:
    repo = Path(root).resolve()
    repository = stagei.validate_repository(repo, implementation_commit)
    stagei.validate_seed_contract()
    dataset_root = repo / stagei.DATASET_RELATIVE_ROOT
    dataset_root.mkdir(parents=True, exist_ok=False)
    marker = {
        "phase": stagei.PHASE,
        "schema": "phase314b_r259_stagei_attempt_started_v1",
        "implementation_commit": implementation_commit,
        "visible_seed_start": stagei.FRESH_VISIBLE_SEED_START,
        "visible_seed_count": stagei.FRESH_VISIBLE_SEED_COUNT,
        "rerun_authorized": False,
    }
    _write_json(repo / stagei.ATTEMPT_MARKER_RELATIVE, marker)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo),
            str(repo / "scripts"),
            str(repo / "external/deformable-ravens"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PHASE313_ALLOW_DATA_GENERATION"] = "1"
    env["PHASE313_DATA_GENERATION_CONFIRMED"] = "1"
    _run(
        [
            str(python_bin),
            "scripts/phase3_13_generate_raw.py",
            "--root",
            str(repo),
            "--split",
            stagei.FRESH_SPLIT_NAME,
            "--seed-start",
            str(stagei.FRESH_VISIBLE_SEED_START),
            "--num-seeds",
            str(stagei.FRESH_VISIBLE_SEED_COUNT),
            "--output-root",
            stagei.RAW_RELATIVE_ROOT,
            "--hz",
            str(stagei.EXPECTED_HZ),
            "--workers",
            str(max(1, int(workers))),
        ],
        cwd=repo,
        env=env,
    )
    stagei.assert_clean_worktree(repo, "main after raw generation")
    stagei.assert_clean_worktree(repo / "external/deformable-ravens", "submodule after raw generation")
    raw_audit = stagei.audit_raw_dataset(
        repo / stagei.RAW_RELATIVE_ROOT,
        implementation_commit=implementation_commit,
        submodule_commit=stagei.EXPECTED_SUBMODULE,
    )
    window_manifest = stagei.build_fresh_windows(repo)
    stagei.assert_clean_worktree(repo, "main after window build")
    inventory = stagei.build_inventory(dataset_root)
    _write_json(repo / stagei.INVENTORY_RELATIVE, inventory)
    stagei.validate_inventory(dataset_root, inventory)
    seal = stagei.build_seal(
        root=repo,
        repository=repository,
        raw_audit=raw_audit,
        window_manifest=window_manifest,
        inventory=inventory,
    )
    _write_json(repo / stagei.SEAL_RELATIVE, seal)
    summary = stagei.build_summary(repository=repository, seal=seal)
    _write_json(repo / stagei.SUCCESS_REPORT, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--implementation-commit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(json.dumps({"import_smoke": True, "phase": stagei.PHASE}, sort_keys=True))
        return 0
    if not args.implementation_commit:
        raise SystemExit("--implementation-commit is required")
    repo = Path(args.root).resolve()
    repository = None
    try:
        summary = execute(
            repo,
            Path(args.python_bin).resolve(),
            str(args.implementation_commit),
            int(args.workers),
        )
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        try:
            if repo.is_dir():
                repository = {
                    "root": str(repo),
                    "head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=str(repo), text=True
                    ).strip(),
                    "submodule_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(repo / "external/deformable-ravens"),
                        text=True,
                    ).strip(),
                }
        except BaseException:
            repository = None
        blocked = stagei.blocked_report(
            repository=repository,
            error=error,
            dataset_root=repo / stagei.DATASET_RELATIVE_ROOT,
        )
        blocked_path = repo / stagei.BLOCKED_REPORT
        if not blocked_path.exists():
            stagei.atomic_write_once(blocked_path, stagei.stable_json_bytes(blocked))
        print(json.dumps(blocked, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
