#!/usr/bin/env python3
"""Controller for r2.6.0 Stage-A data-universe generation and seal."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccda_phase3 import phase314b_r260_stagea_data_universe_genesis as stagea  # noqa: E402


def tree_identity(path: Path) -> str:
    root = Path(path)
    if not root.exists():
        return stagea.sha256_bytes(stagea.stable_json_bytes([]))
    records = []
    for item in sorted((value for value in root.rglob("*") if value.is_file()), key=lambda value: value.relative_to(root).as_posix()):
        records.append(
            {
                "path": item.relative_to(root).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": stagea.sha256_file(item),
            }
        )
    return stagea.sha256_bytes(stagea.stable_json_bytes(records))


def _worker_command(
    *,
    root: Path,
    dataset_root: Path,
    write_ahead: Path,
    role: str,
    seed: int,
    implementation_commit: str,
) -> Mapping[str, Any]:
    command = [
        sys.executable,
        str(root / "scripts/phase3_14b_r260_stagea_worker.py"),
        "--root",
        str(root),
        "--dataset-root",
        str(dataset_root),
        "--write-ahead",
        str(write_ahead),
        "--role",
        role,
        "--seed",
        str(seed),
        "--implementation-commit",
        implementation_commit,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(root),
            str(root / "scripts"),
            str(root / "external/deformable-ravens"),
            environment.get("PYTHONPATH", ""),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise stagea.StageAError("pair worker produced no result")
    return json.loads(lines[-1])


def execute(root: Path, implementation_commit: str, workers: int) -> Mapping[str, Any]:
    if (
        os.environ.get("PHASE314B_R260_ALLOW_DATA_GENERATION") != "1"
        or os.environ.get("PHASE314B_R260_DATA_GENERATION_CONFIRMED") != "1"
    ):
        raise stagea.StageAError("r2.6.0 data-generation authorization gates are missing")
    repo = Path(root).resolve()
    repository = stagea.validate_repository(repo, implementation_commit)
    stagea.role_contract()
    dataset_root = repo / stagea.DATASET_RELATIVE_ROOT
    write_ahead = Path(str(repo) + stagea.WRITE_AHEAD_NAME)
    legacy_write_ahead = Path(str(repo) + stagea.LEGACY_STAGEJ_WRITE_AHEAD_NAME)
    legacy_before = tree_identity(legacy_write_ahead)
    if write_ahead.exists():
        raise stagea.StageAError("Stage-A write-ahead already exists")
    if dataset_root.exists():
        raise stagea.StageAError("Stage-A dataset root already exists")
    write_ahead.mkdir(parents=True, exist_ok=False)
    dataset_root.mkdir(parents=True, exist_ok=False)
    marker = {
        "phase": stagea.PHASE,
        "schema": "phase314b_r260_stagea_attempt_started_v1",
        "implementation_commit": implementation_commit,
        "submodule_commit": stagea.EXPECTED_SUBMODULE,
        "role_contract": stagea.role_contract(),
        "rerun_authorized": False,
        "resume_authorized": False,
    }
    stagea.atomic_write_once(repo / stagea.ATTEMPT_MARKER_RELATIVE, stagea.stable_json_bytes(marker))
    source_lock = stagea.build_source_lock(repo, implementation_commit)
    stagea.atomic_write_once(repo / stagea.SOURCE_LOCK_RELATIVE, stagea.stable_json_bytes(source_lock))

    jobs = [
        (spec.name, seed)
        for spec in stagea.ROLE_SPECS
        for seed in spec.seeds()
    ]
    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_job = {
            executor.submit(
                _worker_command,
                root=repo,
                dataset_root=dataset_root,
                write_ahead=write_ahead,
                role=role,
                seed=seed,
                implementation_commit=implementation_commit,
            ): (role, seed)
            for role, seed in jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(future_to_job), 1):
            role, seed = future_to_job[future]
            receipt = future.result()
            receipts.append(receipt)
            print(
                "[r2.6.0 Stage A] {} seed {} ({}/{})".format(
                    role, seed, index, len(jobs)
                ),
                flush=True,
            )
    if len(receipts) != stagea.TOTAL_VISIBLE_SEEDS:
        raise stagea.StageAError("pair-worker receipt count changed")

    for spec in stagea.ROLE_SPECS:
        role_receipts = []
        for seed in spec.seeds():
            role_receipts.append(
                stagea.validate_pair_directory(
                    repo / stagea.RAW_RELATIVE_ROOT / spec.name / "seed_{}".format(seed),
                    role=spec.name,
                    seed=seed,
                    implementation_commit=implementation_commit,
                )
            )
        receipt_summary = {
            "schema": "phase314b_r260_stagea_role_receipts_v1",
            "role": spec.name,
            "seed_count": spec.seed_count,
            "receipt_sha256_population": stagea.sha256_strings(
                str(value["receipt_sha256"]) for value in role_receipts
            ),
            "records": role_receipts,
        }
        stagea.atomic_write_once(
            repo / stagea.RECEIPTS_RELATIVE_ROOT / "{}.json".format(spec.name),
            stagea.stable_json_bytes(receipt_summary),
        )

    codec = None
    role_arrays: Dict[str, Mapping[str, Any]] = {}
    role_sources: Dict[str, Mapping[str, int]] = {}
    role_manifests: Dict[str, Mapping[str, Any]] = {}
    windows_root = repo / stagea.WINDOWS_RELATIVE_ROOT
    windows_root.mkdir(parents=True, exist_ok=True)
    for spec in stagea.ROLE_SPECS:
        arrays, codec, sources = stagea.build_role_windows(
            root=repo, role=spec.name, codec=codec
        )
        role_arrays[spec.name] = arrays
        role_sources[spec.name] = sources
        npz_path = windows_root / "{}.npz".format(spec.name)
        stagea.write_npz_once(npz_path, arrays)
        reloaded = stagea.load_npz_strict(npz_path)
        if set(reloaded) != set(arrays):
            raise stagea.StageAError("reloaded role NPZ key population changed")
        for key in arrays:
            if stagea.sha256_array(reloaded[key]) != stagea.sha256_array(arrays[key]):
                raise stagea.StageAError("reloaded role array identity changed")
    if codec is None:
        raise stagea.StageAError("action codec was not built")
    stagea.save_action_template(repo / stagea.ACTION_TEMPLATE_RELATIVE, codec)

    audit = stagea.audit_roles(role_arrays, repo)
    for spec in stagea.ROLE_SPECS:
        manifest = stagea.build_role_manifest(
            root=repo,
            role=spec.name,
            arrays=role_arrays[spec.name],
            npz_path=windows_root / "{}.npz".format(spec.name),
            robot_sources=role_sources[spec.name],
        )
        stagea.atomic_write_once(
            repo / stagea.ROLE_MANIFEST_RELATIVE_ROOT / "{}.json".format(spec.name),
            stagea.stable_json_bytes(manifest),
        )
        role_manifests[spec.name] = manifest

    inventory = stagea.build_inventory(dataset_root)
    stagea.atomic_write_once(repo / stagea.INVENTORY_RELATIVE, stagea.stable_json_bytes(inventory))
    stagea.validate_inventory(dataset_root, inventory)
    exports = stagea.build_exports(repo)
    seal = stagea.build_seal(
        repository=repository,
        source_lock=source_lock,
        role_manifests=role_manifests,
        audit=audit,
        inventory=inventory,
        exports=exports,
    )
    stagea.atomic_write_once(repo / stagea.SEAL_RELATIVE, stagea.stable_json_bytes(seal))
    legacy_after = tree_identity(legacy_write_ahead)
    summary = stagea.build_summary(
        repository=repository,
        seal=seal,
        legacy_write_ahead_before_sha256=legacy_before,
        legacy_write_ahead_after_sha256=legacy_after,
    )
    stagea.atomic_write_once(repo / stagea.SUCCESS_REPORT, stagea.stable_json_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(stagea.SCHEMA)
        return 0
    if not args.implementation_commit:
        raise SystemExit("--implementation-commit is required")
    repo = Path(args.root).resolve()
    repository = None
    try:
        summary = execute(repo, str(args.implementation_commit), int(args.workers))
        print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        try:
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
        blocked = stagea.blocked_report(
            repository=repository,
            error=error,
            dataset_root=repo / stagea.DATASET_RELATIVE_ROOT,
            write_ahead=Path(str(repo) + stagea.WRITE_AHEAD_NAME),
        )
        blocked_path = repo / stagea.BLOCKED_REPORT
        if not blocked_path.exists():
            stagea.atomic_write_once(blocked_path, stagea.stable_json_bytes(blocked))
        print(json.dumps(blocked, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
