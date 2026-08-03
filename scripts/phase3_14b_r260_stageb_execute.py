#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_FROM_FILE = Path(__file__).resolve().parents[1]
if str(ROOT_FROM_FILE) not in sys.path:
    sys.path.insert(0, str(ROOT_FROM_FILE))

from ccda_phase3 import phase314b_r260_stageb_external_backup_attestation as stageb


def _load_inputs(contract_path: Path, site_a_path: Path, site_b_path: Path):
    contract = stageb.load_json(contract_path)
    site_a = stageb.load_json(site_a_path)
    site_b = stageb.load_json(site_b_path)
    return contract, site_a, site_b


def preflight(root: Path, contract_path: Path, site_a_path: Path, site_b_path: Path):
    contract, site_a, site_b = _load_inputs(contract_path, site_a_path, site_b_path)
    stageb.validate_contract_against_source(contract, root)
    audit = stageb.adjudicate_sites(contract, site_a, site_b)
    return contract, site_a, site_b, audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--contract")
    parser.add_argument("--site-a")
    parser.add_argument("--site-b")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--import-smoke", action="store_true")
    args = parser.parse_args()
    if args.import_smoke:
        print(stageb.SCHEMA)
        return 0
    for name, value in (
        ("--contract", args.contract),
        ("--site-a", args.site_a),
        ("--site-b", args.site_b),
    ):
        if not value:
            raise SystemExit("{} is required".format(name))
    root = Path(args.root).resolve()
    contract_path = Path(args.contract).resolve()
    site_a_path = Path(args.site_a).resolve()
    site_b_path = Path(args.site_b).resolve()
    repository = None
    try:
        contract, site_a, site_b, audit = preflight(
            root, contract_path, site_a_path, site_b_path
        )
        if args.preflight_only:
            print(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False))
            return 0
        if not args.implementation_commit:
            raise stageb.StageBError("--implementation-commit is required")
        repository = stageb.validate_repository(root, str(args.implementation_commit))
        payload = stageb.build_summary(
            repository=repository,
            contract=contract,
            site_a=site_a,
            site_b=site_b,
        )
        stageb.atomic_write_once(root / stageb.SUCCESS_REPORT, stageb.stable_json_bytes(payload))
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except BaseException as error:
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "execution_verdict": "BLOCKED",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        try:
            if repository is None and root.is_dir():
                repository = {
                    "root": str(root),
                    "head": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=str(root), text=True
                    ).strip(),
                    "submodule_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(root / "external/deformable-ravens"),
                        text=True,
                    ).strip(),
                }
        except BaseException:
            repository = None
        blocked = stageb.blocked_report(repository, error)
        blocked_path = root / stageb.BLOCKED_REPORT
        if not blocked_path.exists():
            stageb.atomic_write_once(blocked_path, stageb.stable_json_bytes(blocked))
        print(json.dumps(blocked, indent=2, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
