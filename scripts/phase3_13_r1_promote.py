#!/usr/bin/env python3
"""Atomically promote a provenance-locked staging dataset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ccda_phase3.provenance_v2 import (
    sha256_file,
    strict_json_dump,
)


def run_verify(
    root: Path,
    formal_root: Path,
    audit_summary: str,
) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/phase3_13_r1_verify.py",
            "--root",
            str(root),
            "--formal-root",
            str(formal_root.relative_to(root)),
            "--audit-summary",
            audit_summary,
            "--require-exact-heads",
            "--output",
            "reports/phase3_13_r1_promotion_verify.json",
            "--report",
            "reports/phase3_13_r1_promotion_verify.md",
        ],
        cwd=str(root),
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--staging-root",
        default="data/phase3_state_v2_slack_r1_staging",
    )
    parser.add_argument(
        "--formal-root",
        default="data/phase3_state_v2_slack",
    )
    parser.add_argument(
        "--audit-summary",
        default=(
            "reports/"
            "phase3_13_r1_dataset_audit_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_13_r1_promotion_summary.json",
    )
    parser.add_argument(
        "--delete-quarantine",
        action="store_true",
    )
    args = parser.parse_args()

    if os.environ.get("PHASE313_R1_ALLOW_PROMOTION") != "1":
        raise SystemExit("PHASE313_R1_ALLOW_PROMOTION must be 1")
    if os.environ.get("PHASE313_R1_PROMOTION_CONFIRMED") != "1":
        raise SystemExit("PHASE313_R1_PROMOTION_CONFIRMED must be 1")

    root = Path(args.root).resolve()
    staging = root / args.staging_root
    formal = root / args.formal_root
    if not staging.is_dir():
        raise SystemExit("staging dataset is missing")
    if not formal.is_dir():
        raise SystemExit("current formal dataset is missing")

    run_verify(root, staging, args.audit_summary)

    old_manifest_hash = sha256_file(formal / "manifest.json")
    quarantine = formal.parent / (
        formal.name
        + "_invalid_provenance_"
        + old_manifest_hash[:12]
    )
    if quarantine.exists():
        raise SystemExit("quarantine path already exists: %s" % quarantine)

    formal.rename(quarantine)
    promoted = False
    try:
        staging.rename(formal)
        run_verify(root, formal, args.audit_summary)
        promoted = True
    except Exception:
        if formal.exists():
            failed = formal.parent / (
                formal.name + "_failed_promotion"
            )
            if failed.exists():
                shutil.rmtree(failed)
            formal.rename(failed)
        quarantine.rename(formal)
        raise

    quarantine_deleted = False
    if promoted and args.delete_quarantine:
        if (
            os.environ.get(
                "PHASE313_R1_DELETE_INVALID_DATASET_CONFIRM",
                "",
            )
            != "DELETE_INVALID_PROVENANCE_DATASET"
        ):
            raise SystemExit(
                "Set PHASE313_R1_DELETE_INVALID_DATASET_CONFIRM="
                "DELETE_INVALID_PROVENANCE_DATASET"
            )
        shutil.rmtree(quarantine)
        quarantine_deleted = True

    payload = {
        "verdict": "PASS",
        "root_cause": (
            "phase313_r1_provenance_locked_dataset_promoted"
        ),
        "formal_root": str(formal),
        "old_manifest_sha256": old_manifest_hash,
        "quarantine": str(quarantine),
        "quarantine_deleted": quarantine_deleted,
        "rollback_available": not quarantine_deleted,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
