#!/usr/bin/env python3
"""Isolated Stage-A contract worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_stagea_contract import build_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = build_artifacts(
        root=Path(args.root),
        output_root=Path(args.output_root),
    )
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "root_cause":
                    manifest["idm_identifiability_audit"]["root_cause"],
                "required_next_path":
                    manifest["idm_identifiability_audit"][
                        "required_next_path"
                    ],
                "contract_sha256": manifest["contract_sha256"],
                "train_view_sha256": manifest["train_view_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
