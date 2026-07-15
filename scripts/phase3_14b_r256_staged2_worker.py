#!/usr/bin/env python3
"""Isolated Stage-D.2 cable-XY collapse provenance worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r256_staged2_collapse_provenance import (
    atomic_write_once,
    run_audit,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_audit(root=Path(args.root))
    output = Path(args.output).resolve()
    atomic_write_once(output, stable_json_bytes(result))
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path": result["required_next_path"],
                "raw_xyz_sha256":
                    result["raw_reconstruction_audit"][
                        "raw_target_xyz_sha256"
                    ],
                "driver_category":
                    result["classification"]["driver_category"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
