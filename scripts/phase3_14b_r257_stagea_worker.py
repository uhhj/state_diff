#!/usr/bin/env python3
"""Isolated r2.5.7 Stage-A upper-objective calibration worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r257_stagea_upper_objective import (
    atomic_write_once,
    run_calibration,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = run_calibration(root=Path(args.root))
    output = Path(args.output).resolve()
    atomic_write_once(output, stable_json_bytes(result))
    selected = result["selected_configuration"]
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path": result["required_next_path"],
                "selected_lambda": (
                    None
                    if selected is None
                    else selected["lambda_upper"]
                ),
                "selection_sha256":
                    result["train_only_selection"][
                        "selection_sha256"
                    ],
                "objective_contract_sha256":
                    result["objective_contract"][
                        "contract_sha256"
                    ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
