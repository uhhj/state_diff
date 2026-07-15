#!/usr/bin/env python3
"""Isolated r2.5.7 Stage-D Resume1 worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r257_staged_resume1_control_trace import (
    atomic_write_once,
    run_resume1,
    stable_json_bytes,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    args = parser.parse_args()

    result = run_resume1(root=Path(args.root))
    output = Path(args.output).resolve()
    atomic_write_once(
        output,
        stable_json_bytes(result),
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path":
                    result["required_next_path"],
                "control_replay_exact":
                    result["control_replay_exact"],
                "preamble_calibration_sha256":
                    result["control_trace_correction"][
                        "preamble_calibration_sha256"
                    ],
                "contract_sha256":
                    result["calibration_contract"][
                        "contract_sha256"
                    ],
                "selection_sha256":
                    result["selection"][
                        "selection_sha256"
                    ],
                "selected_configuration":
                    result["selected_configuration"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
