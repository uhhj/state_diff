#!/usr/bin/env python3
"""Isolated r2.5.7 Stage-D Resume3 worker."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r257_staged_resume3_calibration_trace import (
    atomic_write_once,
    load_json,
    run_manual_calibration_diff,
    run_resume3,
    stable_json_bytes,
    validate_resume2_failure,
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
    parser.add_argument(
        "--mode",
        choices=("run", "manual-diff"),
        default="run",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    validate_resume2_failure(root)

    if args.mode == "manual-diff":
        result = run_manual_calibration_diff(
            root=root
        )
        atomic_write_once(
            output,
            stable_json_bytes(result),
        )
        print(
            json.dumps(
                {
                    "completed": result["completed"],
                    "manual_replay_exact":
                        result["manual_replay_exact"],
                    "expected_calibration_sha256":
                        result[
                            "expected_calibration_sha256"
                        ],
                    "observed_calibration_sha256":
                        result[
                            "observed_calibration_sha256"
                        ],
                    "difference_summary":
                        result["difference_summary"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r257_resume3_manual_",
            dir="/tmp",
        )
    )
    manual_path = temporary_root / "manual_diff.json"
    try:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--output",
                str(manual_path),
                "--mode",
                "manual-diff",
            ],
            cwd=str(root),
            check=True,
        )
        manual_diff = load_json(manual_path)
        result = run_resume3(
            root=root,
            manual_diff=manual_diff,
        )
        atomic_write_once(
            output,
            stable_json_bytes(result),
        )
    finally:
        shutil.rmtree(
            temporary_root,
            ignore_errors=True,
        )

    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "root_cause": result["root_cause"],
                "required_next_path":
                    result["required_next_path"],
                "manual_difference_locus":
                    result[
                        "resume3_calibration_attribution"
                    ][
                        "manual_calibration_diff"
                    ][
                        "difference_summary"
                    ][
                        "primary_difference_locus"
                    ],
                "stagec_entrypoint_calibration_exact":
                    result[
                        "resume3_calibration_attribution"
                    ][
                        "stagec_entrypoint_capture"
                    ][
                        "calibration_exact"
                    ],
                "control_replay_exact":
                    result["control_replay_exact"],
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
