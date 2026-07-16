#!/usr/bin/env python3
"""Isolated Stage-D Resume4 RTX-3090 worker."""
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

from ccda_phase3.phase314b_r257_staged_resume4_3090 import (
    atomic_write_once,
    environment_probe,
    load_json,
    run_resume4,
    stable_json_bytes,
    validate_resume3_failure,
)
from ccda_phase3.phase314b_r257_staged_resume3_calibration_trace import (
    run_manual_calibration_diff,
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
        choices=(
            "run",
            "environment-probe",
            "manual-diff",
        ),
        default="run",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if args.mode == "environment-probe":
        result = environment_probe()
        atomic_write_once(
            output,
            stable_json_bytes(result),
        )
        print(
            json.dumps(
                {
                    "contract_pass":
                        result["contract_pass"],
                    "torch_device_name":
                        result["torch_device_name"],
                    "compute_capability":
                        result["compute_capability"],
                    "driver_version":
                        result["nvidia_smi"][
                            "driver_version"
                        ],
                    "environment_sha256":
                        result["environment_sha256"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    validate_resume3_failure(root)

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
            prefix="phase314b_r257_resume4_3090_",
            dir="/tmp",
        )
    )
    environment_path = (
        temporary_root / "environment.json"
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
                str(environment_path),
                "--mode",
                "environment-probe",
            ],
            cwd=str(root),
            check=True,
        )
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
        environment = load_json(
            environment_path
        )
        manual_diff = load_json(manual_path)
        result = run_resume4(
            root=root,
            environment=environment,
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
                "gpu":
                    result[
                        "resume4_3090_environment"
                    ][
                        "environment_probe"
                    ][
                        "torch_device_name"
                    ],
                "environment_sha256":
                    result[
                        "resume4_3090_environment"
                    ][
                        "environment_probe"
                    ][
                        "environment_sha256"
                    ],
                "cold_main_worker_context":
                    result[
                        "resume4_3090_environment"
                    ][
                        "cold_main_worker_context"
                    ][
                        "torch_cuda_is_initialized"
                    ]
                    is False,
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
