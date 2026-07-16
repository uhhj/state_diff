#!/usr/bin/env python3
"""Isolated Phase3.14b-r2.5.8 Stage-A worker."""
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

from ccda_phase3.phase314b_r258_stagea_conflict_projected_k16 import (
    atomic_write_once,
    load_json,
    probe_portable_environment,
    run_calibration,
    stable_json_bytes,
    validate_resume4_evidence,
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
        ),
        default="run",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if args.mode == "environment-probe":
        result = probe_portable_environment(root=root)
        atomic_write_once(
            output,
            stable_json_bytes(result),
        )
        print(
            json.dumps(
                {
                    "compatibility_pass":
                        result["compatibility_pass"],
                    "torch_device_name":
                        result["hardware_observation"][
                            "torch_device_name"
                        ],
                    "compute_capability":
                        result["hardware_observation"][
                            "compute_capability"
                        ],
                    "compatibility_sha256":
                        result["compatibility_sha256"],
                    "observation_sha256":
                        result["observation_sha256"],
                    "required_operation_pass":
                        result["required_operation_dry_run"][
                            "pass"
                        ],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    validate_resume4_evidence(root)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix="phase314b_r258_stagea_",
            dir="/tmp",
        )
    )
    environment_path = (
        temporary_root / "environment.json"
    )
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
        environment = load_json(
            environment_path
        )
        result = run_calibration(
            root=root,
            environment=environment,
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
                "verdict":
                    result["verdict"],
                "root_cause":
                    result["root_cause"],
                "required_next_path":
                    result[
                        "required_next_path"
                    ],
                "compatibility_sha256":
                    result["environment"][
                        "compatibility_sha256"
                    ],
                "observation_sha256":
                    result["environment"][
                        "observation_sha256"
                    ],
                "gpu":
                    result["environment"][
                        "hardware_observation"
                    ]["torch_device_name"],
                "control_replay_byte_exact":
                    result["control_replay_exact"],
                "control_reference_equivalent":
                    result["control_reference_equivalent"],
                "calibration_sha256":
                    result[
                        "calibration_contract"
                    ][
                        "calibration"
                    ][
                        "calibration_sha256"
                    ],
                "contract_sha256":
                    result[
                        "calibration_contract"
                    ][
                        "contract_sha256"
                    ],
                "selection_sha256":
                    result["selection"][
                        "selection_sha256"
                    ],
                "selected_configuration":
                    result[
                        "selected_configuration"
                    ],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
