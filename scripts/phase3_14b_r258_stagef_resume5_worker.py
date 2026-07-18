#!/usr/bin/env python3
"""Isolated Stage-F Resume2 worker."""
from __future__ import annotations

# Resume5 direct-entrypoint import bootstrap.
import sys as _resume5_sys
from pathlib import Path as _Resume5Path
_RESUME5_REPO_ROOT = _Resume5Path(__file__).resolve().parents[1]
if str(_RESUME5_REPO_ROOT) not in _resume5_sys.path:
    _resume5_sys.path.insert(0, str(_RESUME5_REPO_ROOT))

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
    probe_portable_environment,
)
from ccda_phase3.phase314b_r258_stagef_constraint_aware_surrogate import (
    run_calibration,
)
from ccda_phase3.phase314b_r258_stagef_resume2_porcelain_recovery import (
    atomic_write_once,
    load_json,
    stable_json_bytes,
    validate_resume2_implementation_commit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("run", "environment-probe"),
        default="run",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = Path(args.output).resolve()
    if args.mode == "environment-probe":
        result = probe_portable_environment(root=root)
        atomic_write_once(output, stable_json_bytes(result))
        print(
            json.dumps(
                {
                    "compatibility_pass": result["compatibility_pass"],
                    "compatibility_sha256": result["compatibility_sha256"],
                    "observation_sha256": result["observation_sha256"],
                    "gpu": result["hardware_observation"]["torch_device_name"],
                    "required_operation_pass": result[
                        "required_operation_dry_run"
                    ]["pass"],
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return

    validate_resume2_implementation_commit(root)
    temporary_root = Path(
        tempfile.mkdtemp(prefix="phase314b_r258_stagef_resume2_", dir="/tmp")
    )
    environment_path = temporary_root / "environment.json"
    if environment_path.exists():
        raise FileExistsError("temporary environment output must not exist")
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
        environment = load_json(environment_path)
        result = run_calibration(root=root, environment=environment)
        result["resume2_porcelain_recovery"] = {
            "phase": "Phase3.14b-r2.5.8 Stage F Resume5",
            "porcelain_format": "v1-z",
            "stdout_trimmed": False,
            "nul_delimited": True,
            "rename_copy_supported": True,
            "translation_fix_changed": False,
            "candidate_matrix_changed": False,
            "integrator_changed": False,
            "scientific_thresholds_changed": False,
        }
        atomic_write_once(output, stable_json_bytes(result))
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "scientific_status": result["scientific_status"],
                "root_cause": result["root_cause"],
                "required_next_path": result["required_next_path"],
                "compatibility_sha256": result["environment"][
                    "compatibility_sha256"
                ],
                "observation_sha256": result["environment"][
                    "observation_sha256"
                ],
                "contract_sha256": result["constraint_aware_contract"][
                    "contract_sha256"
                ],
                "selection_sha256": result["selection"]["selection_sha256"],
                "selected_configuration": result["selected_configuration"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
