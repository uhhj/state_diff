#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MINIMAL_BLOCKED = (
    "reports/phase3_14b_r258_stagex_resume1_tail_robust_nested_group_oof_"
    "blocked_summary.json"
)


def _write_minimal_blocked(root: Path, error: BaseException) -> None:
    target = root / MINIMAL_BLOCKED
    if target.exists():
        return
    payload = {
        "phase": "Phase3.14b-r2.5.8 Stage X Resume1",
        "schema": "phase314b_r258_stagex_resume1_import_crossfit_recovery_blocked_v1",
        "execution_verdict": "BLOCKED",
        "scientific_status": "BLOCKED",
        "root_cause": "phase314b_r258_stagex_resume1_controller_import_failed_after_bootstrap",
        "required_next_path": "AUDIT_STAGEX_RESUME1_CONTROLLER_IMPORT_WITHOUT_SCIENCE_RERUN",
        "primary_failure_locus": "controller_import",
        "selected_configuration": None,
        "train_only_recommendation": None,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "selection_holdout_evaluation_count_added": 0,
        "cumulative_selection_holdout_evaluation_count": 1,
        "rerun_authorized": False,
        "complete_nested_oof_population_claimed": False,
        "selection_holdout_reaccessed": False,
        "selection_holdout_target_loaded_for_stagex": False,
        "selection_holdout_used_for_fit_or_selection": False,
        "frozen_probe_accessed": False,
        "formal_training_run": False,
        "reverse_sampling_run": False,
        "idm_run": False,
        "candidate_execution": False,
        "deformable_ravens_executed": False,
        "phase4": False,
        "cps": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "descriptor_tensor_persisted": False,
        "risk_probability_tensor_persisted": False,
        "npz_saved": False,
        "cache_saved": False,
        "image_saved": False,
        "video_saved": False,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def _child(command, root: Path, label: str, env: Mapping[str, str]) -> None:
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "{} rc={} stdout={!r} stderr={!r}".format(
                label, completed.returncode, completed.stdout, completed.stderr
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--import-smoke", action="store_true")
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin", default="/miniforge3/envs/coord_bimanual/bin/python"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        from ccda_phase3.phase314b_r258_stagex_resume1_import_crossfit_recovery import (
            BLOCKED_REPORT,
            SUCCESS_REPORT,
            StageXError,
            blocked_report,
            child_environment,
            sha256_bytes,
            stable_json_bytes,
            validate_repository,
            validate_worker_payload,
            write_once,
        )
    except BaseException as error:
        if not args.import_smoke:
            _write_minimal_blocked(root, error)
        return 2
    if args.import_smoke:
        return 0
    repository = None
    try:
        from ccda_phase3.phase314b_r258_stagex_resume1_import_crossfit_recovery import (
            stagex,
        )
        stagex.validate_environment_variables()
        repository = validate_repository(root)
        env = child_environment(root)
        with tempfile.TemporaryDirectory(
            prefix="phase314b_r258_stagex_resume1_"
        ) as directory:
            temp = Path(directory)
            probe_path = temp / "probe.json"
            worker_path = temp / "worker.json"
            _child(
                [
                    args.python_bin,
                    str(root / "scripts/phase3_14b_r258_stages_resume2a_worker.py"),
                    "--root",
                    str(root),
                    "--mode",
                    "environment-probe",
                    "--output",
                    str(probe_path),
                ],
                root,
                "environment probe",
                env,
            )
            _child(
                [
                    args.python_bin,
                    str(root / "scripts/phase3_14b_r258_stagex_resume1_worker.py"),
                    "--root",
                    str(root),
                    "--probe",
                    str(probe_path),
                    "--repository-head",
                    repository["head"],
                    "--stageu-contract",
                    str(root / stagex.STAGEU_CONTRACT),
                    "--output",
                    str(worker_path),
                ],
                root,
                "cold Stage-X Resume1 worker",
                env,
            )
            probe = json.loads(probe_path.read_text(encoding="utf-8"))
            worker = json.loads(worker_path.read_text(encoding="utf-8"))
            validate_worker_payload(worker)
            probe_pid = int(probe["probe"]["process_id"])
            worker_pid = int(worker["process_id"])
            if probe_pid == worker_pid:
                raise StageXError("probe and Resume1 worker PIDs are not distinct")
        summary = copy.deepcopy(worker)
        summary["schema"] = (
            "phase314b_r258_stagex_resume1_tail_robust_nested_group_oof_summary_v1"
        )
        summary["repository"] = repository
        summary["process_topology"] = {
            "environment_probe_process_count": 1,
            "cold_science_worker_process_count": 1,
            "probe_process_id": probe_pid,
            "science_worker_process_id": worker_pid,
            "processes_distinct": True,
            "temporary_payloads_deleted": True,
            "child_pythonpath_explicit": True,
        }
        summary.pop("scientific_result_sha256", None)
        summary["scientific_result_sha256"] = sha256_bytes(
            stable_json_bytes(summary)
        )
        write_once(root / SUCCESS_REPORT, stable_json_bytes(summary) + b"\n")
        return 0
    except BaseException as error:
        try:
            payload = blocked_report(repository, error)
            write_once(root / BLOCKED_REPORT, stable_json_bytes(payload) + b"\n")
        except BaseException as nested:
            _write_minimal_blocked(root, nested)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
