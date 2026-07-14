#!/usr/bin/env python3
"""Run the r2.5.4 shared-prior repeat and historical fingerprint audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r23_diagnostics import (
    require_repository_state,
    write_json_once,
)
from ccda_phase3.phase314b_r254_resume1_prior_determinism import (
    EXPECTED_CACHE_SHA256,
    EXPECTED_CONTRACT_SHA256,
    EXPECTED_PYTHON,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    REPEAT_RUN_IDS,
    SCHEMA,
    WORKER_SCHEMA,
    PriorEquivalenceSpec,
    assert_only_allowed_worktree_paths,
    classify_prior_audit,
    decode_tensor_state,
    historical_functional_fingerprint,
    pairwise_repeat_audit,
    source_sha256,
    strip_worker_payload,
    tensor_state_sha256_numpy,
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def run_worker(
    *,
    root: Path,
    run_id: str,
    prior_steps: int,
    batch_size: int,
    prior_learning_rate: float,
) -> Dict[str, Any]:
    executable = os.path.realpath(sys.executable)
    expected = os.path.realpath(EXPECTED_PYTHON)
    if executable != expected:
        raise RuntimeError(
            f"worker interpreter mismatch: expected={expected}, observed={executable}"
        )
    worker = root / "scripts/phase3_14b_r254_resume1_prior_worker.py"
    command = [
        executable,
        str(worker),
        "--root",
        str(root),
        "--run-id",
        str(run_id),
        "--prior-steps",
        str(int(prior_steps)),
        "--batch-size",
        str(int(batch_size)),
        "--prior-learning-rate",
        repr(float(prior_learning_rate)),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + "\n" + completed.stderr).splitlines()[-120:])
        raise RuntimeError(
            f"prior worker {run_id} failed with {completed.returncode}:\n{tail}"
        )
    text = completed.stdout.strip()
    if not text:
        raise RuntimeError(f"prior worker {run_id} returned empty stdout")
    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        tail = "\n".join(text.splitlines()[-20:])
        raise RuntimeError(f"prior worker {run_id} emitted invalid JSON:\n{tail}") from exc
    if not isinstance(report, dict):
        raise RuntimeError(f"prior worker {run_id} JSON root is not an object")
    if report.get("schema") != WORKER_SCHEMA:
        raise RuntimeError(f"prior worker {run_id} schema mismatch")
    if report.get("run_id") != run_id:
        raise RuntimeError(f"prior worker {run_id} identity mismatch")
    state = decode_tensor_state(report["_state_payload"])
    recomputed = tensor_state_sha256_numpy(state)
    if recomputed != report.get("prior_state_sha256"):
        raise RuntimeError(
            f"prior worker {run_id} state payload/hash mismatch: "
            f"{recomputed} != {report.get('prior_state_sha256')}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--preflight-report",
        default="reports/phase3_14b_r254_resume1_preflight_summary.json",
    )
    parser.add_argument(
        "--evidence-output",
        default="reports/phase3_14b_r254_resume1_prior_repeat_evidence.json",
    )
    parser.add_argument(
        "--output",
        default="reports/phase3_14b_r254_resume1_prior_audit_summary.json",
    )
    parser.add_argument("--prior-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--prior-learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    preflight_path = resolve(root, args.preflight_report)
    evidence_path = resolve(root, args.evidence_output)
    output_path = resolve(root, args.output)
    if evidence_path.exists():
        raise RuntimeError(f"refusing to overwrite repeat evidence: {evidence_path}")
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite prior audit: {output_path}")
    preflight_relative = preflight_path.relative_to(root).as_posix()

    repository = require_repository_state(root, require_clean=False)
    assert_only_allowed_worktree_paths(root, (preflight_relative,))
    if repository["submodule_commit"] != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("submodule commit mismatch")
    if repository["cache_sha256"] != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache SHA mismatch")
    if repository["frozen_contract_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError("frozen contract SHA mismatch")

    preflight = load_json(preflight_path)
    if preflight.get("verdict") != "PASS":
        raise RuntimeError("Resume1 preflight did not pass")
    if preflight.get("source_sha256") != source_sha256(root):
        raise RuntimeError("Resume1 source changed after preflight")
    spec = PriorEquivalenceSpec(**preflight["audit_spec"])
    spec.validate()
    historical = preflight["historical_prior"]
    expected_contract = preflight["dataset"]["paired_row_contract"]

    records: List[Dict[str, Any]] = []
    try:
        for run_id in REPEAT_RUN_IDS:
            record = run_worker(
                root=root,
                run_id=run_id,
                prior_steps=int(args.prior_steps),
                batch_size=int(args.batch_size),
                prior_learning_rate=float(args.prior_learning_rate),
            )
            if record.get("paired_row_contract") != expected_contract:
                raise RuntimeError(f"paired-row contract changed in worker {run_id}")
            records.append(record)
    except Exception as exc:
        partial = {
            "phase": PHASE,
            "schema": "phase314b_r254_resume1_prior_repeat_evidence_v1",
            "verdict": "BLOCKED",
            "failure": str(exc),
            "preflight_report": preflight_relative,
            "source_sha256": source_sha256(root),
            "completed_run_count": len(records),
            "runs": [strip_worker_payload(record) for record in records],
            "train_only_recommendation": None,
            "selected_configuration": None,
            "validation_targets_used": False,
            "formal_test_read": False,
            "formal_training": False,
            "checkpoint_saved": False,
            "idm": False,
            "candidate_execution": False,
            "phase4": False,
            "cps": False,
        }
        write_json_once(evidence_path, partial)
        raise

    raw_evidence = {
        "phase": PHASE,
        "schema": "phase314b_r254_resume1_prior_repeat_evidence_v1",
        "verdict": "PASS",
        "preflight_report": preflight_relative,
        "source_sha256": source_sha256(root),
        "completed_run_count": len(records),
        "expected_historical_prior_sha256": historical["prior_state_sha256"],
        "runs": [strip_worker_payload(record) for record in records],
        "observed_sha_persisted_before_gate": True,
        "robot_proxy_attribution_run": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    write_json_once(evidence_path, raw_evidence)

    environments = [record["environment_after_fit"] for record in records]
    gpu_names = {str(value.get("gpu_name")) for value in environments}
    capabilities = {
        tuple(int(item) for item in value.get("gpu_capability", []))
        for value in environments
    }
    if len(gpu_names) != 1 or len(capabilities) != 1:
        raise RuntimeError("repeat workers did not use one stable CUDA device class")

    repeat = pairwise_repeat_audit(records, spec=spec)
    historical_fingerprint = historical_functional_fingerprint(
        records, historical, spec=spec
    )
    report: Dict[str, Any] = {
        "phase": PHASE,
        "schema": SCHEMA,
        "verdict": "PASS",
        "preflight_report": preflight_relative,
        "repeat_evidence_report": evidence_path.relative_to(root).as_posix(),
        "source_sha256": source_sha256(root),
        "audit_spec": asdict(spec),
        "environment": {
            "gpu_name": next(iter(gpu_names)),
            "gpu_capability": list(next(iter(capabilities))),
            "repeat_environment_identical": bool(
                all(value == environments[0] for value in environments[1:])
            ),
            "records": environments,
        },
        "historical_prior": historical,
        "runs": [strip_worker_payload(record) for record in records],
        "same_device_repeats": repeat,
        "historical_functional_fingerprint": historical_fingerprint,
        "observed_sha_persisted_before_gate": True,
        "robot_proxy_attribution_run": False,
        "reverse_sampling_rerun": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "formal_training": False,
        "candidate_eligible": False,
        "checkpoint_saved": False,
        "idm": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
    }
    report.update(classify_prior_audit(report))
    write_json_once(output_path, report)
    print(json.dumps({
        "verdict": report["verdict"],
        "root_cause": report["root_cause"],
        "functional_prior_contract_supported": report[
            "functional_prior_contract_supported"
        ],
        "output": str(output_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
