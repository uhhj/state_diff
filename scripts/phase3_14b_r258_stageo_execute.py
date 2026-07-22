#!/usr/bin/env python3
"""Execute one Stage-O dual-oracle targeted replay or environment probe."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3 import phase314b_r258_stageo_dual_oracle_lower_multiplier_admission as stageo


def write_once(path: Path, payload: bytes) -> None:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite write-once output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON payload is not a mapping: {path}")
    return value


def resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--mode",
        choices=("environment-probe", "run"),
        default="run",
    )
    parser.add_argument("--environment-json")
    parser.add_argument("--output")
    parser.add_argument("--blocked-output")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.mode == "environment-probe":
        if not args.output:
            parser.error("--output is required in environment-probe mode")
        output = resolve(root, args.output)
        environment = stageo.probe_environment(root)
        write_once(output, stageo.stable_json_bytes(environment))
        print(
            json.dumps(
                {
                    "mode": "environment-probe",
                    "compatibility_pass": environment.get("compatibility_pass"),
                    "compatibility_sha256": environment.get("compatibility_sha256"),
                    "observation_sha256": environment.get("observation_sha256"),
                    "output": str(output),
                },
                sort_keys=True,
            )
        )
        return 0

    if not args.environment_json:
        parser.error("--environment-json is required in run mode")
    success = resolve(root, args.output or stageo.SUCCESS_REPORT)
    blocked = resolve(root, args.blocked_output or stageo.BLOCKED_REPORT)
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-O output already exists", file=sys.stderr)
        return 2

    repository = None
    try:
        repository = stageo.validate_repository(root)
        environment = load_json(Path(args.environment_json).resolve())
        scientific = stageo.run_targeted_replay(
            root=root,
            environment=environment,
            repository=repository,
        )
        single_run_sha = stageo.sha256_bytes(stageo.stable_json_bytes(scientific))
        report = {
            "phase": stageo.PHASE,
            "schema": "phase314b_r258_stageo_execution_summary_v1",
            "execution_verdict": "PASS",
            "scientific_status": "BLOCKED",
            "root_cause": scientific["root_cause"],
            "required_next_path": scientific["required_next_path"],
            "primary_failure_locus": scientific["primary_failure_locus"],
            "selected_configuration": None,
            "train_only_recommendation": None,
            "repository": repository,
            "execution": {
                "runner_attempt_count": 1,
                "targeted_scientific_execution_count": 1,
                "targeted_callback_off_on_pair_count": stageo.EXPECTED_REPLAY_PAIR_COUNT,
                "stage_l_132_callback_pairs_rerun": False,
                "oof_surrogate_refit": False,
                "runner_return_code": 0,
                "single_run_result_sha256": single_run_sha,
            },
            "scientific_result": scientific,
            **{key: False for key in stageo.FALSE_BOUNDARIES},
        }
        write_once(success, stageo.stable_json_bytes(report))
        print(
            json.dumps(
                {
                    "execution_verdict": "PASS",
                    "scientific_status": "BLOCKED",
                    "root_cause": report["root_cause"],
                    "required_next_path": report["required_next_path"],
                    "single_run_result_sha256": single_run_sha,
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        payload = stageo.blocked_report(repository=repository, error=error)
        try:
            write_once(blocked, stageo.stable_json_bytes(payload))
        except BaseException as write_error:
            print(f"failed to write Stage-O blocked report: {write_error}", file=sys.stderr)
        print(f"BLOCKED: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
