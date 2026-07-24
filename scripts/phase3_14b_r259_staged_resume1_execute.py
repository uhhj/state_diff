#!/usr/bin/env python3
"""Controller and disposable child entry point for Stage-D Resume1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def run_child(
    command: Sequence[str], *, root: Path, env: Mapping[str, str], label: str
) -> None:
    completed = subprocess.run(
        list(command),
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
    parser.add_argument("--mode", choices=("controller", "probe", "science"), default="controller")
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default="/miniforge3/envs/coord_bimanual/bin/python")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--write-ahead-dir")
    parser.add_argument("--probe")
    parser.add_argument("--repository-head")
    parser.add_argument("--stageu-contract")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.import_smoke:
        print(json.dumps({"import_smoke": "PASS", "repository_root": str(REPOSITORY_ROOT)}))
        return 0

    from ccda_phase3.phase314b_r259_staged_resume1_state_dimension_contract_recovery import (
        BLOCKED_REPORT,
        PROBE_EVIDENCE,
        SUCCESS_REPORT,
        WORKER_EVIDENCE,
        atomic_write_once,
        blocked_report,
        build_summary,
        load_json,
        make_probe_evidence,
        promote_write_ahead,
        run_science_worker,
        sha256_bytes,
        stable_json_bytes,
        validate_probe_evidence,
        validate_repository,
        validate_worker_evidence,
    )

    root = Path(args.root).resolve()
    if args.mode == "probe":
        if not args.output:
            parser.error("probe mode requires --output")
        try:
            payload = make_probe_evidence(root)
            atomic_write_once(Path(args.output).resolve(), stable_json_bytes(payload) + b"\n")
            print(json.dumps({"mode": "probe", "process_id": payload["process_id"]}, sort_keys=True))
            return 0
        except BaseException as error:
            print("BLOCKED probe: {}".format(error), file=sys.stderr)
            return 2

    if args.mode == "science":
        missing = [
            name
            for name, value in (
                ("probe", args.probe),
                ("repository-head", args.repository_head),
                ("stageu-contract", args.stageu_contract),
                ("output", args.output),
            )
            if not value
        ]
        if missing:
            parser.error("science mode missing: {}".format(", ".join(missing)))
        try:
            payload = dict(
                run_science_worker(
                    root=root,
                    probe_payload=load_json(Path(args.probe).resolve()),
                    repository_head=str(args.repository_head),
                    stageu_contract=load_json(Path(args.stageu_contract).resolve()),
                )
            )
            payload.pop("worker_result_sha256", None)
            payload["worker_result_sha256"] = sha256_bytes(stable_json_bytes(payload))
            validate_worker_evidence(payload)
            atomic_write_once(Path(args.output).resolve(), stable_json_bytes(payload) + b"\n")
            print(
                json.dumps(
                    {
                        "mode": "science",
                        "process_id": payload["process_id"],
                        "scientific_status": payload["scientific_status"],
                        "worker_result_sha256": payload["worker_result_sha256"],
                    },
                    sort_keys=True,
                )
            )
            return 0
        except BaseException as error:
            print("BLOCKED science: {}".format(error), file=sys.stderr)
            return 2

    if not args.implementation_commit:
        print("BLOCKED: controller requires --implementation-commit", file=sys.stderr)
        return 2

    write_ahead = (
        Path(args.write_ahead_dir).resolve()
        if args.write_ahead_dir
        else root.parent / (root.name + ".phase314b_r259_staged_resume1_write_ahead")
    )
    try:
        write_ahead.relative_to(root)
    except ValueError:
        pass
    else:
        print("BLOCKED: write-ahead directory must be outside Git worktree", file=sys.stderr)
        return 2

    repo_probe = root / PROBE_EVIDENCE
    repo_worker = root / WORKER_EVIDENCE
    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    ahead_probe = write_ahead / "environment_probe_evidence.json"
    ahead_worker = write_ahead / "science_worker_evidence.json"
    if success.exists() or blocked.exists():
        print("BLOCKED: Resume1 terminal report already exists", file=sys.stderr)
        return 2

    repository = None
    try:
        repository = validate_repository(root, args.implementation_commit)
        if write_ahead.exists():
            raise RuntimeError("Resume1 write-ahead already exists; do not rerun science")
        write_ahead.mkdir(parents=True, exist_ok=False)
        from ccda_phase3.phase314b_r259_stagec_fold_resolved_tail_attribution import (
            child_environment,
        )

        env = child_environment(root)
        script = Path(__file__).resolve()
        run_child(
            [
                args.python_bin,
                str(script),
                "--mode",
                "probe",
                "--root",
                str(root),
                "--output",
                str(ahead_probe),
            ],
            root=root,
            env=env,
            label="Stage-D Resume1 portable probe",
        )
        run_child(
            [
                args.python_bin,
                str(script),
                "--mode",
                "science",
                "--root",
                str(root),
                "--probe",
                str(ahead_probe),
                "--repository-head",
                repository["head"],
                "--stageu-contract",
                str(root / "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"),
                "--output",
                str(ahead_worker),
            ],
            root=root,
            env=env,
            label="Stage-D Resume1 cold science worker",
        )
        promote_write_ahead(ahead_probe, repo_probe, validate_probe_evidence)
        promote_write_ahead(ahead_worker, repo_worker, validate_worker_evidence)
        summary = build_summary(
            repository=repository,
            probe_path=repo_probe,
            worker_path=repo_worker,
        )
        atomic_write_once(success, stable_json_bytes(summary) + b"\n")
        print(
            json.dumps(
                {
                    "execution_verdict": summary["execution_verdict"],
                    "scientific_status": summary["scientific_status"],
                    "root_cause": summary["root_cause"],
                    "primary_failure_locus": summary["primary_failure_locus"],
                    "required_next_path": summary["required_next_path"],
                    "summary_sha256": summary["summary_sha256"],
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        for source, destination, validator in (
            (ahead_probe, repo_probe, validate_probe_evidence),
            (ahead_worker, repo_worker, validate_worker_evidence),
        ):
            try:
                if source.is_file() and not destination.exists():
                    promote_write_ahead(source, destination, validator)
            except BaseException:
                pass
        try:
            payload = blocked_report(
                repository=repository,
                error=error,
                probe_path=repo_probe if repo_probe.is_file() else ahead_probe,
                worker_path=repo_worker if repo_worker.is_file() else ahead_worker,
            )
            atomic_write_once(blocked, stable_json_bytes(payload) + b"\n")
        except BaseException as nested:
            print(
                "BLOCKED: {}; failed to persist blocked evidence: {}".format(error, nested),
                file=sys.stderr,
            )
            return 2
        print("BLOCKED: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
