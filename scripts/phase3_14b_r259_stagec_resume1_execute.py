#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def run_child(command, *, root: Path, env: Mapping[str, str], label: str) -> None:
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
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--python-bin", default="/miniforge3/envs/coord_bimanual/bin/python")
    parser.add_argument("--implementation-commit")
    parser.add_argument("--write-ahead-dir")
    args = parser.parse_args()
    if args.import_smoke:
        return 0
    if not args.implementation_commit:
        print("BLOCKED: --implementation-commit is required", file=sys.stderr)
        return 2

    from ccda_phase3.phase314b_r259_stagec_resume1_portable_gpu_compatibility_recovery import (
        BLOCKED_REPORT,
        PROBE_EVIDENCE,
        SUCCESS_REPORT,
        WORKER_A_EVIDENCE,
        WORKER_B_EVIDENCE,
        atomic_write_once,
        blocked_report,
        build_summary,
        promote_write_ahead,
        stable_json_bytes,
        validate_probe_evidence,
        validate_repository,
        validate_worker_evidence,
    )
    from ccda_phase3.phase314b_r259_stagec_fold_resolved_tail_attribution import (
        child_environment,
    )

    root = Path(args.root).resolve()
    write_ahead = (
        Path(args.write_ahead_dir).resolve()
        if args.write_ahead_dir
        else root.parent / (root.name + ".phase314b_r259_stagec_resume1_write_ahead")
    )
    try:
        write_ahead.relative_to(root)
    except ValueError:
        pass
    else:
        print("BLOCKED: write-ahead directory must be outside Git worktree", file=sys.stderr)
        return 2

    success = root / SUCCESS_REPORT
    blocked = root / BLOCKED_REPORT
    repo_probe = root / PROBE_EVIDENCE
    repo_worker_a = root / WORKER_A_EVIDENCE
    repo_worker_b = root / WORKER_B_EVIDENCE
    ahead_probe = write_ahead / "environment_probe_evidence.json"
    ahead_worker_a = write_ahead / "science_worker_a_evidence.json"
    ahead_worker_b = write_ahead / "science_worker_b_evidence.json"
    if success.exists() or blocked.exists():
        print("BLOCKED: Stage-C Resume1 terminal report already exists", file=sys.stderr)
        return 2

    repository = None
    try:
        repository = validate_repository(root, args.implementation_commit)
        if write_ahead.exists():
            raise RuntimeError("Resume1 write-ahead directory already exists; do not rerun science")
        write_ahead.mkdir(parents=True, exist_ok=False)
        env = child_environment(root)
        script = root / "scripts/phase3_14b_r259_stagec_resume1_worker.py"
        run_child(
            [
                args.python_bin,
                str(script),
                "--mode",
                "environment-probe",
                "--root",
                str(root),
                "--output",
                str(ahead_probe),
            ],
            root=root,
            env=env,
            label="external durable Stage-C Resume1 portable environment probe",
        )
        for slot, output in (("A", ahead_worker_a), ("B", ahead_worker_b)):
            run_child(
                [
                    args.python_bin,
                    str(script),
                    "--mode",
                    "science",
                    "--worker-slot",
                    slot,
                    "--root",
                    str(root),
                    "--probe",
                    str(ahead_probe),
                    "--repository-head",
                    repository["head"],
                    "--stageu-contract",
                    str(root / "reports/phase3_14b_r258_stageu_candidate_frontier_contract.json"),
                    "--output",
                    str(output),
                ],
                root=root,
                env=env,
                label="external durable Stage-C Resume1 cold science worker {}".format(slot),
            )
        promote_write_ahead(ahead_probe, repo_probe, validate_probe_evidence)
        promote_write_ahead(ahead_worker_a, repo_worker_a, validate_worker_evidence)
        promote_write_ahead(ahead_worker_b, repo_worker_b, validate_worker_evidence)
        summary = build_summary(
            repository=repository,
            probe_path=repo_probe,
            worker_a_path=repo_worker_a,
            worker_b_path=repo_worker_b,
        )
        atomic_write_once(success, stable_json_bytes(summary) + b"\n")
        print(
            json.dumps(
                {
                    "execution_verdict": summary["execution_verdict"],
                    "scientific_status": summary["scientific_status"],
                    "root_cause": summary["root_cause"],
                    "required_next_path": summary["required_next_path"],
                    "summary_sha256": summary["summary_sha256"],
                    "write_ahead_dir": str(write_ahead),
                    "output": str(success),
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException as error:
        for source, destination, validator in (
            (ahead_probe, repo_probe, validate_probe_evidence),
            (ahead_worker_a, repo_worker_a, validate_worker_evidence),
            (ahead_worker_b, repo_worker_b, validate_worker_evidence),
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
                worker_a_path=repo_worker_a if repo_worker_a.is_file() else ahead_worker_a,
                worker_b_path=repo_worker_b if repo_worker_b.is_file() else ahead_worker_b,
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
