#!/usr/bin/env python3
"""Build the immutable state-v3 cache and run train-only attribution."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ccda_phase3.phase314b_r255_stagec_cache import (
    BASE_EVIDENCE_COMMIT,
    CACHE_FILE,
    CACHE_MANIFEST_FILE,
    EXPECTED_SUBMODULE_COMMIT,
    PHASE,
    TRAIN_VIEW_FILE,
    atomic_write_once,
    compare_build_directories,
    directory_manifest,
    sha256_file,
    stable_json_bytes,
    validate_stage_b_artifacts,
)

TEST_GATE = "reports/phase3_14b_r255_stagec_test_gate_summary.json"
ATTRIBUTION_REPORT = "reports/phase3_14b_r255_stagec_attribution.json"
SUMMARY_REPORT = "reports/phase3_14b_r255_stagec_summary.json"
MARKDOWN_REPORT = "reports/phase3_14b_r255_stagec_report.md"
FINAL_CACHE_ROOT = "data/phase3_14_cache_v3"
EXPECTED_TEST_PASSED = 463
EXPECTED_TEST_FILES = 30


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=str(root), text=True
    ).strip()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path}")
    return value


def fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def assert_repository_contract(
    root: Path,
    test_gate_path: Path,
    *,
    allowed_untracked_paths: Sequence[str] = (),
) -> Dict[str, Any]:
    if git(root, "branch", "--show-current") != "Experiment1":
        raise RuntimeError("Stage C requires Experiment1")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_EVIDENCE_COMMIT, "HEAD"],
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Stage B evidence commit is not an ancestor")
    submodule = root / "external/deformable-ravens"
    if git(submodule, "rev-parse", "HEAD") != EXPECTED_SUBMODULE_COMMIT:
        raise RuntimeError("DeformableRavens commit changed")
    if git(submodule, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("DeformableRavens worktree is dirty")

    status = [
        line
        for line in git(
            root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line.strip()
    ]
    allowed = {test_gate_path.relative_to(root).as_posix()}
    for value in allowed_untracked_paths:
        candidate = Path(value)
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )
        try:
            allowed.add(candidate.relative_to(root).as_posix())
        except ValueError as exc:
            raise RuntimeError(
                f"allowed worktree path is outside repository: {candidate}"
            ) from exc
    unexpected: List[str] = []
    for line in status:
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path not in allowed:
            unexpected.append(line)
    if unexpected:
        raise RuntimeError(
            "unexpected worktree paths before Stage C build: "
            + repr(unexpected)
        )
    return {
        "branch": "Experiment1",
        "implementation_head": git(root, "rev-parse", "HEAD"),
        "base_evidence_commit": BASE_EVIDENCE_COMMIT,
        "submodule_commit": EXPECTED_SUBMODULE_COMMIT,
    }


def validate_test_gate(root: Path, path: Path) -> Dict[str, Any]:
    report = load_json(path)
    if report.get("verdict") != "PASS":
        raise RuntimeError("Stage C test gate is not PASS")
    if int(report.get("passed_test_count", -1)) != EXPECTED_TEST_PASSED:
        raise RuntimeError("Stage C test pass count changed")
    manifest = report.get("test_manifest_sha256")
    if not isinstance(manifest, Mapping) or len(manifest) != EXPECTED_TEST_FILES:
        raise RuntimeError("Stage C test manifest changed")
    for relative, expected in manifest.items():
        observed = sha256_file(root / str(relative))
        if observed != str(expected):
            raise RuntimeError(f"test changed after gate: {relative}")
    return report


def deterministic_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def run_command(command: List[str], *, cwd: Path, environment: Mapping[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with code {completed.returncode}: {command}"
        )
    return completed.stdout


def render_markdown(summary: Mapping[str, Any]) -> str:
    cache = summary["cache"]
    attribution = summary["attribution"]
    classification = attribution["classification"]
    lines = [
        "# Phase3.14b-r2.5.5 Stage C State-v3 Cache and Attribution",
        "",
        f"- Audit verdict: `{summary['verdict']}`",
        f"- Scientific status: `{summary['scientific_status']}`",
        f"- Root cause: `{summary['root_cause']}`",
        f"- Required next path: `{summary['required_next_path']}`",
        "",
        "## Immutable cache",
        "",
        f"- Cache rows: `{cache['rows']}`",
        f"- Train rows: `{cache['train_rows']}`",
        f"- Paired window keys: `{cache['pair_keys']}`",
        f"- Cache SHA256: `{cache['cache_sha256']}`",
        f"- Train-only view SHA256: `{cache['train_attribution_view_sha256']}`",
        f"- Two independent cache builds exact: `{str(cache['independent_builds_exact']).lower()}`",
        f"- Promotion exact: `{str(cache['promotion_exact']).lower()}`",
        "",
        "## Train-only robot-proxy attribution",
        "",
        f"- Attribution workers exact: `{str(summary['attribution_workers_exact']).lower()}`",
        f"- State-v3 schema contract: `{str(attribution['schema_audit']['schema_contract_pass']).lower()}`",
        f"- Best deployable predictor: `{classification['best_deployable_model']}`",
        f"- Best deployable NMSE: `{classification['best_deployable_nmse']:.9g}`",
        f"- Action-conditioned gain: `{classification['action_conditioned_gain']:.9g}`",
        f"- Robot future action-materiality gain: `{classification['robot_action_incremental_gain']:.9g}`",
        "",
        "## Boundary",
        "",
        "- The legacy state-v2 cache and Stage-B artifacts were not modified.",
        "- Attribution used only the train-only view and entire paired episode groups stayed in one fold.",
        "- No diffusion model or historical state-v2 prediction was used.",
        "- No reverse sampling, formal IDM training, candidate execution, Phase4, or CPS was run.",
        "- `train_only_recommendation=None` and `selected_configuration=None`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--python-bin",
        default="/miniforge3/envs/coord_bimanual/bin/python",
    )
    parser.add_argument("--cache-root", default=FINAL_CACHE_ROOT)
    parser.add_argument(
        "--attribution-report",
        default=ATTRIBUTION_REPORT,
    )
    parser.add_argument(
        "--summary-report",
        default=SUMMARY_REPORT,
    )
    parser.add_argument(
        "--markdown-report",
        default=MARKDOWN_REPORT,
    )
    parser.add_argument(
        "--allowed-untracked-path",
        action="append",
        default=[],
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    def resolve_output(value: str) -> Path:
        candidate = Path(value)
        return (
            candidate.resolve()
            if candidate.is_absolute()
            else (root / candidate).resolve()
        )

    final_cache_root = resolve_output(args.cache_root)
    test_gate_path = root / TEST_GATE
    attribution_path = resolve_output(args.attribution_report)
    summary_path = resolve_output(args.summary_report)
    markdown_path = resolve_output(args.markdown_report)
    for output in (final_cache_root, attribution_path, summary_path, markdown_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite Stage C output: {output}")

    repository = assert_repository_contract(
        root,
        test_gate_path,
        allowed_untracked_paths=args.allowed_untracked_path,
    )
    test_gate = validate_test_gate(root, test_gate_path)
    stage_b_inputs = validate_stage_b_artifacts(root)
    environment = deterministic_environment()

    parent = final_cache_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    worker_a = parent / f".{final_cache_root.name}.worker_a.{os.getpid()}"
    worker_b = parent / f".{final_cache_root.name}.worker_b.{os.getpid()}"
    if worker_a.exists() or worker_b.exists():
        raise RuntimeError("Stage C cache worker path already exists")
    try:
        for output_root in (worker_a, worker_b):
            run_command(
                [
                    args.python_bin,
                    str(root / "scripts/phase3_14b_r255_stagec_worker.py"),
                    "--root",
                    str(root),
                    "--output-root",
                    str(output_root),
                ],
                cwd=root,
                environment=environment,
            )
        cache_comparison = compare_build_directories(worker_a, worker_b)
        if not cache_comparison["exact"]:
            raise RuntimeError("independent Stage C cache builds differ")
        worker_b_manifest = directory_manifest(worker_b)
        os.replace(worker_a, final_cache_root)
        fsync_directory(parent)
        promotion = {
            "exact": directory_manifest(final_cache_root) == worker_b_manifest,
            "promoted": directory_manifest(final_cache_root),
            "reference": worker_b_manifest,
        }
        if not promotion["exact"]:
            raise RuntimeError("promoted Stage C cache differs from worker output")
    finally:
        shutil.rmtree(worker_a, ignore_errors=True)
        shutil.rmtree(worker_b, ignore_errors=True)

    cache_manifest_path = final_cache_root / CACHE_MANIFEST_FILE
    cache_manifest = load_json(cache_manifest_path)
    if cache_manifest.get("cache_sha256") != sha256_file(final_cache_root / CACHE_FILE):
        raise RuntimeError("promoted cache manifest binding failed")
    if cache_manifest.get("train_attribution_view_sha256") != sha256_file(
        final_cache_root / TRAIN_VIEW_FILE
    ):
        raise RuntimeError("promoted train-view manifest binding failed")

    reports_parent = attribution_path.parent
    reports_parent.mkdir(parents=True, exist_ok=True)
    attribution_a = reports_parent / f".{attribution_path.name}.worker_a.{os.getpid()}"
    attribution_b = reports_parent / f".{attribution_path.name}.worker_b.{os.getpid()}"
    try:
        for output in (attribution_a, attribution_b):
            run_command(
                [
                    args.python_bin,
                    str(
                        root
                        / "scripts/phase3_14b_r255_stagec_attribution_worker.py"
                    ),
                    "--train-view",
                    str(final_cache_root / TRAIN_VIEW_FILE),
                    "--output",
                    str(output),
                ],
                cwd=root,
                environment=environment,
            )
        attribution_workers_exact = bool(
            sha256_file(attribution_a) == sha256_file(attribution_b)
        )
        if not attribution_workers_exact:
            raise RuntimeError("independent attribution workers differ")
        attribution = load_json(attribution_a)
        atomic_write_once(attribution_path, attribution_a.read_bytes())
    finally:
        for output in (attribution_a, attribution_b):
            if output.exists():
                output.unlink()

    if attribution.get("verdict") != "PASS":
        raise RuntimeError("train-only attribution did not pass")
    if attribution.get("scientific_status") != "BLOCKED":
        raise RuntimeError("Stage C scientific boundary changed")
    if attribution.get("train_only_recommendation") is not None:
        raise RuntimeError("Stage C selected a train-only recommendation")
    if attribution.get("selected_configuration") is not None:
        raise RuntimeError("Stage C selected a configuration")

    cache_section = {
        "root": final_cache_root.relative_to(root).as_posix(),
        "manifest": cache_manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(cache_manifest_path),
        "cache": (final_cache_root / CACHE_FILE).relative_to(root).as_posix(),
        "cache_sha256": sha256_file(final_cache_root / CACHE_FILE),
        "train_attribution_view": (
            final_cache_root / TRAIN_VIEW_FILE
        ).relative_to(root).as_posix(),
        "train_attribution_view_sha256": sha256_file(
            final_cache_root / TRAIN_VIEW_FILE
        ),
        "rows": int(cache_manifest["rows"]),
        "train_rows": int(cache_manifest["train_rows"]),
        "pair_keys": int(cache_manifest["pair_keys"]),
        "independent_builds_exact": bool(cache_comparison["exact"]),
        "promotion_exact": bool(promotion["exact"]),
        "file_manifest": directory_manifest(final_cache_root),
    }
    summary: Dict[str, Any] = {
        "phase": PHASE,
        "schema": "phase314b_r255_stagec_summary_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": attribution["root_cause"],
        "required_next_path": attribution["required_next_path"],
        "repository": repository,
        "test_gate": test_gate,
        "test_gate_sha256": sha256_file(test_gate_path),
        "stage_b_inputs": stage_b_inputs,
        "cache_build_root_cause": (
            "phase314b_r255_stagec_state_v3_cache_materialized_and_byte_reproducible"
        ),
        "cache": cache_section,
        "attribution_report": attribution_path.relative_to(root).as_posix(),
        "attribution_report_sha256": sha256_file(attribution_path),
        "attribution_workers_exact": attribution_workers_exact,
        "attribution": attribution,
        "legacy_cache_modified": False,
        "stage_b_artifacts_modified": False,
        "state_v3_cache_written": True,
        "state_v3_robot_proxy_attribution_interpretable": True,
        "legacy_state_v2_robot_proxy_attribution_interpretable": False,
        "validation_targets_used_for_attribution": False,
        "formal_test_targets_used_for_attribution": False,
        "model_attribution_performed": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "idm_training": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    atomic_write_once(summary_path, stable_json_bytes(summary))
    atomic_write_once(markdown_path, render_markdown(summary).encode("utf-8"))
    print(
        json.dumps(
            {
                "verdict": "PASS",
                "scientific_status": "BLOCKED",
                "root_cause": summary["root_cause"],
                "required_next_path": summary["required_next_path"],
                "cache_sha256": cache_section["cache_sha256"],
                "attribution_sha256": summary["attribution_report_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
