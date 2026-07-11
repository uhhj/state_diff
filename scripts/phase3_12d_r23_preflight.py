#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3_12d_r22_integrity import sha256_file, strict_json_dump


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def check_import(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "error": "",
        }
    except Exception as exc:
        return {"ok": False, "version": "", "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--r22-summary",
        default="reports/phase3_12d_r22_observation_leakage_summary.json",
    )
    parser.add_argument(
        "--r22-data",
        default="data/phase3_12d_r22_observation_leakage/paired_observations.npz",
    )
    parser.add_argument(
        "--windows",
        default="data/phase3_state_diff_windows/phase3_windows.npz",
    )
    parser.add_argument(
        "--out-json",
        default="reports/phase3_12d_r23_preflight_summary.json",
    )
    parser.add_argument(
        "--out-md",
        default="reports/phase3_12d_r23_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append(
            {"level": level, "name": name, "detail": str(detail)}
        )

    branch = git(root, "branch", "--show-current")
    main_head = git(root, "rev-parse", "HEAD")
    main_status = git(root, "status", "--short")
    submodule_head = git(
        root,
        "-C",
        "external/deformable-ravens",
        "rev-parse",
        "HEAD",
    )
    submodule_status = git(
        root,
        "-C",
        "external/deformable-ravens",
        "status",
        "--short",
    )

    checks["main_branch_experiment1"] = branch == "Experiment1"
    checks["submodule_clean"] = not bool(submodule_status)
    checks["main_worktree_clean"] = not bool(main_status)
    if not checks["main_branch_experiment1"]:
        issue("FAIL", "wrong_main_branch", branch)
    if not checks["submodule_clean"]:
        issue("FAIL", "submodule_dirty", submodule_status)
    if main_status:
        issue(
            "WARN",
            "main_worktree_contains_preserved_changes",
            main_status,
        )

    for name in [
        "numpy",
        "torch",
        "pybullet",
        "ccda_phase3.data_io",
        "ccda_phase3.rollout",
        "ccda_phase3.observation_contract",
        "phase3_12d_r22_integrity",
        "phase3_12d_r22_collect_observation_leakage",
        "phase3_12d_r22_analyze_observation_leakage",
        "phase3_12d_r2_environment_audit",
    ]:
        result = check_import(name)
        checks["import_" + name.replace(".", "_")] = bool(result["ok"])
        if not result["ok"]:
            issue("FAIL", "import_failed", f"{name}: {result['error']}")

    r22_summary_path = root / args.r22_summary
    r22_data_path = root / args.r22_data
    windows_path = root / args.windows
    for label, path in [
        ("r22_summary", r22_summary_path),
        ("r22_data", r22_data_path),
        ("windows", windows_path),
    ]:
        checks[label + "_exists"] = path.exists()
        if not path.exists():
            issue("FAIL", "required_file_missing", f"{label}: {path}")

    r22_summary: Dict[str, Any] = {}
    if r22_summary_path.exists():
        r22_summary = json.loads(r22_summary_path.read_text())
        checks["r22_expected_root_cause"] = (
            r22_summary.get("root_cause")
            == "phase312d_r22_latent_condition_observably_leaked"
        )
        if not checks["r22_expected_root_cause"]:
            issue(
                "FAIL",
                "unexpected_r22_root_cause",
                r22_summary.get("root_cause"),
            )

    dimensions: Dict[str, int] = {}
    if windows_path.exists():
        with np.load(windows_path, allow_pickle=False) as data:
            th = int(np.asarray(data["th"]).reshape(-1)[0])
            action_dim = int(np.asarray(data["action_dim"]).reshape(-1)[0])
            n_beads = int(np.asarray(data["n_beads"]).reshape(-1)[0])
            from ccda_phase3.observation_contract import schema_dimensions

            dimensions = schema_dimensions(
                n_beads=n_beads,
                th=th,
                action_dim=action_dim,
            )
            checks["legacy_state_dim_135"] = (
                dimensions["privileged_state_dim"] == 135
            )
            checks["proposed_state_dim_87"] = (
                dimensions["position_proprio_state_dim"] == 87
            )
            checks["proposed_model_x_dim_303"] = (
                dimensions["position_proprio_state_action_x_dim"] == 303
            )

    try:
        import phase3_12d_r22_collect_observation_leakage as collector

        required = {
            "capture_track": collector.capture_track,
            "rows_to_arrays": collector.rows_to_arrays,
            "discover_used_seeds": collector.discover_used_seeds,
        }
        signatures = {
            name: str(inspect.signature(value))
            for name, value in required.items()
        }
        checks["r22_collector_reuse_api_available"] = True
    except Exception as exc:
        signatures = {}
        checks["r22_collector_reuse_api_available"] = False
        issue("FAIL", "r22_collector_api_missing", repr(exc))

    collector_source = (
        root / "scripts/phase3_12d_r22_collect_observation_leakage.py"
    )
    analyzer_source = (
        root / "scripts/phase3_12d_r22_analyze_observation_leakage.py"
    )
    task_source = (
        root
        / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    )

    source_text = (
        collector_source.read_text(errors="replace")
        if collector_source.exists()
        else ""
    )
    task_text = (
        task_source.read_text(errors="replace")
        if task_source.exists()
        else ""
    )
    checks["r22_model_x_repeats_single_state"] = (
        "pad_history([state], args.th)" in source_text
    )
    checks["task_logs_pybullet_base_velocity"] = (
        "getBaseVelocity" in task_text
        and '"bead_velocities"' in task_text
    )
    if checks["r22_model_x_repeats_single_state"]:
        issue(
            "WARN",
            "r22_model_x_is_not_real_causal_history",
            "pad_history([state], th) repeats one state",
        )
    if checks["task_logs_pybullet_base_velocity"]:
        issue(
            "WARN",
            "simulator_bead_velocity_is_privileged_state",
            "bead velocity comes directly from PyBullet getBaseVelocity",
        )

    forbidden = [
        name
        for name in sys.modules
        if name == "tensorflow" or name.startswith("tensorflow.")
    ]
    checks["tensorflow_not_loaded"] = not forbidden
    if forbidden:
        issue("FAIL", "tensorflow_loaded", forbidden[:20])

    verdict = (
        "FAIL"
        if any(entry["level"] == "FAIL" for entry in issues)
        else "WARN"
        if issues
        else "PASS"
    )
    collection_allowed = verdict != "FAIL"

    payload = {
        "verdict": verdict,
        "collection_allowed": collection_allowed,
        "checks": checks,
        "issues": issues,
        "main_branch": branch,
        "main_head": main_head,
        "main_status": main_status,
        "submodule_head": submodule_head,
        "python": sys.executable,
        "dimensions": dimensions,
        "r22_summary": {
            "verdict": r22_summary.get("verdict"),
            "root_cause": r22_summary.get("root_cause"),
            "backend": r22_summary.get("backend"),
        },
        "reused_function_signatures": signatures,
        "hashes": {
            str(path.relative_to(root)): sha256_file(path)
            for path in [
                collector_source,
                analyzer_source,
                task_source,
                root / "ccda_phase3/observation_contract.py",
            ]
            if path.exists()
        },
        "scope": {
            "model_training": False,
            "candidate_matrix": False,
            "phase4": False,
            "cps": False,
            "submodule_changes": False,
        },
    }
    strict_json_dump(root / args.out_json, payload)

    lines = [
        "# Phase3.12d-r2.3 Observation Contract Preflight",
        "",
        f"- Verdict: `{verdict}`",
        f"- Collection allowed: `{collection_allowed}`",
        f"- Main HEAD: `{main_head}`",
        f"- Submodule: `{submodule_head}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for entry in issues:
            lines.append(
                f"| `{entry['level']}` | `{entry['name']}` | "
                f"{entry['detail'].replace('|', '/')} |"
            )
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    lines += [
        "",
        "## Decision",
        "",
        "- This phase does not change the physical task.",
        "- It does not change the production training schema yet.",
        "- It collects real no-action trajectories and separates privileged "
        "simulator velocity from observable position/proprio history.",
        "- No candidate matrix, Phase4, or CPS is allowed.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[r2.3] preflight failed")


if __name__ == "__main__":
    main()
