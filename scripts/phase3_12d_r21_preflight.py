#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_FILES = [
    "scripts/phase3_12d_r2_common.py",
    "scripts/phase3_12d_r2_environment_audit.py",
    "scripts/phase3_12d_r2_query_local_snapshot.py",
    "scripts/phase3_12d_r2_analyze.py",
    "scripts/phase3_12d_r2_postprocess.py",
    "scripts/phase3_12d_r21_paired_horizon_audit.py",
    "scripts/phase3_12d_r21_static_selftest.py",
    "scripts/phase3_12d_r21_run.sh",
]

FORBIDDEN_PREFIXES = [
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
]


def run(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd),
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception as exc:
        return repr(exc)


def import_check(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
            "error": None,
        }
    except Exception as exc:
        return {"ok": False, "version": None, "error": repr(exc)}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--r2-summary",
        default="reports/phase3_12d_r2_environment_audit_summary.json",
    )
    parser.add_argument(
        "--out-json",
        default="reports/phase3_12d_r21_preflight_summary.json",
    )
    parser.add_argument(
        "--out-md",
        default="reports/phase3_12d_r21_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Initialize the established TensorFlow-free Ravens runtime before import
    # checks so this comparator-only audit does not pull TensorFlow transitively.
    try:
        import phase3_policy_rollout as p34

        p34.patch_pybullet_pkg_resources_metadata()
        p34.import_ravens_runtime(root)
    except Exception:
        # Module checks below report the exact failure.
        pass

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    git = {
        "branch": run(["git", "branch", "--show-current"], root),
        "head": run(["git", "rev-parse", "HEAD"], root),
        "origin_experiment1": run(
            ["git", "ls-remote", "origin", "Experiment1"], root
        ).split("\t")[0],
        "status": run(["git", "status", "--short"], root),
        "submodule": run(
            ["git", "submodule", "status", "external/deformable-ravens"], root
        ),
        "submodule_status": run(
            ["git", "-C", "external/deformable-ravens", "status", "--short"],
            root,
        ),
    }

    for name in [
        "numpy",
        "torch",
        "pybullet",
        "scipy",
        "ravens.tasks",
        "ravens.environment",
        "phase3_12d_r2_common",
        "phase3_12d_r2_environment_audit",
        "phase3_12d_r2_query_local_snapshot",
        "phase3_12d_r2_analyze",
        "phase3_12d_r21_paired_horizon_audit",
    ]:
        result = import_check(name)
        key = "import_" + name.replace(".", "_")
        checks[key] = bool(result["ok"])
        if not result["ok"]:
            issue("FAIL", "import_failed", {"module": name, **result})

    bad_loaded: List[str] = []
    for module_name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if module_name == prefix or module_name.startswith(prefix + "."):
                bad_loaded.append(module_name)
    checks["forbidden_modules_not_loaded"] = not bad_loaded
    if bad_loaded:
        issue("FAIL", "forbidden_modules_loaded", sorted(set(bad_loaded)))

    for rel in REQUIRED_FILES:
        exists = (root / rel).exists()
        checks["exists_" + Path(rel).name.replace(".", "_")] = exists
        if not exists:
            issue("FAIL", "required_file_missing", rel)

    r2_audit_path = root / "scripts/phase3_12d_r2_environment_audit.py"
    r2_common_path = root / "scripts/phase3_12d_r2_common.py"
    r2_run_path = root / "scripts/phase3_12d_r2_run.sh"

    r2_audit_text = r2_audit_path.read_text(errors="replace") if r2_audit_path.exists() else ""
    r2_common_text = r2_common_path.read_text(errors="replace") if r2_common_path.exists() else ""
    r2_run_text = r2_run_path.read_text(errors="replace") if r2_run_path.exists() else ""

    # Historical r2 bug: hidden(t=N) was compared with free(t=0).
    old_comparator_present = bool(
        re.search(
            r"difference_stats\s*\(\s*xy\s*,\s*free_pre\s*\)",
            r2_audit_text,
            flags=re.MULTILINE,
        )
    )
    checks["r2_historical_absolute_drift_comparator_present"] = old_comparator_present
    if old_comparator_present:
        issue(
            "WARN",
            "r2_post_arm_gate_compares_hidden_horizon_to_free_t0",
            "Historical r2 report cannot establish condition-specific excess drift.",
        )
    else:
        issue(
            "WARN",
            "r2_historical_comparator_pattern_not_found",
            "Inspect local r2 audit manually before trusting the correction assumptions.",
        )

    checks["r2_common_exports_post_arm_xy"] = '"post_arm_xy"' in r2_common_text
    checks["r2_common_exports_post_arm_velocity"] = (
        '"post_arm_velocity"' in r2_common_text
    )
    checks["r2_common_exports_pre_arm_xy"] = '"pre_arm_xy"' in r2_common_text
    checks["r2_common_uses_direct_physics_steps"] = "direct_physics_step" in r2_common_text

    for key in [
        "r2_common_exports_post_arm_xy",
        "r2_common_exports_post_arm_velocity",
        "r2_common_exports_pre_arm_xy",
        "r2_common_uses_direct_physics_steps",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

    checks["r2_runner_hardcodes_old_environment_audit"] = (
        "reports/phase3_12d_r2_environment_audit_summary.json" in r2_run_text
    )
    if checks["r2_runner_hardcodes_old_environment_audit"]:
        issue(
            "WARN",
            "do_not_run_original_r2_runner_for_r21",
            "The original runner re-executes the historical comparator and will block again.",
        )

    r2_summary = load_json(root / args.r2_summary)
    checks["r2_summary_exists"] = bool(r2_summary)
    checks["r2_summary_failed"] = r2_summary.get("verdict") == "FAIL"
    checks["r2_query_matrix_was_blocked"] = not bool(
        r2_summary.get("query_local_snapshot_allowed", True)
    )
    if not checks["r2_summary_exists"]:
        issue("FAIL", "r2_summary_missing", args.r2_summary)

    checks["submodule_clean"] = git["submodule_status"] == ""
    if not checks["submodule_clean"]:
        issue("FAIL", "submodule_has_uncommitted_changes", git["submodule_status"])

    checks["running_in_coord_bimanual"] = (
        "coord_bimanual" in sys.executable
        or os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual"
    )
    if not checks["running_in_coord_bimanual"]:
        issue("WARN", "not_running_in_coord_bimanual", sys.executable)

    # r2.1 must not patch the physical task; it only corrects the comparator.
    r21_runner_path = root / "scripts/phase3_12d_r21_run.sh"
    r21_runner_text = (
        r21_runner_path.read_text(errors="replace") if r21_runner_path.exists() else ""
    )
    checks["r21_runner_does_not_patch_submodule"] = (
        "phase3_12d_r2_patch_submodule.py" not in r21_runner_text
        and "git -C external/deformable-ravens" not in r21_runner_text
    )
    if not checks["r21_runner_does_not_patch_submodule"]:
        issue(
            "FAIL",
            "r21_must_not_modify_submodule",
            "r2.1 is an audit correction, not another physical-environment patch.",
        )

    has_fail = any(item["level"] == "FAIL" for item in issues)
    has_warn = any(item["level"] == "WARN" for item in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "git": git,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "r2_summary": {
            "verdict": r2_summary.get("verdict"),
            "root_cause": r2_summary.get("root_cause"),
            "query_local_snapshot_allowed": r2_summary.get(
                "query_local_snapshot_allowed"
            ),
        },
        "decision": (
            "WARN is expected for the historical r2 comparator. Any FAIL blocks r2.1."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.12d-r2.1 Paired-Horizon Audit Preflight",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        f"- Conda env: `{os.environ.get('CONDA_DEFAULT_ENV')}`",
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
        for item in issues:
            detail = str(item["detail"]).replace("|", "/")
            lines.append(f"| `{item['level']}` | `{item['name']}` | {detail} |")
    else:
        lines.append("| `PASS` | `none` | No issue found. |")

    lines += [
        "",
        "## Decision",
        "",
        "- The historical r2 absolute-drift failure is not condition-specific evidence.",
        "- r2.1 compares free and hidden at the same no-action horizon.",
        "- r2.1 does not modify the DeformableRavens submodule.",
        "- No training, Phase4, or CPS is allowed.",
    ]
    out_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.12d-r2.1][FAIL] preflight failed")


if __name__ == "__main__":
    main()
