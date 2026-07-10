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

import numpy as np


FORBIDDEN_PREFIXES = [
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
]


def run(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        )
    except Exception as exc:
        return repr(exc)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def check_import(name: str) -> Dict[str, Any]:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--windows",
        default="data/phase3_state_diff_windows/phase3_windows.npz",
    )
    parser.add_argument(
        "--out-json", default="reports/phase3_12d_r2_preflight_summary.json"
    )
    parser.add_argument(
        "--out-md", default="reports/phase3_12d_r2_preflight_report.md"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for path in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Initialize the established TensorFlow-free Ravens runtime before the
    # module-by-module import checks.
    try:
        import phase3_policy_rollout as p34

        p34.patch_pybullet_pkg_resources_metadata()
        p34.import_ravens_runtime(root)
    except Exception:
        # The import checks below report the precise failure.
        pass

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    for name in [
        "numpy",
        "torch",
        "pybullet",
        "scipy",
        "ravens.tasks",
        "ravens.environment",
        "ccda_phase3.rollout",
        "ccda_phase3.data_io",
        "ccda_phase3.train_utils",
        "phase3_policy_rollout",
        "phase3_12c_matched_reset_common",
        "phase3_12d_r1_common",
        "phase3_12d_r2_common",
    ]:
        result = check_import(name)
        key = "import_" + name.replace(".", "_")
        checks[key] = bool(result["ok"])
        if not result["ok"]:
            issue("FAIL", "import_failed", f"{name}: {result.get('error')}")

    bad = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    checks["forbidden_modules_not_loaded"] = not bad
    if bad:
        issue("FAIL", "forbidden_modules_loaded", sorted(set(bad)))

    task_path = (
        root
        / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    )
    env_path = root / "external/deformable-ravens/ravens/environment.py"
    task_text = task_path.read_text(errors="replace") if task_path.exists() else ""
    env_text = env_path.read_text(errors="replace") if env_path.exists() else ""

    source_tokens = {
        "task_deferred_arming_marker": "PHASE3_12D_R2_DEFERRED_ARMING",
        "task_arm_method": "def arm_hidden_contact_after_settle",
        "task_pending_state": "_hidden_contact_pending",
        "task_armed_state": "_hidden_contact_armed",
        "task_defer_env": "CCDA_DEFER_HIDDEN_CONTACT_ARMING",
        "task_post_arm_settle_env": "CCDA_POST_ARM_SETTLE_SECONDS",
        "task_breakaway_guard": "if not self._hidden_contact_armed:",
        "task_zero_default_breakaway_damping": 'CCDA_BREAKAWAY_DAMPING", "0.0"',
    }
    for key, token in source_tokens.items():
        checks[key] = token in task_text
        if not checks[key]:
            issue("FAIL", key, token)

    checks["environment_step_lock_present"] = "_ccda_step_lock" in env_text
    checks["environment_physics_hook_present"] = "physics_step_hook" in env_text
    checks["environment_hook_error_surface_present"] = (
        "_ccda_physics_hook_error" in env_text
    )
    for key in [
        "environment_step_lock_present",
        "environment_physics_hook_present",
        "environment_hook_error_surface_present",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

    damping = float(os.environ.get("CCDA_BREAKAWAY_DAMPING", "0.0"))
    checks["runtime_breakaway_damping_zero"] = abs(damping) <= 1e-12
    if not checks["runtime_breakaway_damping_zero"]:
        issue(
            "FAIL",
            "nonzero_breakaway_damping_leaves_residual_condition_after_release",
            damping,
        )

    windows_path = root / args.windows
    checks["windows_loadable"] = windows_path.exists()
    if windows_path.exists():
        data = np.load(windows_path, allow_pickle=True)
        for key in [
            "state_action_x",
            "y_state",
            "y_action",
            "condition_name",
            "split_name",
            "th",
            "action_dim",
            "n_beads",
        ]:
            checks["has_" + key] = key in data.files
            if key not in data.files:
                issue("FAIL", "missing_window_key", key)
        if "action_dim" in data.files:
            action_dim = int(np.asarray(data["action_dim"]).reshape(-1)[0])
            checks["action_dim_14"] = action_dim == 14
            if action_dim != 14:
                issue("FAIL", "action_dim_not_14", action_dim)
        if "y_action" in data.files:
            checks["y_action_dim_14"] = (
                data["y_action"].ndim == 2 and data["y_action"].shape[1] == 14
            )
            if not checks["y_action_dim_14"]:
                issue("FAIL", "y_action_dim_not_14", data["y_action"].shape)
    else:
        issue("FAIL", "windows_missing", windows_path)

    # The current windows/checkpoints predate the deferred-arming repair.
    # This is a scientific-scope warning, not a code-integrity failure.
    checks["legacy_checkpoint_bridge_required"] = True
    issue(
        "WARN",
        "checkpoint_environment_semantics_changed",
        "Existing Phase3 windows/checkpoints were generated before deferred "
        "zero-offset arming. Negative candidate-headroom results require a "
        "repaired-data repeat before architecture-level conclusions.",
    )

    r1_summary = load_json(
        root / "reports/phase3_12d_r1_environment_audit_summary.json"
    )
    checks["r1_environment_audit_exists"] = bool(r1_summary)
    checks["r1_failed_for_visible_geometry"] = any(
        item.get("name") == "hidden_condition_changes_initial_visible_geometry_too_much"
        for item in r1_summary.get("issues", [])
    )
    if not checks["r1_environment_audit_exists"]:
        issue("FAIL", "r1_environment_audit_missing", "")
    if not checks["r1_failed_for_visible_geometry"]:
        issue("FAIL", "unexpected_r1_failure_mode", r1_summary.get("root_cause"))

    for filename in [
        "phase3_12d_r1_common.py",
        "phase3_12d_r1_query_local_snapshot.py",
        "phase3_12d_r1_analyze.py",
        "phase3_12d_r2_common.py",
        "phase3_12d_r2_static_selftest.py",
        "phase3_12d_r2_environment_audit.py",
        "phase3_12d_r2_query_local_snapshot.py",
        "phase3_12d_r2_analyze.py",
        "phase3_12d_r2_postprocess.py",
    ]:
        path = root / "scripts" / filename
        checks["exists_" + filename] = path.exists()
        if not path.exists():
            issue("FAIL", "required_script_missing", filename)

    query_path = root / "scripts/phase3_12d_r2_query_local_snapshot.py"
    if query_path.exists():
        query_text = query_path.read_text(errors="replace")
        checks["query_uses_r2_common"] = "phase3_12d_r2_common" in query_text
        checks["query_uses_deferred_arm_reset"] = (
            "common.deterministic_reset_deferred_arm(" in query_text
        )
        checks["query_does_not_use_r1_reset"] = (
            "p12c.deterministic_reset(" not in query_text
        )
        checks["query_one_worker_per_query_preserved"] = (
            "def run_query" in query_text
            and "for candidate in candidates" in query_text
            and "QuerySnapshot.create" in query_text
            and "snapshot.restore" in query_text
        )
        for key in [
            "query_uses_r2_common",
            "query_uses_deferred_arm_reset",
            "query_does_not_use_r1_reset",
            "query_one_worker_per_query_preserved",
        ]:
            if not checks[key]:
                issue("FAIL", key, key)

    checks["running_in_coord_bimanual"] = (
        os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual"
        or "coord_bimanual" in sys.executable
    )
    if not checks["running_in_coord_bimanual"]:
        issue("WARN", "not_coord_bimanual", sys.executable)

    git = {
        "main_branch": run(["git", "branch", "--show-current"], root).strip(),
        "main_head": run(["git", "rev-parse", "HEAD"], root).strip(),
        "main_status": run(["git", "status", "--short"], root),
        "submodule": run(
            ["git", "submodule", "status", "external/deformable-ravens"], root
        ).strip(),
        "submodule_status": run(
            ["git", "status", "--short"],
            root / "external/deformable-ravens",
        ),
    }

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
        "query_local_snapshot_allowed": not has_fail,
        "scope": "Phase3.12d-r2 paired-visible arming repair preflight",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.12d-r2 Paired-Visible Arming Preflight",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        f"- Query-local snapshot allowed after environment audit: `{not has_fail}`",
        "",
        "## Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key, value in checks.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += ["", "## Issues", "", "| Level | Name | Detail |", "|---|---|---|"]
    if issues:
        for item in issues:
            lines.append(
                f"| `{item['level']}` | `{item['name']}` | "
                f"{str(item['detail']).replace('|', '/')} |"
            )
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    lines += [
        "",
        "## Decision",
        "",
        "- Any FAIL blocks the environment audit.",
        "- The environment audit remains a separate hard gate before candidate execution.",
        "- No training, Phase4, or CPS is permitted.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.12d-r2][FAIL] preflight failed")


if __name__ == "__main__":
    main()
