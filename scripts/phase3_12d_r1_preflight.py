#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import phase3_12d_r1_common as common


def command(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True
        )
    except Exception as exc:
        return repr(exc)


def check_import(name: str) -> Dict[str, Any]:
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
    return json.loads(path.read_text()) if path.exists() else {}


def first_state_checkpoint(root: Path, checkpoint_root: str) -> Optional[Path]:
    base = root / checkpoint_root / "state_action"
    if not base.exists():
        return None
    for candidate in sorted(x for x in base.glob("fold_*_seed_*") if x.is_dir()):
        if (candidate / "state_model.pt").exists():
            return candidate / "state_model.pt"
    return None


def repaired_checkpoint(root: Path, summary_path: str, ablation: str) -> Optional[Path]:
    payload = load_json(root / summary_path)
    value = payload.get("results", {}).get(ablation, {}).get("checkpoint")
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None


def codec_summary(root: Path, data: Any, fallback: str) -> Dict[str, Any]:
    candidates: List[Path] = []
    if "action_template_json_or_pickle_path" in data.files:
        candidates.append(
            Path(str(common.scalar_from_npz(data, "action_template_json_or_pickle_path")))
        )
    candidates.append(Path(fallback))
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.exists():
            continue
        try:
            from ccda_phase3.data_io import load_action_codec_from_template

            codec = load_action_codec_from_template(path)
        except Exception:
            with path.open("rb") as file:
                codec = pickle.load(file)
        dim_attr = getattr(codec, "dim", None)
        dim = int(dim_attr() if callable(dim_attr) else dim_attr)
        return {
            "path": str(path),
            "dim": dim,
            "summary": codec.summary() if hasattr(codec, "summary") else {},
        }
    return {"path": "", "dim": -1, "summary": {}}


def source_checks(root: Path) -> Dict[str, Any]:
    env_path = root / "external/deformable-ravens/ravens/environment.py"
    task_path = root / "external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"
    query_path = root / "scripts/phase3_12d_r1_query_local_snapshot.py"
    common_path = root / "scripts/phase3_12d_r1_common.py"
    env_text = env_path.read_text(errors="replace") if env_path.exists() else ""
    task_text = task_path.read_text(errors="replace") if task_path.exists() else ""
    query_text = query_path.read_text(errors="replace") if query_path.exists() else ""
    common_text = common_path.read_text(errors="replace") if common_path.exists() else ""
    return {
        "environment_step_lock_present": "self._ccda_step_lock = threading.RLock()" in env_text,
        "environment_physics_hook_present": "physics_step_hook" in env_text,
        "environment_hook_error_surface_present": "_ccda_physics_hook_error" in env_text,
        "task_physics_hook_present": "def physics_step_hook" in task_text,
        "task_breakaway_physics_counter_present": "_ccda_physics_step_count" in task_text,
        "task_breakaway_release_physics_step_present": "_breakaway_release_physics_step" in task_text,
        "task_breakaway_constraint_cleared_on_release": "self._breakaway_constraint_id = None" in task_text,
        "query_uses_one_worker_per_query": "def run_query" in query_text and "for candidate in candidates" in query_text,
        "query_uses_save_state": "QuerySnapshot.create" in query_text and "p.saveState" in common_text,
        "query_uses_restore_state": "p.restoreState" in common_text,
        "query_does_not_cross_process_replay_candidates": "prefix_actions_completed" in query_text and "snapshot.restore" in query_text,
        "action_history_appended_after_step": (
            query_text.find("execute_action(env, predicted[\"action\"])")
            < query_text.find("action_hist.append")
            if "execute_action(env, predicted[\"action\"])" in query_text
            and "action_hist.append" in query_text
            else False
        ),
        "condition_label_only_in_diagnostic_candidates": (
            "condition_nearest_upper" in query_text
            and "uses_condition_label=(selector == \"condition_nearest_upper\")" in query_text
        ),
    }


def script_hazards(root: Path) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_12d_r1_common.py",
        root / "scripts/phase3_12d_r1_patch_environment.py",
        root / "scripts/phase3_12d_r1_environment_audit.py",
        root / "scripts/phase3_12d_r1_query_local_snapshot.py",
        root / "scripts/phase3_12d_r1_analyze.py",
        root / "scripts/phase3_12d_r1_preflight.py",
    ]
    for path in targets:
        if not path.exists():
            issues.append({"level": "FAIL", "name": "required_script_missing", "detail": str(path)})
            continue
        text = path.read_text(errors="replace")
        for module in ("tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"):
            if re.search(rf"^\s*(?:import|from)\s+{re.escape(module)}", text, flags=re.MULTILINE):
                issues.append({"level": "FAIL", "name": "forbidden_import", "detail": f"{path}:{module}"})
        if re.search(r"^\s*import\s+ravens\s*$", text, flags=re.MULTILINE):
            issues.append({"level": "FAIL", "name": "top_level_import_ravens", "detail": str(path)})
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action-template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint-root", default="checkpoints/phase3")
    parser.add_argument("--phase39b-summary", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--best-ablation", default="xy_only_high_weight")
    parser.add_argument("--phase312c-summary", default="reports/phase3_12c_paired_selector_summary.json")
    parser.add_argument("--phase312d-summary", default="reports/phase3_12d_candidate_headroom_summary.json")
    parser.add_argument("--out-json", default="reports/phase3_12d_r1_preflight_summary.json")
    parser.add_argument("--out-md", default="reports/phase3_12d_r1_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    common.add_repo_paths(root)

    # Reuse the established TensorFlow-free Ravens runtime shim before the
    # module-by-module import checks below.
    try:
        import phase3_policy_rollout as p34

        p34.patch_pybullet_pkg_resources_metadata()
        p34.import_ravens_runtime(root)
    except Exception:
        # Import checks below report the exact runtime failure.
        pass

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    imports = [
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
        "phase3_12b_proxy_score_rollout",
        "phase3_12c_matched_reset_common",
        "phase3_12d_r1_common",
    ]
    for name in imports:
        result = check_import(name)
        checks["import_" + name.replace(".", "_")] = result["ok"]
        if not result["ok"]:
            issue("FAIL", "import_failed", f"{name}: {result['error']}")

    try:
        common.assert_no_forbidden_modules("preflight")
        checks["forbidden_modules_not_loaded"] = True
    except Exception as exc:
        checks["forbidden_modules_not_loaded"] = False
        issue("FAIL", "forbidden_modules_loaded", repr(exc))

    windows_path = root / args.windows
    if not windows_path.exists():
        issue("FAIL", "windows_missing", str(windows_path))
        data = None
    else:
        data = np.load(windows_path, allow_pickle=True)
        checks["windows_loadable"] = True
        for key in ("state_action_x", "y_state", "y_action", "condition_name", "split_name", "th", "action_dim", "n_beads"):
            checks["has_" + key] = key in data.files
            if key not in data.files:
                issue("FAIL", "missing_window_key", key)
        if "action_dim" in data.files:
            checks["action_dim_14"] = int(common.scalar_from_npz(data, "action_dim")) == 14
        if "y_action" in data.files:
            checks["y_action_dim_14"] = data["y_action"].ndim == 2 and data["y_action"].shape[1] == 14

    codec = codec_summary(root, data, args.action_template) if data is not None else {"dim": -1, "summary": {}}
    checks["codec_dim_14"] = codec["dim"] == 14
    checks["codec_no_camera_config"] = int(codec["summary"].get("num_camera_config_paths", 0) or 0) == 0

    state_checkpoint = first_state_checkpoint(root, args.checkpoint_root)
    idm_checkpoint = repaired_checkpoint(root, args.phase39b_summary, args.best_ablation)
    checks["state_checkpoint_exists"] = state_checkpoint is not None
    checks["repaired_idm_exists"] = idm_checkpoint is not None

    phase312c = load_json(root / args.phase312c_summary)
    checks["phase312c_summary_exists"] = bool(phase312c)
    checks["phase312c_matched_reset_passed"] = bool(phase312c.get("matched_reset_integrity_passed", False))

    phase312d = load_json(root / args.phase312d_summary)
    checks["phase312d_failure_is_engineering_not_scientific"] = (
        not phase312d
        or phase312d.get("root_cause") == "phase312d_code_or_prefix_replay_integrity_failed"
    )
    if phase312d:
        issue(
            "WARN",
            "prior_phase312d_invalid_due_to_prefix_replay",
            {"verdict": phase312d.get("verdict"), "root_cause": phase312d.get("root_cause")},
        )

    source = source_checks(root)
    for key, value in source.items():
        checks[key] = value
        if not value:
            issue("FAIL", key, value)

    for hazard in script_hazards(root):
        issues.append(hazard)

    checks["running_in_coord_bimanual"] = (
        os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual"
        or "coord_bimanual" in sys.executable
    )
    if not checks["running_in_coord_bimanual"]:
        issue("WARN", "not_coord_bimanual", sys.executable)

    for key in (
        "action_dim_14",
        "y_action_dim_14",
        "codec_dim_14",
        "codec_no_camera_config",
        "state_checkpoint_exists",
        "repaired_idm_exists",
        "phase312c_summary_exists",
        "phase312c_matched_reset_passed",
    ):
        if not checks.get(key, False):
            issue("FAIL", key, checks.get(key))

    git = {
        "main_branch": command(["git", "branch", "--show-current"], root).strip(),
        "main_head": command(["git", "rev-parse", "HEAD"], root).strip(),
        "main_status": command(["git", "status", "--short"], root),
        "submodule": command(["git", "submodule", "status", "external/deformable-ravens"], root).strip(),
        "submodule_status": command(["git", "status", "--short"], root / "external/deformable-ravens"),
    }

    has_fail = any(x["level"] == "FAIL" for x in issues)
    has_warn = any(x["level"] == "WARN" for x in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "codec": codec,
        "state_checkpoint": str(state_checkpoint) if state_checkpoint else None,
        "repaired_idm": str(idm_checkpoint) if idm_checkpoint else None,
        "source_checks": source,
        "git": git,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "query_local_snapshot_allowed": not has_fail,
    }
    common.write_json_atomic(root / args.out_json, payload)

    lines = [
        "# Phase3.12d-r1 Query-Local Snapshot Preflight",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        f"- Query-local snapshot allowed: `{not has_fail}`",
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
                "| `{}` | `{}` | {} |".format(
                    item["level"], item["name"], str(item["detail"]).replace("|", "/")
                )
            )
    else:
        lines.append("| `PASS` | `none` | No issues. |")
    lines += [
        "",
        "## Decision",
        "",
        "- Any FAIL blocks environment audit and candidate execution.",
        "- The prior Phase3.12d report is invalid scientific evidence because prefix replay was incomplete.",
        "- No model training, Phase4, or CPS is permitted.",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if has_fail:
        raise SystemExit("[Phase3.12d-r1] preflight failed")


if __name__ == "__main__":
    main()
