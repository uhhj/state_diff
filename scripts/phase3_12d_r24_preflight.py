#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict

from phase3_12d_r22_integrity import strict_json_dump

EXPECTED_MAIN = "0184eb65495f7cec2f3527f077003f808e7b0e94"
EXPECTED_SUBMODULE = "e5525384af4af5bd0a02c3d1fd33ae84323d44e2"
CONDITION = "hidden_slack_breakaway_pin_v2"


def git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install_ravens_stub(submodule: Path) -> None:
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(submodule / "ravens")]
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--out-json", default="reports/phase3_12d_r24_preflight_summary.json")
    ap.add_argument("--out-md", default="reports/phase3_12d_r24_preflight_report.md")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    sub = root / "external/deformable-ravens"
    env_path = sub / "ravens/environment.py"
    task_path = sub / "ravens/tasks/ccda_hidden_contact_cable.py"
    helper_path = sub / "ravens/tasks/ccda_slack_breakaway.py"
    env_text = env_path.read_text()
    task_text = task_path.read_text()
    helper_text = helper_path.read_text()
    discovery = json.loads((root / "reports/phase3_12d_r24_repo_discovery.json").read_text())
    previous = json.loads((root / "reports/phase3_12d_r23_observation_contract_summary.json").read_text())

    status = git(sub, "status", "--short")
    changed_paths = {line.split()[-1] for line in status.splitlines() if line.split()}
    expected_paths = {
        "ravens/environment.py",
        "ravens/tasks/ccda_hidden_contact_cable.py",
        "ravens/tasks/ccda_slack_breakaway.py",
    }
    pre = env_text.index("physics_pre_step_hook")
    step = env_text.index("p.stepSimulation", pre)
    post = env_text.index("physics_step_hook", step)
    forbidden_calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(ast.parse(helper_text))
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }.intersection({"createConstraint", "createCollisionShape", "createMultiBody"})

    install_ravens_stub(sub)
    import ravens.tasks as tasks
    from ravens.tasks.ccda_slack_breakaway import UnilateralSlackBreakaway

    checks: Dict[str, bool] = {
        "main_branch": git(root, "branch", "--show-current") == "Experiment1",
        "main_head": git(root, "rev-parse", "HEAD") == EXPECTED_MAIN,
        "submodule_branch": git(sub, "branch", "--show-current") == "ccda-cable",
        "submodule_baseline": git(sub, "rev-parse", "HEAD") == EXPECTED_SUBMODULE,
        "submodule_only_expected_patch": changed_paths == expected_paths,
        "helper_import": UnilateralSlackBreakaway is not None,
        "prehook_before_step": pre < step,
        "posthook_after_step": step < post,
        "condition_registered": CONDITION in tasks.names["hidden-contact-cable-line"].CONDITIONS,
        "v2_no_body_constraint_calls": not forbidden_calls,
        "old_source_baseline_recorded": discovery["source_sha256_before"]["external/deformable-ravens/ravens/tasks/ccda_hidden_contact_cable.py"] == "5e35b093701476d3f7e6d7bd029c2004e3ec961a40b617a295afe423c47e1625",
        "state_schema_not_activated": "ccda_state_v2_position_proprio" in (root / "GOALS.md").read_text() and git(root, "diff", "--", "ccda_phase3") == "",
        "r23_root_cause": previous.get("root_cause") == "phase312d_r23_latent_condition_leaks_through_observable_motion",
        "tensorflow_not_loaded": "tensorflow" not in sys.modules,
        "scope_is_environment_only": all(os.environ.get(k) not in {"1", "true", "TRUE"} for k in ["PHASE3_ALLOW_ROLLOUT", "PHASE3_ROLLOUT_CONFIRMED"]),
        "goals_has_gates": "Phase3.12d-r2.4" in (root / "GOALS.md").read_text() and "Phase 3.13" in (root / "GOALS.md").read_text(),
        "submodule_import_smoke": hasattr(tasks, "names"),
        "hook_error_field": "_ccda_physics_hook_error" in env_text and "physics_pre_step_hook: " in env_text,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    payload: Dict[str, Any] = {
        "verdict": verdict,
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "main_head": git(root, "rev-parse", "HEAD"),
        "submodule_head": git(sub, "rev-parse", "HEAD"),
        "submodule_status": status,
        "source_sha256_after": {str(p.relative_to(root)): sha(p) for p in [env_path, task_path, helper_path]},
        "parameter_sweep_allowed": verdict == "PASS",
        "scope": {"training": False, "candidate_matrix": False, "phase4": False, "cps": False},
    }
    strict_json_dump(root / args.out_json, payload)
    lines = ["# Phase3.12d-r2.4 Preflight", "", f"- Verdict: `{verdict}`", "", "| Check | Result |", "|---|---:|"]
    lines += [f"| `{k}` | `{v}` |" for k, v in checks.items()]
    lines += ["", "No training, candidate matrix, Phase4, or CPS is authorized."]
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if verdict != "PASS":
        raise SystemExit("[r2.4] preflight failed")


if __name__ == "__main__":
    main()
