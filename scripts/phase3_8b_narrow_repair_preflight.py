#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]


def run(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return repr(exc)


def check_import(name: str) -> Dict[str, Any]:
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown"), "error": None}
    except Exception as exc:
        return {"ok": False, "version": None, "error": repr(exc)}

def install_ravens_stub(defravens: Path) -> None:
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg

def forbidden_loaded() -> List[str]:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    return sorted(set(bad))


def load_npz_meta(path: Path) -> tuple[Any, Dict[str, Any], Dict[str, List[int]]]:
    data = np.load(path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
            meta = json.loads(text)
        except Exception:
            meta = {}
    shapes = {k: list(data[k].shape) for k in data.files if hasattr(data[k], "shape")}
    return data, meta, shapes


def scan_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_6_matched_action_replay_diag.py",
        root / "scripts/phase3_8_idm_geometry_repair_probe.py",
        root / "scripts/phase3_8b_narrow_pose1_repair_probe.py",
    ]
    forbidden_input_terms = [
        "hidden_condition",
        "hidden_contact_meta",
        "recoverability_params",
        "breakaway_released",
        "breakaway_release_step",
        "breakaway_max_disp_seen",
        "condition_id",
        "condition_name",
        "success",
        "final_fraction",
    ]
    for path in targets:
        if not path.exists():
            continue
        txt = path.read_text(errors="replace")
        if re.search(r"^\s*import\s+ravens\s*$", txt, flags=re.MULTILINE):
            findings.append({"level": "FAIL", "name": "top_level_import_ravens", "file": str(path)})
        for term in ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]:
            if re.search(rf"^\s*(import|from)\s+{re.escape(term)}", txt, flags=re.MULTILINE):
                findings.append({"level": "FAIL", "name": "forbidden_import", "file": str(path), "term": term})
        for term in forbidden_input_terms:
            for m in re.finditer(term, txt):
                ctx = txt[max(0, m.start() - 180): min(len(txt), m.end() + 180)]
                if any(k in ctx for k in ["model_x", "idm_x", "policy_input", "np.concatenate", "torch.cat", "x ="]):
                    findings.append({
                        "level": "FAIL",
                        "name": "possible_forbidden_metadata_in_model_input",
                        "file": str(path),
                        "term": term,
                        "context": ctx.replace("\n", " ")[:300],
                    })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--phase38_sensitivity", default="reports/phase3_8_action_geometry_sensitivity_summary.json")
    parser.add_argument("--phase38_repair_report", default="reports/phase3_8_idm_geometry_repair_report.md")
    parser.add_argument("--phase37_summary", default="reports/phase3_7_learned_action_alignment_summary.json")
    parser.add_argument("--phase36_summary", default="reports/phase3_6_matched_replay_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_8b_narrow_repair_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_8b_narrow_repair_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    install_ravens_stub(defravens)

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    git = {
        "branch": run(["git", "branch", "--show-current"], root).strip(),
        "head": run(["git", "rev-parse", "HEAD"], root).strip(),
        "origin_experiment1": run(["git", "ls-remote", "origin", "Experiment1"], root).strip().split("\t")[0],
        "status": run(["git", "status", "--short"], root),
        "submodule": run(["git", "submodule", "status", "external/deformable-ravens"], root).strip(),
    }

    imports = {
        "numpy": check_import("numpy"),
        "torch": check_import("torch"),
        "pybullet": check_import("pybullet"),
        "ravens.tasks": check_import("ravens.tasks"),
        "ravens.environment": check_import("ravens.environment"),
        "phase3_6_matched_action_replay_diag": check_import("phase3_6_matched_action_replay_diag"),
        "phase3_8_idm_geometry_repair_probe": check_import("phase3_8_idm_geometry_repair_probe"),
    }
    for name, result in imports.items():
        checks["import_" + name.replace(".", "_")] = bool(result["ok"])
        if not result["ok"]:
            issue("FAIL", "import_failed", f"{name}: {result['error']}")

    bad = forbidden_loaded()
    checks["forbidden_modules_not_loaded"] = not bad
    if bad:
        issue("FAIL", "forbidden_modules_loaded", bad)

    try:
        tasks = importlib.import_module("ravens.tasks")
        checks["task_registered"] = "hidden-contact-cable-line" in getattr(tasks, "names", {})
        task = tasks.names["hidden-contact-cable-line"]()
        conds = list(getattr(task, "CONDITIONS", []))
        checks["task_instantiates"] = True
        checks["task_has_required_conditions"] = all(c in conds for c in REQUIRED_CONDITIONS)
        checks["task_has_hidden_breakaway_pin"] = PRIMARY in conds
    except Exception as exc:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_required_conditions"] = False
        checks["task_has_hidden_breakaway_pin"] = False
        issue("FAIL", "task_probe_failed", repr(exc))

    data, meta, shapes = load_npz_meta(root / args.windows)
    checks["windows_loadable"] = True
    checks["windows_conditions_match"] = meta.get("conditions") == REQUIRED_CONDITIONS
    checks["windows_primary_match"] = meta.get("primary_hidden_condition") == PRIMARY
    checks["windows_diagnostic_match"] = meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
    checks["has_paper_x"] = "paper_x" in data.files
    checks["has_state_action_x"] = "state_action_x" in data.files
    checks["has_y_action"] = "y_action" in data.files
    checks["y_action_dim_14"] = "y_action" in data.files and len(data["y_action"].shape) == 2 and int(data["y_action"].shape[1]) == 14
    checks["has_condition_name"] = "condition_name" in data.files
    checks["has_visible_seed"] = "visible_seed" in data.files
    checks["has_window_t"] = "window_t" in data.files
    checks["has_source_file"] = "source_file" in data.files

    for key in [
        "windows_conditions_match",
        "windows_primary_match",
        "windows_diagnostic_match",
        "has_paper_x",
        "has_state_action_x",
        "has_y_action",
        "y_action_dim_14",
        "has_condition_name",
        "has_visible_seed",
        "has_window_t",
        "has_source_file",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

    p36_path = root / args.phase36_summary
    p37_path = root / args.phase37_summary
    p38_sens_path = root / args.phase38_sensitivity
    p38_report_path = root / args.phase38_repair_report

    p36 = json.loads(p36_path.read_text()) if p36_path.exists() else {}
    p37 = json.loads(p37_path.read_text()) if p37_path.exists() else {}
    p38_sens = json.loads(p38_sens_path.read_text()) if p38_sens_path.exists() else {}

    checks["phase36_summary_exists"] = bool(p36)
    checks["phase36_raw_prefix_available"] = "raw_action_file_high_confidence" in p36.get("prefix_sources", [])
    checks["phase36_state_match"] = bool(p36.get("state_match", False))
    checks["phase37_summary_exists"] = bool(p37)
    checks["phase37_root_geometry_blocker"] = p37.get("root_cause") == "idm_action_execution_geometry_blocker"
    checks["phase37_verdict_fail_expected"] = p37.get("verdict") == "FAIL"
    checks["phase38_sensitivity_exists"] = bool(p38_sens)
    checks["phase38_sensitivity_pass"] = p38_sens.get("verdict") == "PASS"
    checks["phase38_repair_fail_report_exists"] = p38_report_path.exists()

    for key in [
        "phase36_summary_exists",
        "phase36_raw_prefix_available",
        "phase36_state_match",
        "phase37_summary_exists",
        "phase37_root_geometry_blocker",
        "phase37_verdict_fail_expected",
        "phase38_sensitivity_exists",
        "phase38_sensitivity_pass",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

    top_dims = p38_sens.get("top_dims", [])
    checks["phase38_top_dim_mentions_pose1_y"] = any(
        str(x.get("path", "")).endswith("params/pose1/0/1") or str(x.get("path", "")) == "params/pose1/0/1"
        for x in top_dims[:8]
    )
    if not checks["phase38_top_dim_mentions_pose1_y"]:
        issue("WARN", "pose1_y_not_in_top_dims", top_dims[:8])

    for finding in scan_scripts(root):
        issues.append(finding)

    checks["script_import_and_input_hazard_free"] = not any(
        x["level"] == "FAIL"
        for x in issues
        if x["name"] in {"top_level_import_ravens", "forbidden_import", "possible_forbidden_metadata_in_model_input"}
    )

    checks["running_in_coord_bimanual"] = "coord_bimanual" in sys.executable or os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual"
    if not checks["running_in_coord_bimanual"]:
        issue("WARN", "not_coord_bimanual", sys.executable)

    has_fail = any(x["level"] == "FAIL" for x in issues)
    has_warn = any(x["level"] == "WARN" for x in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "git": git,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "windows_meta": meta,
        "shapes": shapes,
        "phase37_summary": {
            "verdict": p37.get("verdict"),
            "root_cause": p37.get("root_cause"),
            "gt_exec_mean_delta_final_fraction": p37.get("gt_exec_mean_delta_final_fraction"),
            "idm_gt_exec_mean_delta_final_fraction": p37.get("idm_gt_exec_mean_delta_final_fraction"),
        },
        "phase38_sensitivity_top_dims": top_dims[:8],
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.8b Narrow Repair Preflight",
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
    for k, v in checks.items():
        lines.append(f"| `{k}` | `{v}` |")

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for item in issues:
            lines.append(f"| `{item['level']}` | `{item['name']}` | {str(item['detail']).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No preflight issues found. |")

    lines += [
        "",
        "## Scope",
        "",
        "- This preflight does not run repair execution.",
        "- Phase3.8b requires explicit gates.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.8b][FAIL] preflight failed")


if __name__ == "__main__":
    main()
