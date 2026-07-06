#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]

def run(cmd, cwd):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        return repr(e)

def install_ravens_stub(defravens: Path):
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg

def check_import(name):
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown"), "error": None}
    except Exception as e:
        return {"ok": False, "version": None, "error": repr(e)}

def forbidden_loaded():
    bad = []
    for m in sys.modules:
        for p in FORBIDDEN_PREFIXES:
            if m == p or m.startswith(p + "."):
                bad.append(m)
    return sorted(bad)

def inspect_action_template(path):
    info = {"exists": Path(path).exists()}
    if not info["exists"]:
        return info
    with open(path, "rb") as f:
        codec = pickle.load(f)
    info["class"] = codec.__class__.__name__
    dim_attr = getattr(codec, "dim", None)
    if callable(dim_attr):
        dim_val = dim_attr()
    elif dim_attr is not None:
        dim_val = dim_attr
    else:
        dim_val = getattr(codec, "action_dim", -1)
    info["dim"] = int(dim_val)
    paths = getattr(codec, "path_strings", getattr(codec, "paths", []))
    info["num_paths"] = len(paths) if paths is not None else None
    info["num_camera_config_paths"] = sum("camera_config" in str(p) for p in paths) if paths is not None else None
    return info

def scan_scripts(root):
    findings = []
    target_paths = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_5_action_execution_diag.py",
        root / "scripts/phase3_6_matched_action_replay_diag.py",
    ]
    for p in target_paths:
        if not p.exists():
            continue
        txt = p.read_text(errors="replace")
        if re.search(r"^\s*import\s+ravens\s*$", txt, flags=re.MULTILINE):
            findings.append({"level": "FAIL", "name": "top_level_import_ravens", "file": str(p)})
        for term in ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]:
            if re.search(rf"^\s*(import|from)\s+{re.escape(term)}", txt, flags=re.MULTILINE):
                findings.append({"level": "FAIL", "name": "forbidden_import", "file": str(p), "term": term})
    return findings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    ap.add_argument("--schema_json", default="reports/phase3_6_window_schema_audit_summary.json")
    ap.add_argument("--out_json", default="reports/phase3_6_matched_replay_preflight_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_6_matched_replay_preflight_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    install_ravens_stub(defravens)

    issues = []
    checks = {}

    def issue(level, name, detail):
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
    }
    for k, v in imports.items():
        checks[f"import_{k.replace('.', '_')}"] = v["ok"]
        if not v["ok"]:
            issue("FAIL", "import_failed", f"{k}: {v['error']}")

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
        checks["task_has_hidden_breakaway_pin"] = "hidden_breakaway_pin" in conds
    except Exception as e:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_required_conditions"] = False
        checks["task_has_hidden_breakaway_pin"] = False
        issue("FAIL", "task_probe_failed", repr(e))

    data = np.load(root / args.windows, allow_pickle=True)
    meta = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0]))
        except Exception as e:
            issue("WARN", "meta_json_parse_failed", repr(e))

    checks["windows_loadable"] = True
    checks["windows_conditions_match"] = meta.get("conditions") == REQUIRED_CONDITIONS
    checks["windows_primary_match"] = meta.get("primary_hidden_condition") == PRIMARY
    checks["windows_diagnostic_match"] = meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
    checks["has_y_action"] = "y_action" in data.files
    checks["y_action_dim_14"] = "y_action" in data.files and len(data["y_action"].shape) == 2 and int(data["y_action"].shape[1]) == 14

    action_template = inspect_action_template(root / args.action_template)
    checks["action_template_exists"] = bool(action_template.get("exists"))
    checks["action_template_dim_14"] = action_template.get("dim") == 14
    checks["action_template_no_camera_config"] = action_template.get("num_camera_config_paths") == 0

    schema_path = root / args.schema_json
    schema = {}
    if schema_path.exists():
        schema = json.loads(schema_path.read_text())
        checks["schema_audit_exists"] = True
        checks["schema_audit_not_fail"] = schema.get("verdict") in {"PASS", "WARN"}
        checks["has_visible_seed_key"] = bool(schema.get("visible_seed_key"))
        checks["has_window_t_key"] = bool(schema.get("window_t_key"))
        checks["has_condition_key"] = bool(schema.get("condition_key"))
        checks["has_high_conf_source_key"] = bool(schema.get("source_key"))
        checks["can_low_conf_prefix"] = bool((schema.get("checks") or {}).get("can_use_state_action_tail_as_low_conf_prefix"))
    else:
        checks["schema_audit_exists"] = False
        checks["schema_audit_not_fail"] = False
        checks["has_visible_seed_key"] = False
        checks["has_window_t_key"] = False
        checks["has_condition_key"] = False
        checks["has_high_conf_source_key"] = False
        checks["can_low_conf_prefix"] = False
        issue("FAIL", "missing_schema_audit", str(schema_path))

    checks["has_any_prefix_source"] = checks["has_high_conf_source_key"] or checks["can_low_conf_prefix"]
    if not checks["has_any_prefix_source"]:
        issue("FAIL", "no_prefix_replay_source", "Need raw source/action file or state_action_x tail fallback.")

    p35 = root / "reports/phase3_5_action_execution_diagnostic_summary.json"
    if p35.exists():
        p35j = json.loads(p35.read_text())
        checks["phase35_exists"] = True
        checks["phase35_verdict_fail_expected"] = p35j.get("verdict") == "FAIL"
        checks["phase35_root_cause_recorded"] = bool(p35j.get("root_cause"))
    else:
        checks["phase35_exists"] = False
        checks["phase35_verdict_fail_expected"] = False
        checks["phase35_root_cause_recorded"] = False
        issue("WARN", "phase35_report_missing", str(p35))

    for f in scan_scripts(root):
        issues.append(f)
    checks["script_import_hazard_free"] = not any(i["level"] == "FAIL" for i in issues if i["name"] in {"top_level_import_ravens", "forbidden_import"})

    checks["running_in_coord_bimanual"] = "coord_bimanual" in sys.executable or os.environ.get("CONDA_DEFAULT_ENV") == "coord_bimanual"
    if not checks["running_in_coord_bimanual"]:
        issue("WARN", "not_coord_bimanual", sys.executable)

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "git": git,
        "python": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "imports": imports,
        "windows_meta": meta,
        "action_template": action_template,
        "schema_summary": {
            "condition_key": schema.get("condition_key"),
            "visible_seed_key": schema.get("visible_seed_key"),
            "window_t_key": schema.get("window_t_key"),
            "source_key": schema.get("source_key"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.6 Matched Replay Preflight",
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
        for i in issues:
            lines.append(f"| `{i['level']}` | `{i['name']}` | {str(i['detail']).replace('|','/')} |")
    else:
        lines.append("| `PASS` | `none` | No preflight issues found. |")

    lines += [
        "",
        "## Scope",
        "",
        "- This preflight does not execute matched replay.",
        "- Matched replay requires explicit gates.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.6][FAIL] matched replay preflight failed")

if __name__ == "__main__":
    main()
