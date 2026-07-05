#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import traceback
import types
from pathlib import Path

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FORBIDDEN_MODULE_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]
FORBIDDEN_POLICY_INPUT_TERMS = [
    "hidden_contact_meta",
    "recoverability_params",
    "breakaway_released",
    "breakaway_release_step",
    "breakaway_max_disp_seen",
    "condition_id",
    "condition_name",
    "hidden_condition",
    "success",
    "final_fraction",
]


def run(cmd, cwd):
    try:
        return subprocess.check_output(cmd, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
    except Exception as exc:
        return repr(exc)


def check_import(name):
    try:
        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", "unknown"), "error": None}
    except Exception as exc:
        return {"ok": False, "version": None, "error": repr(exc), "traceback": traceback.format_exc()}


def install_ravens_stub(defravens: Path):
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg


def loaded_forbidden_modules():
    out = []
    for name in sys.modules:
        for prefix in FORBIDDEN_MODULE_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                out.append(name)
    return sorted(out)


def load_windows_meta(path):
    import numpy as np

    data = np.load(path, allow_pickle=True)
    meta = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0]))
        except Exception:
            meta = {}
    shapes = {k: list(v.shape) for k, v in data.items() if hasattr(v, "shape")}
    return data, meta, shapes


def inspect_action_template(path):
    info = {"exists": Path(path).exists()}
    if not info["exists"]:
        return info
    with open(path, "rb") as f:
        obj = pickle.load(f)
    dim = getattr(obj, "dim", None)
    info["class"] = obj.__class__.__name__
    info["dim"] = int(dim()) if callable(dim) else int(getattr(obj, "action_dim", -1))
    paths = getattr(obj, "path_strings", getattr(obj, "paths", []))
    info["num_paths"] = len(paths) if paths is not None else None
    info["num_camera_config_paths"] = sum("camera_config" in str(p) for p in paths) if paths is not None else None
    return info


def checkpoint_summary(root):
    root = Path(root)
    out = {}
    for baseline in ["paper_state", "state_action"]:
        bdir = root / baseline
        dirs = sorted([p for p in bdir.glob("fold_*_seed_*") if p.is_dir()]) if bdir.exists() else []
        complete = []
        for d in dirs:
            required = [d / "config.json", d / "state_model.pt", d / "inverse_dynamics.pt"]
            if all(p.exists() for p in required):
                complete.append(str(d))
        out[baseline] = {
            "exists": bdir.exists(),
            "num_complete": len(complete),
            "first_complete": complete[0] if complete else None,
        }
    return out


def scan_rollout_script(path):
    txt = Path(path).read_text(errors="replace")
    code_no_comments = "\n".join(line for line in txt.splitlines() if not line.lstrip().startswith("#"))
    findings = []

    def add(level, name, detail):
        findings.append({"level": level, "name": name, "detail": detail})

    forbidden_patterns = [
        (r"^\s*import\s+ravens\s*$", "top_level_import_ravens"),
        (r"^\s*import\s+tensorflow\b", "tensorflow_import"),
        (r"^\s*from\s+tensorflow\b", "tensorflow_import"),
        (r"ravens\.agents", "ravens_agents_reference"),
        (r"ravens\.models", "ravens_models_reference"),
        (r"ravens\.datasets", "ravens_datasets_reference"),
    ]
    for pattern, name in forbidden_patterns:
        for match in re.finditer(pattern, code_no_comments, flags=re.MULTILINE):
            line = code_no_comments.splitlines()[code_no_comments[:match.start()].count("\n")]
            if "FORBIDDEN" in line or "name.startswith" in line:
                continue
            add("FAIL", name, f"Forbidden pattern found: {line}")

    required = [
        "primary_hidden_condition",
        "diagnostic_hidden_condition",
        "hidden_breakaway_pin",
        "PHASE3_ALLOW_ROLLOUT",
        "PHASE3_ROLLOUT_CONFIRMED",
        "primary_success_gap",
        "diagnostic_success_gap",
        "assert_no_tensorflow_loaded",
    ]
    for text in required:
        if text not in txt:
            add("FAIL", "required_rollout_string_missing", text)

    for term in FORBIDDEN_POLICY_INPUT_TERMS:
        for match in re.finditer(term, code_no_comments):
            start = max(0, match.start() - 180)
            end = min(len(code_no_comments), match.end() + 180)
            ctx = code_no_comments[start:end]
            if any(k in ctx for k in ["policy_input", "obs_vec", "state_vec", "np.concatenate", "torch.cat", "x ="]):
                add("FAIL", "possible_forbidden_metadata_in_policy_input", f"{term}: {ctx[:320].replace(chr(10), ' ')}")
    return findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--out_json", default="reports/phase3_4_rollout_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_4_rollout_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))

    checks = {}
    issues = []

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
    }
    install_ravens_stub(defravens)
    imports["ravens.tasks"] = check_import("ravens.tasks")
    imports["ravens.environment"] = check_import("ravens.environment")

    for key, val in imports.items():
        checks[f"import_{key.replace('.', '_')}"] = val["ok"]
        if not val["ok"]:
            issue("FAIL", f"import_{key}_failed", val.get("error"))

    forbidden_loaded = loaded_forbidden_modules()
    checks["forbidden_modules_not_loaded_after_imports"] = not forbidden_loaded
    if forbidden_loaded:
        issue("FAIL", "forbidden_modules_loaded", forbidden_loaded)

    try:
        tasks_mod = importlib.import_module("ravens.tasks")
        checks["task_registered"] = "hidden-contact-cable-line" in getattr(tasks_mod, "names", {})
        task = tasks_mod.names["hidden-contact-cable-line"]()
        conditions = list(getattr(task, "CONDITIONS", []))
        checks["task_instantiates"] = True
        checks["task_conditions_exact_or_superset"] = all(c in conditions for c in REQUIRED_CONDITIONS)
        checks["task_has_hidden_breakaway_pin"] = "hidden_breakaway_pin" in conditions
        if not checks["task_registered"]:
            issue("FAIL", "hidden_contact_task_not_registered", sorted(getattr(tasks_mod, "names", {}).keys())[:20])
        if not checks["task_conditions_exact_or_superset"]:
            issue("FAIL", "task_conditions_missing_required", conditions)
    except Exception as exc:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_conditions_exact_or_superset"] = False
        checks["task_has_hidden_breakaway_pin"] = False
        issue("FAIL", "task_instantiate_failed", repr(exc))

    try:
        data, meta, shapes = load_windows_meta(root / args.windows)
        checks["windows_loadable"] = True
        checks["windows_conditions_match"] = meta.get("conditions") == REQUIRED_CONDITIONS
        checks["windows_primary_match"] = meta.get("primary_hidden_condition") == PRIMARY
        checks["windows_diagnostic_match"] = meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
        checks["y_action_dim_14"] = shapes.get("y_action", [None, None])[1] == 14
        if not checks["windows_conditions_match"]:
            issue("FAIL", "windows_conditions_mismatch", meta.get("conditions"))
        if not checks["windows_primary_match"]:
            issue("FAIL", "windows_primary_mismatch", meta.get("primary_hidden_condition"))
        if not checks["windows_diagnostic_match"]:
            issue("FAIL", "windows_diagnostic_mismatch", meta.get("diagnostic_hidden_condition"))
        if not checks["y_action_dim_14"]:
            issue("FAIL", "y_action_dim_not_14", shapes.get("y_action"))
    except Exception as exc:
        meta = {}
        shapes = {}
        checks["windows_loadable"] = False
        issue("FAIL", "windows_load_failed", repr(exc))

    action_template = inspect_action_template(root / args.action_template)
    checks["action_template_exists"] = bool(action_template.get("exists"))
    checks["action_template_dim_14"] = action_template.get("dim") == 14
    checks["action_template_no_camera_config"] = action_template.get("num_camera_config_paths") == 0
    if not checks["action_template_exists"]:
        issue("FAIL", "action_template_missing", args.action_template)
    if not checks["action_template_dim_14"]:
        issue("FAIL", "action_template_dim_not_14", action_template)
    if not checks["action_template_no_camera_config"]:
        issue("FAIL", "action_template_camera_config_paths", action_template)

    ckpt = checkpoint_summary(root / args.checkpoint_root)
    checks["paper_state_checkpoints_exist"] = ckpt["paper_state"]["num_complete"] > 0
    checks["state_action_checkpoints_exist"] = ckpt["state_action"]["num_complete"] > 0
    if not checks["paper_state_checkpoints_exist"]:
        issue("FAIL", "paper_state_checkpoints_missing", ckpt["paper_state"])
    if not checks["state_action_checkpoints_exist"]:
        issue("FAIL", "state_action_checkpoints_missing", ckpt["state_action"])

    phase33b = {}
    p33 = root / "reports/phase3_3b_runtime_verdict_summary.json"
    if p33.exists():
        phase33b = json.loads(p33.read_text())
    checks["phase33b_pass"] = phase33b.get("verdict") == "PASS"
    if not checks["phase33b_pass"]:
        issue("FAIL", "phase33b_not_pass", phase33b.get("verdict"))

    rollout_findings = scan_rollout_script(root / "scripts/phase3_policy_rollout.py")
    checks["rollout_script_hazard_free"] = not any(f["level"] == "FAIL" for f in rollout_findings)
    issues.extend(rollout_findings)

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
        "forbidden_loaded": forbidden_loaded,
        "windows_meta": {
            "conditions": meta.get("conditions"),
            "primary_hidden_condition": meta.get("primary_hidden_condition"),
            "diagnostic_hidden_condition": meta.get("diagnostic_hidden_condition"),
        },
        "shapes": shapes,
        "action_template": action_template,
        "checkpoint_summary": ckpt,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.4 Rollout Smoke Preflight",
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
    for key, val in checks.items():
        lines.append(f"| `{key}` | `{val}` |")
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
        "- This preflight does not run rollout.",
        "- TensorFlow / Ravens agents / Ravens models / Ravens datasets must remain unloaded.",
        "- Rollout smoke requires explicit gates in the shell wrapper.",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.4][FAIL] rollout preflight failed")


if __name__ == "__main__":
    main()
