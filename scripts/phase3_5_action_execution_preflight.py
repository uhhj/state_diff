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

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"

FORBIDDEN_PREFIXES = [
    "tensorflow",
    "ravens.agents",
    "ravens.models",
    "ravens.datasets",
]

FORBIDDEN_INPUT_TERMS = [
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
    out = []
    for m in sys.modules:
        for p in FORBIDDEN_PREFIXES:
            if m == p or m.startswith(p + "."):
                out.append(m)
    return sorted(out)

def load_npz_meta(path):
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

def scan_policy_scripts(root):
    findings = []
    paths = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_5_action_execution_diag.py",
    ]
    for p in paths:
        if not p.exists():
            continue
        txt = p.read_text(errors="replace")
        code_lines = [line for line in txt.splitlines() if not line.lstrip().startswith("#")]
        code_no_comments = "\n".join(code_lines)
        if re.search(r"^\s*import\s+ravens\s*$", code_no_comments, flags=re.MULTILINE):
            findings.append({"level": "FAIL", "name": "top_level_import_ravens", "file": str(p), "detail": "import ravens"})
        for term in ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]:
            if re.search(rf"^\s*(import|from)\s+{re.escape(term)}", code_no_comments, flags=re.MULTILINE):
                findings.append({"level": "FAIL", "name": "forbidden_import", "file": str(p), "term": term, "detail": term})
        input_markers = ["policy_input", "model_x", "obs_vec", "state_vec", "np.concatenate", "torch.cat"]
        for line in code_lines:
            stripped = line.strip()
            if not any(marker in stripped for marker in input_markers):
                continue
            for term in FORBIDDEN_INPUT_TERMS:
                if term in stripped:
                    findings.append({
                        "level": "FAIL",
                        "name": "possible_forbidden_metadata_in_policy_input",
                        "file": str(p),
                        "term": term,
                        "detail": stripped[:300],
                    })
    return findings

def checkpoint_summary(root):
    root = Path(root)
    out = {}
    for baseline in ["paper_state", "state_action"]:
        bdir = root / baseline
        dirs = sorted([p for p in bdir.glob("fold_*_seed_*") if p.is_dir()]) if bdir.exists() else []
        complete = []
        for d in dirs:
            if (d / "state_model.pt").exists() and (d / "inverse_dynamics.pt").exists() and (d / "config.json").exists():
                complete.append(str(d))
        out[baseline] = {
            "exists": bdir.exists(),
            "num_complete": len(complete),
            "first_complete": complete[0] if complete else None,
        }
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    ap.add_argument("--checkpoint_root", default="checkpoints/phase3")
    ap.add_argument("--out_json", default="reports/phase3_5_action_execution_preflight_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_5_action_execution_preflight_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    install_ravens_stub(defravens)

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
        "ravens.tasks": check_import("ravens.tasks"),
        "ravens.environment": check_import("ravens.environment"),
    }
    for k, r in imports.items():
        checks["import_" + k.replace(".", "_")] = r["ok"]
        if not r["ok"]:
            issue("FAIL", "import_failed", f"{k}: {r['error']}")

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

    try:
        data, meta, shapes = load_npz_meta(root / args.windows)
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
    except Exception as e:
        meta, shapes = {}, {}
        checks["windows_loadable"] = False
        issue("FAIL", "windows_load_failed", repr(e))

    action_template = inspect_action_template(root / args.action_template)
    checks["action_template_exists"] = bool(action_template.get("exists"))
    checks["action_template_dim_14"] = action_template.get("dim") == 14
    checks["action_template_no_camera_config"] = action_template.get("num_camera_config_paths") == 0
    if not checks["action_template_exists"]:
        issue("FAIL", "action_template_missing", args.action_template)
    if not checks["action_template_dim_14"]:
        issue("FAIL", "action_template_dim_not_14", action_template)
    if not checks["action_template_no_camera_config"]:
        issue("FAIL", "action_template_has_camera_config", action_template)

    ckpt = checkpoint_summary(root / args.checkpoint_root)
    checks["paper_state_checkpoints_complete"] = ckpt["paper_state"]["num_complete"] > 0
    checks["state_action_checkpoints_complete"] = ckpt["state_action"]["num_complete"] > 0
    if not checks["paper_state_checkpoints_complete"]:
        issue("FAIL", "paper_state_checkpoints_missing", ckpt["paper_state"])
    if not checks["state_action_checkpoints_complete"]:
        issue("FAIL", "state_action_checkpoints_missing", ckpt["state_action"])

    p34 = root / "reports/phase3_4_rollout_result_audit_summary.json"
    if p34.exists():
        p34j = json.loads(p34.read_text())
        checks["phase34_exists"] = True
        checks["phase34_warn_or_pass"] = p34j.get("verdict") in {"WARN", "PASS"}
        checks["phase34_rows_positive"] = int(p34j.get("num_rows", 0)) > 0
    else:
        p34j = {}
        checks["phase34_exists"] = False
        checks["phase34_warn_or_pass"] = False
        checks["phase34_rows_positive"] = False
        issue("WARN", "phase34_report_missing", str(p34))

    for f in scan_policy_scripts(root):
        issues.append(f)
    checks["policy_script_no_hazard"] = not any(i["level"] == "FAIL" for i in issues if i["name"] in {"top_level_import_ravens", "forbidden_import", "possible_forbidden_metadata_in_policy_input"})

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
        "windows_meta": {
            "conditions": meta.get("conditions"),
            "primary_hidden_condition": meta.get("primary_hidden_condition"),
            "diagnostic_hidden_condition": meta.get("diagnostic_hidden_condition"),
        },
        "shapes": shapes,
        "action_template": action_template,
        "checkpoint_summary": ckpt,
        "phase34_summary": {
            "verdict": p34j.get("verdict"),
            "num_rows": p34j.get("num_rows"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.5 Action Execution Preflight",
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
            lines.append(f"| `{i.get('level')}` | `{i.get('name')}` | {str(i.get('detail', '')).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No preflight issues found. |")

    lines += [
        "",
        "## Scope",
        "",
        "- This preflight does not execute actions.",
        "- Phase3.5 action diagnostic requires explicit gates.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.5][FAIL] preflight failed")

if __name__ == "__main__":
    main()
