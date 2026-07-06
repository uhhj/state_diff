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


def forbidden_loaded() -> List[str]:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    return sorted(set(bad))



def install_ravens_stub(defravens: Path) -> None:
    """Expose ravens.tasks/environment without executing ravens/__init__.py.

    The coord_bimanual env intentionally does not install TensorFlow. The
    DeformableRavens package __init__ imports TensorFlow-heavy modules, but the
    Phase3.9b preflight only needs tasks/environment registration checks.
    """
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg

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


def load_codec_summary(root: Path, data: Any, fallback: str) -> Dict[str, Any]:
    candidates: List[Path] = []
    if "action_template_json_or_pickle_path" in data.files:
        raw = data["action_template_json_or_pickle_path"]
        val = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
        candidates.append(Path(val))
    candidates.append(Path(fallback))
    for cand in candidates:
        p = cand if cand.is_absolute() else root / cand
        if not p.exists():
            continue
        try:
            from ccda_phase3.data_io import load_action_codec_from_template
            codec = load_action_codec_from_template(p)
        except Exception:
            with p.open("rb") as f:
                codec = pickle.load(f)
        summary = codec.summary() if hasattr(codec, "summary") else {}
        dim_attr = getattr(codec, "dim", None)
        dim = int(dim_attr() if callable(dim_attr) else dim_attr)
        return {"path": str(p), "dim": dim, "summary": summary}
    return {"path": "", "dim": -1, "summary": {}}


def scan_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_9_train_geometry_idm.py",
        root / "scripts/phase3_9_controlled_retry.py",
        root / "scripts/phase3_9_analyze_controlled_retry.py",
        root / "scripts/phase3_9b_run_ablation_matrix.py",
        root / "scripts/phase3_9b_analyze_ablation_matrix.py",
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
            for match in re.finditer(term, txt):
                ctx = txt[max(0, match.start() - 180): min(len(txt), match.end() + 180)]
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
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--phase39_summary", default="reports/phase3_9_geometry_idm_controlled_retry_summary.json")
    parser.add_argument("--phase39_train_summary", default="reports/phase3_9_geometry_idm_train_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_9b_geometry_ablation_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_9b_geometry_ablation_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    install_ravens_stub(root / "external/deformable-ravens")

    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    git = {
        "branch": run(["git", "branch", "--show-current"], root).strip(),
        "head": run(["git", "rev-parse", "HEAD"], root).strip(),
        "origin_experiment1": run(["git", "ls-remote", "origin", "Experiment1"], root).strip().split("\t")[0],
        "status": run(["git", "status", "--short"], root),
        "submodule": run(["git", "submodule", "status", "external/deformable-ravens"], root).strip(),
    }

    for name in [
        "numpy",
        "torch",
        "pybullet",
        "ravens.tasks",
        "ravens.environment",
        "ccda_phase3.models",
        "ccda_phase3.train_utils",
        "phase3_9_train_geometry_idm",
        "phase3_9_controlled_retry",
        "phase3_9_analyze_controlled_retry",
    ]:
        result = check_import(name)
        checks["import_" + name.replace(".", "_")] = result["ok"]
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
    checks["has_split_name"] = "split_name" in data.files

    for key in [
        "windows_conditions_match",
        "windows_primary_match",
        "windows_diagnostic_match",
        "has_paper_x",
        "has_state_action_x",
        "has_y_action",
        "y_action_dim_14",
        "has_split_name",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

    codec = load_codec_summary(root, data, args.action_template)
    checks["action_codec_dim_14"] = codec["dim"] == 14
    checks["action_codec_no_camera_config"] = int(codec["summary"].get("num_camera_config_paths", 0) or 0) == 0
    if not checks["action_codec_dim_14"]:
        issue("FAIL", "action_codec_dim_not_14", codec)
    if not checks["action_codec_no_camera_config"]:
        issue("FAIL", "action_codec_has_camera_config", codec)

    p39_path = root / args.phase39_summary
    p39_train_path = root / args.phase39_train_summary
    p39 = json.loads(p39_path.read_text()) if p39_path.exists() else {}
    p39_train = json.loads(p39_train_path.read_text()) if p39_train_path.exists() else {}

    checks["phase39_summary_exists"] = bool(p39)
    checks["phase39_repair_supported"] = p39.get("root_cause") == "phase39_geometry_idm_repair_supported"
    checks["phase39_primary_improved"] = bool(p39.get("primary_improved", False))
    checks["phase39_hidden_breakaway_improved"] = "hidden_breakaway_pin" in p39.get("improved_conditions", [])
    checks["phase39_train_summary_exists"] = bool(p39_train)
    checks["phase39_train_inverse_only"] = p39_train.get("scope") == "phase3_9_geometry_aware_idm_train_no_phase4_no_cps"

    for key in [
        "phase39_summary_exists",
        "phase39_repair_supported",
        "phase39_primary_improved",
        "phase39_hidden_breakaway_improved",
        "phase39_train_summary_exists",
        "phase39_train_inverse_only",
    ]:
        if not checks[key]:
            issue("FAIL", key, key)

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
        "codec": codec,
        "windows_meta": meta,
        "shapes": shapes,
        "phase39_summary": {
            "verdict": p39.get("verdict"),
            "root_cause": p39.get("root_cause"),
            "improved_conditions": p39.get("improved_conditions"),
            "primary_improved": p39.get("primary_improved"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.9b Geometry IDM Ablation Preflight",
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
        "- This preflight does not train.",
        "- Phase3.9b trains inverse dynamics ablations only.",
        "- No future DDPM training.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.9b][FAIL] preflight failed")


if __name__ == "__main__":
    main()