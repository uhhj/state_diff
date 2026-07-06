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
CONFIRM_CONDITIONS = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
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
        dim = getattr(codec, "dim", None)
        dim = int(dim() if callable(dim) else dim)
        return {"path": str(p), "dim": dim, "summary": summary}
    return {"path": "", "dim": -1, "summary": {}}


def first_checkpoint(root: Path, checkpoint_root: str, baseline: str) -> Path | None:
    base = root / checkpoint_root / baseline
    if not base.exists():
        return None
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    return None


def scan_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_6_matched_action_replay_diag.py",
        root / "scripts/phase3_8b_narrow_pose1_repair_probe.py",
        root / "scripts/phase3_9_train_geometry_idm.py",
        root / "scripts/phase3_9_controlled_retry.py",
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
    parser.add_argument("--checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase38c_summary", default="reports/phase3_8c_pose0_confirmation_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_9_geometry_idm_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_9_geometry_idm_preflight_report.md")
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
        "phase3_8b_narrow_pose1_repair_probe",
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
    checks["has_condition_name"] = "condition_name" in data.files
    checks["has_visible_seed"] = "visible_seed" in data.files
    checks["has_window_t"] = "window_t" in data.files
    checks["has_source_file"] = "source_file" in data.files
    checks["has_split_name"] = "split_name" in data.files

    for key in [
        "windows_conditions_match",
        "windows_primary_match",
        "windows_diagnostic_match",
        "has_paper_x",
        "has_y_action",
        "y_action_dim_14",
        "has_condition_name",
        "has_visible_seed",
        "has_window_t",
        "has_source_file",
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

    ckpt = first_checkpoint(root, args.checkpoint_root, "state_action")
    checks["state_action_checkpoint_complete"] = ckpt is not None
    if ckpt is None:
        issue("FAIL", "state_action_checkpoint_missing", args.checkpoint_root)

    p38c_path = root / args.phase38c_summary
    if p38c_path.exists():
        p38c = json.loads(p38c_path.read_text())
    else:
        p38c = {}
        issue("FAIL", "phase38c_summary_missing", str(p38c_path))
    checks["phase38c_summary_exists"] = bool(p38c)
    checks["phase38c_root_coupled"] = p38c.get("root_cause") == "coupled_pick_place_xy_geometry_blocker_supported"
    checks["phase38c_warn_expected"] = p38c.get("verdict") == "WARN"
    if not checks["phase38c_root_coupled"]:
        issue("FAIL", "phase38c_root_not_coupled", p38c.get("root_cause"))

    for finding in scan_scripts(root):
        issues.append(finding)

    checks["script_import_and_input_hazard_free"] = not any(x["level"] == "FAIL" for x in issues if x["name"] in {
        "top_level_import_ravens",
        "forbidden_import",
        "possible_forbidden_metadata_in_model_input",
    })

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
        "checkpoint": str(ckpt) if ckpt else None,
        "windows_meta": meta,
        "shapes": shapes,
        "phase38c_summary": {
            "verdict": p38c.get("verdict"),
            "root_cause": p38c.get("root_cause"),
            "pose0_supported_conditions": p38c.get("pose0_supported_conditions"),
            "pose1_supported_conditions": p38c.get("pose1_supported_conditions"),
            "both_xy_supported_conditions": p38c.get("both_xy_supported_conditions"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.9 Geometry-Aware IDM Preflight",
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
        "- Phase3.9 trains inverse dynamics only.",
        "- No future DDPM training.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.9][FAIL] preflight failed")


if __name__ == "__main__":
    main()
