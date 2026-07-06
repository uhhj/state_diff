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


def codec_dim(codec: Any) -> int:
    dim = getattr(codec, "dim", None)
    if callable(dim):
        return int(dim())
    if dim is not None:
        return int(dim)
    return int(getattr(codec, "action_dim", -1))


def codec_summary(codec: Any) -> Dict[str, Any]:
    summary = getattr(codec, "summary", None)
    if callable(summary):
        try:
            out = summary()
            return out if isinstance(out, dict) else {"summary": str(out)}
        except Exception as exc:
            return {"summary_error": repr(exc)}
    paths = getattr(codec, "path_strings", getattr(codec, "paths", []))
    return {
        "num_paths": len(paths) if paths is not None else None,
        "num_camera_config_paths": sum("camera_config" in str(p) for p in paths) if paths is not None else None,
    }


def inspect_action_template(root: Path, npz_data: Any, fallback: str) -> Dict[str, Any]:
    candidates: List[Path] = []
    if "action_template_json_or_pickle_path" in npz_data.files:
        raw = npz_data["action_template_json_or_pickle_path"]
        val = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
        candidates.append(Path(val))
    candidates.append(Path(fallback))
    for cand in candidates:
        p = cand if cand.is_absolute() else root / cand
        if p.exists():
            try:
                from ccda_phase3.data_io import load_action_codec_from_template

                codec = load_action_codec_from_template(p)
            except Exception:
                with p.open("rb") as f:
                    codec = pickle.load(f)
            summary = codec_summary(codec)
            return {
                "exists": True,
                "path": str(p),
                "class": codec.__class__.__name__,
                "dim": codec_dim(codec),
                "summary": summary,
                "num_camera_config_paths": int(summary.get("num_camera_config_paths", 0) or 0),
            }
    return {"exists": False, "path": str(candidates)}


def checkpoint_summary(root: Path, checkpoint_root: str) -> Dict[str, Any]:
    base = Path(checkpoint_root)
    if not base.is_absolute():
        base = root / base
    out: Dict[str, Any] = {}
    for baseline in ["paper_state", "state_action"]:
        bdir = base / baseline
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


def scan_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_6_matched_action_replay_diag.py",
        root / "scripts/phase3_7_learned_action_alignment_diag.py",
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
    parser.add_argument("--phase36_summary", default="reports/phase3_6_matched_replay_summary.json")
    parser.add_argument("--out_json", default="reports/phase3_7_learned_action_alignment_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_7_learned_action_alignment_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    defravens = root / "external/deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    install_ravens_stub(defravens)

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

    imports = {
        "numpy": check_import("numpy"),
        "torch": check_import("torch"),
        "pybullet": check_import("pybullet"),
        "ravens.tasks": check_import("ravens.tasks"),
        "ravens.environment": check_import("ravens.environment"),
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
        checks["task_has_hidden_breakaway_pin"] = "hidden_breakaway_pin" in conds
    except Exception as exc:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_required_conditions"] = False
        checks["task_has_hidden_breakaway_pin"] = False
        issue("FAIL", "task_probe_failed", repr(exc))

    windows_path = root / args.windows
    try:
        data, meta, shapes = load_npz_meta(windows_path)
        checks["windows_loadable"] = True
        checks["windows_conditions_match"] = meta.get("conditions") == REQUIRED_CONDITIONS
        checks["windows_primary_match"] = meta.get("primary_hidden_condition") == PRIMARY
        checks["windows_diagnostic_match"] = meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
        checks["has_paper_x"] = "paper_x" in data.files
        checks["has_state_action_x"] = "state_action_x" in data.files
        checks["has_y_action"] = "y_action" in data.files
        checks["has_y_state_or_y_final"] = "y_state" in data.files or "y_final_state" in data.files
        checks["has_condition_name"] = "condition_name" in data.files
        checks["has_visible_seed"] = "visible_seed" in data.files
        checks["has_window_t"] = "window_t" in data.files
        checks["has_source_file"] = "source_file" in data.files
        checks["y_action_dim_14"] = "y_action" in data.files and len(data["y_action"].shape) == 2 and int(data["y_action"].shape[1]) == 14
        if not checks["windows_conditions_match"]:
            issue("FAIL", "windows_conditions_mismatch", meta.get("conditions"))
        if not checks["windows_primary_match"]:
            issue("FAIL", "windows_primary_mismatch", meta.get("primary_hidden_condition"))
        if not checks["windows_diagnostic_match"]:
            issue("FAIL", "windows_diagnostic_mismatch", meta.get("diagnostic_hidden_condition"))
        for key in ["has_paper_x", "has_state_action_x", "has_y_action", "has_y_state_or_y_final", "has_condition_name", "has_visible_seed", "has_window_t", "has_source_file", "y_action_dim_14"]:
            if not checks[key]:
                issue("FAIL", key + "_failed", key)
    except Exception as exc:
        data = None
        meta = {}
        shapes = {}
        checks["windows_loadable"] = False
        issue("FAIL", "windows_load_failed", repr(exc))

    if data is not None:
        action_template = inspect_action_template(root, data, args.action_template)
    else:
        action_template = {"exists": False}
    checks["action_template_exists"] = bool(action_template.get("exists"))
    checks["action_template_dim_14"] = action_template.get("dim") == 14
    checks["action_template_no_camera_config"] = int(action_template.get("num_camera_config_paths", -1)) == 0
    if not checks["action_template_exists"]:
        issue("FAIL", "action_template_missing", args.action_template)
    if not checks["action_template_dim_14"]:
        issue("FAIL", "action_template_dim_not_14", action_template)
    if not checks["action_template_no_camera_config"]:
        issue("FAIL", "action_template_has_camera_config", action_template)

    ckpt = checkpoint_summary(root, args.checkpoint_root)
    checks["paper_state_checkpoints_complete"] = ckpt["paper_state"]["num_complete"] > 0
    checks["state_action_checkpoints_complete"] = ckpt["state_action"]["num_complete"] > 0
    if not checks["paper_state_checkpoints_complete"]:
        issue("FAIL", "paper_state_checkpoint_missing", ckpt["paper_state"])
    if not checks["state_action_checkpoints_complete"]:
        issue("FAIL", "state_action_checkpoint_missing", ckpt["state_action"])

    p36_path = root / args.phase36_summary
    if p36_path.exists():
        p36 = json.loads(p36_path.read_text())
        checks["phase36_summary_exists"] = True
        checks["phase36_warn_or_pass"] = p36.get("verdict") in {"WARN", "PASS"}
        checks["phase36_raw_prefix_available"] = bool(p36.get("high_conf_raw_prefix_available", False)) or "raw_action_file_high_confidence" in p36.get("prefix_sources", [])
        checks["phase36_matched_gt_progress"] = float(p36.get("gt_mean_delta_final_fraction", 0.0) or 0.0) > 0.0
    else:
        p36 = {}
        checks["phase36_summary_exists"] = False
        checks["phase36_warn_or_pass"] = False
        checks["phase36_raw_prefix_available"] = False
        checks["phase36_matched_gt_progress"] = False
        issue("FAIL", "phase36_summary_missing", str(p36_path))

    for finding in scan_scripts(root):
        issues.append(finding)
    checks["script_import_and_input_hazard_free"] = not any(i["level"] == "FAIL" for i in issues if i["name"] in {"top_level_import_ravens", "forbidden_import", "possible_forbidden_metadata_in_model_input"})

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
        "shapes": shapes,
        "action_template": action_template,
        "checkpoint_summary": ckpt,
        "phase36_summary": {
            "verdict": p36.get("verdict"),
            "root_cause": p36.get("root_cause"),
            "gt_mean_delta_final_fraction": p36.get("gt_mean_delta_final_fraction"),
            "oracle_mean_delta_final_fraction": p36.get("oracle_mean_delta_final_fraction"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.7 Learned Action Alignment Preflight",
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
            lines.append(f"| `{item['level']}` | `{item['name']}` | {str(item['detail']).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No preflight issues found. |")

    lines += [
        "",
        "## Scope",
        "",
        "- This preflight does not run action alignment.",
        "- Phase3.7 requires explicit gates.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.7][FAIL] preflight failed")


if __name__ == "__main__":
    main()
