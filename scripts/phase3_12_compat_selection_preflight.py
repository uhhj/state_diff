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
    bad = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    return sorted(set(bad))


def load_npz_meta(path: Path) -> Tuple[Any, Dict[str, Any], Dict[str, List[int]]]:
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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def first_old_checkpoint(root: Path, checkpoint_root: str, baseline: str = "state_action") -> Optional[Path]:
    base = root / checkpoint_root / baseline
    if not base.exists():
        return None
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    return None


def checkpoint_from_phase39b_raw(root: Path, raw_path: str, ablation: str) -> Optional[Path]:
    raw = load_json(root / raw_path)
    p = raw.get("results", {}).get(ablation, {}).get("checkpoint")
    if not p:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = root / pp
    return pp if pp.exists() else None


def scan_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    targets = [
        root / "scripts/phase3_policy_rollout.py",
        root / "scripts/phase3_12_compat_selection_rollout.py",
        root / "scripts/phase3_12_analyze_compat_selection.py",
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
                ctx = txt[max(0, match.start() - 280): min(len(txt), match.end() + 280)]
                if any(k in ctx for k in ["model_x", "idm_x", "policy_input", "np.concatenate", "torch.cat", "input_vec"]):
                    findings.append({
                        "level": "FAIL",
                        "name": "possible_forbidden_metadata_in_model_input",
                        "file": str(path),
                        "term": term,
                        "context": ctx.replace("\n", " ")[:500],
                    })
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--phase39b_raw", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--phase311b_summary", default="reports/phase3_11b_retrieval_feasibility_summary.json")
    parser.add_argument("--best_ablation", default="xy_only_high_weight")
    parser.add_argument("--out_json", default="reports/phase3_12_compat_selection_preflight_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_12_compat_selection_preflight_report.md")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    for p in [root, root / "scripts", root / "external/deformable-ravens"]:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        # Reuse the TensorFlow-free minimal DeformableRavens runtime used by rollout.
        # This lets preflight verify ravens.tasks / ravens.environment without
        # importing ravens.agents, ravens.models, ravens.datasets, or TensorFlow.
        import phase3_policy_rollout as _phase3_policy_rollout

        _phase3_policy_rollout.patch_pybullet_pkg_resources_metadata()
        _phase3_policy_rollout.import_ravens_runtime(root)
    except Exception:
        pass

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
        "ccda_phase3.rollout",
        "ccda_phase3.train_utils",
        "phase3_policy_rollout",
    ]:
        result = check_import(name)
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
        checks["task_has_primary"] = PRIMARY in conds
    except Exception as exc:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_required_conditions"] = False
        checks["task_has_primary"] = False
        issue("FAIL", "task_probe_failed", repr(exc))

    data, meta, shapes = load_npz_meta(root / args.windows)
    checks["windows_loadable"] = True
    checks["windows_conditions_match"] = meta.get("conditions") == REQUIRED_CONDITIONS
    checks["windows_primary_match"] = meta.get("primary_hidden_condition") == PRIMARY
    checks["windows_diagnostic_match"] = meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
    for key in ["state_action_x", "y_state", "y_action", "condition_name", "split_name", "th", "action_dim", "n_beads"]:
        checks["has_" + key] = key in data.files
        if key not in data.files:
            issue("FAIL", "missing_window_key", key)

    checks["y_action_dim_14"] = "y_action" in data.files and len(data["y_action"].shape) == 2 and int(data["y_action"].shape[1]) == 14
    if not checks["y_action_dim_14"]:
        issue("FAIL", "y_action_dim_not_14", list(data["y_action"].shape) if "y_action" in data.files else "missing")

    for key in ["windows_conditions_match", "windows_primary_match", "windows_diagnostic_match"]:
        if not checks[key]:
            issue("FAIL", key, key)

    codec = load_codec_summary(root, data, args.action_template)
    checks["action_codec_dim_14"] = codec["dim"] == 14
    checks["action_codec_no_camera_config"] = int(codec["summary"].get("num_camera_config_paths", 0) or 0) == 0
    if not checks["action_codec_dim_14"]:
        issue("FAIL", "action_codec_dim_not_14", codec)
    if not checks["action_codec_no_camera_config"]:
        issue("FAIL", "action_codec_has_camera_config", codec)

    old_ckpt = first_old_checkpoint(root, args.old_checkpoint_root, "state_action")
    best_idm = checkpoint_from_phase39b_raw(root, args.phase39b_raw, args.best_ablation)
    checks["old_state_action_checkpoint_complete"] = old_ckpt is not None
    checks["phase39b_best_idm_exists"] = best_idm is not None
    if old_ckpt is None:
        issue("FAIL", "old_state_action_checkpoint_missing", args.old_checkpoint_root)
    if best_idm is None:
        issue("FAIL", "phase39b_best_idm_missing", f"{args.phase39b_raw}:{args.best_ablation}")

    p311b = load_json(root / args.phase311b_summary)
    checks["phase311b_summary_exists"] = bool(p311b)
    checks["phase311b_live_source_context_mismatch"] = p311b.get("root_cause") == "phase311b_live_source_context_mismatch_supported"
    checks["phase311b_no_phase4_scope"] = (root / "reports/phase3_11b_no_phase4_confirmation.md").exists()

    for key in ["phase311b_summary_exists", "phase311b_live_source_context_mismatch", "phase311b_no_phase4_scope"]:
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
        "checkpoints": {
            "old_state_action": str(old_ckpt) if old_ckpt else None,
            "phase39b_best": str(best_idm) if best_idm else None,
            "best_ablation": args.best_ablation,
        },
        "phase311b_summary": {
            "verdict": p311b.get("verdict"),
            "root_cause": p311b.get("root_cause"),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.12 Compatibility-Aware Future Selection Preflight",
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
        "- This preflight does not run rollout.",
        "- Phase3.12 runs compatibility-aware future target selection diagnostics only.",
        "- No model training.",
        "- No future DDPM training.",
        "- No Phase4 or CPS is allowed.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.12][FAIL] preflight failed")


if __name__ == "__main__":
    main()
