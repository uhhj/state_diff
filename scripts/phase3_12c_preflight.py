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

import phase3_12c_matched_reset_common as common


REQUIRED_CONDITIONS = [
    "free",
    "hidden_pin",
    "hidden_high_friction",
    "hidden_breakaway_pin",
]

PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"


def run(cmd: List[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(cwd),
            stderr=subprocess.STDOUT,
            text=True,
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
        return {
            "ok": False,
            "version": None,
            "error": repr(exc),
        }


def load_npz_meta(
    path: Path,
) -> Tuple[Any, Dict[str, Any], Dict[str, List[int]]]:
    data = np.load(path, allow_pickle=True)

    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        try:
            raw = common.scalar_from_npz(data, "meta_json")
            meta = json.loads(str(raw))
        except Exception:
            meta = {}

    shapes = {
        key: list(data[key].shape)
        for key in data.files
        if hasattr(data[key], "shape")
    }
    return data, meta, shapes


def load_codec_summary(
    root: Path,
    data: Any,
    fallback: str,
) -> Dict[str, Any]:
    candidates: List[Path] = []

    if "action_template_json_or_pickle_path" in data.files:
        try:
            value = common.scalar_from_npz(
                data,
                "action_template_json_or_pickle_path",
            )
            candidates.append(Path(str(value)))
        except Exception:
            pass

    candidates.append(Path(fallback))

    for candidate in candidates:
        path = candidate if candidate.is_absolute() else root / candidate
        if not path.exists():
            continue

        try:
            from ccda_phase3.data_io import (
                load_action_codec_from_template,
            )

            codec = load_action_codec_from_template(path)
        except Exception:
            with path.open("rb") as file:
                codec = pickle.load(file)

        dim_attr = getattr(codec, "dim", None)
        dim = int(dim_attr() if callable(dim_attr) else dim_attr)
        summary = (
            codec.summary()
            if hasattr(codec, "summary")
            else {}
        )

        return {
            "path": str(path),
            "dim": dim,
            "summary": summary,
        }

    return {
        "path": "",
        "dim": -1,
        "summary": {},
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def first_old_checkpoint(
    root: Path,
    checkpoint_root: str,
) -> Optional[Path]:
    base = root / checkpoint_root / "state_action"
    if not base.exists():
        return None

    for candidate in sorted(
        path
        for path in base.glob("fold_*_seed_*")
        if path.is_dir()
    ):
        if (
            (candidate / "state_model.pt").exists()
            and (candidate / "inverse_dynamics.pt").exists()
        ):
            return candidate

    return None


def phase39b_checkpoint(
    root: Path,
    raw_path: str,
    ablation: str,
) -> Optional[Path]:
    payload = load_json(root / raw_path)
    value = (
        payload
        .get("results", {})
        .get(ablation, {})
        .get("checkpoint")
    )
    if not value:
        return None

    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path if path.exists() else None


def scan_current_code(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    p12b = root / "scripts/phase3_12b_proxy_score_rollout.py"
    a12b = root / "scripts/phase3_12b_analyze_proxy_score.py"

    if p12b.exists():
        text = p12b.read_text(errors="replace")

        selector_pair_group = (
            'f"phase3_12b_{selector}_{condition}_{visible_seed}"'
            in text
            or "phase3_12b_{selector}_{condition}_{visible_seed}"
            in text
        )
        findings.append({
            "level": "WARN" if selector_pair_group else "PASS",
            "name": "phase312b_selector_dependent_pair_group_present",
            "detail": selector_pair_group,
        })

        seed_calls = {
            "random.seed": "random.seed(" in text,
            "np.random.seed": "np.random.seed(" in text,
            "torch.manual_seed": "torch.manual_seed(" in text,
        }
        if not all(seed_calls.values()):
            findings.append({
                "level": "WARN",
                "name": "phase312b_explicit_rng_seeding_incomplete",
                "detail": seed_calls,
            })

    if a12b.exists():
        text = a12b.read_text(errors="replace")
        compares_final = (
            'metric(episodes, "ddpm_mean", condition, "final_fraction")'
            in text
        )
        findings.append({
            "level": "WARN" if compares_final else "PASS",
            "name": "phase312b_primary_comparison_uses_final_fraction",
            "detail": compares_final,
        })

    return findings


def scan_new_scripts(root: Path) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    targets = [
        root / "scripts/phase3_12c_matched_reset_common.py",
        root / "scripts/phase3_12c_preflight.py",
        root / "scripts/phase3_12c_matched_reset_experiment.py",
        root / "scripts/phase3_12c_analyze.py",
    ]

    forbidden_imports = [
        "tensorflow",
        "ravens.agents",
        "ravens.models",
        "ravens.datasets",
    ]

    for path in targets:
        if not path.exists():
            findings.append({
                "level": "FAIL",
                "name": "required_script_missing",
                "detail": str(path),
            })
            continue

        text = path.read_text(errors="replace")

        if re.search(
            r"^\s*import\s+ravens\s*$",
            text,
            flags=re.MULTILINE,
        ):
            findings.append({
                "level": "FAIL",
                "name": "top_level_import_ravens",
                "detail": str(path),
            })

        for module in forbidden_imports:
            if re.search(
                rf"^\s*(import|from)\s+{re.escape(module)}",
                text,
                flags=re.MULTILINE,
            ):
                findings.append({
                    "level": "FAIL",
                    "name": "forbidden_import",
                    "detail": f"{path}: {module}",
                })

    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--windows",
        default="data/phase3_state_diff_windows/phase3_windows.npz",
    )
    parser.add_argument(
        "--action_template",
        default=(
            "data/phase3_state_diff_windows/"
            "phase3_action_template.pkl"
        ),
    )
    parser.add_argument(
        "--old_checkpoint_root",
        default="checkpoints/phase3",
    )
    parser.add_argument(
        "--phase39b_raw",
        default="reports/phase3_9b_ablation_raw_summary.json",
    )
    parser.add_argument(
        "--phase312b_summary",
        default="reports/phase3_12b_proxy_score_summary.json",
    )
    parser.add_argument(
        "--best_ablation",
        default="xy_only_high_weight",
    )
    parser.add_argument(
        "--out_json",
        default="reports/phase3_12c_preflight_summary.json",
    )
    parser.add_argument(
        "--out_md",
        default="reports/phase3_12c_preflight_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    common.add_repo_paths(root)

    # Initialize the established minimal runtime before probing Ravens. This
    # deliberately avoids ravens.agents/models/datasets and TensorFlow.
    try:
        import phase3_policy_rollout as _phase3_policy_rollout

        _phase3_policy_rollout.patch_pybullet_pkg_resources_metadata()
        _phase3_policy_rollout.import_ravens_runtime(root)
    except Exception:
        pass

    checks: Dict[str, Any] = {}
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({
            "level": level,
            "name": name,
            "detail": str(detail),
        })

    git = {
        "branch": run(
            ["git", "branch", "--show-current"],
            root,
        ).strip(),
        "head": run(
            ["git", "rev-parse", "HEAD"],
            root,
        ).strip(),
        "origin_experiment1": run(
            ["git", "ls-remote", "origin", "Experiment1"],
            root,
        ).strip().split("\t")[0],
        "status": run(
            ["git", "status", "--short"],
            root,
        ),
        "submodule": run(
            [
                "git",
                "submodule",
                "status",
                "external/deformable-ravens",
            ],
            root,
        ).strip(),
    }

    for module_name in [
        "numpy",
        "torch",
        "pybullet",
        "ravens.tasks",
        "ravens.environment",
        "ccda_phase3.rollout",
        "ccda_phase3.metrics",
        "ccda_phase3.train_utils",
        "phase3_policy_rollout",
        "phase3_12b_proxy_score_rollout",
        "phase3_12c_matched_reset_common",
    ]:
        result = check_import(module_name)
        key = "import_" + module_name.replace(".", "_")
        checks[key] = bool(result["ok"])
        if not result["ok"]:
            issue(
                "FAIL",
                "import_failed",
                f"{module_name}: {result['error']}",
            )

    try:
        common.assert_no_forbidden_modules("preflight")
        checks["forbidden_modules_not_loaded"] = True
    except Exception as exc:
        checks["forbidden_modules_not_loaded"] = False
        issue("FAIL", "forbidden_modules_loaded", repr(exc))

    try:
        tasks = importlib.import_module("ravens.tasks")
        checks["task_registered"] = (
            "hidden-contact-cable-line"
            in getattr(tasks, "names", {})
        )

        task = tasks.names["hidden-contact-cable-line"]()
        conditions = list(getattr(task, "CONDITIONS", []))

        checks["task_instantiates"] = True
        checks["task_has_required_conditions"] = all(
            condition in conditions
            for condition in REQUIRED_CONDITIONS
        )
        checks["task_has_primary"] = PRIMARY in conditions
        checks["task_has_settle_attribute"] = hasattr(
            task,
            "_settle_secs",
        )
    except Exception as exc:
        checks["task_registered"] = False
        checks["task_instantiates"] = False
        checks["task_has_required_conditions"] = False
        checks["task_has_primary"] = False
        checks["task_has_settle_attribute"] = False
        issue("FAIL", "task_probe_failed", repr(exc))

    data, meta, shapes = load_npz_meta(root / args.windows)
    checks["windows_loadable"] = True
    checks["windows_primary_match"] = (
        meta.get("primary_hidden_condition") == PRIMARY
    )
    checks["windows_diagnostic_match"] = (
        meta.get("diagnostic_hidden_condition") == DIAGNOSTIC
    )

    for key in [
        "state_action_x",
        "y_state",
        "y_action",
        "condition_name",
        "split_name",
        "th",
        "action_dim",
        "n_beads",
    ]:
        value = key in data.files
        checks["has_" + key] = value
        if not value:
            issue("FAIL", "missing_window_key", key)

    checks["y_action_dim_14"] = (
        "y_action" in data.files
        and len(data["y_action"].shape) == 2
        and int(data["y_action"].shape[1]) == 14
    )

    codec = load_codec_summary(
        root,
        data,
        args.action_template,
    )
    checks["action_codec_dim_14"] = codec["dim"] == 14
    checks["action_codec_no_camera_config"] = (
        int(
            codec["summary"].get(
                "num_camera_config_paths",
                0,
            )
            or 0
        )
        == 0
    )

    old_checkpoint = first_old_checkpoint(
        root,
        args.old_checkpoint_root,
    )
    repaired_checkpoint = phase39b_checkpoint(
        root,
        args.phase39b_raw,
        args.best_ablation,
    )

    checks["old_state_action_checkpoint_complete"] = (
        old_checkpoint is not None
    )
    checks["phase39b_best_idm_exists"] = (
        repaired_checkpoint is not None
    )

    phase312b = load_json(root / args.phase312b_summary)
    checks["phase312b_summary_exists"] = bool(phase312b)
    checks["phase312b_completed"] = (
        phase312b.get("progress", {}).get("status")
        == "completed"
    )
    checks["phase312b_no_phase4_scope"] = (
        root / "reports/phase3_12b_no_phase4_confirmation.md"
    ).exists()

    for key, value in checks.items():
        if key.startswith("import_") and not value:
            continue
        if key in {
            "task_registered",
            "task_instantiates",
            "task_has_required_conditions",
            "task_has_primary",
            "task_has_settle_attribute",
            "windows_primary_match",
            "windows_diagnostic_match",
            "y_action_dim_14",
            "action_codec_dim_14",
            "action_codec_no_camera_config",
            "old_state_action_checkpoint_complete",
            "phase39b_best_idm_exists",
            "phase312b_summary_exists",
            "phase312b_completed",
            "phase312b_no_phase4_scope",
        } and not value:
            issue("FAIL", key, key)

    for finding in scan_current_code(root):
        if finding["level"] != "PASS":
            issues.append(finding)

    for finding in scan_new_scripts(root):
        issues.append(finding)

    checks["running_in_coord_bimanual"] = (
        "coord_bimanual" in sys.executable
        or os.environ.get("CONDA_DEFAULT_ENV")
        == "coord_bimanual"
    )
    if not checks["running_in_coord_bimanual"]:
        issue(
            "WARN",
            "not_coord_bimanual",
            sys.executable,
        )

    has_fail = any(
        entry["level"] == "FAIL"
        for entry in issues
    )
    has_warn = any(
        entry["level"] == "WARN"
        for entry in issues
    )

    verdict = (
        "FAIL"
        if has_fail
        else "WARN"
        if has_warn
        else "PASS"
    )

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
            "old_state_action": (
                str(old_checkpoint)
                if old_checkpoint
                else None
            ),
            "phase39b_best": (
                str(repaired_checkpoint)
                if repaired_checkpoint
                else None
            ),
            "best_ablation": args.best_ablation,
        },
        "phase312b": {
            "verdict": phase312b.get("verdict"),
            "root_cause": phase312b.get("root_cause"),
            "num_episode_rows": phase312b.get(
                "num_episode_rows"
            ),
            "num_step_rows": phase312b.get(
                "num_step_rows"
            ),
        },
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )

    lines = [
        "# Phase3.12c Matched-Reset Preflight",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Python: `{sys.executable}`",
        (
            "- Conda env: "
            f"`{os.environ.get('CONDA_DEFAULT_ENV')}`"
        ),
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
        for entry in issues:
            detail = str(entry["detail"]).replace("|", "/")
            lines.append(
                f"| `{entry['level']}` | "
                f"`{entry['name']}` | {detail} |"
            )
    else:
        lines.append(
            "| `PASS` | `none` | No issues found. |"
        )

    lines += [
        "",
        "## Scope",
        "",
        "- Preflight only.",
        "- No rollout.",
        "- No model training.",
        "- No future DDPM training.",
        "- No Phase4.",
        "- No CPS.",
        "",
        "## Important",
        "",
        (
            "- WARN findings about Phase3.12b are expected "
            "diagnostic findings and do not block Phase3.12c."
        ),
        (
            "- Any FAIL finding blocks reset audit and paired "
            "selector evaluation."
        ),
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit(
            "[Phase3.12c][FAIL] preflight failed"
        )


if __name__ == "__main__":
    main()
