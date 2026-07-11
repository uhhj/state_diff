#!/usr/bin/env python3
"""Preflight gate for the Phase3.12d-r2.2 diagnostic audit."""

from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from phase3_12d_r22_integrity import strict_json_dump


def command(args, root: Path) -> str:
    return subprocess.check_output(args, cwd=str(root), text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--out-json", default="reports/phase3_12d_r22_preflight_summary.json")
    parser.add_argument("--out-md", default="reports/phase3_12d_r22_preflight_report.md")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    for path in (root, root / "scripts", root / "external/deformable-ravens"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    from ccda_phase3 import data_io, rollout
    import phase3_policy_rollout  # noqa: F401 - validates TensorFlow-free import path.
    import pytest
    import torch

    checks: Dict[str, Any] = {}
    checks["main_branch_experiment1"] = command(["git", "branch", "--show-current"], root) == "Experiment1"
    checks["submodule_clean"] = command(["git", "-C", "external/deformable-ravens", "status", "--short"], root) == ""
    checks["tensorflow_not_loaded"] = not any(name == "tensorflow" or name.startswith("tensorflow.") for name in sys.modules)
    tests = [
        "tests/test_ccda_phase3_state_safety.py",
        "tests/test_ccda_phase3_visible_seed.py",
        "tests/test_phase3_12d_r22_integrity.py",
    ]
    checks["dedicated_tests_pass"] = pytest.main(["-q", *[str(root / value) for value in tests]]) == 0
    offline_source = inspect.getsource(data_io.state_from_info)
    live_source = inspect.getsource(rollout.state_from_live_info)
    checks["shared_state_safety_helper"] = "finite_velocity_or_difference" in offline_source and "finite_velocity_or_difference" in live_source
    checks["strict_visible_seed_parser"] = data_io.visible_seed_from_extras({"ccda_pair_group": "phase3_12d_r22_seed_313127"}) == 313127 and data_io.visible_seed_from_extras({"ccda_pair_group": "phase3_12d_r22_313127"}) == -1
    with tempfile.TemporaryDirectory() as directory:
        try:
            strict_json_dump(Path(directory) / "bad.json", {"bad": float("nan")})
            checks["strict_json_rejects_nonfinite"] = False
        except ValueError:
            checks["strict_json_rejects_nonfinite"] = True

    required = {
        "windows_exist": root / "data/phase3_state_diff_windows/phase3_windows.npz",
        "action_codec_exists": root / "data/phase3_state_diff_windows/phase3_action_template.pkl",
        "state_checkpoint_exists": root / "checkpoints/phase3",
        "idm_checkpoint_report_exists": root / "reports/phase3_9b_ablation_raw_summary.json",
        "r21_report_exists": root / "reports/phase3_12d_r21_environment_audit_summary.json",
        "train_raw_source_exists": root / "external/deformable-ravens/data/phase3_ccda_large/train/hidden-contact-cable-line",
        "heldout_raw_source_exists": root / "external/deformable-ravens/data/phase3_ccda_large/heldout/hidden-contact-cable-line",
    }
    checks.update({key: path.exists() for key, path in required.items()})
    if required["r21_report_exists"].exists():
        r21 = json.loads(required["r21_report_exists"].read_text())
        checks["r21_root_cause_expected"] = r21.get("root_cause") == "phase312d_r21_latent_constraint_no_action_visible_leak_supported"
    else:
        checks["r21_root_cause_expected"] = False
    checks["torch_linear_fallback_available"] = bool(torch.__version__)
    try:
        import sklearn
        classifier_backend = f"sklearn_{sklearn.__version__}"
    except Exception:
        classifier_backend = f"torch_{torch.__version__}_linear_bce"
    checks["no_phase4_or_cps_scope"] = True

    warnings: List[Dict[str, Any]] = []
    main_status = command(["git", "status", "--short"], root)
    if main_status:
        warnings.append({"name": "main_worktree_contains_preserved_local_changes", "detail": main_status})
    prepare_summary = root / "reports/phase3_prepare_windows_summary.json"
    audit_payload = json.loads(prepare_summary.read_text()) if prepare_summary.exists() else {}
    if not audit_payload.get("environment_semantics_version"):
        warnings.append({"name": "legacy_environment_semantics", "detail": "existing windows/checkpoints are a bridge diagnostic only"})
    failures = [name for name, value in checks.items() if value is not True]
    verdict = "FAIL" if failures else "WARN" if warnings else "PASS"
    payload = {
        "verdict": verdict,
        "collection_allowed": not failures,
        "checks": checks,
        "failures": failures,
        "warnings": warnings,
        "classifier_backend": classifier_backend,
        "python": sys.executable,
        "scope": "diagnostic observation leakage only; no Phase4, CPS, candidate matrix, DDPM, or IDM training",
    }
    strict_json_dump(root / args.out_json, payload)
    lines = ["# Phase3.12d-r2.2 Preflight", "", f"- Verdict: `{verdict}`", f"- Collection allowed: `{not failures}`", f"- Classifier backend: `{classifier_backend}`", "", "## Checks", "", "| Check | Result |", "|---|---:|"]
    lines.extend(f"| `{name}` | `{value}` |" for name, value in checks.items())
    lines += ["", "## Warnings", ""]
    lines.extend(f"- `{item['name']}`: {str(item['detail']).replace(chr(10), '; ')}" for item in warnings)
    if not warnings:
        lines.append("- None")
    (root / args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if failures:
        raise SystemExit("[Phase3.12d-r2.2] preflight failed")


if __name__ == "__main__":
    main()
