#!/usr/bin/env python3
"""Audit Phase3 dynamic hidden-condition integration before training/medium."""

import sys
from pathlib import Path as _Phase3Path
ROOT = _Phase3Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ccda_phase3.data_io import FORBIDDEN_METADATA_NOT_IN_X, normalize_conditions


def add_issue(issues: List[Dict[str, Any]], level: str, name: str, detail: str) -> None:
    issues.append({"level": level, "name": name, "detail": detail})


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def group_presence(split, cond, seed, wt):
    groups = {}
    for i in range(len(cond)):
        key = (str(split[i]), int(seed[i]), int(wt[i]))
        groups.setdefault(key, set()).add(str(cond[i]))
    return groups


def max_pair_diff(leak: Dict[str, Any]) -> float:
    vals = []
    for item in (leak.get("pair_consistency_by_hidden_condition") or {}).values():
        for key in ["max_pair_paper_x_max_abs_diff", "max_pair_state_action_x_max_abs_diff"]:
            v = item.get(key)
            if v is not None:
                vals.append(float(v))
    return max(vals) if vals else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--data", required=True)
    ap.add_argument("--leak_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--primary_hidden_condition", default=None)
    ap.add_argument("--diagnostic_hidden_condition", default=None)
    ap.add_argument("--pair_threshold", type=float, default=1e-8)
    args = ap.parse_args()

    data = np.load(args.data, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))
    leak = load_json(Path(args.leak_json))
    conditions = normalize_conditions(args.conditions or meta.get("conditions"))
    primary = args.primary_hidden_condition or meta.get("primary_hidden_condition", "hidden_breakaway_pin")
    diagnostic = args.diagnostic_hidden_condition or meta.get("diagnostic_hidden_condition", "hidden_pin")
    issues: List[Dict[str, Any]] = []

    if conditions != meta.get("conditions"):
        add_issue(issues, "FAIL", "conditions_mismatch", f"CLI conditions={conditions}; window meta={meta.get('conditions')}")
    if primary != meta.get("primary_hidden_condition"):
        add_issue(issues, "FAIL", "primary_hidden_mismatch", f"CLI primary={primary}; meta={meta.get('primary_hidden_condition')}")
    if diagnostic != meta.get("diagnostic_hidden_condition"):
        add_issue(issues, "FAIL", "diagnostic_hidden_mismatch", f"CLI diagnostic={diagnostic}; meta={meta.get('diagnostic_hidden_condition')}")
    if primary not in conditions:
        add_issue(issues, "FAIL", "primary_missing_from_conditions", f"{primary} not in {conditions}")
    if diagnostic not in conditions:
        add_issue(issues, "FAIL", "diagnostic_missing_from_conditions", f"{diagnostic} not in {conditions}")
    if "hidden_side_jam" in conditions:
        add_issue(issues, "FAIL", "removed_condition_present", "hidden_side_jam must not be used in Phase3.")

    expected_default = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
    if conditions != expected_default:
        add_issue(issues, "WARN", "conditions_not_default_phase2_5c_set", f"conditions={conditions}; expected={expected_default}")

    if int(meta.get("action_dim", -1)) != 14:
        add_issue(issues, "FAIL", "action_dim_not_14", f"action_dim={meta.get('action_dim')}")
    codec = meta.get("action_codec_summary") or meta.get("action_codec") or {}
    if int(codec.get("dim", -1)) != 14:
        add_issue(issues, "FAIL", "codec_dim_not_14", f"codec={codec}")
    if int(codec.get("num_camera_config_paths", -1)) != 0:
        add_issue(issues, "FAIL", "codec_camera_config_paths", f"codec={codec}")
    y_shape = meta.get("y_action_shape") or []
    if len(y_shape) != 2 or int(y_shape[1]) != 14:
        add_issue(issues, "FAIL", "y_action_dim_not_14", f"y_action_shape={y_shape}")

    forbidden = set(meta.get("forbidden_metadata_not_in_x") or meta.get("feature_schema", {}).get("forbidden_not_in_x", []))
    missing_forbidden = sorted(set(FORBIDDEN_METADATA_NOT_IN_X) - forbidden)
    if missing_forbidden:
        add_issue(issues, "FAIL", "forbidden_metadata_guard_incomplete", f"missing={missing_forbidden}")
    x_schema = []
    schema = meta.get("feature_schema", {})
    for key in ["paper_x", "state_action_x"]:
        x_schema.extend(str(x) for x in schema.get(key, []))
    leaked_names = sorted(name for name in forbidden if any(name in field for field in x_schema))
    if leaked_names:
        add_issue(issues, "FAIL", "forbidden_metadata_in_x_schema", f"leaked={leaked_names}")

    windows_per_condition = {str(k): int(v) for k, v in (meta.get("windows_per_condition") or {}).items()}
    missing_conditions = [c for c in conditions if windows_per_condition.get(c, 0) <= 0]
    if missing_conditions:
        add_issue(issues, "FAIL", "condition_windows_missing", f"missing={missing_conditions}; counts={windows_per_condition}")
    count_values = [windows_per_condition.get(c, 0) for c in conditions]
    if len(set(count_values)) > 1:
        add_issue(issues, "FAIL", "condition_window_counts_unbalanced", f"counts={windows_per_condition}")

    if meta.get("train_heldout_seed_overlap"):
        add_issue(issues, "FAIL", "train_heldout_seed_overlap", str(meta.get("train_heldout_seed_overlap")))

    split = data["split_name"].astype(str)
    cond = data["condition_name"].astype(str)
    seed = data["visible_seed"].astype(int)
    wt = data["window_t"].astype(int)
    groups = group_presence(split, cond, seed, wt)
    heldout_groups = {k: v for k, v in groups.items() if k[0] == "heldout"}
    missing_primary_refs = [k for k, present in heldout_groups.items() if not {"free", primary}.issubset(present)]
    missing_diagnostic_refs = [k for k, present in heldout_groups.items() if not {"free", diagnostic}.issubset(present)]
    if missing_primary_refs:
        add_issue(issues, "FAIL", "heldout_primary_pair_refs_missing", f"count={len(missing_primary_refs)} example={missing_primary_refs[:3]}")
    if missing_diagnostic_refs:
        add_issue(issues, "FAIL", "heldout_diagnostic_pair_refs_missing", f"count={len(missing_diagnostic_refs)} example={missing_diagnostic_refs[:3]}")

    if not leak.get("pass"):
        add_issue(issues, "FAIL", "input_leakage_report_failed", f"pass={leak.get('pass')}")
    pair_max = max_pair_diff(leak)
    if pair_max > args.pair_threshold:
        add_issue(issues, "FAIL", "post_canonical_pair_diff_nonzero", f"max_pair_diff={pair_max}")
    leak_primary = leak.get("primary_hidden_condition")
    if leak_primary and leak_primary != primary:
        add_issue(issues, "FAIL", "leakage_primary_mismatch", f"leak={leak_primary}; primary={primary}")

    verdict = "FAIL" if any(i["level"] == "FAIL" for i in issues) else ("WARN" if any(i["level"] == "WARN" for i in issues) else "PASS")
    payload = {
        "verdict": verdict,
        "conditions": conditions,
        "primary_hidden_condition": primary,
        "diagnostic_hidden_condition": diagnostic,
        "primary_branch_pair": f"free_vs_{primary}",
        "diagnostic_branch_pair": f"free_vs_{diagnostic}",
        "phase2_5_selected_config": meta.get("phase2_5_selected_config"),
        "windows_per_condition": windows_per_condition,
        "num_windows": int(meta.get("num_windows", len(cond))),
        "train_visible_seed_count": int(meta.get("train_visible_seed_count", 0)),
        "heldout_visible_seed_count": int(meta.get("heldout_visible_seed_count", 0)),
        "heldout_primary_pair_ref_groups": int(len(heldout_groups) - len(missing_primary_refs)),
        "heldout_diagnostic_pair_ref_groups": int(len(heldout_groups) - len(missing_diagnostic_refs)),
        "max_post_canonical_pair_diff": pair_max,
        "input_leakage_pass": bool(leak.get("pass")),
        "action_dim": int(meta.get("action_dim", -1)),
        "y_action_shape": meta.get("y_action_shape"),
        "action_codec_summary": codec,
        "state_action_extra_std": meta.get("state_action_extra_std"),
        "forbidden_metadata_count": len(forbidden),
        "issues": issues,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3 Condition Integration Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Conditions: `{', '.join(conditions)}`",
        f"- Primary branch pair: `free_vs_{primary}`",
        f"- Diagnostic branch pair: `free_vs_{diagnostic}`",
        f"- Phase2.5 selected config: `{payload['phase2_5_selected_config']}`",
        "",
        "## Window Balance",
        "",
        "| Condition | Windows |",
        "|---|---:|",
    ]
    for c in conditions:
        lines.append(f"| `{c}` | {windows_per_condition.get(c, 0)} |")
    lines += [
        "",
        "## Leakage Guard",
        "",
        f"- Input leakage pass: `{payload['input_leakage_pass']}`",
        f"- Max post-canonical pair diff: `{pair_max}`",
        f"- Action dim: `{payload['action_dim']}`",
        f"- y_action shape: `{payload['y_action_shape']}`",
        f"- State-action extra std: `{payload['state_action_extra_std']}`",
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for issue in issues:
            lines.append(f"| `{issue['level']}` | `{issue['name']}` | {issue['detail']} |")
    else:
        lines.append("| `PASS` | `none` | No blocking issue found. |")
    lines += [
        "",
        "## Conclusion",
        "",
    ]
    if verdict == "PASS":
        lines.append("PASS: Phase3 is configured for the Phase2.5c recoverable branch without exposing hidden-contact labels, branch names, success labels, or recoverability parameters to model inputs.")
    elif verdict == "WARN":
        lines.append("WARN: Phase3 integration has no blocking failure, but warnings should be inspected before medium.")
    else:
        lines.append("FAIL: Do not run Phase3 medium until the listed condition integration or leakage issues are fixed.")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3][FAIL] condition integration audit failed")


if __name__ == "__main__":
    main()
