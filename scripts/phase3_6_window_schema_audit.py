#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"

CANDIDATE_CONDITION_KEYS = ["condition_name", "condition", "conditions"]
CANDIDATE_VISIBLE_SEED_KEYS = ["visible_seed", "ccda_visible_seed", "seed", "episode_seed"]
CANDIDATE_WINDOW_T_KEYS = ["window_t", "t", "time_idx", "step_idx"]
CANDIDATE_SOURCE_KEYS = ["source_file", "episode_file", "info_file", "action_file", "episode_id", "demo_id"]

def arr_preview(arr, n=5):
    try:
        flat = np.asarray(arr).reshape(-1)
        return [str(x) for x in flat[:n]]
    except Exception:
        return []

def find_first_key(files, keys):
    for k in keys:
        if k in files:
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--out_json", default="reports/phase3_6_window_schema_audit_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_6_window_schema_audit_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    path = root / args.windows
    issues = []

    def issue(level, name, detail):
        issues.append({"level": level, "name": name, "detail": str(detail)})

    if not path.exists():
        payload = {"verdict": "FAIL", "issues": [{"level": "FAIL", "name": "missing_windows", "detail": str(path)}]}
        (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit("[Phase3.6][FAIL] windows npz missing")

    data = np.load(path, allow_pickle=True)
    keys = list(data.files)
    meta = {}
    if "meta_json" in keys:
        raw = data["meta_json"]
        try:
            meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0]))
        except Exception as e:
            issue("WARN", "meta_json_parse_failed", repr(e))

    shape_table = {}
    previews = {}
    for k in keys:
        arr = data[k]
        shape_table[k] = list(arr.shape) if hasattr(arr, "shape") else []
        previews[k] = arr_preview(arr)

    condition_key = find_first_key(keys, CANDIDATE_CONDITION_KEYS)
    visible_seed_key = find_first_key(keys, CANDIDATE_VISIBLE_SEED_KEYS)
    window_t_key = find_first_key(keys, CANDIDATE_WINDOW_T_KEYS)
    source_key = find_first_key(keys, CANDIDATE_SOURCE_KEYS)

    checks = {
        "windows_exists": True,
        "has_paper_x": "paper_x" in keys,
        "has_state_action_x": "state_action_x" in keys,
        "has_y_action": "y_action" in keys,
        "has_condition_key": condition_key is not None,
        "has_visible_seed_key": visible_seed_key is not None,
        "has_window_t_key": window_t_key is not None,
        "has_source_key": source_key is not None,
        "meta_conditions_match": meta.get("conditions") == REQUIRED_CONDITIONS,
        "meta_primary_match": meta.get("primary_hidden_condition") == PRIMARY,
        "meta_diagnostic_match": meta.get("diagnostic_hidden_condition") == DIAGNOSTIC,
    }

    if "y_action" in keys:
        checks["y_action_dim_14"] = len(data["y_action"].shape) == 2 and int(data["y_action"].shape[1]) == 14
    else:
        checks["y_action_dim_14"] = False

    if "state_action_x" in keys and "paper_x" in keys and "y_action" in keys:
        extra = int(data["state_action_x"].shape[1] - data["paper_x"].shape[1])
        checks["state_action_extra_dim"] = extra
        checks["state_action_extra_dim_multiple_of_14"] = (extra % 14 == 0)
        checks["can_use_state_action_tail_as_low_conf_prefix"] = extra >= 14 and extra % 14 == 0
    else:
        checks["state_action_extra_dim"] = None
        checks["state_action_extra_dim_multiple_of_14"] = False
        checks["can_use_state_action_tail_as_low_conf_prefix"] = False

    if not checks["has_condition_key"]:
        issue("FAIL", "missing_condition_key", keys)
    if not checks["has_visible_seed_key"]:
        issue("FAIL", "missing_visible_seed_key", keys)
    if not checks["has_window_t_key"]:
        issue("FAIL", "missing_window_t_key", keys)
    if not checks["has_y_action"]:
        issue("FAIL", "missing_y_action", keys)
    if not checks["has_paper_x"]:
        issue("FAIL", "missing_paper_x", keys)
    if not checks["y_action_dim_14"]:
        issue("FAIL", "y_action_dim_not_14", shape_table.get("y_action"))

    # source_key is desirable but not required because state_action_x tail can be used as low-confidence fallback.
    if not checks["has_source_key"]:
        issue("WARN", "missing_source_key", "No source_file/action_file metadata found; raw prefix replay may be unavailable.")
    if not checks["can_use_state_action_tail_as_low_conf_prefix"]:
        issue("WARN", "state_action_tail_prefix_unavailable", "state_action_x tail cannot be interpreted as action history.")

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    payload = {
        "verdict": verdict,
        "keys": keys,
        "shape_table": shape_table,
        "previews": previews,
        "meta": meta,
        "condition_key": condition_key,
        "visible_seed_key": visible_seed_key,
        "window_t_key": window_t_key,
        "source_key": source_key,
        "checks": checks,
        "issues": issues,
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.6 Window Schema Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- condition_key: `{condition_key}`",
        f"- visible_seed_key: `{visible_seed_key}`",
        f"- window_t_key: `{window_t_key}`",
        f"- source_key: `{source_key}`",
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
        "## Key Shapes",
        "",
        "| Key | Shape | Preview |",
        "|---|---|---|",
    ]
    for k in keys:
        lines.append(f"| `{k}` | `{shape_table[k]}` | `{previews[k]}` |")

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
        lines.append("| `PASS` | `none` | No schema issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- Raw source/action-file prefix is high-confidence matched replay.",
        "- state_action_x tail prefix is low-confidence because canonicalization may have copied free inputs.",
        "- If no prefix source exists, Phase3.6 must report BLOCKED instead of claiming codec failure.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.6][FAIL] window schema audit failed")

if __name__ == "__main__":
    main()
