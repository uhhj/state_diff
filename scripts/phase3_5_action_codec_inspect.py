#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]

def safe_jsonable(x):
    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)

def get_array_str(data, *names):
    for n in names:
        if n in data.files:
            arr = data[n]
            return np.array([str(x) for x in arr])
    return None

def flatten_action(action):
    out = {}
    out["primitive"] = action.get("primitive") if isinstance(action, dict) else None
    params = action.get("params", {}) if isinstance(action, dict) else {}
    out["has_params"] = isinstance(params, dict)
    for key in ["pose0", "pose1"]:
        val = params.get(key) if isinstance(params, dict) else None
        if val is not None:
            try:
                pos = np.asarray(val[0], dtype=float).reshape(-1)
                rot = np.asarray(val[1], dtype=float).reshape(-1)
                out[f"{key}_pos"] = pos.tolist()
                out[f"{key}_rot"] = rot.tolist()
                out[f"{key}_x"] = float(pos[0]) if len(pos) > 0 else float("nan")
                out[f"{key}_y"] = float(pos[1]) if len(pos) > 1 else float("nan")
                out[f"{key}_z"] = float(pos[2]) if len(pos) > 2 else float("nan")
                out[f"{key}_quat_norm"] = float(np.linalg.norm(rot)) if len(rot) else float("nan")
            except Exception as e:
                out[f"{key}_parse_error"] = repr(e)
    return out

def in_workspace(x, y):
    if not math.isfinite(x) or not math.isfinite(y):
        return False
    return 0.0 <= x <= 1.0 and -1.0 <= y <= 1.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    ap.add_argument("--samples_per_condition", type=int, default=8)
    ap.add_argument("--out_csv", default="reports/phase3_5_action_codec_decode_samples.csv")
    ap.add_argument("--out_json", default="reports/phase3_5_action_codec_inspect_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_5_action_codec_inspect_report.md")
    args = ap.parse_args()

    root = Path(args.root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    data = np.load(root / args.windows, allow_pickle=True)
    with open(root / args.action_template, "rb") as f:
        codec = pickle.load(f)

    y = np.asarray(data["y_action"], dtype=np.float32)
    conds = get_array_str(data, "condition_name", "condition", "conditions")
    if conds is None:
        # Fallback: evenly inspect without condition labels.
        conds = np.array(["unknown"] * len(y))

    rows = []
    issues = []

    for cond in REQUIRED_CONDITIONS:
        idxs = np.where(conds == cond)[0]
        if len(idxs) == 0:
            issues.append({"level": "FAIL", "name": "missing_condition_in_windows", "detail": cond})
            continue
        for idx in idxs[: args.samples_per_condition]:
            vec = y[idx]
            try:
                action = codec.decode(vec)
                flat = flatten_action(action)
                primitive_ok = flat.get("primitive") == "pick_place"
                pose0_ok = in_workspace(flat.get("pose0_x", float("nan")), flat.get("pose0_y", float("nan")))
                pose1_ok = in_workspace(flat.get("pose1_x", float("nan")), flat.get("pose1_y", float("nan")))
                quat0 = flat.get("pose0_quat_norm", float("nan"))
                quat1 = flat.get("pose1_quat_norm", float("nan"))
                quat_ok = (not math.isfinite(quat0) or abs(quat0 - 1.0) < 1e-3) and (not math.isfinite(quat1) or abs(quat1 - 1.0) < 1e-3)
                row = {
                    "idx": int(idx),
                    "condition": cond,
                    "decode_ok": True,
                    "primitive": flat.get("primitive"),
                    "primitive_ok": primitive_ok,
                    "pose0_workspace_ok": pose0_ok,
                    "pose1_workspace_ok": pose1_ok,
                    "quat_norm_ok": quat_ok,
                    "vec_norm": float(np.linalg.norm(vec)),
                    "vec_min": float(np.min(vec)),
                    "vec_max": float(np.max(vec)),
                    "pose0_x": flat.get("pose0_x", float("nan")),
                    "pose0_y": flat.get("pose0_y", float("nan")),
                    "pose0_z": flat.get("pose0_z", float("nan")),
                    "pose1_x": flat.get("pose1_x", float("nan")),
                    "pose1_y": flat.get("pose1_y", float("nan")),
                    "pose1_z": flat.get("pose1_z", float("nan")),
                    "action_repr": json.dumps(safe_jsonable(action), default=str)[:1000],
                    "error": "",
                }
                if not primitive_ok:
                    issues.append({"level": "FAIL", "name": "decoded_primitive_not_pick_place", "detail": row})
                if not pose0_ok or not pose1_ok:
                    issues.append({"level": "WARN", "name": "decoded_pose_outside_broad_workspace", "detail": row})
                rows.append(row)
            except Exception as e:
                rows.append({
                    "idx": int(idx),
                    "condition": cond,
                    "decode_ok": False,
                    "primitive": "",
                    "primitive_ok": False,
                    "pose0_workspace_ok": False,
                    "pose1_workspace_ok": False,
                    "quat_norm_ok": False,
                    "vec_norm": float(np.linalg.norm(vec)),
                    "vec_min": float(np.min(vec)),
                    "vec_max": float(np.max(vec)),
                    "pose0_x": "",
                    "pose0_y": "",
                    "pose0_z": "",
                    "pose1_x": "",
                    "pose1_y": "",
                    "pose1_z": "",
                    "action_repr": "",
                    "error": repr(e),
                })
                issues.append({"level": "FAIL", "name": "decode_failed", "detail": f"{cond} idx={idx}: {repr(e)}"})

    action_dim_std = np.std(y, axis=0)
    near_zero_count = int(np.sum(action_dim_std < 1e-8))

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    payload = {
        "verdict": verdict,
        "num_rows": len(rows),
        "near_zero_action_dims": near_zero_count,
        "action_dim_std": action_dim_std.tolist(),
        "issues": issues,
        "interpretation": "Codec decode sanity only; not action execution evidence.",
    }

    (root / args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.5 Action Codec Inspect",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Decoded samples: `{len(rows)}`",
        f"- near_zero_action_dims: `{near_zero_count}`",
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if issues:
        for i in issues[:50]:
            lines.append(f"| `{i['level']}` | `{i['name']}` | {str(i['detail']).replace('|', '/')} |")
    else:
        lines.append("| `PASS` | `none` | No codec decode issues found. |")
    lines += [
        "",
        "## Output",
        "",
        f"- CSV: `{args.out_csv}`",
    ]
    (root / args.out_md).write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.5][FAIL] action codec inspect failed")

if __name__ == "__main__":
    main()
