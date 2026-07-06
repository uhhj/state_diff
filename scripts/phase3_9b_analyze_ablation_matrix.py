#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def sf(x: Any) -> float:
    try:
        if x is None or x == "":
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def mean(xs: List[float]) -> float:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def get_condition_row(summary: Dict[str, Any], condition: str) -> Dict[str, Any]:
    for row in summary.get("condition_table", []):
        if row.get("condition") == condition:
            return row
    return {}


def score_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    conditions = ["free", "hidden_breakaway_pin", "hidden_high_friction"]
    rows = {c: get_condition_row(summary, c) for c in conditions}
    improvements = {c: sf(rows[c].get("improvement")) for c in conditions}
    phase39_delta = {c: sf(rows[c].get("phase39_idm_delta")) for c in conditions}
    old_delta = {c: sf(rows[c].get("old_idm_delta")) for c in conditions}
    gt_delta = {c: sf(rows[c].get("gt_reference_delta")) for c in conditions}
    improved_conditions = list(summary.get("improved_conditions", []))
    return {
        "verdict": summary.get("verdict"),
        "root_cause": summary.get("root_cause"),
        "improved_conditions": improved_conditions,
        "primary_improved": bool(summary.get("primary_improved", False)),
        "num_improved_conditions": len(improved_conditions),
        "hidden_breakaway_improvement": improvements["hidden_breakaway_pin"],
        "mean_improvement": mean(list(improvements.values())),
        "min_improvement": min([v for v in improvements.values() if math.isfinite(v)], default=float("nan")),
        "mean_phase39_delta": mean(list(phase39_delta.values())),
        "mean_old_delta": mean(list(old_delta.values())),
        "mean_gt_delta": mean(list(gt_delta.values())),
        "condition_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--raw_summary", default="reports/phase3_9b_ablation_raw_summary.json")
    parser.add_argument("--out_dir", default="reports/phase3_9b_ablation")
    parser.add_argument("--out_json", default="reports/phase3_9b_ablation_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_9b_ablation_report.md")
    parser.add_argument("--support_threshold", type=float, default=0.05)
    parser.add_argument("--ablation_gap_threshold", type=float, default=0.05)
    args = parser.parse_args()

    root = Path(args.root)
    raw = load_json(root / args.raw_summary)
    issues: List[Dict[str, Any]] = []

    def issue(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": str(detail)})

    ablations = raw.get("ablations", [])
    if not ablations:
        issue("FAIL", "no_ablations", "No ablations found in raw summary.")

    records: List[Dict[str, Any]] = []
    scores: Dict[str, Dict[str, Any]] = {}

    for ablation in ablations:
        retry_summary_rel = raw.get("results", {}).get(ablation, {}).get("retry_summary")
        summary_path = root / retry_summary_rel if retry_summary_rel else root / args.out_dir / ablation / "controlled_retry_summary.json"
        summary = load_json(summary_path)
        if not summary:
            issue("FAIL", "missing_ablation_summary", f"{ablation}: {summary_path}")
            score = {
                "verdict": "MISSING",
                "root_cause": "missing",
                "improved_conditions": [],
                "primary_improved": False,
                "num_improved_conditions": 0,
                "hidden_breakaway_improvement": float("nan"),
                "mean_improvement": float("nan"),
                "min_improvement": float("nan"),
                "mean_phase39_delta": float("nan"),
                "mean_old_delta": float("nan"),
                "mean_gt_delta": float("nan"),
                "condition_rows": {},
            }
        else:
            score = score_summary(summary)
        score["ablation"] = ablation
        score["summary_path"] = str(summary_path)
        scores[ablation] = score
        records.append(score)

    default = scores.get("default_geometry", {})
    no_pull = scores.get("no_pull_loss", {})
    no_coupled = scores.get("no_coupled_xy_loss", {})
    xy_high = scores.get("xy_only_high_weight", {})
    no_quat = scores.get("no_quat_loss", {})

    default_robust = (
        default.get("primary_improved", False)
        and int(default.get("num_improved_conditions", 0)) >= 2
        and sf(default.get("hidden_breakaway_improvement")) > args.support_threshold
    )

    if default_robust:
        issue("WARN", "default_geometry_repair_robust", {
            "improved_conditions": default.get("improved_conditions"),
            "hidden_breakaway_improvement": default.get("hidden_breakaway_improvement"),
        })
    else:
        issue("FAIL", "default_geometry_not_robust", default)

    coupled_needed = False
    pull_needed = False
    quat_not_needed = False
    xy_high_competitive = False

    if default and no_coupled:
        gap = sf(default.get("hidden_breakaway_improvement")) - sf(no_coupled.get("hidden_breakaway_improvement"))
        coupled_needed = math.isfinite(gap) and gap > args.ablation_gap_threshold
        if coupled_needed:
            issue("WARN", "coupled_xy_loss_supported_by_ablation", f"hidden_breakaway improvement gap={gap:.6f}")

    if default and no_pull:
        gap = sf(default.get("hidden_breakaway_improvement")) - sf(no_pull.get("hidden_breakaway_improvement"))
        pull_needed = math.isfinite(gap) and gap > args.ablation_gap_threshold
        if pull_needed:
            issue("WARN", "pull_xy_loss_supported_by_ablation", f"hidden_breakaway improvement gap={gap:.6f}")

    if default and no_quat:
        gap = abs(sf(default.get("hidden_breakaway_improvement")) - sf(no_quat.get("hidden_breakaway_improvement")))
        quat_not_needed = math.isfinite(gap) and gap <= args.ablation_gap_threshold
        if quat_not_needed:
            issue("WARN", "quat_loss_not_critical_in_ablation", f"hidden_breakaway improvement gap={gap:.6f}")

    if default and xy_high:
        gap = sf(xy_high.get("hidden_breakaway_improvement")) - sf(default.get("hidden_breakaway_improvement"))
        xy_high_competitive = math.isfinite(gap) and gap >= -args.ablation_gap_threshold
        if xy_high_competitive:
            issue("WARN", "xy_only_high_weight_competitive", f"hidden_breakaway improvement gap={gap:.6f}")

    sorted_records = sorted(
        records,
        key=lambda r: (
            int(bool(r.get("primary_improved", False))),
            int(r.get("num_improved_conditions", 0)),
            sf(r.get("hidden_breakaway_improvement")),
            sf(r.get("mean_improvement")),
        ),
        reverse=True,
    )
    best = sorted_records[0] if sorted_records else {}

    if default_robust and (coupled_needed or pull_needed):
        root_cause = "phase39b_geometry_loss_ablation_supported"
    elif default_robust:
        root_cause = "phase39b_default_geometry_repair_robust"
    elif bool(best.get("primary_improved", False)):
        root_cause = "phase39b_nondefault_repair_candidate"
        issue("WARN", "nondefault_repair_candidate", best)
    else:
        root_cause = "phase39b_repair_not_robust"

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail and not bool(best.get("primary_improved", False)) else ("WARN" if has_warn or has_fail else "PASS")

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "best_ablation": best.get("ablation"),
        "default_robust": default_robust,
        "coupled_needed": coupled_needed,
        "pull_needed": pull_needed,
        "quat_not_needed": quat_not_needed,
        "xy_high_competitive": xy_high_competitive,
        "records": records,
        "issues": issues,
        "raw_summary": raw,
        "important_note": "This is still one-step matched-prefix diagnostic only. No Phase4/CPS or closed-loop policy evidence.",
        "recommendation": (
            "Do not enter Phase4/CPS. Next run Phase3.10 controlled learned rollout retry with repaired IDM only if default geometry remains robust."
            if default_robust else
            "Do not enter Phase4/CPS. Tune geometry loss weights or inspect primitive-level action geometry before rollout."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.9b Geometry IDM Robustness + Loss Ablation Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Best ablation: `{best.get('ablation')}`",
        f"- Default robust: `{default_robust}`",
        f"- Coupled loss needed: `{coupled_needed}`",
        f"- Pull loss needed: `{pull_needed}`",
        f"- Quat loss not critical: `{quat_not_needed}`",
        f"- XY-only high-weight competitive: `{xy_high_competitive}`",
        "",
        "## Ablation Summary",
        "",
        "| Ablation | Verdict | Root cause | Improved conditions | Primary improved | Hidden breakaway improvement | Mean improvement | Mean Phase3.9 Δ | Mean old Δ |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]

    for r in records:
        lines.append(
            f"| `{r.get('ablation')}` | `{r.get('verdict')}` | `{r.get('root_cause')}` | "
            f"`{r.get('improved_conditions')}` | `{r.get('primary_improved')}` | "
            f"{sf(r.get('hidden_breakaway_improvement')):.4f} | {sf(r.get('mean_improvement')):.4f} | "
            f"{sf(r.get('mean_phase39_delta')):.4f} | {sf(r.get('mean_old_delta')):.4f} |"
        )

    lines += [
        "",
        "## Per-Ablation Per-Condition",
        "",
        "| Ablation | Condition | GT Δ | Old IDM Δ | New IDM Δ | Improvement | Improved |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for r in records:
        for cond in ["free", "hidden_breakaway_pin", "hidden_high_friction"]:
            row = r.get("condition_rows", {}).get(cond, {})
            lines.append(
                f"| `{r.get('ablation')}` | `{cond}` | {sf(row.get('gt_reference_delta')):.4f} | "
                f"{sf(row.get('old_idm_delta')):.4f} | {sf(row.get('phase39_idm_delta')):.4f} | "
                f"{sf(row.get('improvement')):.4f} | `{row.get('improved')}` |"
            )

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
        lines.append("| `PASS` | `none` | No issues found. |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- This is one-step matched-prefix diagnostic only.",
        "- No Phase4 or CPS was run.",
        "- No future DDPM was trained.",
        "- Checkpoints are local diagnostic artifacts and must not be committed.",
        "- If default geometry is robust, the next step is Phase3.10 controlled learned rollout retry, not Phase4/CPS.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict == "FAIL":
        raise SystemExit("[Phase3.9b][FAIL] ablation matrix did not support a robust repair")


if __name__ == "__main__":
    main()