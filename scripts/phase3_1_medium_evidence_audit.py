#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

PRIMARY = "hidden_breakaway_pin"
DIAGNOSTIC = "hidden_pin"
FREE = "free"
REQUIRED_CONDITIONS = [FREE, DIAGNOSTIC, "hidden_high_friction", PRIMARY]
FORBIDDEN_NAMES = [
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
    "source_file",
    "ccda_pair_group",
    "visible_seed",
]


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None or x == "":
            return default
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["empty"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def finite(vals: Iterable[Any]) -> List[float]:
    out = [safe_float(v) for v in vals]
    return [v for v in out if math.isfinite(v)]


def mean(vals: Iterable[Any]) -> float:
    xs = finite(vals)
    return float(np.mean(xs)) if xs else float("nan")


def std(vals: Iterable[Any]) -> float:
    xs = finite(vals)
    return float(np.std(xs)) if xs else float("nan")


def q(vals: Iterable[Any], prob: float) -> float:
    xs = finite(vals)
    return float(np.quantile(xs, prob)) if xs else float("nan")


def ci_bootstrap_mean(vals: Iterable[Any], n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    xs = np.asarray(finite(vals), dtype=np.float64)
    if xs.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    samples = rng.choice(xs, size=(n_boot, xs.size), replace=True).mean(axis=1)
    return {"mean": float(xs.mean()), "lo": float(np.quantile(samples, 0.025)), "hi": float(np.quantile(samples, 0.975))}


def grouped(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    out: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[tuple(r.get(k, "") for k in keys)].append(r)
    return out


def nested_pair_max(canon: Dict[str, Any], stage: str, field: str) -> Any:
    stats = canon.get(stage, {}) if isinstance(canon, dict) else {}
    by_cond = stats.get("by_hidden_condition", {}) if isinstance(stats, dict) else {}
    vals = []
    for item in by_cond.values():
        block = item.get(field, {}) if isinstance(item, dict) else {}
        if isinstance(block, dict) and block.get("max") is not None:
            vals.append(float(block["max"]))
    return max(vals) if vals else None


def scalar_meta(data: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    raw = np.asarray(data["meta_json"])
    return json.loads(str(raw.item() if raw.shape == () else raw.reshape(-1)[0]))


def summarize_metric_rows(rows: List[Dict[str, str]]) -> Dict[str, float]:
    return {
        "count": len(rows),
        "wrong_branch_mean": mean(r.get("sample_wrong_branch_rate") for r in rows),
        "branch_accuracy_mean": mean(r.get("branch_accuracy") for r in rows),
        "p_free_mean": mean(r.get("p_free_branch") for r in rows),
        "p_primary_mean": mean(r.get("p_primary_hidden_branch", r.get("p_pin_branch")) for r in rows),
        "future_error_mean": mean(r.get("future_chamfer_to_true") for r in rows),
        "action_mse_mean": mean(r.get("action_mse") for r in rows),
        "action_ood_mean": mean(r.get("action_ood_score") for r in rows),
    }


def pair_delta_rows(rows: List[Dict[str, str]], primary: str) -> List[Dict[str, Any]]:
    out = []
    index = {}
    for r in rows:
        key = (r.get("baseline"), r.get("fold"), r.get("seed"), r.get("visible_seed"), r.get("window_t"), r.get("condition"))
        index[key] = r
    base_keys = sorted({(r.get("baseline"), r.get("fold"), r.get("seed"), r.get("visible_seed"), r.get("window_t")) for r in rows})
    for key in base_keys:
        free = index.get(key + (FREE,))
        hidden = index.get(key + (primary,))
        if not free or not hidden:
            continue
        out.append({
            "baseline": key[0],
            "fold": key[1],
            "seed": key[2],
            "visible_seed": key[3],
            "window_t": key[4],
            "delta_hidden_minus_free_sample_wrong_branch_rate": safe_float(hidden.get("sample_wrong_branch_rate")) - safe_float(free.get("sample_wrong_branch_rate")),
            "delta_hidden_minus_free_future_chamfer_to_true": safe_float(hidden.get("future_chamfer_to_true")) - safe_float(free.get("future_chamfer_to_true")),
            "delta_hidden_minus_free_action_mse": safe_float(hidden.get("action_mse")) - safe_float(free.get("action_mse")),
        })
    return out


def action_dim_review(data_path: Path, out_csv: Path) -> List[Dict[str, Any]]:
    data = np.load(data_path, allow_pickle=True)
    y = np.asarray(data["y_action"], dtype=np.float64)
    rows = []
    for i in range(y.shape[1]):
        col = y[:, i]
        rows.append({
            "dim": i,
            "mean": float(np.mean(col)),
            "std": float(np.std(col)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
            "p99_abs": float(np.quantile(np.abs(col), 0.99)),
            "near_zero_std": bool(np.std(col) < 1e-8),
        })
    write_csv(out_csv, rows)
    return rows


def ood_topk(rows: List[Dict[str, str]], out_csv: Path, k: int = 50) -> None:
    ordered = sorted(rows, key=lambda r: safe_float(r.get("action_ood_score"), -1.0), reverse=True)[:k]
    keep = []
    for r in ordered:
        keep.append({
            "baseline": r.get("baseline"),
            "fold": r.get("fold"),
            "seed": r.get("seed"),
            "condition": r.get("condition"),
            "visible_seed": r.get("visible_seed"),
            "window_t": r.get("window_t"),
            "action_ood_score": r.get("action_ood_score"),
            "action_mse": r.get("action_mse"),
        })
    write_csv(out_csv, keep)


def make_markdown(payload: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Phase3.1 Medium Evidence Audit",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Recommendation: {payload['recommendation']}",
        f"- Conditions: `{' '.join(payload['conditions'])}`",
        f"- Primary pair: `free_vs_{payload['meta_primary_hidden_condition']}`",
        f"- Diagnostic pair: `free_vs_{payload['meta_diagnostic_hidden_condition']}`",
        "",
        "## Gate Checks",
        "",
        "| Check | Result |",
        "|---|---|",
    ]
    for k, v in payload["checks"].items():
        lines.append(f"| `{k}` | `{v}` |")
    lines += [
        "",
        "## Primary Pair Statistics",
        "",
        "| Baseline | Condition | Count | Wrong-Branch | Branch-Accuracy | p_free | p_primary | Future Error | Action MSE | Action OOD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, stats in sorted(payload["primary_condition_summary"].items()):
        baseline, cond = key.split("||")
        lines.append(
            f"| `{baseline}` | `{cond}` | {stats['count']} | {stats['wrong_branch_mean']:.6g} | {stats['branch_accuracy_mean']:.6g} | "
            f"{stats['p_free_mean']:.6g} | {stats['p_primary_mean']:.6g} | {stats['future_error_mean']:.6g} | {stats['action_mse_mean']:.6g} | {stats['action_ood_mean']:.6g} |"
        )
    lines += [
        "",
        "## Paired Deltas",
        "",
        "| Baseline | Count | Delta wrong hidden-free | 95% CI | Delta future hidden-free | 95% CI | Delta action MSE hidden-free |",
        "|---|---:|---:|---|---:|---|---:|",
    ]
    for baseline, stats in sorted(payload["paired_delta_summary"].items()):
        dw = stats["delta_wrong_ci95"]
        df = stats["delta_future_ci95"]
        lines.append(
            f"| `{baseline}` | {stats['count']} | {stats['delta_wrong_mean']:.6g} | [{dw['lo']:.6g}, {dw['hi']:.6g}] | "
            f"{stats['delta_future_mean']:.6g} | [{df['lo']:.6g}, {df['hi']:.6g}] | {stats['delta_action_mse_mean']:.6g} |"
        )
    lines += [
        "",
        "## Diagnostic Hidden Pin",
        "",
        "| Baseline | Free future error | Hidden pin future error | Hidden pin - free |",
        "|---|---:|---:|---:|",
    ]
    for baseline, stats in sorted(payload["diagnostic_summary"].items()):
        lines.append(f"| `{baseline}` | {stats['free_future_error_mean']:.6g} | {stats['hidden_pin_future_error_mean']:.6g} | {stats['hidden_pin_minus_free_future_error']:.6g} |")
    lines += [
        "",
        "## Warnings And Failures",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]
    if payload["issues"]:
        for issue in payload["issues"]:
            lines.append(f"| `{issue['level']}` | `{issue['name']}` | {str(issue['detail']).replace('|','/')} |")
    else:
        lines.append("| `PASS` | `none` | No warnings or failures. |")
    lines += [
        "",
        "## Interpretation",
        "",
        "- No rollout, Phase4, or CPS was run by this audit.",
        "- `WARN` items are documented evidence-review items, not automatic FAILs.",
        "- Rollout smoke may be prepared only after user approval and explicit `PHASE3_ALLOW_ROLLOUT=1`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/state_diff2")
    ap.add_argument("--data", default="data/phase3_state_diff_windows/phase3_windows.npz")
    ap.add_argument("--pred_csv", default="reports/phase3_baseline_eval_predictions.csv")
    ap.add_argument("--eval_summary", default="reports/phase3_baseline_eval_summary.json")
    ap.add_argument("--leak_json", default="reports/phase3_input_leakage_summary.json")
    ap.add_argument("--integration_json", default="reports/phase3_condition_integration_audit_summary.json")
    ap.add_argument("--canonical_json", default="reports/phase3_canonicalization_summary.json")
    ap.add_argument("--action_debug_json", default="reports/phase3_action_idm_debug_summary.json")
    ap.add_argument("--sanity_json", default="reports/phase3_sanity_check_summary.json")
    ap.add_argument("--out_json", default="reports/phase3_1_medium_evidence_summary.json")
    ap.add_argument("--out_md", default="reports/phase3_1_medium_evidence_report.md")
    ap.add_argument("--out_pair_delta_csv", default="reports/phase3_1_primary_pair_deltas.csv")
    ap.add_argument("--out_action_dim_csv", default="reports/phase3_1_action_dim_review.csv")
    ap.add_argument("--out_ood_topk_csv", default="reports/phase3_1_idm_ood_topk.csv")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    data_path = root / args.data
    rows = read_csv(root / args.pred_csv)
    eval_summary = read_json(root / args.eval_summary)
    leak = read_json(root / args.leak_json)
    integration = read_json(root / args.integration_json)
    canon = read_json(root / args.canonical_json)
    action_debug = read_json(root / args.action_debug_json)
    sanity = read_json(root / args.sanity_json)
    data = np.load(data_path, allow_pickle=True)
    meta = scalar_meta(data)

    issues: List[Dict[str, Any]] = []
    checks: Dict[str, Any] = {}

    def add(level: str, name: str, detail: Any) -> None:
        issues.append({"level": level, "name": name, "detail": detail})

    conditions = list(meta.get("conditions", []))
    pred_conditions = sorted({r.get("condition", "") for r in rows})
    checks["conditions_match_required"] = conditions == REQUIRED_CONDITIONS or set(conditions) == set(REQUIRED_CONDITIONS)
    if set(conditions) != set(REQUIRED_CONDITIONS):
        add("FAIL", "conditions_not_required_set", conditions)
    if PRIMARY not in pred_conditions:
        add("FAIL", "primary_condition_missing_from_predictions", pred_conditions)
    if len(rows) < 300:
        add("FAIL", "prediction_rows_below_300", len(rows))

    backends = Counter(r.get("training_backend", "") for r in rows)
    checks["all_torch"] = set(backends.keys()) == {"torch"}
    if not checks["all_torch"]:
        add("FAIL", "non_torch_prediction_rows", dict(backends))

    primary_values = sorted({r.get("primary_hidden_condition", "") for r in rows})
    checks["prediction_primary_recorded"] = primary_values == [PRIMARY]
    if primary_values and primary_values != [PRIMARY]:
        add("FAIL", "prediction_primary_mismatch", primary_values)
    elif not primary_values:
        add("WARN", "prediction_primary_column_missing", "CSV lacks primary_hidden_condition; infer from primary_pair.")

    primary_pair_values = sorted({r.get("primary_pair", "") or r.get("primary_branch_pair", "") for r in rows})
    checks["primary_pair_correct"] = primary_pair_values == [f"free_vs_{PRIMARY}"]
    if "free_vs_hidden_pin" in primary_pair_values:
        add("FAIL", "primary_pair_still_hidden_pin", primary_pair_values)
    elif not checks["primary_pair_correct"]:
        add("WARN", "primary_pair_unexpected_values", primary_pair_values)

    checks["leakage_pass"] = leak.get("pass") is True
    if not checks["leakage_pass"]:
        add("FAIL", "input_leakage_not_pass", leak.get("pass"))
    checks["integration_pass"] = integration.get("verdict") == "PASS"
    if not checks["integration_pass"]:
        add("FAIL", "condition_integration_not_pass", integration.get("verdict"))

    post_paper = nested_pair_max(canon, "post_pair_stats", "paper_x_max_abs")
    post_sa = nested_pair_max(canon, "post_pair_stats", "state_action_x_max_abs")
    raw_paper = nested_pair_max(canon, "raw_pair_stats", "paper_x_max_abs")
    checks["canonical_post_zero"] = post_paper == 0.0 and post_sa == 0.0
    if not checks["canonical_post_zero"]:
        add("FAIL", "canonical_post_diff_nonzero_or_missing", {"post_paper": post_paper, "post_state_action": post_sa})
    if raw_paper is not None and raw_paper > 0.1:
        add("WARN", "canonicalization_strong_intervention", f"raw max pair paper_x diff={raw_paper}")

    feature_schema = meta.get("feature_schema", {})
    explicit_schema_text = json.dumps({k: v for k, v in feature_schema.items() if k != "forbidden_not_in_x"}, sort_keys=True)
    forbidden_explicit = [x for x in FORBIDDEN_NAMES if x in explicit_schema_text]
    checks["forbidden_not_in_explicit_feature_schema"] = not forbidden_explicit
    if forbidden_explicit:
        add("FAIL", "forbidden_metadata_in_feature_schema", forbidden_explicit)

    shapes = {
        "paper_x": list(data["paper_x"].shape),
        "state_action_x": list(data["state_action_x"].shape),
        "y_action": list(data["y_action"].shape),
    }
    checks["y_action_dim_14"] = len(shapes["y_action"]) == 2 and shapes["y_action"][1] == 14
    if not checks["y_action_dim_14"]:
        add("FAIL", "bad_y_action_shape", shapes["y_action"])

    exact_rate = action_debug.get("action_history_target_exact_match_rate")
    if exact_rate is None:
        exact_rate = (action_debug.get("state_action_extra_block") or {}).get("action_history_target_exact_match_rate")
    checks["action_history_no_exact_target_leak"] = exact_rate is None or safe_float(exact_rate, 0.0) == 0.0
    if not checks["action_history_no_exact_target_leak"]:
        add("FAIL", "action_history_target_exact_match_nonzero", exact_rate)

    checks["action_debug_not_fail"] = action_debug.get("verdict") != "FAIL"
    if not checks["action_debug_not_fail"]:
        add("FAIL", "action_debug_fail", action_debug.get("issues", []))
    checks["sanity_not_fail"] = sanity.get("verdict") != "FAIL"
    if not checks["sanity_not_fail"]:
        add("FAIL", "phase3_sanity_fail", sanity.get("issues", []))

    cond_summary = {}
    for (baseline, cond), rs in grouped([r for r in rows if r.get("condition") in {FREE, PRIMARY}], ["baseline", "condition"]).items():
        cond_summary[f"{baseline}||{cond}"] = summarize_metric_rows(rs)

    deltas = pair_delta_rows(rows, PRIMARY)
    write_csv(root / args.out_pair_delta_csv, deltas)
    paired_summary = {}
    for (baseline,), rs in grouped(deltas, ["baseline"]).items():
        wrong = [r["delta_hidden_minus_free_sample_wrong_branch_rate"] for r in rs]
        future = [r["delta_hidden_minus_free_future_chamfer_to_true"] for r in rs]
        action = [r["delta_hidden_minus_free_action_mse"] for r in rs]
        paired_summary[baseline] = {
            "count": len(rs),
            "delta_wrong_mean": mean(wrong),
            "delta_wrong_ci95": ci_bootstrap_mean(wrong),
            "delta_future_mean": mean(future),
            "delta_future_ci95": ci_bootstrap_mean(future),
            "delta_action_mse_mean": mean(action),
        }
        if abs(paired_summary[baseline]["delta_future_mean"]) < 0.005:
            add("WARN", "low_primary_future_error_contrast", f"{baseline}: hidden-free future error delta={paired_summary[baseline]['delta_future_mean']:.6f}")

    diag_summary = {}
    for baseline in sorted({r.get("baseline", "") for r in rows}):
        free_rows = [r for r in rows if r.get("baseline") == baseline and r.get("condition") == FREE]
        pin_rows = [r for r in rows if r.get("baseline") == baseline and r.get("condition") == DIAGNOSTIC]
        f = mean(r.get("future_chamfer_to_true") for r in free_rows)
        h = mean(r.get("future_chamfer_to_true") for r in pin_rows)
        diag_summary[baseline] = {"free_future_error_mean": f, "hidden_pin_future_error_mean": h, "hidden_pin_minus_free_future_error": h - f}

    baseline_compare = {}
    for cond in REQUIRED_CONDITIONS:
        p_rows = [r for r in rows if r.get("baseline") == "paper_state" and r.get("condition") == cond]
        s_rows = [r for r in rows if r.get("baseline") == "state_action" and r.get("condition") == cond]
        pf = mean(r.get("future_chamfer_to_true") for r in p_rows)
        sf = mean(r.get("future_chamfer_to_true") for r in s_rows)
        diff = abs(sf - pf)
        baseline_compare[cond] = {"paper_future_error": pf, "state_action_future_error": sf, "abs_diff": diff}
        if math.isfinite(diff) and diff < 1e-3:
            add("WARN", "paper_state_state_action_nearly_identical", f"{cond}: abs future error diff={diff:.6g}")

    action_rows = action_dim_review(data_path, root / args.out_action_dim_csv)
    near_zero_count = sum(1 for r in action_rows if r["near_zero_std"])
    if near_zero_count >= 8:
        add("WARN", "many_near_zero_action_dims", f"{near_zero_count}/14 dims std < 1e-8")

    ood_vals = finite(r.get("action_ood_score") for r in rows)
    ood_quantiles = {"p50": q(ood_vals, 0.50), "p90": q(ood_vals, 0.90), "p95": q(ood_vals, 0.95), "p99": q(ood_vals, 0.99), "max": max(ood_vals) if ood_vals else float("nan")}
    ood_topk(rows, root / args.out_ood_topk_csv)
    if math.isfinite(ood_quantiles["max"]) and ood_quantiles["max"] > 3.0:
        add("WARN", "idm_ood_max_high", ood_quantiles["max"])

    if action_debug.get("verdict") == "WARN":
        add("WARN", "action_debug_warn", "See phase3_action_idm_debug_report.md")
    if sanity.get("verdict") == "WARN":
        add("WARN", "phase3_sanity_warn", "See phase3_sanity_check_report.md")

    has_fail = any(i["level"] == "FAIL" for i in issues)
    has_warn = any(i["level"] == "WARN" for i in issues)
    verdict = "FAIL" if has_fail else ("WARN" if has_warn else "PASS")
    payload = {
        "verdict": verdict,
        "checks": checks,
        "issues": issues,
        "prediction_rows": len(rows),
        "backend_counts": dict(backends),
        "conditions": conditions,
        "prediction_conditions": pred_conditions,
        "data_shapes": shapes,
        "meta_primary_hidden_condition": meta.get("primary_hidden_condition"),
        "meta_diagnostic_hidden_condition": meta.get("diagnostic_hidden_condition"),
        "primary_condition_summary": cond_summary,
        "paired_delta_summary": paired_summary,
        "diagnostic_summary": diag_summary,
        "baseline_compare": baseline_compare,
        "canonicalization": {"raw_pair_paper_max": raw_paper, "post_pair_paper_max": post_paper, "post_pair_state_action_max": post_sa},
        "action_dim_near_zero_count": near_zero_count,
        "action_ood_quantiles": ood_quantiles,
        "outputs": {"pair_delta_csv": args.out_pair_delta_csv, "action_dim_csv": args.out_action_dim_csv, "ood_topk_csv": args.out_ood_topk_csv},
        "recommendation": "No FAIL. Rollout smoke may be prepared, but only after user approval." if not has_fail else "Do not run rollout or Phase4 until FAIL items are fixed.",
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    make_markdown(payload, out_md)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if verdict == "FAIL":
        raise SystemExit("[Phase3.1][FAIL] medium evidence audit failed")


if __name__ == "__main__":
    main()
