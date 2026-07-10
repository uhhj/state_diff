#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import phase3_12c_matched_reset_common as common


def sf(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def si(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    array = np.asarray(
        [
            float(value)
            for value in values
            if math.isfinite(float(value))
        ],
        dtype=np.float64,
    )

    if array.size == 0:
        return float("nan"), float("nan")

    if array.size == 1:
        value = float(array[0])
        return value, value

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        array.size,
        size=(int(samples), array.size),
    )
    means = np.mean(array[indices], axis=1)

    lower = float(
        np.quantile(means, alpha / 2.0)
    )
    upper = float(
        np.quantile(means, 1.0 - alpha / 2.0)
    )
    return lower, upper


def write_csv(
    path: Path,
    rows: List[Dict[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fields
            })


def integrity_audit(
    rows: List[Dict[str, str]],
    *,
    selectors: Sequence[str],
    conditions: Sequence[str],
    visible_seeds: Sequence[int],
    baseline_selector: str,
    max_abs_threshold: float,
    mae_threshold: float,
    fraction_threshold: float,
    curve_threshold: float,
) -> Tuple[
    bool,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    issues: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []

    okay = [
        row
        for row in rows
        if (
            row.get("status") == "ok"
            and not row.get("failure_reason")
        )
    ]

    groups: Dict[
        Tuple[str, int],
        Dict[str, List[Dict[str, str]]],
    ] = defaultdict(lambda: defaultdict(list))

    for row in okay:
        key = (
            str(row.get("condition", "")),
            si(row.get("visible_seed")),
        )
        groups[key][
            str(row.get("selector", ""))
        ].append(row)

    hard_failure = False

    for condition in conditions:
        for visible_seed in visible_seeds:
            group = groups.get(
                (condition, int(visible_seed)),
                {},
            )

            for selector in selectors:
                count = len(group.get(selector, []))
                if count != 1:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "missing_or_duplicate_reset_row"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "count": count,
                        },
                    })

            if hard_failure and (
                baseline_selector not in group
                or len(
                    group.get(
                        baseline_selector,
                        [],
                    )
                )
                != 1
            ):
                continue

            baseline_rows = group.get(
                baseline_selector,
                [],
            )
            if len(baseline_rows) != 1:
                continue

            baseline = baseline_rows[0]

            try:
                baseline_state = (
                    common.parse_state_json(
                        baseline.get(
                            "initial_state_json"
                        )
                    )
                )
            except Exception as exc:
                hard_failure = True
                issues.append({
                    "level": "FAIL",
                    "name": (
                        "baseline_initial_state_parse_failed"
                    ),
                    "detail": {
                        "condition": condition,
                        "visible_seed": int(
                            visible_seed
                        ),
                        "error": repr(exc),
                    },
                })
                continue

            expected_pair_group = (
                f"phase3_12c_seed_{int(visible_seed)}"
            )

            for selector in selectors:
                candidates = group.get(selector, [])
                if len(candidates) != 1:
                    continue

                row = candidates[0]

                try:
                    state = common.parse_state_json(
                        row.get(
                            "initial_state_json"
                        )
                    )
                except Exception as exc:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "initial_state_parse_failed"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "error": repr(exc),
                        },
                    })
                    continue

                if state.shape != baseline_state.shape:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "initial_state_shape_mismatch"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "baseline_shape": list(
                                baseline_state.shape
                            ),
                            "selector_shape": list(
                                state.shape
                            ),
                        },
                    })
                    continue

                difference = state - baseline_state

                max_abs = float(
                    np.max(np.abs(difference))
                )
                mae = float(
                    np.mean(np.abs(difference))
                )
                fraction_diff = abs(
                    sf(row.get("initial_fraction"))
                    - sf(
                        baseline.get(
                            "initial_fraction"
                        )
                    )
                )
                curve_diff = abs(
                    sf(row.get("initial_curve"))
                    - sf(
                        baseline.get(
                            "initial_curve"
                        )
                    )
                )

                round_hash_equal = (
                    row.get(
                        "initial_state_round_sha256"
                    )
                    == baseline.get(
                        "initial_state_round_sha256"
                    )
                )

                pair_group_ok = (
                    row.get(
                        "initial_pair_group_logged"
                    )
                    == expected_pair_group
                    and row.get("pair_group")
                    == expected_pair_group
                )

                condition_log_ok = (
                    row.get(
                        "initial_condition_logged"
                    )
                    == condition
                )

                numeric_pass = (
                    max_abs <= max_abs_threshold
                    and mae <= mae_threshold
                    and fraction_diff
                    <= fraction_threshold
                    and curve_diff
                    <= curve_threshold
                )

                if not numeric_pass:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "matched_reset_numeric_mismatch"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "max_abs": max_abs,
                            "mae": mae,
                            "fraction_diff": (
                                fraction_diff
                            ),
                            "curve_diff": curve_diff,
                        },
                    })

                if not pair_group_ok:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "pair_group_metadata_mismatch"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "expected": (
                                expected_pair_group
                            ),
                            "spec_pair_group": row.get(
                                "pair_group"
                            ),
                            "logged_pair_group": (
                                row.get(
                                    "initial_pair_group_logged"
                                )
                            ),
                        },
                    })

                if not condition_log_ok:
                    hard_failure = True
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "condition_metadata_mismatch"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                            "logged": row.get(
                                "initial_condition_logged"
                            ),
                        },
                    })

                if numeric_pass and not round_hash_equal:
                    issues.append({
                        "level": "WARN",
                        "name": (
                            "rounded_hash_differs_"
                            "within_numeric_tolerance"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                        },
                    })

                pair_rows.append({
                    "condition": condition,
                    "visible_seed": int(
                        visible_seed
                    ),
                    "seed_cohort": (
                        common.seed_cohort(
                            int(visible_seed)
                        )
                    ),
                    "baseline_selector": (
                        baseline_selector
                    ),
                    "selector": selector,
                    "initial_state_max_abs_diff": (
                        max_abs
                    ),
                    "initial_state_mae": mae,
                    "initial_fraction_diff": (
                        fraction_diff
                    ),
                    "initial_curve_diff": (
                        curve_diff
                    ),
                    "round_hash_equal": int(
                        round_hash_equal
                    ),
                    "pair_group_ok": int(
                        pair_group_ok
                    ),
                    "condition_log_ok": int(
                        condition_log_ok
                    ),
                    "settle_steps_baseline": si(
                        baseline.get(
                            "settle_steps_used"
                        )
                    ),
                    "settle_steps_selector": si(
                        row.get(
                            "settle_steps_used"
                        )
                    ),
                    "settle_static_baseline": si(
                        baseline.get(
                            "settle_static"
                        ),
                        0,
                    ),
                    "settle_static_selector": si(
                        row.get(
                            "settle_static"
                        ),
                        0,
                    ),
                    "numeric_pass": int(
                        numeric_pass
                    ),
                })

    return (
        not hard_failure,
        pair_rows,
        issues,
    )


def analyze_reset(args: argparse.Namespace) -> None:
    root = Path(args.root)
    rows = read_csv_rows(root / args.input_csv)
    progress = read_json(root / args.progress_json)
    raw = read_json(root / args.raw_json)

    passed, pair_rows, issues = integrity_audit(
        rows,
        selectors=args.selectors,
        conditions=args.conditions,
        visible_seeds=args.visible_seeds,
        baseline_selector=args.baseline_selector,
        max_abs_threshold=args.max_abs_threshold,
        mae_threshold=args.mae_threshold,
        fraction_threshold=args.fraction_threshold,
        curve_threshold=args.curve_threshold,
    )

    if not rows:
        passed = False
        issues.append({
            "level": "FAIL",
            "name": "no_reset_rows",
            "detail": args.input_csv,
        })

    if progress.get("status") != "completed":
        passed = False
        issues.append({
            "level": "FAIL",
            "name": "reset_probe_not_completed",
            "detail": progress.get("status"),
        })

    non_static = [
        row
        for row in rows
        if si(row.get("settle_static"), 0) != 1
    ]
    if non_static:
        issues.append({
            "level": "WARN",
            "name": "reset_did_not_reach_static_threshold",
            "detail": len(non_static),
        })

    max_state_diff = max(
        [
            sf(
                row.get(
                    "initial_state_max_abs_diff"
                )
            )
            for row in pair_rows
        ]
        or [float("nan")]
    )
    max_fraction_diff = max(
        [
            sf(row.get("initial_fraction_diff"))
            for row in pair_rows
        ]
        or [float("nan")]
    )

    verdict = "PASS" if passed else "FAIL"
    root_cause = (
        "phase312c_matched_reset_integrity_passed"
        if passed
        else "phase312c_matched_reset_integrity_failed"
    )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_rows": len(rows),
        "num_pair_rows": len(pair_rows),
        "max_initial_state_abs_diff": (
            max_state_diff
        ),
        "max_initial_fraction_diff": (
            max_fraction_diff
        ),
        "thresholds": {
            "max_abs_threshold": (
                args.max_abs_threshold
            ),
            "mae_threshold": args.mae_threshold,
            "fraction_threshold": (
                args.fraction_threshold
            ),
            "curve_threshold": (
                args.curve_threshold
            ),
        },
        "issues": issues,
        "progress": progress,
        "raw": raw,
        "paired_selector_rollout_allowed": bool(
            passed
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md
    pairs_csv = root / args.pairs_csv

    out_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )

    pair_fields = [
        "condition",
        "visible_seed",
        "seed_cohort",
        "baseline_selector",
        "selector",
        "initial_state_max_abs_diff",
        "initial_state_mae",
        "initial_fraction_diff",
        "initial_curve_diff",
        "round_hash_equal",
        "pair_group_ok",
        "condition_log_ok",
        "settle_steps_baseline",
        "settle_steps_selector",
        "settle_static_baseline",
        "settle_static_selector",
        "numeric_pass",
    ]
    write_csv(pairs_csv, pair_rows, pair_fields)

    lines = [
        "# Phase3.12c Matched-Reset Integrity Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        f"- Reset rows: `{len(rows)}`",
        f"- Pair rows: `{len(pair_rows)}`",
        (
            "- Maximum initial state absolute "
            f"difference: `{max_state_diff}`"
        ),
        (
            "- Maximum initial fraction "
            f"difference: `{max_fraction_diff}`"
        ),
        (
            "- Paired selector rollout allowed: "
            f"`{passed}`"
        ),
        "",
        "## Integrity thresholds",
        "",
        "| Threshold | Value |",
        "|---|---:|",
        (
            "| Initial state max abs | "
            f"{args.max_abs_threshold} |"
        ),
        (
            "| Initial state MAE | "
            f"{args.mae_threshold} |"
        ),
        (
            "| Initial fraction diff | "
            f"{args.fraction_threshold} |"
        ),
        (
            "| Initial curve diff | "
            f"{args.curve_threshold} |"
        ),
        "",
        "## Pair summary",
        "",
        (
            "| Condition | Seed | Selector | "
            "Max abs | MAE | Fraction diff | "
            "Curve diff | Hash equal | Pass |"
        ),
        (
            "|---|---:|---|---:|---:|---:|"
            "---:|---:|---:|"
        ),
    ]

    for row in pair_rows:
        lines.append(
            f"| `{row['condition']}` | "
            f"{row['visible_seed']} | "
            f"`{row['selector']}` | "
            f"{sf(row['initial_state_max_abs_diff']):.8g} | "
            f"{sf(row['initial_state_mae']):.8g} | "
            f"{sf(row['initial_fraction_diff']):.8g} | "
            f"{sf(row['initial_curve_diff']):.8g} | "
            f"{row['round_hash_equal']} | "
            f"{row['numeric_pass']} |"
        )

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]

    if issues:
        for entry in issues:
            lines.append(
                f"| `{entry['level']}` | "
                f"`{entry['name']}` | "
                f"{str(entry['detail']).replace('|', '/')} |"
            )
    else:
        lines.append(
            "| `PASS` | `none` | "
            "All reset groups matched. |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "- Selector quality must not be analyzed "
            "unless this report is PASS."
        ),
        (
            "- Reset groups are keyed by "
            "`condition + visible_seed`."
        ),
        (
            "- Pair group is selector-independent and "
            "condition-independent."
        ),
        (
            "- Wall-clock cable settling was replaced "
            "with deterministic PyBullet stepping."
        ),
        "- No Phase4 or CPS was run.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))

    if not passed:
        raise SystemExit(
            "[Phase3.12c][FAIL] matched reset integrity failed"
        )


def analyze_rollout(args: argparse.Namespace) -> None:
    root = Path(args.root)
    episodes = read_csv_rows(root / args.input_csv)
    steps = read_csv_rows(root / args.step_csv)
    progress = read_json(root / args.progress_json)
    raw = read_json(root / args.raw_json)

    integrity_passed, integrity_pairs, integrity_issues = (
        integrity_audit(
            episodes,
            selectors=args.selectors,
            conditions=args.conditions,
            visible_seeds=args.visible_seeds,
            baseline_selector=(
                args.baseline_selector
            ),
            max_abs_threshold=(
                args.max_abs_threshold
            ),
            mae_threshold=args.mae_threshold,
            fraction_threshold=(
                args.fraction_threshold
            ),
            curve_threshold=(
                args.curve_threshold
            ),
        )
    )

    issues: List[Dict[str, Any]] = list(
        integrity_issues
    )

    if not episodes:
        integrity_passed = False
        issues.append({
            "level": "FAIL",
            "name": "no_episode_rows",
            "detail": args.input_csv,
        })

    if progress.get("status") != "completed":
        integrity_passed = False
        issues.append({
            "level": "FAIL",
            "name": "paired_rollout_not_completed",
            "detail": progress.get("status"),
        })

    okay = [
        row
        for row in episodes
        if (
            row.get("status") == "ok"
            and not row.get("failure_reason")
        )
    ]

    groups: Dict[
        Tuple[str, int],
        Dict[str, Dict[str, str]],
    ] = defaultdict(dict)

    duplicate_keys: List[Tuple[str, int, str]] = []

    for row in okay:
        key = (
            str(row.get("condition", "")),
            si(row.get("visible_seed")),
        )
        selector = str(row.get("selector", ""))

        if selector in groups[key]:
            duplicate_keys.append(
                (key[0], key[1], selector)
            )
        else:
            groups[key][selector] = row

    if duplicate_keys:
        integrity_passed = False
        issues.append({
            "level": "FAIL",
            "name": "duplicate_paired_episode_rows",
            "detail": duplicate_keys,
        })

    paired_rows: List[Dict[str, Any]] = []

    for condition in args.conditions:
        for visible_seed in args.visible_seeds:
            group = groups.get(
                (condition, int(visible_seed)),
                {},
            )

            baseline = group.get(
                args.baseline_selector
            )
            if baseline is None:
                integrity_passed = False
                issues.append({
                    "level": "FAIL",
                    "name": "missing_paired_baseline",
                    "detail": {
                        "condition": condition,
                        "visible_seed": int(
                            visible_seed
                        ),
                    },
                })
                continue

            baseline_delta = sf(
                baseline.get("delta_fraction")
            )
            baseline_final = sf(
                baseline.get("final_fraction")
            )
            baseline_success = sf(
                baseline.get("success")
            )

            for selector in args.selectors:
                row = group.get(selector)
                if row is None:
                    integrity_passed = False
                    issues.append({
                        "level": "FAIL",
                        "name": (
                            "missing_paired_selector"
                        ),
                        "detail": {
                            "condition": condition,
                            "visible_seed": int(
                                visible_seed
                            ),
                            "selector": selector,
                        },
                    })
                    continue

                selector_delta = sf(
                    row.get("delta_fraction")
                )
                selector_final = sf(
                    row.get("final_fraction")
                )
                selector_success = sf(
                    row.get("success")
                )

                paired_rows.append({
                    "condition": condition,
                    "visible_seed": int(
                        visible_seed
                    ),
                    "seed_cohort": (
                        common.seed_cohort(
                            int(visible_seed)
                        )
                    ),
                    "baseline_selector": (
                        args.baseline_selector
                    ),
                    "selector": selector,
                    "baseline_delta": baseline_delta,
                    "selector_delta": selector_delta,
                    "paired_delta_gain": (
                        selector_delta
                        - baseline_delta
                    ),
                    "baseline_final": baseline_final,
                    "selector_final": selector_final,
                    "paired_final_gain": (
                        selector_final
                        - baseline_final
                    ),
                    "baseline_success": (
                        baseline_success
                    ),
                    "selector_success": (
                        selector_success
                    ),
                    "paired_success_gain": (
                        selector_success
                        - baseline_success
                    ),
                    "initial_fraction": sf(
                        row.get(
                            "initial_fraction"
                        )
                    ),
                    "selector_pull": sf(
                        row.get(
                            "mean_pull_len"
                        )
                    ),
                    "baseline_pull": sf(
                        baseline.get(
                            "mean_pull_len"
                        )
                    ),
                    "selector_future_match": sf(
                        row.get(
                            "mean_future_match"
                        )
                    ),
                    "baseline_future_match": sf(
                        baseline.get(
                            "mean_future_match"
                        )
                    ),
                })

    selector_summary: List[Dict[str, Any]] = []

    for condition in args.conditions:
        for selector in args.selectors:
            rows = [
                row
                for row in paired_rows
                if (
                    row["condition"] == condition
                    and row["selector"] == selector
                )
            ]

            gains = [
                sf(row["paired_delta_gain"])
                for row in rows
            ]
            finite_gains = [
                gain
                for gain in gains
                if math.isfinite(gain)
            ]

            ci_low, ci_high = bootstrap_mean_ci(
                finite_gains,
                samples=args.bootstrap_samples,
                seed=(
                    args.bootstrap_seed
                    + sum(ord(char) for char in condition)
                    + sum(ord(char) for char in selector)
                ),
            )

            positive = sum(
                gain > args.tie_tolerance
                for gain in finite_gains
            )
            negative = sum(
                gain < -args.tie_tolerance
                for gain in finite_gains
            )
            ties = (
                len(finite_gains)
                - positive
                - negative
            )

            mean_gain = common.finite_mean(
                finite_gains
            )
            median_gain = common.finite_median(
                finite_gains
            )

            required_positive = int(
                math.ceil(
                    args.min_positive_fraction
                    * max(1, len(finite_gains))
                )
            )

            robust_improvement = (
                selector != args.baseline_selector
                and len(finite_gains)
                >= args.min_pairs
                and math.isfinite(mean_gain)
                and mean_gain
                > args.improve_threshold
                and math.isfinite(ci_low)
                and ci_low > 0.0
                and positive >= required_positive
            )

            selector_summary.append({
                "condition": condition,
                "selector": selector,
                "num_pairs": len(finite_gains),
                "mean_delta": common.finite_mean(
                    [
                        sf(row["selector_delta"])
                        for row in rows
                    ]
                ),
                "baseline_mean_delta": (
                    common.finite_mean(
                        [
                            sf(row["baseline_delta"])
                            for row in rows
                        ]
                    )
                ),
                "paired_mean_gain": mean_gain,
                "paired_median_gain": median_gain,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "positive_pairs": positive,
                "negative_pairs": negative,
                "ties": ties,
                "required_positive_pairs": (
                    required_positive
                ),
                "robust_improvement": bool(
                    robust_improvement
                ),
                "mean_selector_pull": (
                    common.finite_mean(
                        [
                            sf(row["selector_pull"])
                            for row in rows
                        ]
                    )
                ),
                "mean_baseline_pull": (
                    common.finite_mean(
                        [
                            sf(row["baseline_pull"])
                            for row in rows
                        ]
                    )
                ),
                "mean_selector_future_match": (
                    common.finite_mean(
                        [
                            sf(
                                row[
                                    "selector_future_match"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),
                "mean_baseline_future_match": (
                    common.finite_mean(
                        [
                            sf(
                                row[
                                    "baseline_future_match"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),
            })

    cohort_summary: List[Dict[str, Any]] = []

    cohorts = sorted({
        str(row["seed_cohort"])
        for row in paired_rows
    })

    for cohort in cohorts:
        for condition in args.conditions:
            for selector in args.selectors:
                rows = [
                    row
                    for row in paired_rows
                    if (
                        row["seed_cohort"] == cohort
                        and row["condition"]
                        == condition
                        and row["selector"]
                        == selector
                    )
                ]

                gains = [
                    sf(row["paired_delta_gain"])
                    for row in rows
                ]

                cohort_summary.append({
                    "seed_cohort": cohort,
                    "condition": condition,
                    "selector": selector,
                    "num_pairs": len(gains),
                    "paired_mean_gain": (
                        common.finite_mean(gains)
                    ),
                    "paired_median_gain": (
                        common.finite_median(gains)
                    ),
                    "positive_pairs": sum(
                        gain > args.tie_tolerance
                        for gain in gains
                        if math.isfinite(gain)
                    ),
                    "negative_pairs": sum(
                        gain < -args.tie_tolerance
                        for gain in gains
                        if math.isfinite(gain)
                    ),
                })

    primary = args.primary_condition

    summary_lookup = {
        (
            row["condition"],
            row["selector"],
        ): row
        for row in selector_summary
    }

    condition_upper = summary_lookup.get(
        (primary, "condition_nearest_upper"),
        {},
    )
    state_proxy = summary_lookup.get(
        (primary, "proxy_state_motion_nn"),
        {},
    )
    compat_proxy = summary_lookup.get(
        (
            primary,
            "proxy_combined_topk_action_geom",
        ),
        {},
    )

    condition_upper_supported = bool(
        condition_upper.get(
            "robust_improvement",
            False,
        )
    )
    state_proxy_supported = bool(
        state_proxy.get(
            "robust_improvement",
            False,
        )
    )
    compat_proxy_supported = bool(
        compat_proxy.get(
            "robust_improvement",
            False,
        )
    )

    if not integrity_passed:
        verdict = "FAIL"
        root_cause = (
            "phase312c_matched_reset_integrity_failed"
        )

    else:
        verdict = "WARN"

        if compat_proxy_supported:
            root_cause = (
                "phase312c_observable_compat_"
                "selector_supported"
            )
            issues.append({
                "level": "WARN",
                "name": (
                    "observable_compat_selector_"
                    "robustly_improves_primary"
                ),
                "detail": compat_proxy,
            })

        elif state_proxy_supported:
            root_cause = (
                "phase312c_observable_state_"
                "motion_proxy_supported"
            )
            issues.append({
                "level": "WARN",
                "name": (
                    "observable_state_motion_proxy_"
                    "robustly_improves_primary"
                ),
                "detail": state_proxy,
            })

        elif condition_upper_supported:
            root_cause = (
                "phase312c_condition_upper_bound_"
                "robustly_supported"
            )
            issues.append({
                "level": "WARN",
                "name": (
                    "condition_upper_bound_"
                    "robustly_improves_primary"
                ),
                "detail": condition_upper,
            })

        else:
            root_cause = (
                "phase312c_no_selector_robustly_"
                "improves_ddpm_under_paired_reset"
            )
            issues.append({
                "level": "WARN",
                "name": (
                    "no_selector_robustly_improves_"
                    "primary_under_paired_reset"
                ),
                "detail": {
                    "condition_upper": condition_upper,
                    "state_proxy": state_proxy,
                    "compat_proxy": compat_proxy,
                },
            })

    paired_fields = [
        "condition",
        "visible_seed",
        "seed_cohort",
        "baseline_selector",
        "selector",
        "initial_fraction",
        "baseline_delta",
        "selector_delta",
        "paired_delta_gain",
        "baseline_final",
        "selector_final",
        "paired_final_gain",
        "baseline_success",
        "selector_success",
        "paired_success_gain",
        "baseline_pull",
        "selector_pull",
        "baseline_future_match",
        "selector_future_match",
    ]

    summary_fields = [
        "condition",
        "selector",
        "num_pairs",
        "mean_delta",
        "baseline_mean_delta",
        "paired_mean_gain",
        "paired_median_gain",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "positive_pairs",
        "negative_pairs",
        "ties",
        "required_positive_pairs",
        "robust_improvement",
        "mean_selector_pull",
        "mean_baseline_pull",
        "mean_selector_future_match",
        "mean_baseline_future_match",
    ]

    cohort_fields = [
        "seed_cohort",
        "condition",
        "selector",
        "num_pairs",
        "paired_mean_gain",
        "paired_median_gain",
        "positive_pairs",
        "negative_pairs",
    ]

    write_csv(
        root / args.pairs_csv,
        paired_rows,
        paired_fields,
    )
    write_csv(
        root / args.selector_summary_csv,
        selector_summary,
        summary_fields,
    )
    write_csv(
        root / args.cohort_summary_csv,
        cohort_summary,
        cohort_fields,
    )

    payload = {
        "verdict": verdict,
        "root_cause": root_cause,
        "num_episode_rows": len(episodes),
        "num_step_rows": len(steps),
        "matched_reset_integrity_passed": bool(
            integrity_passed
        ),
        "integrity_pair_rows": len(
            integrity_pairs
        ),
        "paired_rows": len(paired_rows),
        "selector_summary": selector_summary,
        "cohort_summary": cohort_summary,
        "diagnostic_flags": {
            "condition_upper_supported": (
                condition_upper_supported
            ),
            "state_proxy_supported": (
                state_proxy_supported
            ),
            "compat_proxy_supported": (
                compat_proxy_supported
            ),
        },
        "paired_criteria": {
            "baseline_selector": (
                args.baseline_selector
            ),
            "primary_condition": (
                args.primary_condition
            ),
            "improve_threshold": (
                args.improve_threshold
            ),
            "min_pairs": args.min_pairs,
            "min_positive_fraction": (
                args.min_positive_fraction
            ),
            "bootstrap_samples": (
                args.bootstrap_samples
            ),
            "bootstrap_ci_requires_low_above_zero": (
                True
            ),
        },
        "issues": issues,
        "progress": progress,
        "raw": raw,
        "important_note": (
            "Main selector comparisons use paired "
            "delta_fraction under matched condition + seed. "
            "condition_nearest_upper uses condition labels "
            "and remains a diagnostic upper bound."
        ),
        "recommendation": (
            "Do not enter Phase4/CPS. Base the next step "
            "only on matched-reset paired evidence."
        ),
    }

    out_json = root / args.out_json
    out_md = root / args.out_md

    out_json.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
    )

    lines = [
        "# Phase3.12c Matched-Reset Paired Selector Report",
        "",
        "## Verdict",
        "",
        f"- Verdict: `{verdict}`",
        f"- Root cause: `{root_cause}`",
        (
            "- Matched-reset integrity passed: "
            f"`{integrity_passed}`"
        ),
        f"- Episode rows: `{len(episodes)}`",
        f"- Step rows: `{len(steps)}`",
        f"- Paired rows: `{len(paired_rows)}`",
        (
            "- Progress status: "
            f"`{progress.get('status', 'missing')}`"
        ),
        "",
        "## Paired selector summary",
        "",
        (
            "| Condition | Selector | Pairs | "
            "Mean Δ | Baseline mean Δ | "
            "Paired mean gain | Median gain | "
            "95% CI | Positive | Negative | "
            "Robust |"
        ),
        (
            "|---|---|---:|---:|---:|---:|---:|"
            "---|---:|---:|---:|"
        ),
    ]

    for row in selector_summary:
        lines.append(
            f"| `{row['condition']}` | "
            f"`{row['selector']}` | "
            f"{row['num_pairs']} | "
            f"{sf(row['mean_delta']):.4f} | "
            f"{sf(row['baseline_mean_delta']):.4f} | "
            f"{sf(row['paired_mean_gain']):.4f} | "
            f"{sf(row['paired_median_gain']):.4f} | "
            f"[{sf(row['bootstrap_ci_low']):.4f}, "
            f"{sf(row['bootstrap_ci_high']):.4f}] | "
            f"{row['positive_pairs']} | "
            f"{row['negative_pairs']} | "
            f"`{row['robust_improvement']}` |"
        )

    lines += [
        "",
        "## Seed-cohort comparison",
        "",
        (
            "| Cohort | Condition | Selector | "
            "Pairs | Paired mean gain | "
            "Median gain | Positive | Negative |"
        ),
        (
            "|---|---|---|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in cohort_summary:
        lines.append(
            f"| `{row['seed_cohort']}` | "
            f"`{row['condition']}` | "
            f"`{row['selector']}` | "
            f"{row['num_pairs']} | "
            f"{sf(row['paired_mean_gain']):.4f} | "
            f"{sf(row['paired_median_gain']):.4f} | "
            f"{row['positive_pairs']} | "
            f"{row['negative_pairs']} |"
        )

    lines += [
        "",
        "## Issues",
        "",
        "| Level | Name | Detail |",
        "|---|---|---|",
    ]

    if issues:
        for entry in issues:
            lines.append(
                f"| `{entry['level']}` | "
                f"`{entry['name']}` | "
                f"{str(entry['detail']).replace('|', '/')} |"
            )
    else:
        lines.append(
            "| `PASS` | `none` | No issues. |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        (
            "- Initial state equivalence is a hard "
            "prerequisite for selector comparison."
        ),
        (
            "- Main gains are paired differences in "
            "`delta_fraction`, not unpaired final fractions."
        ),
        (
            "- The two historical seed blocks are "
            "reported separately."
        ),
        (
            "- `condition_nearest_upper` uses true "
            "condition labels and is not deployable."
        ),
        "- No model training was run.",
        "- No future DDPM training was run.",
        "- No Phase4 or CPS was run.",
    ]

    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))

    if verdict == "FAIL":
        raise SystemExit(
            "[Phase3.12c][FAIL] paired analysis invalid"
        )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="/data/state_diff2",
    )
    parser.add_argument(
        "--stage",
        choices=["reset", "rollout"],
        required=True,
    )
    parser.add_argument(
        "--input_csv",
        required=True,
    )
    parser.add_argument(
        "--step_csv",
        default="",
    )
    parser.add_argument(
        "--progress_json",
        required=True,
    )
    parser.add_argument(
        "--raw_json",
        required=True,
    )

    parser.add_argument(
        "--selectors",
        nargs="+",
        default=common.SELECTORS,
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=common.CONDITIONS,
    )
    parser.add_argument(
        "--visible_seeds",
        nargs="+",
        type=int,
        default=common.VISIBLE_SEEDS,
    )
    parser.add_argument(
        "--baseline_selector",
        default="ddpm_mean",
    )
    parser.add_argument(
        "--primary_condition",
        default="hidden_breakaway_pin",
    )

    parser.add_argument(
        "--max_abs_threshold",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--mae_threshold",
        type=float,
        default=1e-7,
    )
    parser.add_argument(
        "--fraction_threshold",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--curve_threshold",
        type=float,
        default=1e-7,
    )

    parser.add_argument(
        "--improve_threshold",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--min_pairs",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--min_positive_fraction",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-9,
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=312012,
    )

    parser.add_argument(
        "--out_json",
        required=True,
    )
    parser.add_argument(
        "--out_md",
        required=True,
    )
    parser.add_argument(
        "--pairs_csv",
        required=True,
    )
    parser.add_argument(
        "--selector_summary_csv",
        default=(
            "reports/"
            "phase3_12c_selector_summary.csv"
        ),
    )
    parser.add_argument(
        "--cohort_summary_csv",
        default=(
            "reports/"
            "phase3_12c_cohort_summary.csv"
        ),
    )

    args = parser.parse_args()

    if args.stage == "reset":
        analyze_reset(args)
    else:
        if not args.step_csv:
            raise SystemExit(
                "--step_csv is required for rollout analysis"
            )
        analyze_rollout(args)


if __name__ == "__main__":
    main()
