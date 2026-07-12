#!/usr/bin/env python3
"""Select a stable Phase3.14b-r2 repair configuration using validation only."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ccda_phase3.phase314a_contract import (
    sha256_file,
    strict_json_dump,
    strict_json_load,
)
from ccda_phase3.phase314b_r2_contract import (
    CACHE_SHA256,
    MIN_STABLE_SEEDS_PER_CONFIG,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument(
        "--training",
        default="reports/phase3_14b_r2_training_summary.json",
    )
    parser.add_argument(
        "--selection",
        default="reports/phase3_14b_r2_selected_model.json",
    )
    parser.add_argument(
        "--summary",
        default="reports/phase3_14b_r2_selection_summary.json",
    )
    parser.add_argument(
        "--report",
        default="reports/phase3_14b_r2_selection_report.md",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    training = strict_json_load(root / args.training)
    if training.get("verdict") != "PASS":
        raise RuntimeError("r2 training did not complete")
    if training.get("cache_sha256") != CACHE_SHA256:
        raise RuntimeError("r2 training used another cache")

    by_config: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for run in training["runs"]:
        checkpoint = root / str(run["checkpoint"])
        if sha256_file(checkpoint) != run["checkpoint_sha256"]:
            raise RuntimeError(
                f"checkpoint SHA256 mismatch: {checkpoint}"
            )
        by_config[str(run["repair_config"])].append(run)

    configuration_summary = {}
    eligible_configs = []
    for name, runs in sorted(by_config.items()):
        stable = [
            run
            for run in runs
            if bool(run["stability_gate"]["stable"])
        ]
        median_best8 = (
            float(
                np.median(
                    [
                        run["stability_gate"]["values"]["best8"]
                        for run in stable
                    ]
                )
            )
            if stable
            else None
        )
        median_k1 = (
            float(
                np.median(
                    [
                        run["stability_gate"]["values"]["k1"]
                        for run in stable
                    ]
                )
            )
            if stable
            else None
        )
        eligible = len(stable) >= MIN_STABLE_SEEDS_PER_CONFIG
        summary = {
            "run_count": len(runs),
            "stable_run_count": len(stable),
            "eligible": eligible,
            "median_stable_best8": median_best8,
            "median_stable_k1": median_k1,
            "stable_seeds": [
                int(run["training_seed"]) for run in stable
            ],
        }
        configuration_summary[name] = summary
        if eligible:
            eligible_configs.append((name, summary, stable))

    if not eligible_configs:
        payload = {
            "verdict": "FAIL",
            "root_cause": "phase314b_r2_no_stable_configuration",
            "cache_sha256": CACHE_SHA256,
            "minimum_stable_seeds": MIN_STABLE_SEEDS_PER_CONFIG,
            "configurations": configuration_summary,
            "selected_model": None,
            "test_used_for_selection": False,
            "test_evaluation_allowed": False,
            "idm": False,
            "phase4": False,
            "cps": False,
        }
        strict_json_dump(root / args.summary, payload)
        lines = [
            "# Phase3.14b-r2 Validation Selection",
            "",
            "- Verdict: `FAIL`",
            "- Root cause: `phase314b_r2_no_stable_configuration`",
            "- Test evaluation allowed: `False`",
        ]
        (root / args.report).write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        raise SystemExit(2)

    selected_config_name, selected_config_summary, stable_runs = min(
        eligible_configs,
        key=lambda item: (
            item[1]["median_stable_best8"],
            item[1]["median_stable_k1"],
            item[0],
        ),
    )
    selected_run = min(
        stable_runs,
        key=lambda run: (
            run["stability_gate"]["values"]["best8"],
            run["stability_gate"]["values"]["k1"],
            int(run["training_seed"]),
        ),
    )
    selection = {
        "selection_version": "phase3_14b_r2_val_only_v1",
        "cache_sha256": CACHE_SHA256,
        "repair_config": selected_run["repair_config"],
        "prediction_type": selected_run["prediction_type"],
        "schedule_kind": selected_run["schedule_kind"],
        "training_seed": selected_run["training_seed"],
        "checkpoint": selected_run["checkpoint"],
        "checkpoint_sha256": selected_run["checkpoint_sha256"],
        "best_epoch": selected_run["best_epoch"],
        "scheduler": selected_run["scheduler"],
        "stability_gate": selected_run["stability_gate"],
        "configuration_summary": selected_config_summary,
        "selection_split": "val",
        "selection_subset": (
            "full_horizon_pre_engagement_paired"
        ),
        "test_used_for_selection": False,
        "test_evaluation_allowed": True,
    }
    strict_json_dump(root / args.selection, selection)

    payload = {
        "verdict": "PASS",
        "root_cause": "phase314b_r2_stable_configuration_selected",
        "cache_sha256": CACHE_SHA256,
        "minimum_stable_seeds": MIN_STABLE_SEEDS_PER_CONFIG,
        "configurations": configuration_summary,
        "selected_model": selection,
        "test_used_for_selection": False,
        "test_evaluation_allowed": True,
        "idm": False,
        "phase4": False,
        "cps": False,
    }
    strict_json_dump(root / args.summary, payload)

    lines = [
        "# Phase3.14b-r2 Validation Selection",
        "",
        "- Verdict: `PASS`",
        (
            "- Root cause: "
            "`phase314b_r2_stable_configuration_selected`"
        ),
        f"- Selected config: `{selection['repair_config']}`",
        f"- Selected seed: `{selection['training_seed']}`",
        "- Test used for selection: `False`",
        "- Test evaluation allowed: `True`",
        "",
        "| Config | Stable seeds | Eligible | Median best-8 |",
        "|---|---:|---|---:|",
    ]
    for name, item in configuration_summary.items():
        lines.append(
            f"| `{name}` | {item['stable_run_count']} | "
            f"`{item['eligible']}` | "
            f"{item['median_stable_best8']} |"
        )
    (root / args.report).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
