#!/usr/bin/env python3
"""Audit fixed-step repeatability of exact CCDA counterfactual pairs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.calibration_common import (  # noqa: E402
    build_candidate_config,
)
from scripts.experiment2.phase0.common import (  # noqa: E402
    canonical_json_sha256,
    compute_pair_metrics,
)
from scripts.experiment2.phase0.determinism_common import (  # noqa: E402
    compare_repeat_metadata,
    compare_traces,
    max_numeric_error,
)
from scripts.experiment2.phase0.exact_counterfactual import (  # noqa: E402
    run_exact_counterfactual_pair,
)


def _git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _save_trace(path: Path, trace: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trace)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment2/phase0/deterministic_replay_audit.json",
    )
    parser.add_argument(
        "--output",
        default="reports/experiment2/phase0_hidden_friction/determinism",
    )
    parser.add_argument("--disp", action="store_true")
    args = parser.parse_args()

    audit_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    base_path = (REPO_ROOT / audit["base_config"]).resolve()
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    candidate_config = build_candidate_config(base_config, audit["candidate"])
    execution = dict(audit["execution"])
    seeds = [int(seed) for seed in audit["seeds"]]
    repeats = int(audit["repeats"])
    numeric_atol = float(audit["numeric_atol"])
    if repeats < 2:
        raise ValueError("repeats must be at least 2")

    seed_summaries: List[Dict[str, Any]] = []
    for seed in seeds:
        repeat_rows: List[Dict[str, Any]] = []
        repeat_traces: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []

        for repeat in range(repeats):
            group_id = f"det_{seed:06d}_r{repeat}"
            result = run_exact_counterfactual_pair(
                candidate_config,
                seed=seed,
                group_id=group_id,
                disp=args.disp,
                execution=execution,
            )
            free_trace, hidden_trace = result[0], result[1]
            free_meta, hidden_meta = result[2], result[3]
            pair_meta = result[5]
            metrics = compute_pair_metrics(
                free_trace,
                hidden_trace,
                free_meta,
                hidden_meta,
                hz=float(candidate_config["hz"]),
                trace_stride=int(candidate_config["trace_stride"]),
            )

            raw_dir = output_root / "raw" / f"seed_{seed:06d}"
            _save_trace(raw_dir / f"repeat_{repeat}_free.npz", free_trace)
            _save_trace(
                raw_dir / f"repeat_{repeat}_hidden_high_friction.npz",
                hidden_trace,
            )
            row = {
                "repeat": repeat,
                "group_id": group_id,
                **pair_meta,
                "metrics": metrics,
            }
            _write_json(raw_dir / f"repeat_{repeat}.json", row)
            repeat_rows.append(row)
            repeat_traces.append((free_trace, hidden_trace))
            print(
                f"completed seed={seed} repeat={repeat}: "
                f"free_len={pair_meta['free_trace_length']} "
                f"hidden_len={pair_meta['hidden_trace_length']} "
                f"FDE={metrics['main_branch_fde']:.8f}",
                flush=True,
            )

        metadata_comparison = compare_repeat_metadata(repeat_rows)
        free_comparisons = []
        hidden_comparisons = []
        for repeat in range(1, repeats):
            free_comparisons.append(
                {
                    "reference_repeat": 0,
                    "candidate_repeat": repeat,
                    **compare_traces(
                        repeat_traces[0][0],
                        repeat_traces[repeat][0],
                        numeric_atol,
                    ),
                }
            )
            hidden_comparisons.append(
                {
                    "reference_repeat": 0,
                    "candidate_repeat": repeat,
                    **compare_traces(
                        repeat_traces[0][1],
                        repeat_traces[repeat][1],
                        numeric_atol,
                    ),
                }
            )

        trace_pass = all(
            row["passed"] for row in free_comparisons + hidden_comparisons
        )
        byte_exact = all(
            row["byte_exact"] for row in free_comparisons + hidden_comparisons
        )
        metrics_keys = (
            "preload_end_max_abs_xy",
            "contact_impulse_gap",
            "main_branch_ade",
            "main_branch_fde",
            "post_main_branch_fde",
        )
        metric_spread = {
            key: {
                "min": float(
                    np.min([row["metrics"][key] for row in repeat_rows])
                ),
                "max": float(
                    np.max([row["metrics"][key] for row in repeat_rows])
                ),
                "range": float(
                    np.ptp([row["metrics"][key] for row in repeat_rows])
                ),
            }
            for key in metrics_keys
        }
        seed_summary = {
            "seed": seed,
            "repeats": repeat_rows,
            "metadata_comparison": metadata_comparison,
            "free_trace_comparisons": free_comparisons,
            "hidden_trace_comparisons": hidden_comparisons,
            "max_numeric_error": max_numeric_error(
                free_comparisons + hidden_comparisons
            ),
            "byte_exact": byte_exact,
            "metric_spread": metric_spread,
            "passed": bool(metadata_comparison["passed"] and trace_pass),
        }
        _write_json(
            output_root
            / "raw"
            / f"seed_{seed:06d}"
            / "seed_summary.json",
            seed_summary,
        )
        seed_summaries.append(seed_summary)

    overall_pass = all(row["passed"] for row in seed_summaries)
    summary = {
        "stage": "Experiment2 Phase 0D",
        "main_repository_sha": _git_sha(REPO_ROOT),
        "submodule_sha": _git_sha(SUBMODULE_ROOT),
        "audit_config": str(audit_path.relative_to(REPO_ROOT)),
        "audit_config_hash": canonical_json_sha256(audit),
        "candidate_id": audit["candidate"]["id"],
        "candidate_config_hash": canonical_json_sha256(candidate_config),
        "execution": execution,
        "seeds": seeds,
        "repeats": repeats,
        "numeric_atol": numeric_atol,
        "seed_results": seed_summaries,
        "passed": overall_pass,
        "interpretation": (
            "Fixed-step replay audit only. Parameter eligibility and formal "
            "CCDA gates remain pending."
        ),
    }
    _write_json(output_root / "determinism_summary.json", summary)

    lines = [
        "# Phase 0D Fixed-Step Determinism Audit",
        "",
        f"- Candidate: `{audit['candidate']['id']}`",
        f"- Seeds: `{seeds}`",
        f"- Repeats per seed: `{repeats}`",
        f"- Numeric tolerance: `{numeric_atol}`",
        f"- Overall pass: `{overall_pass}`",
    ]
    for row in seed_summaries:
        lines.append(
            f"- Seed {row['seed']}: pass={row['passed']}, "
            f"byte_exact={row['byte_exact']}, "
            f"max_error={row['max_numeric_error']}"
        )
    lines.append(
        "- Status: deterministic replay audit only; no formal CCDA claim."
    )
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(output_root / "determinism_summary.json")
    if not overall_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
