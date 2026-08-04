#!/usr/bin/env python3
"""Calibrate Hidden-Friction Cable using exact restored pairs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULE_ROOT = REPO_ROOT / "external" / "deformable-ravens"
for path in (REPO_ROOT, SUBMODULE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.experiment2.phase0.calibration_common import (  # noqa: E402
    build_candidate_config,
    rank_candidates,
    summarize_candidate,
)
from scripts.experiment2.phase0.common import (  # noqa: E402
    canonical_json_sha256,
    compute_pair_metrics,
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


def _trace_difference(
    first: Dict[str, np.ndarray],
    second: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    keys = (
        "bead_positions",
        "bead_velocities",
        "joint_positions",
        "joint_velocities",
        "ee_position",
        "contact_force_norm",
    )
    result: Dict[str, Any] = {}
    for key in keys:
        a = np.asarray(first[key])
        b = np.asarray(second[key])
        if a.shape != b.shape:
            result[key] = {
                "shape_match": False,
                "first_shape": list(a.shape),
                "second_shape": list(b.shape),
                "max_abs": None,
            }
        else:
            result[key] = {
                "shape_match": True,
                "first_shape": list(a.shape),
                "second_shape": list(b.shape),
                "max_abs": (
                    float(np.max(np.abs(a - b))) if a.size else 0.0
                ),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-config",
        default=(
            "configs/experiment2/phase0/"
            "hidden_friction_calibration.json"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "reports/experiment2/phase0_hidden_friction/calibration"
        ),
    )
    parser.add_argument("--disp", action="store_true")
    args = parser.parse_args()

    calibration_path = (REPO_ROOT / args.calibration_config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    base_path = (REPO_ROOT / calibration["base_config"]).resolve()
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in calibration["seeds"]]
    targets = dict(calibration["selection_targets"])

    summaries: List[Dict[str, Any]] = []
    for candidate in calibration["candidates"]:
        candidate_id = str(candidate["id"])
        candidate_config = build_candidate_config(
            base_config, candidate
        )
        candidate_dir = output_root / "raw" / candidate_id
        rows: List[Dict[str, Any]] = []

        for seed in seeds:
            group_id = f"hf_{seed:06d}"
            result = run_exact_counterfactual_pair(
                candidate_config,
                seed=seed,
                group_id=group_id,
                disp=args.disp,
            )
            (
                free_trace,
                hidden_trace,
                free_meta,
                hidden_meta,
                action_script,
                pair_meta,
            ) = result

            metrics = compute_pair_metrics(
                free_trace,
                hidden_trace,
                free_meta,
                hidden_meta,
                hz=float(candidate_config["hz"]),
                trace_stride=int(candidate_config["trace_stride"]),
            )
            row = {
                "candidate_id": candidate_id,
                "group_id": group_id,
                "seed": seed,
                **metrics,
                **pair_meta,
            }
            rows.append(row)

            _save_trace(
                candidate_dir / f"{group_id}_free.npz",
                free_trace,
            )
            _save_trace(
                candidate_dir
                / f"{group_id}_hidden_high_friction.npz",
                hidden_trace,
            )
            _write_json(
                candidate_dir / f"{group_id}_pair.json",
                {
                    "candidate_id": candidate_id,
                    "candidate_config": candidate_config,
                    "action_script": action_script,
                    "free_metadata": free_meta,
                    "hidden_metadata": hidden_meta,
                    "pair_metadata": pair_meta,
                    "metrics": metrics,
                },
            )
            print(
                f"{candidate_id} {group_id} "
                f"preload={metrics['preload_end_max_abs_xy']:.6f} "
                f"contact={metrics['contact_impulse_gap']:.6f} "
                f"fde={metrics['main_branch_fde']:.6f}",
                flush=True,
            )

        summary = summarize_candidate(
            candidate_id,
            candidate,
            rows,
            targets,
        )
        summary["candidate_config_hash"] = canonical_json_sha256(
            candidate_config
        )
        summaries.append(summary)
        _write_json(
            candidate_dir / "candidate_summary.json",
            summary,
        )

    ranked = rank_candidates(summaries)
    selected = ranked[0]
    selected_config = build_candidate_config(
        base_config, selected["candidate"]
    )
    selected_config["calibration"] = {
        "candidate_id": selected["candidate_id"],
        "eligible": bool(selected["eligible"]),
        "score": selected["score"],
        "source_calibration_config": str(
            calibration_path.relative_to(REPO_ROOT)
        ),
        "source_calibration_hash": canonical_json_sha256(
            calibration
        ),
        "selection_targets": targets,
        "note": (
            "Provisional Phase 0C selection; "
            "not a formal CCDA gate pass."
        ),
    }

    _write_json(output_root / "selected_config.json", selected_config)
    calibrated_path = (
        REPO_ROOT
        / "configs/experiment2/phase0/"
        "hidden_friction_cable_calibrated.json"
    )
    if selected["eligible"]:
        _write_json(calibrated_path, selected_config)

    det_cfg = calibration["determinism"]
    repeated = []
    pair_metadata = []
    for repeat in range(int(det_cfg["repeats"])):
        result = run_exact_counterfactual_pair(
            selected_config,
            seed=int(det_cfg["seed"]),
            group_id=f"det_{int(det_cfg['seed']):06d}_{repeat}",
            disp=args.disp,
        )
        repeated.append((result[0], result[1]))
        pair_metadata.append(result[5])

    comparisons = []
    for index in range(1, len(repeated)):
        comparisons.append(
            {
                "repeat_a": 0,
                "repeat_b": index,
                "free": _trace_difference(
                    repeated[0][0], repeated[index][0]
                ),
                "hidden": _trace_difference(
                    repeated[0][1], repeated[index][1]
                ),
            }
        )
    _write_json(
        output_root / "determinism.json",
        {
            "candidate_id": selected["candidate_id"],
            "seed": int(det_cfg["seed"]),
            "repeats": int(det_cfg["repeats"]),
            "trace_atol": float(det_cfg["trace_atol"]),
            "pair_metadata": pair_metadata,
            "comparisons": comparisons,
        },
    )

    top = {
        "stage": "Experiment2 Phase 0C",
        "main_repository_sha": _git_sha(REPO_ROOT),
        "submodule_sha": _git_sha(SUBMODULE_ROOT),
        "calibration_config": str(
            calibration_path.relative_to(REPO_ROOT)
        ),
        "calibration_config_hash": canonical_json_sha256(
            calibration
        ),
        "base_config": str(base_path.relative_to(REPO_ROOT)),
        "base_config_hash": canonical_json_sha256(base_config),
        "seeds": seeds,
        "targets": targets,
        "ranked_candidates": ranked,
        "selected_candidate_id": selected["candidate_id"],
        "selected_candidate_eligible": bool(selected["eligible"]),
        "selected_config_path": str(
            (output_root / "selected_config.json").relative_to(
                REPO_ROOT
            )
        ),
        "repo_calibrated_config_written": bool(
            selected["eligible"]
        ),
        "interpretation": (
            "Small exact-pair calibration only. "
            "Formal statistics and leakage classifiers remain pending."
        ),
    }
    _write_json(output_root / "calibration_summary.json", top)

    lines = [
        "# Phase 0C Exact-Pair Hidden-Friction Calibration",
        "",
        f"- Candidates: {len(ranked)}",
        f"- Seeds: {seeds}",
        f"- Selected: `{selected['candidate_id']}`",
        f"- Eligible: `{selected['eligible']}`",
        f"- Score: `{selected['score']}`",
        f"- Median metrics: `{selected['median']}`",
        "- Status: provisional calibration only.",
    ]
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(output_root / "calibration_summary.json")
    if not selected["eligible"]:
        print(
            "No candidate met every calibration target.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
