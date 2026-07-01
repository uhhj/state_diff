from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_diff.env.ccda_hose.audit import (
    AuditThresholds,
    compute_pair_metrics,
    plot_pair_metric_bars,
    write_pair_metrics_csv,
)
from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env import scripted_rollout, summarize_rollouts


def candidate_configs(base: HoseEnvConfig) -> List[HoseEnvConfig]:
    """Small calibration grid.

    The objective is not physical realism. The objective is to avoid the bad
    regime where jam force is huge but final insertion depth stays almost the
    same as free insertion. The first tier deliberately probes entrance-near
    hidden jams, because the pair audit looks at the first future window after
    the pre-insertion audit state.
    """
    candidates: List[HoseEnvConfig] = []

    for stiffness in [0.02, 0.03, 0.04]:
        for max_delta in [0.0008, 0.0010, 0.0012]:
            for jam_x, jam_y, jam_size_x, jam_size_y in [
                (0.015, 0.0000, 0.090, 0.0090),
                (0.020, 0.0005, 0.100, 0.0095),
                (0.025, 0.0010, 0.100, 0.0100),
            ]:
                candidates.append(
                    replace(
                        base,
                        hose_joint_stiffness=stiffness,
                        hose_joint_damping=0.035,
                        max_delta_per_env_step=max_delta,
                        push_steps=140,
                        push_distance=0.120,
                        jam_block_x=jam_x,
                        jam_block_y=jam_y,
                        jam_block_z=base.socket_center_z,
                        jam_block_size=(jam_size_x, jam_size_y, 0.0110),
                        jam_friction=(18.0, 0.30, 0.08),
                        lateral_offset_threshold=0.0020,
                    )
                )

    # Keep the original milder search tier as a fallback so the calibration
    # report can still compare against less intrusive hidden-jam settings.
    for stiffness in [0.08, 0.12, 0.16]:
        for max_delta in [0.0011, 0.0015, 0.0019]:
            for jam_y in [0.0048, 0.0058, 0.0068]:
                for jam_friction in [2.0, 3.2, 4.5]:
                    candidates.append(
                        replace(
                            base,
                            hose_joint_stiffness=stiffness,
                            hose_joint_damping=0.045,
                            max_delta_per_env_step=max_delta,
                            push_steps=110,
                            push_distance=0.105,
                            jam_block_x=0.020,
                            jam_block_y=jam_y,
                            jam_block_z=base.socket_center_z,
                            jam_block_size=(0.014, 0.0028, 0.0075),
                            jam_friction=(jam_friction, 0.08, 0.02),
                        )
                    )
    return candidates


def _save_npz(rollouts, out_path: Path) -> Dict[str, np.ndarray]:
    traces = [r["trace"] for r in rollouts]
    keys = list(traces[0].keys())
    arrays = {}
    for k in keys:
        arrays[k] = np.stack([t[k] for t in traces], axis=0)
    arrays["condition"] = np.array([r["condition"] for r in rollouts])
    arrays["seed"] = np.array([int(r["seed"]) for r in rollouts])
    arrays["audit_index"] = np.array([int(r["audit_index"]) for r in rollouts])
    arrays["final_success"] = np.array([float(r["final_success"]) for r in rollouts])
    arrays["final_branch"] = np.array([str(r["final_branch"]) for r in rollouts])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    return arrays


def evaluate_config(cfg: HoseEnvConfig, episodes_per_condition: int, seed: int) -> Dict[str, object]:
    rollouts = []
    for i in range(episodes_per_condition):
        rollouts.append(scripted_rollout(FREE_INSERT, seed=seed + i, config=cfg, record_frames=False))
        rollouts.append(scripted_rollout(RIGHT_HIDDEN_JAM, seed=seed + 10000 + i, config=cfg, record_frames=False))

    summary = summarize_rollouts(rollouts)
    by = summary["by_condition"]
    a = by.get(FREE_INSERT, {})
    b = by.get(RIGHT_HIDDEN_JAM, {})

    traces = [r["trace"] for r in rollouts]
    data = {k: np.stack([t[k] for t in traces], axis=0) for k in traces[0].keys()}
    data["condition"] = np.array([r["condition"] for r in rollouts])
    data["seed"] = np.array([int(r["seed"]) for r in rollouts])
    data["audit_index"] = np.array([int(r["audit_index"]) for r in rollouts])
    data["final_success"] = np.array([float(r["final_success"]) for r in rollouts])
    data["final_branch"] = np.array([str(r["final_branch"]) for r in rollouts])

    thresholds = AuditThresholds()
    rows, pair_summary = compute_pair_metrics(
        data,
        history_steps=8,
        future_steps=32,
        thresholds=thresholds,
        max_pairs_per_type=1000,
    )
    ab = pair_summary["by_pair_type"].get("A_vs_B", {})

    sr_a = float(a.get("success_rate", 0.0))
    sr_b = float(b.get("success_rate", 0.0))
    depth_gap = float(a.get("mean_final_insert_depth", 0.0)) - float(b.get("mean_final_insert_depth", 0.0))
    curvature_gap = float(b.get("mean_final_max_curvature", 0.0)) - float(a.get("mean_final_max_curvature", 0.0))
    lateral_gap = abs(float(b.get("mean_final_lateral_offset", 0.0))) - abs(float(a.get("mean_final_lateral_offset", 0.0)))
    ab_future = float(ab.get("mean_d_future", 0.0))
    ab_vis = float(ab.get("mean_d_vis", 999.0))
    ab_contact = float(ab.get("mean_d_contact", 0.0))
    success_gap = sr_a - sr_b

    score = 0.0
    score += 3.0 * max(0.0, sr_a - 0.75)
    score += 2.0 * max(0.0, success_gap)
    score += 2.0 * min(ab_future / 0.04, 2.0)
    score += 1.5 * min(ab_contact / 1.0, 2.0)
    score += 1.0 * max(0.0, min(depth_gap / 0.03, 1.5))
    score += 1.0 * max(0.0, min(curvature_gap / 15.0, 1.5))
    score += 0.7 * max(0.0, min(lateral_gap / 0.008, 1.5))
    score -= 2.0 * max(0.0, ab_vis - 0.025) / 0.025

    if sr_b < 0.05:
        score -= 0.6
    if sr_b > 0.65:
        score -= 2.0

    return {
        "score": float(score),
        "config": asdict(cfg),
        "condition_summary": summary,
        "pair_summary": pair_summary,
        "pair_rows": rows,
        "data": data,
    }


def write_candidate_report(results: List[Dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(results, key=lambda x: float(x["score"]), reverse=True)

    compact = []
    for rank, r in enumerate(ranked, start=1):
        by = r["condition_summary"]["by_condition"]
        ab = r["pair_summary"]["by_pair_type"].get("A_vs_B", {})
        cfg = r["config"]
        compact.append(
            {
                "rank": rank,
                "score": r["score"],
                "free_sr": by.get(FREE_INSERT, {}).get("success_rate", 0.0),
                "jam_sr": by.get(RIGHT_HIDDEN_JAM, {}).get("success_rate", 0.0),
                "ab_d_vis": ab.get("mean_d_vis", 0.0),
                "ab_d_contact": ab.get("mean_d_contact", 0.0),
                "ab_d_future": ab.get("mean_d_future", 0.0),
                "ab_success_diff": ab.get("success_diff_rate", 0.0),
                "hose_joint_stiffness": cfg["hose_joint_stiffness"],
                "max_delta_per_env_step": cfg["max_delta_per_env_step"],
                "lateral_offset_threshold": cfg["lateral_offset_threshold"],
                "jam_block_x": cfg["jam_block_x"],
                "jam_block_y": cfg["jam_block_y"],
                "jam_block_size": cfg["jam_block_size"],
                "jam_friction": cfg["jam_friction"][0],
            }
        )

    (out_dir / "calibration_summary.json").write_text(json.dumps(compact, indent=2), encoding="utf-8")
    (out_dir / "best_config.json").write_text(json.dumps(ranked[0]["config"], indent=2), encoding="utf-8")

    md_lines = [
        "# Stage 1.5 Calibration Report",
        "",
        "## Purpose",
        "",
        "This script searches a small parameter grid for the Hidden Lateral-Jam Hose Insertion environment.",
        "The target is not realism. The target is a clean CCDA precondition: similar visible/proprio/action history, but different hidden contact and future hose state.",
        "",
        "## Ranked candidates",
        "",
        "| Rank | Score | Free SR | Jam SR | A/B d_vis | A/B d_contact | A/B d_future | A/B success diff | stiffness | max delta | lat threshold | jam_x | jam_y | jam size | jam friction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for x in compact[:20]:
        jam_size = ", ".join(f"{v:.4f}" for v in x["jam_block_size"])
        md_lines.append(
            f"| {x['rank']} | {x['score']:.3f} | {x['free_sr']:.3f} | {x['jam_sr']:.3f} | "
            f"{x['ab_d_vis']:.5f} | {x['ab_d_contact']:.5f} | {x['ab_d_future']:.5f} | "
            f"{x['ab_success_diff']:.3f} | {x['hose_joint_stiffness']:.3f} | "
            f"{x['max_delta_per_env_step']:.4f} | {x['lateral_offset_threshold']:.4f} | "
            f"{x['jam_block_x']:.4f} | {x['jam_block_y']:.4f} | `{jam_size}` | {x['jam_friction']:.2f} |"
        )
    md_lines += [
        "",
        "## Best config",
        "",
        "See `best_config.json`.",
        "",
        "## Next command",
        "",
        "Use the best config values to update `HoseEnvConfig` defaults only after checking the generated visualizations.",
    ]
    (out_dir / "calibration_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    top = compact[:20]
    plt.figure(figsize=(9, 4.5))
    plt.bar([str(x["rank"]) for x in top], [float(x["score"]) for x in top])
    plt.xlabel("Candidate rank")
    plt.ylabel("Score")
    plt.title("Top calibration candidates")
    plt.tight_layout()
    plt.savefig(out_dir / "candidate_scores.png", dpi=170)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-per-condition", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--out-dir", type=str, default="reports/ccda_hose_stage1/calibration")
    parser.add_argument("--save-best-dataset", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    base = HoseEnvConfig()
    candidates = candidate_configs(base)[: args.max_candidates]

    results = []
    for idx, cfg in enumerate(candidates):
        print(f"[{idx + 1}/{len(candidates)}] evaluating config...")
        result = evaluate_config(cfg, args.episodes_per_condition, args.seed + idx * 1000)
        results.append(result)
        print("score=", result["score"])

    write_candidate_report(results, out_dir)

    ranked = sorted(results, key=lambda x: float(x["score"]), reverse=True)
    best = ranked[0]

    write_pair_metrics_csv(best["pair_rows"], out_dir / "best_pair_metrics.csv")
    plot_pair_metric_bars(best["pair_summary"], out_dir / "best_pair_plots")

    if args.save_best_dataset:
        data = best["data"]
        dataset_path = out_dir / "best_calibration_dataset.npz"
        np.savez_compressed(dataset_path, **data)
        print(f"Saved best calibration dataset to {dataset_path}")

    print(f"Best score: {best['score']:.3f}")
    print(f"Saved report to {out_dir / 'calibration_report.md'}")
    print(f"Saved best config to {out_dir / 'best_config.json'}")


if __name__ == "__main__":
    main()
