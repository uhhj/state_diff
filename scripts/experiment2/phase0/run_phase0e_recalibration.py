#!/usr/bin/env python3
"""Run the bounded fixed-step Phase 0E protocol recalibration."""
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

from scripts.experiment2.phase0.calibration_common import build_candidate_config
from scripts.experiment2.phase0.common import (
    canonical_json_sha256,
    compute_pair_metrics,
)
from scripts.experiment2.phase0.determinism_common import compare_traces
from scripts.experiment2.phase0.exact_counterfactual import (
    run_exact_counterfactual_pair,
)
from scripts.experiment2.phase0.observation_common import (
    delta_image_feature,
    grouped_ridge_accuracy,
    image_feature,
    load_rgb,
)
from scripts.experiment2.phase0.phase0e_common import (
    contact_feature,
    preload_return_residual,
    rank_phase0e,
    summarize_phase0e_candidate,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def git_sha(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def save_trace(path: Path, trace: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trace)


def run_candidate_seed(
    candidate_id: str,
    candidate_config: Dict[str, Any],
    seed: int,
    execution: Dict[str, Any],
    observation: Dict[str, Any],
    raw_root: Path,
):
    group_id = f"hf_{seed:06d}"
    group_dir = raw_root / candidate_id / group_id
    result = run_exact_counterfactual_pair(
        candidate_config,
        seed=seed,
        group_id=group_id,
        execution=execution,
        observation_output_dir=group_dir / "observations",
        observation_config=observation,
    )
    free_trace, hidden_trace, free_meta, hidden_meta, action_script, pair_meta = result
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
    row["free_preload_return_residual"] = preload_return_residual(free_trace)
    row["hidden_preload_return_residual"] = preload_return_residual(hidden_trace)
    row["branch_amplification"] = (
        float(metrics["main_branch_fde"])
        / max(float(metrics["preload_end_max_abs_xy"]), 1e-9)
    )

    save_trace(group_dir / "free.npz", free_trace)
    save_trace(group_dir / "hidden_high_friction.npz", hidden_trace)
    write_json(
        group_dir / "pair.json",
        {
            "candidate_config": candidate_config,
            "action_script": action_script,
            "free_metadata": free_meta,
            "hidden_metadata": hidden_meta,
            "pair_metadata": pair_meta,
            "metrics": row,
        },
    )
    return result, row


def classifier_samples(
    result: Tuple[Any, ...],
    group_id: str,
    candidate_config: Dict[str, Any],
    observation: Dict[str, Any],
) -> List[Dict[str, Any]]:
    free_trace, hidden_trace, free_meta, hidden_meta = result[:4]
    samples = []
    for condition in ("free", "hidden_high_friction"):
        metadata = free_meta if condition == "free" else hidden_meta
        trace = free_trace if condition == "free" else hidden_trace
        no_action_path = Path(
            metadata["observations"]["no_action_end"]["path"]
        )
        pre_main_path = Path(metadata["observations"]["pre_main"]["path"])
        no_action_rgb = load_rgb(no_action_path)
        pre_main_rgb = load_rgb(pre_main_path)
        label = -1 if condition == "free" else 1
        samples.append(
            {
                "group_id": group_id,
                "condition": condition,
                "label": label,
                "vision_raw": image_feature(
                    pre_main_rgb,
                    observation["feature_width"],
                    observation["feature_height"],
                ),
                "vision_delta": delta_image_feature(
                    no_action_rgb,
                    pre_main_rgb,
                    observation["feature_width"],
                    observation["feature_height"],
                ),
                "contact": contact_feature(
                    trace,
                    hz=candidate_config["hz"],
                    trace_stride=candidate_config["trace_stride"],
                ),
            }
        )
    return samples


def write_roadmap_alignment(path: Path) -> None:
    lines = [
        "# Phase 0E Roadmap Alignment",
        "",
        "| 路线项 | 当前状态 | 判断 |",
        "|---|---|---|",
        "| 主环境使用 DeformableRavens | 已满足 | 对齐路线 |",
        "| 第一优先任务 Hidden-Friction Cable Pull | 已满足 | 对齐路线 |",
        "| 同基础快照复制到两个隐藏条件 | 已满足 | 对齐数据合同 |",
        "| 未来动作逐元素一致 | 已满足 | Gate B 工程证据通过 |",
        "| 固定步进确定性恢复 | 已满足 | Gate H 单机审计通过 |",
        "| 无动作稳定 | 有小样本工程证据 | 正式 Gate C 未完成 |",
        "| 接触信息可分 | pilot 中可分 | 正式 Gate D 未完成 |",
        "| 视觉不可分 | Phase 0E 初步真实视觉分类 | Gate A/D/G 未完成 |",
        "| 稳定未来分支 | 当前候选待本轮判断 | Gate E 未通过 |",
        "| Outcome impact | 未实现 | Gate F 未开始 |",
        "| Leakage audit | 仅人工视频和本轮初筛 | Gate G 未完成 |",
        "| 模型训练 | 尚未开始 | 正确，不应提前进入 M1 |",
        "",
        "Verdict: NOT_OFF_TRACK_BUT_PHASE0_INCOMPLETE",
        "",
        "- Phase 0C 的异步大 FDE 是模拟器调度伪效应，已被 Phase 0D 否定。",
        "- 保留 Phase 0C 失败报告符合路线图，不删除历史失败证据。",
        "- Phase 0B 真实仿真视频是 M0 可视化交付物，不构成研究偏航。",
        "- 当前最大风险是仅凭 privileged bead 指标调参，而缺少完整视觉泄漏与 outcome impact 证据。",
        "- Phase 0E 只执行一次有界扫描；若无合格候选，不扩大网格。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repeat_check(
    ranked: List[Dict[str, Any]],
    seeds: List[int],
    base_config: Dict[str, Any],
    phase0e: Dict[str, Any],
    output_root: Path,
) -> Dict[str, Any]:
    cfg = phase0e["repeat_check"]
    rows = []
    for summary in ranked[: int(cfg["top_k"])]:
        candidate = summary["candidate"]
        candidate_config = build_candidate_config(base_config, candidate)
        for seed in seeds[: int(cfg["seed_count"])]:
            results = []
            for repeat in range(int(cfg["repeats"])):
                results.append(
                    run_exact_counterfactual_pair(
                        candidate_config,
                        seed=seed,
                        group_id=(
                            f"repeat_{summary['candidate_id']}_{seed}_r{repeat}"
                        ),
                        execution=phase0e["execution"],
                    )
                )
            reference = results[0]
            comparisons = []
            for index in range(1, len(results)):
                candidate_result = results[index]
                free_comparison = compare_traces(
                    reference[0], candidate_result[0], 0.0
                )
                hidden_comparison = compare_traces(
                    reference[1], candidate_result[1], 0.0
                )
                metadata_match = bool(
                    reference[5]["base_state_hash"]
                    == candidate_result[5]["base_state_hash"]
                    and reference[5]["action_hash"]
                    == candidate_result[5]["action_hash"]
                    and reference[5]["free_events"]
                    == candidate_result[5]["free_events"]
                    and reference[5]["hidden_events"]
                    == candidate_result[5]["hidden_events"]
                    and reference[5]["free_trace_hash"]
                    == candidate_result[5]["free_trace_hash"]
                    and reference[5]["hidden_trace_hash"]
                    == candidate_result[5]["hidden_trace_hash"]
                )
                comparisons.append(
                    {
                        "repeat": index,
                        "metadata_match": metadata_match,
                        "free": free_comparison,
                        "hidden": hidden_comparison,
                        "passed": bool(
                            metadata_match
                            and free_comparison["byte_exact"]
                            and hidden_comparison["byte_exact"]
                        ),
                    }
                )
            rows.append(
                {
                    "candidate_id": summary["candidate_id"],
                    "seed": seed,
                    "comparisons": comparisons,
                    "passed": all(row["passed"] for row in comparisons),
                }
            )
    result = {"rows": rows, "passed": all(row["passed"] for row in rows)}
    write_json(output_root / "repeat_check.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiment2/phase0/hidden_friction_phase0e.json",
    )
    parser.add_argument(
        "--output",
        default="reports/experiment2/phase0_hidden_friction/phase0e",
    )
    args = parser.parse_args()

    phase0e_path = (REPO_ROOT / args.config).resolve()
    output_root = (REPO_ROOT / args.output).resolve()
    raw_root = output_root / "raw"
    phase0e = json.loads(phase0e_path.read_text(encoding="utf-8"))
    base_path = (REPO_ROOT / phase0e["base_config"]).resolve()
    base_config = json.loads(base_path.read_text(encoding="utf-8"))
    candidates = list(phase0e["candidates"])
    first_candidate = candidates[0]
    first_config = build_candidate_config(base_config, first_candidate)

    search = phase0e["seed_search"]
    selected_seeds: List[int] = []
    rejected_seeds: List[Dict[str, Any]] = []
    first_cache: Dict[int, Tuple[Any, Dict[str, Any]]] = {}
    for offset in range(int(search["max_attempts"])):
        seed = int(search["start"]) + offset
        try:
            result, row = run_candidate_seed(
                first_candidate["id"],
                first_config,
                seed,
                phase0e["execution"],
                phase0e["observation"],
                raw_root,
            )
        except RuntimeError as exc:
            if "neither ordered cable endpoint has a legal fixed pull action" in str(exc):
                rejected_seeds.append({"seed": seed, "reason": str(exc)})
                continue
            raise
        selected_seeds.append(seed)
        first_cache[seed] = (result, row)
        print(f"accepted seed={seed}", flush=True)
        if len(selected_seeds) == int(search["count"]):
            break
    if len(selected_seeds) != int(search["count"]):
        raise RuntimeError(
            f"found {len(selected_seeds)} legal seeds, expected {search['count']}"
        )
    manifest = {
        "selected_seeds": selected_seeds,
        "rejected_seeds": rejected_seeds,
        "start": int(search["start"]),
        "max_attempts": int(search["max_attempts"]),
    }
    write_json(output_root / "seed_manifest.json", manifest)

    summaries = []
    for candidate_index, candidate in enumerate(candidates):
        candidate_id = str(candidate["id"])
        candidate_config = build_candidate_config(base_config, candidate)
        pair_rows = []
        samples = []
        for seed in selected_seeds:
            if candidate_index == 0:
                result, row = first_cache[seed]
            else:
                result, row = run_candidate_seed(
                    candidate_id,
                    candidate_config,
                    seed,
                    phase0e["execution"],
                    phase0e["observation"],
                    raw_root,
                )
            pair_rows.append(row)
            samples.extend(
                classifier_samples(
                    result,
                    row["group_id"],
                    candidate_config,
                    phase0e["observation"],
                )
            )
            print(
                f"completed {candidate_id} seed={seed} "
                f"preload={row['preload_end_max_abs_xy']:.6f} "
                f"FDE={row['main_branch_fde']:.6f}",
                flush=True,
            )

        observation = phase0e["observation"]
        vision_raw = grouped_ridge_accuracy(
            samples, "vision_raw", observation["ridge_l2"]
        )
        vision_delta = grouped_ridge_accuracy(
            samples, "vision_delta", observation["ridge_l2"]
        )
        contact_classifier = grouped_ridge_accuracy(
            samples, "contact", observation["ridge_l2"]
        )
        summary = summarize_phase0e_candidate(
            candidate_id,
            candidate,
            pair_rows,
            phase0e["selection_targets"],
            vision_raw,
            vision_delta,
            contact_classifier,
        )
        summary["candidate_config_hash"] = canonical_json_sha256(
            candidate_config
        )
        summaries.append(summary)
        write_json(output_root / f"candidate_{candidate_id}.json", summary)

    ranked = rank_phase0e(summaries)
    selected = ranked[0]
    selected_path = (
        REPO_ROOT
        / "configs/experiment2/phase0/hidden_friction_cable_phase0e_selected.json"
    )
    if selected["eligible"]:
        selected_config = build_candidate_config(
            base_config, selected["candidate"]
        )
        selected_config["phase0e"] = {
            "candidate_id": selected["candidate_id"],
            "provisional_engineering_screen": True,
            "formal_gate_pass": False,
        }
        write_json(selected_path, selected_config)

    repeat_result = repeat_check(
        ranked, selected_seeds, base_config, phase0e, output_root
    )
    write_roadmap_alignment(output_root / "roadmap_alignment.md")
    top = {
        "stage": "Experiment2 Phase 0E",
        "main_repository_sha": git_sha(REPO_ROOT),
        "submodule_sha": git_sha(SUBMODULE_ROOT),
        "config": str(phase0e_path.relative_to(REPO_ROOT)),
        "config_hash": canonical_json_sha256(phase0e),
        "execution": phase0e["execution"],
        "observation": phase0e["observation"],
        "seed_manifest": manifest,
        "targets": phase0e["selection_targets"],
        "ranked_candidates": ranked,
        "top_candidate_id": selected["candidate_id"],
        "top_candidate_eligible": bool(selected["eligible"]),
        "selected_config_written": bool(selected["eligible"]),
        "repeat_check_passed": bool(repeat_result["passed"]),
        "roadmap_verdict": "NOT_OFF_TRACK_BUT_PHASE0_INCOMPLETE",
        "interpretation": (
            "Bounded fixed-step engineering screen only; formal Gate A-H "
            "statistics, outcome impact, and leakage audit remain pending."
        ),
    }
    write_json(output_root / "summary.json", top)
    if selected["eligible"]:
        result_text = (
            "Phase 0E engineering screen passed for one provisional candidate. "
            "Formal Gate A-H statistics, outcome impact, and leakage audit remain pending."
        )
    else:
        result_text = (
            "Hidden-Friction Cable Pull remains scientifically blocked at Stage M0. "
            "The bounded fixed-step scan did not produce contact-informative, "
            "vision-ambiguous, future-divergent behavior."
        )
    lines = [
        "# Phase 0E Fixed-Step Protocol Recalibration",
        "",
        f"- Selected seeds: `{selected_seeds}`",
        f"- Rejected seeds: `{rejected_seeds}`",
        f"- Candidates: `{len(ranked)}`",
        f"- Top candidate: `{selected['candidate_id']}`",
        f"- Eligible: `{selected['eligible']}`",
        f"- Repeat check: `{repeat_result['passed']}`",
        f"- Roadmap verdict: `NOT_OFF_TRACK_BUT_PHASE0_INCOMPLETE`",
        f"- Result: {result_text}",
    ]
    (output_root / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(output_root / "summary.json")


if __name__ == "__main__":
    main()
