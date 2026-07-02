#!/usr/bin/env python3
"""Inspect Phase1 hidden-contact cable data and write a summary report."""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


CONDITIONS = ["free", "hidden_pin", "hidden_high_friction"]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def parse_ep_len(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except Exception:
        return -1


def get_episode_files(base: Path) -> List[Path]:
    color_dir = base / "color"
    if not color_dir.exists():
        return []
    return sorted(color_dir.glob("*.pkl"))


def get_extras(info: Dict) -> Dict:
    if isinstance(info, dict):
        return info.get("extras", {})
    return {}


def get_bead_positions(info: Dict) -> Optional[np.ndarray]:
    extras = get_extras(info)
    if "bead_positions" in extras:
        arr = np.asarray(extras["bead_positions"], dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]

    if "bead_ids" in extras:
        bead_ids = extras["bead_ids"]
        pos = []
        for bid in bead_ids:
            if bid in info:
                pos.append(info[bid][0])
        if pos:
            return np.asarray(pos, dtype=np.float32)

    int_keys = sorted([k for k in info.keys() if isinstance(k, int)])
    pos = []
    for k in int_keys:
        value = info[k]
        if isinstance(value, tuple) and len(value) >= 1:
            p = value[0]
            if len(p) == 3:
                pos.append(p)
    if pos:
        return np.asarray(pos, dtype=np.float32)

    return None


def curve_metric(beads: Optional[np.ndarray]) -> Optional[float]:
    if beads is None or len(beads) < 3:
        return None
    xy = beads[:, :2]
    second = xy[:-2] - 2 * xy[1:-1] + xy[2:]
    return float(np.mean(np.linalg.norm(second, axis=1)))


def pairwise_chamfer(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    a = a[:, :2]
    b = b[:, :2]
    if len(a) == 0 or len(b) == 0:
        return None
    da = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return float(np.mean(np.sqrt(da.min(axis=1))) + np.mean(np.sqrt(da.min(axis=0))))


def summarize_episode(cond: str, base: Path, color_file: Path) -> Dict:
    ep_len = parse_ep_len(color_file)
    fname = color_file.name

    info_path = base / "info" / fname
    last_info_path = base / "last_info" / fname
    action_path = base / "action" / fname

    info_list = load_pickle(info_path) if info_path.exists() else []
    last_info = load_pickle(last_info_path) if last_info_path.exists() else {}
    actions = load_pickle(action_path) if action_path.exists() else []

    first_info = info_list[0] if len(info_list) > 0 else {}
    first_ex = get_extras(first_info)
    last_ex = get_extras(last_info)

    first_beads = get_bead_positions(first_info)
    last_beads = get_bead_positions(last_info)

    visible_seed = first_ex.get("ccda_visible_seed", last_ex.get("ccda_visible_seed", None))
    pair_group = first_ex.get("ccda_pair_group", last_ex.get("ccda_pair_group", ""))

    nb_beads = last_ex.get("nb_beads", None)
    nb_zone = last_ex.get("nb_zone", None)
    if nb_beads:
        final_fraction = float(nb_zone) / float(nb_beads)
    else:
        final_fraction = last_ex.get("total_rewards", None)

    return {
        "condition": cond,
        "file": fname,
        "episode_len": ep_len,
        "num_actions": len(actions),
        "visible_seed": visible_seed,
        "pair_group": pair_group,
        "hidden_condition_first": first_ex.get("hidden_condition"),
        "hidden_condition_last": last_ex.get("hidden_condition"),
        "hidden_contact_applied": bool(last_ex.get("hidden_contact_applied", False)),
        "success": bool(last_ex.get("task.done", False)),
        "final_fraction": final_fraction,
        "first_curve": curve_metric(first_beads),
        "final_curve": curve_metric(last_beads),
        "first_bead_positions_available": first_beads is not None,
        "last_bead_positions_available": last_beads is not None,
        "hidden_contact_meta": last_ex.get("hidden_contact_meta", {}),
    }


def mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None
    return float(np.mean(xs))


def summarize_condition(cond: str, base: Path) -> Dict:
    files = get_episode_files(base)
    episodes = [summarize_episode(cond, base, f) for f in files]

    return {
        "condition": cond,
        "path": str(base),
        "exists": base.exists(),
        "num_episodes": len(episodes),
        "success_rate": mean([float(e["success"]) for e in episodes]),
        "mean_episode_len": mean([e["episode_len"] for e in episodes]),
        "mean_final_fraction": mean([e["final_fraction"] for e in episodes]),
        "mean_final_curve": mean([e["final_curve"] for e in episodes]),
        "episodes": episodes,
    }


def load_beads_for_episode(base: Path, fname: str, field: str) -> Optional[np.ndarray]:
    if field == "first":
        info_path = base / "info" / fname
        info_list = load_pickle(info_path)
        if not info_list:
            return None
        return get_bead_positions(info_list[0])
    if field == "last":
        return get_bead_positions(load_pickle(base / "last_info" / fname))
    raise ValueError(field)


def paired_metrics(root_data: Path, summaries: Dict[str, Dict]) -> List[Dict]:
    by_seed: Dict[str, Dict[str, Dict]] = {}

    for cond, csum in summaries.items():
        for ep in csum["episodes"]:
            seed = str(ep.get("visible_seed"))
            if seed == "None":
                continue
            by_seed.setdefault(seed, {})[cond] = ep

    out = []
    for seed, cond_eps in sorted(by_seed.items()):
        if len(cond_eps) < 2:
            continue

        free_ep = cond_eps.get("free")
        if not free_ep:
            continue

        free_base = root_data / "free"
        free_first = load_beads_for_episode(free_base, free_ep["file"], "first")
        free_last = load_beads_for_episode(free_base, free_ep["file"], "last")

        for cond, ep in sorted(cond_eps.items()):
            if cond == "free":
                continue
            base = root_data / cond
            other_first = load_beads_for_episode(base, ep["file"], "first")
            other_last = load_beads_for_episode(base, ep["file"], "last")

            out.append(
                {
                    "visible_seed": seed,
                    "pair": ["free", cond],
                    "initial_chamfer": pairwise_chamfer(free_first, other_first),
                    "final_chamfer": pairwise_chamfer(free_last, other_last),
                    "success_diff": bool(free_ep["success"] != ep["success"]),
                    "free_success": free_ep["success"],
                    "other_success": ep["success"],
                    "free_final_fraction": free_ep["final_fraction"],
                    "other_final_fraction": ep["final_fraction"],
                }
            )

    return out


def write_report(report_md: Path, summary: Dict) -> None:
    lines = []
    lines.append("# Phase1 Hidden-Contact Cable Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Task: `{summary['task']}`")
    lines.append(f"- Data root: `{summary['data_root']}`")
    lines.append(f"- Conditions: `{', '.join(summary['conditions'])}`")
    lines.append("")
    lines.append("## Condition Summary")
    lines.append("")
    lines.append("| Condition | Episodes | Success Rate | Mean Final Fraction | Mean Final Curve |")
    lines.append("|---|---:|---:|---:|---:|")
    for cond in summary["conditions"]:
        c = summary["conditions_summary"][cond]
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                cond,
                c["num_episodes"],
                fmt(c["success_rate"]),
                fmt(c["mean_final_fraction"]),
                fmt(c["mean_final_curve"]),
            )
        )
    lines.append("")
    lines.append("## Paired Free-vs-Hidden Metrics")
    lines.append("")
    lines.append("| Seed | Pair | Initial Chamfer | Final Chamfer | Success Diff |")
    lines.append("|---|---|---:|---:|---:|")
    for item in summary["paired_metrics"]:
        lines.append(
            "| {} | {} vs {} | {} | {} | {} |".format(
                item["visible_seed"],
                item["pair"][0],
                item["pair"][1],
                fmt(item["initial_chamfer"]),
                fmt(item["final_chamfer"]),
                item["success_diff"],
            )
        )
    lines.append("")
    lines.append("## Phase1 Interpretation")
    lines.append("")
    lines.append("- This phase verifies task registration, hidden-contact injection, bead-state logging, and qualitative/quantitative divergence.")
    lines.append("- It is not yet the final CCDA audit. Phase2 should generate larger paired rollouts and compute formal thresholds.")
    lines.append("- A good Phase1 signal is low initial Chamfer with higher final Chamfer between `free` and hidden-contact conditions.")
    lines.append("")
    lines.append("## Phase1 Cleanup Conclusion")
    lines.append("")
    lines.append("Phase1 now keeps only two hidden-contact variants: `hidden_pin` and `hidden_high_friction`.")
    lines.append("The invalid side-jam variant was removed because RGB-D observation checks showed visible leakage.")
    lines.append("The previous action-step observation replay visualization was also removed because it did not show continuous robot-cable contact and could be misleading.")
    lines.append("The retained visual checks are bead trajectory overlays and RGB-D observation checks.")
    lines.append("Continuous robot-cable interaction should be inspected only with the continuous PyBullet rollout recorder.")
    lines.append("")
    report_md.write_text("\n".join(lines))


def fmt(x):
    if x is None:
        return "NA"
    try:
        return "{:.4f}".format(float(x))
    except Exception:
        return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--task", default="hidden-contact-cable-line")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data_root = root / "external" / "deformable-ravens" / "data" / args.task
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    condition_summary = {}
    for cond in CONDITIONS:
        condition_summary[cond] = summarize_condition(cond, data_root / cond)

    summary = {
        "task": args.task,
        "data_root": str(data_root),
        "conditions": CONDITIONS,
        "conditions_summary": condition_summary,
        "paired_metrics": paired_metrics(data_root, condition_summary),
    }

    out_json = reports / "phase1_hidden_contact_summary.json"
    out_md = reports / "phase1_hidden_contact_report.md"

    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_report(out_md, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print("[Phase1] wrote", out_json)
    print("[Phase1] wrote", out_md)

    for cond in CONDITIONS:
        if condition_summary[cond]["num_episodes"] == 0:
            raise SystemExit("[Phase1][FAIL] no episodes for condition={}".format(cond))

    for cond in CONDITIONS:
        episodes = condition_summary[cond]["episodes"]
        missing = [e for e in episodes if not e["last_bead_positions_available"]]
        if missing:
            raise SystemExit("[Phase1][FAIL] missing bead positions for {}".format(cond))

    if not summary["paired_metrics"]:
        raise SystemExit("[Phase1][FAIL] no paired metrics found")


if __name__ == "__main__":
    main()
