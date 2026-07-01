from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM


PathLike = Union[str, Path]


@dataclass
class AuditThresholds:
    """Thresholds for CCDA pair classification.

    Defaults are intentionally permissive. For formal reporting, inspect
    pair_metrics.csv and tune these thresholds in the script CLI.
    """

    tau_vis: float = 0.020
    tau_prop: float = 0.010
    tau_act: float = 0.003
    tau_contact: float = 0.200
    tau_future: float = 0.025


def _as_scalar_success(x: np.ndarray) -> float:
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[-1])


def _decode_str_array(arr: np.ndarray) -> List[str]:
    out: List[str] = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _window(arr: np.ndarray, center: int, before: int, after: int = 0) -> np.ndarray:
    """Return arr[start:end] around center, clipped safely."""
    t = arr.shape[0]
    start = max(0, center - before + 1)
    end = min(t, center + after + 1)
    return arr[start:end]


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float("nan")
    aa = a[:n].reshape(n, -1)
    bb = b[:n].reshape(n, -1)
    return float(np.sqrt(np.mean((aa - bb) ** 2)))


def _extract_episode(data: Dict[str, np.ndarray], idx: int) -> Dict[str, object]:
    condition = _decode_str_array(data["condition"])[idx]
    audit_t = int(data["audit_index"][idx])
    final_success = float(data["final_success"][idx])
    final_branch = _decode_str_array(data["final_branch"])[idx]
    return {
        "idx": idx,
        "condition": condition,
        "audit_t": audit_t,
        "final_success": final_success,
        "final_branch": final_branch,
        "visible_state": data["visible_state"][idx],
        "proprio": data["proprio"][idx],
        "action": data["action"][idx],
        "hose_keypoints": data["hose_keypoints"][idx],
        "privileged_contact": data["privileged_contact"][idx],
        "insertion_depth": data["insertion_depth"][idx],
        "lateral_offset": data["lateral_offset"][idx],
        "max_curvature": data["max_curvature"][idx],
    }


def load_npz_dataset(path: PathLike) -> Dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}


def pair_type(cond_i: str, cond_j: str) -> str:
    if cond_i == FREE_INSERT and cond_j == FREE_INSERT:
        return "A_vs_A"
    if cond_i == RIGHT_HIDDEN_JAM and cond_j == RIGHT_HIDDEN_JAM:
        return "B_vs_B"
    if {cond_i, cond_j} == {FREE_INSERT, RIGHT_HIDDEN_JAM}:
        return "A_vs_B"
    return f"{cond_i}_vs_{cond_j}"


def compute_pair_metrics(
    data: Dict[str, np.ndarray],
    history_steps: int = 8,
    future_steps: int = 32,
    thresholds: Optional[AuditThresholds] = None,
    max_pairs_per_type: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Compute CCDA audit pair metrics from a stage-1 .npz dataset."""
    thresholds = thresholds or AuditThresholds()
    n = int(data["visible_state"].shape[0])
    episodes = [_extract_episode(data, i) for i in range(n)]

    rows: List[Dict[str, object]] = []
    type_counts: Dict[str, int] = {}

    for i in range(n):
        for j in range(i + 1, n):
            ei = episodes[i]
            ej = episodes[j]
            ptype = pair_type(str(ei["condition"]), str(ej["condition"]))

            if max_pairs_per_type is not None:
                cnt = type_counts.get(ptype, 0)
                if cnt >= max_pairs_per_type:
                    continue
                type_counts[ptype] = cnt + 1

            ti = int(ei["audit_t"])
            tj = int(ej["audit_t"])

            vi = _window(ei["visible_state"], ti, history_steps)
            vj = _window(ej["visible_state"], tj, history_steps)

            qi = _window(ei["proprio"], ti, history_steps)
            qj = _window(ej["proprio"], tj, history_steps)

            ai = _window(ei["action"], ti, history_steps)
            aj = _window(ej["action"], tj, history_steps)

            ci = np.asarray(ei["privileged_contact"])[ti]
            cj = np.asarray(ej["privileged_contact"])[tj]

            fi = _window(ei["hose_keypoints"], ti + 1, 0, future_steps)
            fj = _window(ej["hose_keypoints"], tj + 1, 0, future_steps)

            d_vis = _rms(vi, vj)
            d_prop = _rms(qi, qj)
            d_act = _rms(ai, aj)
            d_contact = float(np.sqrt(np.mean((ci.reshape(-1) - cj.reshape(-1)) ** 2)))
            d_future = _rms(fi, fj)
            success_diff = float(float(ei["final_success"]) != float(ej["final_success"]))
            branch_diff = float(str(ei["final_branch"]) != str(ej["final_branch"]))

            is_ccda = (
                d_vis < thresholds.tau_vis
                and d_prop < thresholds.tau_prop
                and d_act < thresholds.tau_act
                and d_contact > thresholds.tau_contact
                and d_future > thresholds.tau_future
            )
            is_impact_ccda = bool(is_ccda and success_diff > 0.5)

            rows.append(
                {
                    "i": i,
                    "j": j,
                    "pair_type": ptype,
                    "condition_i": ei["condition"],
                    "condition_j": ej["condition"],
                    "branch_i": ei["final_branch"],
                    "branch_j": ej["final_branch"],
                    "success_i": float(ei["final_success"]),
                    "success_j": float(ej["final_success"]),
                    "d_vis": d_vis,
                    "d_prop": d_prop,
                    "d_act": d_act,
                    "d_contact": d_contact,
                    "d_future": d_future,
                    "success_diff": success_diff,
                    "branch_diff": branch_diff,
                    "is_ccda": float(is_ccda),
                    "is_impact_ccda": float(is_impact_ccda),
                }
            )

    summary = summarize_pair_metrics(rows, thresholds)
    return rows, summary


def summarize_pair_metrics(
    rows: List[Dict[str, object]],
    thresholds: Optional[AuditThresholds] = None,
) -> Dict[str, object]:
    thresholds = thresholds or AuditThresholds()
    out: Dict[str, object] = {
        "thresholds": asdict(thresholds),
        "by_pair_type": {},
    }
    pair_types = sorted(set(str(r["pair_type"]) for r in rows))
    for ptype in pair_types:
        sub = [r for r in rows if r["pair_type"] == ptype]
        if not sub:
            continue

        def mean(key: str) -> float:
            return float(np.mean([float(r[key]) for r in sub]))

        def std(key: str) -> float:
            return float(np.std([float(r[key]) for r in sub]))

        out["by_pair_type"][ptype] = {
            "n_pairs": len(sub),
            "mean_d_vis": mean("d_vis"),
            "std_d_vis": std("d_vis"),
            "mean_d_prop": mean("d_prop"),
            "std_d_prop": std("d_prop"),
            "mean_d_act": mean("d_act"),
            "std_d_act": std("d_act"),
            "mean_d_contact": mean("d_contact"),
            "std_d_contact": std("d_contact"),
            "mean_d_future": mean("d_future"),
            "std_d_future": std("d_future"),
            "success_diff_rate": mean("success_diff"),
            "branch_diff_rate": mean("branch_diff"),
            "ccda_pair_ratio": mean("is_ccda"),
            "impact_ccda_ratio": mean("is_impact_ccda"),
        }

    return out


def write_pair_metrics_csv(rows: List[Dict[str, object]], path: PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_report(
    summary: Dict[str, object],
    rows: List[Dict[str, object]],
    out_dir: PathLike,
    dataset_path: PathLike,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by = summary["by_pair_type"]

    def line(ptype: str) -> str:
        s = by.get(ptype, {})
        return (
            f"| `{ptype}` | {s.get('n_pairs', 0)} | "
            f"{s.get('mean_d_vis', 0):.6f} | "
            f"{s.get('mean_d_prop', 0):.6f} | "
            f"{s.get('mean_d_act', 0):.6f} | "
            f"{s.get('mean_d_contact', 0):.6f} | "
            f"{s.get('mean_d_future', 0):.6f} | "
            f"{s.get('success_diff_rate', 0):.3f} | "
            f"{s.get('branch_diff_rate', 0):.3f} | "
            f"{s.get('ccda_pair_ratio', 0):.3f} | "
            f"{s.get('impact_ccda_ratio', 0):.3f} |"
        )

    md = f"""# Stage 1.5 CCDA Pair Audit

## Dataset

`{dataset_path}`

## Purpose

This audit checks whether the current MuJoCo hose insertion environment produces the intended CCDA structure:

- small visible-history distance;
- small proprioception distance;
- small action-history distance;
- large hidden-contact distance;
- large future hose-state distance;
- different success or branch outcomes.

## Thresholds

```json
{json.dumps(summary["thresholds"], indent=2)}
```

## Pair summary

| Pair type | N pairs | mean d_vis | mean d_prop | mean d_act | mean d_contact | mean d_future | success diff | branch diff | CCDA ratio | Impact-CCDA ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{line("A_vs_A")}
{line("B_vs_B")}
{line("A_vs_B")}

## Interpretation

The desired pattern is:

| Pair type | Expected pattern |
| --- | --- |
| `A_vs_A` | low d_vis, low d_contact, low d_future, low success_diff |
| `B_vs_B` | low d_vis, low/medium d_contact, medium d_future, low/medium success_diff |
| `A_vs_B` | low d_vis, high d_contact, high d_future, high success_diff |

If `A_vs_B` has high d_contact but low d_future, the environment only proves hidden force difference, not future state branch divergence.
If `A_vs_B` has high d_vis, the audit time leaks the hidden condition visually and the setting is not a clean CCDA test.
"""
    (out_dir / "ccda_audit_report.md").write_text(md, encoding="utf-8")
    (out_dir / "ccda_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_pair_metric_bars(summary: Dict[str, object], out_dir: PathLike) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by = summary["by_pair_type"]
    pair_types = [p for p in ["A_vs_A", "B_vs_B", "A_vs_B"] if p in by]
    metrics = [
        ("mean_d_vis", "Visible-history distance"),
        ("mean_d_prop", "Proprio distance"),
        ("mean_d_act", "Action-history distance"),
        ("mean_d_contact", "Hidden-contact distance"),
        ("mean_d_future", "Future hose-state distance"),
        ("success_diff_rate", "Success difference rate"),
        ("branch_diff_rate", "Branch difference rate"),
        ("ccda_pair_ratio", "CCDA pair ratio"),
        ("impact_ccda_ratio", "Impact-CCDA ratio"),
    ]

    for key, title in metrics:
        values = [float(by[p].get(key, 0.0)) for p in pair_types]
        plt.figure(figsize=(7.5, 4.5))
        plt.bar(pair_types, values)
        plt.ylabel(key)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}.png", dpi=170)
        plt.close()


def plot_future_xy_overlay(data: Dict[str, np.ndarray], out_dir: PathLike, future_steps: int = 32) -> None:
    """Plot plug xy future trajectories from audit time."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = _decode_str_array(data["condition"])
    audit_idx = data["audit_index"]
    hose = data["hose_keypoints"]

    plt.figure(figsize=(7.0, 5.0))
    used = set()
    for i, cond in enumerate(conditions):
        t = int(audit_idx[i])
        fut = hose[i, t + 1 : t + 1 + future_steps, 0, :]
        if fut.shape[0] == 0:
            continue
        label = cond if cond not in used else None
        used.add(cond)
        plt.plot(fut[:, 0], fut[:, 1], alpha=0.45, label=label)
        plt.scatter([fut[0, 0]], [fut[0, 1]], s=8)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Future plug XY trajectories after audit time")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "future_plug_xy_overlay.png", dpi=180)
    plt.close()
