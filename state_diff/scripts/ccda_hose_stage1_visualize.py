from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")

from state_diff.env.ccda_hose.config import FREE_INSERT, RIGHT_HIDDEN_JAM, HoseEnvConfig
from state_diff.env.ccda_hose.env import scripted_rollout


def _save_video(frames, path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise RuntimeError("No frames were recorded.")
    imageio.mimsave(path, frames, fps=fps)


def _plot_compare(rollouts, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("insertion_depth", "Insertion depth"),
        ("lateral_offset", "Lateral offset"),
        ("max_curvature", "Max curvature"),
    ]

    for key, ylabel in metrics:
        plt.figure(figsize=(8, 4.5))
        for r in rollouts:
            y = r["trace"][key].reshape(len(r["trace"][key]), -1)[:, 0]
            plt.plot(y, label=f"{r['condition']} / {r['final_branch']}")
            plt.axvline(int(r["audit_index"]), linestyle="--", linewidth=0.8, alpha=0.35)
        plt.xlabel("Environment step")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel}: free_insert vs right_hidden_jam")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_compare.png", dpi=170)
        plt.close()

    plt.figure(figsize=(8, 4.5))
    for r in rollouts:
        y = r["trace"]["privileged_contact"][:, 3]
        plt.plot(y, label=f"{r['condition']} / {r['final_branch']}")
        plt.axvline(int(r["audit_index"]), linestyle="--", linewidth=0.8, alpha=0.35)
    plt.xlabel("Environment step")
    plt.ylabel("Lateral contact force")
    plt.title("Privileged lateral contact force")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "lateral_contact_force_compare.png", dpi=170)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", type=str, default="front", choices=["front", "top", "side"])
    parser.add_argument("--out-dir", type=str, default="reports/ccda_hose_stage1/visual_debug")
    args = parser.parse_args()

    cfg = HoseEnvConfig()
    out_dir = Path(args.out_dir)

    rollouts = [
        scripted_rollout(FREE_INSERT, seed=args.seed, config=cfg, record_frames=True, camera_name=args.camera),
        scripted_rollout(RIGHT_HIDDEN_JAM, seed=args.seed + 1, config=cfg, record_frames=True, camera_name=args.camera),
    ]

    for r in rollouts:
        video_path = out_dir / f"{r['condition']}_{args.camera}.mp4"
        _save_video(r["frames"], video_path)
        print(f"Saved {video_path}; final_branch={r['final_branch']}; success={r['final_success']}")

    _plot_compare(rollouts, out_dir)

    md = f"""# Stage 1 Visual Debug

Camera: `{args.camera}`

Generated videos:

* `{FREE_INSERT}_{args.camera}.mp4`
* `{RIGHT_HIDDEN_JAM}_{args.camera}.mp4`

Generated comparison plots:

* `insertion_depth_compare.png`
* `lateral_offset_compare.png`
* `max_curvature_compare.png`
* `lateral_contact_force_compare.png`

The dashed vertical line in each plot marks the audit time, defined as the end of the approach phase and the start of the shared push phase.
"""
    (out_dir / "visual_debug_report.md").write_text(md, encoding="utf-8")
    print(f"Saved visual report to {out_dir / 'visual_debug_report.md'}")


if __name__ == "__main__":
    main()
