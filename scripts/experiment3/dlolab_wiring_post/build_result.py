from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from scripts.experiment3.dlolab_wiring_post.paths import (
    official_log_dir,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True).strip()


def build(config_path, pair_path):
    config = load_json(config_path)
    pairs = load_json(pair_path)
    minimum = int(config["pair_mining"]["min_candidates_to_continue"])
    count = int(pairs["candidate_count"])
    is_scaleup = "SCALEUP" in Path(pair_path).name

    if count >= minimum:
        verdict = "PB0_WIRING_POST_PAIR_CANDIDATES_FOUND"
        next_task = (
            "Freeze PB0 collection/mining settings and proceed to PB1 "
            "snapshot branching and control-relevance audit on top pairs.")
    elif not is_scaleup:
        verdict = "PB0_WIRING_POST_INITIAL_COLLECTION_INSUFFICIENT"
        next_task = (
            "Run the one allowed collection scale-up with 16 batches. "
            "Do not change physics, observation protocol or pair filters.")
    else:
        verdict = "PB0_WIRING_POST_NO_NATURAL_CCDA_PAIRS"
        next_task = (
            "Stop Wiring-post as the primary CCDA benchmark and evaluate "
            "another published task. Do not modify Wiring-post physics.")

    raw_root = Path(config["outputs"]["raw_root"])
    metadata_path = raw_root / (
        "collection_metadata_scaleup.json"
        if is_scaleup
        else "collection_metadata.json")
    metadata = load_json(metadata_path)
    log_dir = official_log_dir()
    top = pairs["top_candidates"][:10]
    result = {
        "verdict": verdict,
        "candidate_count": count,
        "minimum_to_continue": minimum,
        "dataset": pairs["dataset"],
        "num_rollouts": int(pairs["num_rollouts"]),
        "time_samples": int(pairs["time_samples"]),
        "finite_replay_reward": bool(metadata["finite_replay_reward"]),
        "official_best_traj_exists": bool(
            (log_dir / "best_traj.npy").is_file()),
        "official_best_qpos_exists": bool(
            (log_dir / "best_qpos.npy").is_file()),
        "top_candidates": top,
        "benchmark_physics_modified": False,
        "model_training_started": False,
        "next_task": next_task,
    }

    committed = REPO_ROOT / config["outputs"]["committed_report_dir"]
    committed.mkdir(parents=True, exist_ok=True)
    evidence = {
        "phase_name": config["phase_name"],
        "verdict": verdict,
        "repository": {
            "starting_main_sha": config["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink": git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": result,
    }
    (committed / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# PB0 DLO-Lab Wiring-post",
        "",
        "Verdict: `{}`".format(verdict),
        "",
        "## Scientific status",
        "",
        "- Official DLO-Lab revision: `{}`".format(
            config["benchmark"]["revision"]),
        "- Published task physics modified: No",
        "- Model training started: No",
        "- best_traj exists: {}".format(
            result["official_best_traj_exists"]),
        "- best_qpos exists: {}".format(
            result["official_best_qpos_exists"]),
        "- Finite replay reward: {}".format(
            result["finite_replay_reward"]),
        "- Collection size: {}".format(result["num_rollouts"]),
        "- Time samples: {}".format(result["time_samples"]),
        "- Pair candidates: {}".format(count),
        "- Minimum to continue: {}".format(minimum),
        "",
        "## Top candidates",
        "",
    ]
    for index, row in enumerate(top):
        wrap = max(
            row["hidden_difference"]["wrap_difference_rad"],
            default=0.0)
        lines.append(
            "{}. rollouts {} / {}, t={}, obs={:.6f} m, EE={:.6f} m, "
            "wrap={:.6f} rad, contact-mismatch={}, future={:.6f} m, "
            "ratio={:.3f}, score={:.3f}".format(
                index + 1,
                row["rollout_a"],
                row["rollout_b"],
                row["time_index"],
                row["visible_history_chamfer_m"],
                row["ee_position_distance_m"],
                wrap,
                row["hidden_difference"]["contact_like_mismatch"],
                row["future_visible_chamfer_m"],
                row["future_to_observable_ratio"],
                row["score"],
            ))
    lines.extend(["", "## Next task", "", next_task, ""])
    (committed / "RESULT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pairs", required=True)
    args = parser.parse_args()
    print(build(args.config, args.pairs))


if __name__ == "__main__":
    main()
