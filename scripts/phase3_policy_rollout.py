#!/usr/bin/env python3
import argparse
import csv
import importlib
import json
import os
import random
import sys
import time
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from ccda_phase3.data_io import load_action_codec_from_template
from ccda_phase3.metrics import curve_metric_from_state
from ccda_phase3.rollout import final_fraction_from_info, pad_history, state_from_live_info
from ccda_phase3.train_utils import load_future_model, load_inverse_model

REQUIRED_CONDITIONS = ["free", "hidden_pin", "hidden_high_friction", "hidden_breakaway_pin"]
PRIMARY_HIDDEN = "hidden_breakaway_pin"
DIAGNOSTIC_HIDDEN = "hidden_pin"
SELECTED_RECOVERABLE_CONFIG = "breakaway_force_2p6_disp_0p045_pull_0p36"


def require_rollout_gates() -> None:
    if os.environ.get("PHASE3_ALLOW_ROLLOUT", "0") != "1":
        raise SystemExit("[Phase3.4][BLOCKED] Set PHASE3_ALLOW_ROLLOUT=1 to run rollout.")
    if os.environ.get("PHASE3_ROLLOUT_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.4][BLOCKED] Set PHASE3_ROLLOUT_CONFIRMED=1 after user approval.")


FORBIDDEN_ROLLOUT_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]


def assert_no_tensorflow_loaded(stage: str) -> None:
    bad = []
    for name in sys.modules:
        if name == "tensorflow" or name.startswith("tensorflow."):
            bad.append(name)
        if name.startswith("ravens.agents") or name.startswith("ravens.models") or name.startswith("ravens.datasets"):
            bad.append(name)
    if bad:
        raise SystemExit(f"[Phase3.3b][FAIL] Forbidden rollout modules loaded at {stage}: {bad[:20]}")


def install_minimal_ravens_package(root: Path) -> None:
    defravens = root / "external" / "deformable-ravens"
    if str(defravens) not in sys.path:
        sys.path.insert(0, str(defravens))
    for name in list(sys.modules):
        if name == "ravens" or name.startswith("ravens."):
            del sys.modules[name]
    pkg = types.ModuleType("ravens")
    pkg.__path__ = [str(defravens / "ravens")]
    pkg.__file__ = str(defravens / "ravens" / "__init__.py")
    pkg.__package__ = "ravens"
    sys.modules["ravens"] = pkg


def import_ravens_runtime(root: Path):
    # TensorFlow-free minimal rollout runtime imports.
    # Do not import top-level ravens, ravens.agents, ravens.models, ravens.datasets, or TensorFlow.
    # This DeformableRavens fork exposes Environment at ravens.environment.
    install_minimal_ravens_package(root)
    tasks = importlib.import_module("ravens.tasks")
    env_mod = importlib.import_module("ravens.environment")
    Environment = env_mod.Environment
    assert_no_tensorflow_loaded("after_minimal_ravens_import")
    return tasks, Environment


def require_runtime(root: Path):
    missing = []
    for name in ["torch", "pybullet"]:
        try:
            __import__(name)
        except Exception as exc:
            missing.append(f"{name}: {repr(exc)}")
    if missing:
        raise SystemExit("[Phase3.3b][FAIL] Rollout runtime missing dependencies: " + "; ".join(missing))
    patch_pybullet_pkg_resources_metadata()
    try:
        tasks, Environment = import_ravens_runtime(root)
    except Exception as exc:
        raise SystemExit(f"[Phase3.3b][FAIL] Minimal Ravens runtime import failed: {repr(exc)}") from exc
    if "hidden-contact-cable-line" not in tasks.names:
        raise SystemExit("[Phase3.3b][FAIL] hidden-contact-cable-line not registered in ravens.tasks.names")
    return tasks, Environment




def patch_pybullet_pkg_resources_metadata() -> None:
    # DeformableRavens Environment checks pkg_resources.get_distribution("pybullet").
    # In coord_bimanual, pybullet is importable but may lack setuptools distribution metadata.
    # Provide the version expected by the original environment check without installing packages.
    try:
        import pkg_resources
        pkg_resources.get_distribution("pybullet")
        return
    except Exception:
        pass
    import pkg_resources

    original_get_distribution = pkg_resources.get_distribution

    class _PyBulletDistribution:
        version = "3.0.4"

    def _patched_get_distribution(dist):
        if str(dist) == "pybullet":
            return _PyBulletDistribution()
        return original_get_distribution(dist)

    pkg_resources.get_distribution = _patched_get_distribution


def set_selected_recoverable_env_defaults() -> None:
    os.environ.setdefault("CCDA_BREAKAWAY_FORCE", "2.6")
    os.environ.setdefault("CCDA_BREAKAWAY_DISP", "0.045")
    os.environ.setdefault("CCDA_BREAKAWAY_BEAD_RATIO", "0.45")
    os.environ.setdefault("CCDA_ORACLE_BREAKAWAY_PULL_DIST", "0.36")


def close_env_safely(env: Any) -> None:
    # Pause the upstream daemon simulation loop before disconnecting PyBullet.
    # This avoids post-disconnect background thread exceptions in Phase3.4 logs.
    try:
        env.pause()
    except Exception:
        pass
    try:
        env.running = False
    except Exception:
        pass
    try:
        env.ee = None
    except Exception:
        pass
    time.sleep(0.05)
    try:
        env.stop()
    except Exception:
        pass


def load_windows_meta(path: Path) -> Tuple[Any, Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta_json" in data:
        raw = data["meta_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
            meta = json.loads(text)
        except Exception:
            meta = {}
    return data, meta


def validate_rollout_inputs(args: argparse.Namespace, root: Path) -> Tuple[Any, Dict[str, Any], Path]:
    conditions = list(dict.fromkeys(args.conditions))
    if conditions != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.4][FAIL] rollout conditions must be exactly {REQUIRED_CONDITIONS}, got {conditions}")
    if args.primary_hidden_condition != PRIMARY_HIDDEN:
        raise SystemExit(f"[Phase3.4][FAIL] primary hidden condition must be {PRIMARY_HIDDEN}, got {args.primary_hidden_condition}")
    if args.diagnostic_hidden_condition != DIAGNOSTIC_HIDDEN:
        raise SystemExit(f"[Phase3.4][FAIL] diagnostic hidden condition must be {DIAGNOSTIC_HIDDEN}, got {args.diagnostic_hidden_condition}")

    windows = Path(args.windows)
    if not windows.is_absolute():
        windows = root / windows
    if not windows.exists():
        raise SystemExit(f"[Phase3.4][FAIL] windows file missing: {windows}")

    data, meta = load_windows_meta(windows)
    if meta.get("conditions") != REQUIRED_CONDITIONS:
        raise SystemExit(f"[Phase3.4][FAIL] windows conditions mismatch: {meta.get('conditions')}")
    if meta.get("primary_hidden_condition") != args.primary_hidden_condition:
        raise SystemExit(f"[Phase3.4][FAIL] windows primary mismatch: {meta.get('primary_hidden_condition')}")
    if meta.get("diagnostic_hidden_condition") != args.diagnostic_hidden_condition:
        raise SystemExit(f"[Phase3.4][FAIL] windows diagnostic mismatch: {meta.get('diagnostic_hidden_condition')}")
    if "y_action" not in data or data["y_action"].shape[1] != 14:
        shape = data["y_action"].shape if "y_action" in data else None
        raise SystemExit(f"[Phase3.4][FAIL] y_action dim must be 14, got {shape}")
    if int(data["action_dim"]) != 14:
        raise SystemExit(f"[Phase3.4][FAIL] action_dim must be 14, got {int(data['action_dim'])}")

    template_path = Path(str(data["action_template_json_or_pickle_path"]))
    codec = load_action_codec_from_template(template_path)
    if codec.dim() != 14:
        raise SystemExit(f"[Phase3.4][FAIL] action codec dim must be 14, got {codec.dim()}")
    if codec.summary().get("num_camera_config_paths") != 0:
        raise SystemExit(f"[Phase3.4][FAIL] action codec encodes camera_config: {codec.summary()}")
    return data, meta, template_path


def first_checkpoint(root: Path, baseline: str) -> Path:
    cands = sorted((root / baseline).glob("fold_*_seed_*"))
    for cand in cands:
        if (cand / "config.json").exists() and (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"no complete checkpoint for baseline={baseline} under {root}")


def write_outputs(trials: List[Dict[str, Any]], out_csv: Path, out_json: Path, out_md: Path, args: argparse.Namespace) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(trials[0].keys()) if trials else [
        "baseline", "condition", "visible_seed", "episode_idx", "success", "final_fraction", "num_steps",
        "primary_hidden_condition", "diagnostic_hidden_condition", "primary_pair", "diagnostic_pair",
        "selected_recoverable_config", "rollout_runtime",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in trials:
            writer.writerow(row)

    primary_pair = f"free_vs_{args.primary_hidden_condition}"
    diagnostic_pair = f"free_vs_{args.diagnostic_hidden_condition}"
    groups = defaultdict(list)
    for row in trials:
        groups[(row["baseline"], row["condition"])].append(row)

    summary_rows = []
    by_baseline: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (baseline, condition), rows in sorted(groups.items()):
        success_rate = float(np.mean([float(r["success"]) for r in rows])) if rows else None
        item = {
            "baseline": baseline,
            "condition": condition,
            "num_trials": len(rows),
            "success_rate": success_rate,
            "success_count": int(sum(int(r["success"]) for r in rows)),
            "mean_final_fraction": float(np.nanmean([float(r["final_fraction"]) for r in rows])) if rows else None,
            "mean_final_curve": float(np.nanmean([float(r["final_curve"]) for r in rows])) if rows else None,
            "mean_action_ood_score": float(np.nanmean([float(r["mean_action_ood_score"]) for r in rows])) if rows else None,
            "primary_pair": primary_pair,
            "diagnostic_pair": diagnostic_pair,
        }
        summary_rows.append(item)
        by_baseline[baseline][condition] = item

    gaps = {}
    for baseline, by_condition in sorted(by_baseline.items()):
        free = by_condition.get("free", {}).get("success_rate")
        primary = by_condition.get(args.primary_hidden_condition, {}).get("success_rate")
        diagnostic = by_condition.get(args.diagnostic_hidden_condition, {}).get("success_rate")
        primary_gap = None if free is None or primary is None else float(free - primary)
        diagnostic_gap = None if free is None or diagnostic is None else float(free - diagnostic)
        gaps[baseline] = {
            "primary_pair": primary_pair,
            "diagnostic_pair": diagnostic_pair,
            "primary_success_gap": primary_gap,
            "diagnostic_success_gap": diagnostic_gap,
            "free_vs_hidden_breakaway_pin_success_gap": primary_gap,
            "free_vs_hidden_pin_success_gap": diagnostic_gap,
        }
        for item in summary_rows:
            if item["baseline"] == baseline:
                item.update(gaps[baseline])

    summary = {
        "num_trials": len(trials),
        "conditions": REQUIRED_CONDITIONS,
        "primary_hidden_condition": args.primary_hidden_condition,
        "diagnostic_hidden_condition": args.diagnostic_hidden_condition,
        "primary_pair": primary_pair,
        "diagnostic_pair": diagnostic_pair,
        "selected_recoverable_config": args.selected_recoverable_config,
        "rollout_runtime": "tf_free_learned_policy",
        "scope": "phase3_4_rollout_smoke_only_no_phase4_no_cps",
        "summary_rows": summary_rows,
        "success_gaps_by_baseline": gaps,
        "interpretation": "Rollout smoke only. This is not Phase4 or CPS evidence.",
    }
    assert_no_tensorflow_loaded("before_summary_write")
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))

    lines = [
        "# Phase3.4 Rollout Smoke Report",
        "",
        "## Scope",
        "",
        "- This is small rollout smoke only.",
        "- It is not Phase4 and not CPS evidence.",
        f"- Primary pair: `{primary_pair}`",
        f"- Diagnostic pair: `{diagnostic_pair}`",
        "- `free_vs_hidden_pin` is diagnostic only.",
        "",
        "## Summary",
        "",
        "| Baseline | Condition | Success | Total | Rate | Final fraction mean | Action OOD mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        total = int(row["num_trials"])
        rate = row["success_rate"]
        final_fraction = row["mean_final_fraction"]
        action_ood = row["mean_action_ood_score"]
        lines.append(
            f"| `{row['baseline']}` | `{row['condition']}` | {row['success_count']} | {total} | "
            f"{rate:.3f} | {final_fraction:.3f} | {action_ood:.3f} |"
        )
    lines += [
        "",
        "## Success Gaps",
        "",
        "| Baseline | Primary success gap free-hidden_breakaway | Diagnostic gap free-hidden_pin |",
        "|---|---:|---:|",
    ]
    for baseline, gap in gaps.items():
        pg = gap["primary_success_gap"]
        dg = gap["diagnostic_success_gap"]
        lines.append(f"| `{baseline}` | {pg if pg is not None else float('nan'):.3f} | {dg if dg is not None else float('nan'):.3f} |")
    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--data", "--windows", dest="windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--ckpt_root", "--checkpoint_root", dest="checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--baselines", "--rollout_models", dest="rollout_models", nargs="+", default=["paper_state", "state_action"])
    parser.add_argument("--conditions", nargs="+", default=REQUIRED_CONDITIONS)
    parser.add_argument("--primary_hidden_condition", default=PRIMARY_HIDDEN)
    parser.add_argument("--diagnostic_hidden_condition", default=DIAGNOSTIC_HIDDEN)
    parser.add_argument("--selected_recoverable_config", default=SELECTED_RECOVERABLE_CONFIG)
    parser.add_argument("--seed_start", type=int, default=200000)
    parser.add_argument("--num_seeds", type=int, default=None, help="Backward-compatible alias for max_episodes_per_condition.")
    parser.add_argument("--max_episodes_per_condition", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=16)
    parser.add_argument("--samples_per_step", type=int, default=16)
    parser.add_argument("--motion_timeout", type=float, default=5.0)
    parser.add_argument("--action_clip_std", type=float, default=3.0)
    parser.add_argument("--allow_numpy_fallback", action="store_true", help="Rejected for learned rollout; kept only for CLI compatibility.")
    parser.add_argument("--out_trials", "--out_csv", dest="out_csv", default="reports/phase3_2_rollout_smoke_trials.csv")
    parser.add_argument("--out_summary", "--out_json", dest="out_json", default="reports/phase3_2_rollout_smoke_summary.json")
    parser.add_argument("--out_md", default="reports/phase3_2_rollout_smoke_report.md")
    args = parser.parse_args()

    if args.allow_numpy_fallback:
        raise SystemExit("[Phase3.4][FAIL] NumPy fallback is not allowed for learned rollout.")

    root = Path(args.root).resolve()
    require_rollout_gates()
    set_selected_recoverable_env_defaults()
    tasks, Environment = require_runtime(root)
    data, meta, template_path = validate_rollout_inputs(args, root)
    assert_no_tensorflow_loaded("after_rollout_input_validation")

    codec = load_action_codec_from_template(template_path)
    assert_no_tensorflow_loaded("after_action_codec_load")
    th = int(data["th"])
    action_dim = int(data["action_dim"])
    n_beads = int(data["n_beads"])
    max_episodes = int(args.num_seeds if args.num_seeds is not None else args.max_episodes_per_condition)
    primary_pair = f"free_vs_{args.primary_hidden_condition}"
    diagnostic_pair = f"free_vs_{args.diagnostic_hidden_condition}"

    ckpt_root = Path(args.checkpoint_root)
    if not ckpt_root.is_absolute():
        ckpt_root = root / ckpt_root
    checkpoints = []
    if args.checkpoint_dir and args.baseline:
        checkpoints.append((args.baseline, Path(args.checkpoint_dir)))
    else:
        for baseline in args.rollout_models:
            checkpoints.append((baseline, first_checkpoint(ckpt_root, baseline)))

    trials: List[Dict[str, Any]] = []
    for baseline, ckpt in checkpoints:
        state_model = load_future_model(ckpt / "state_model.pt")
        idm = load_inverse_model(ckpt / "inverse_dynamics.pt")
        assert_no_tensorflow_loaded("after_checkpoint_load")
        for condition in REQUIRED_CONDITIONS:
            for episode_idx in range(max_episodes):
                visible_seed = args.seed_start + episode_idx
                random.seed(visible_seed)
                np.random.seed(visible_seed)
                os.environ["CCDA_HIDDEN_CONDITION"] = condition
                os.environ["CCDA_VISIBLE_SEED"] = str(visible_seed)
                os.environ["CCDA_PAIR_GROUP"] = f"phase3_4_rollout_seed_{visible_seed}"

                task = tasks.names["hidden-contact-cable-line"]()
                task.mode = "train"
                assert_no_tensorflow_loaded("before_environment_create")
                env = Environment(disp=False, hz=240)
                env.t_lim = float(args.motion_timeout)
                assert_no_tensorflow_loaded("after_environment_create")
                info: Dict[str, Any] = {}
                state_hist: List[np.ndarray] = []
                action_hist: List[np.ndarray] = []
                ood_scores: List[float] = []
                failure = ""
                success = False
                final_fraction = float("nan")
                final_curve = float("nan")

                try:
                    assert_no_tensorflow_loaded("before_rollout_loop")
                    env.reset(task)
                    reward_extras = task.reward()[1]
                    info = env.info
                    reward_extras["task.done"] = task.done()
                    info["extras"] = reward_extras
                    prev_xy = None
                    done = False
                    for step in range(int(args.max_steps)):
                        # Policy input must use observable bead/proprio history and past action history only.
                        # Do not concatenate condition labels, hidden_contact_meta, recoverability params,
                        # breakaway release fields, success labels, final_fraction, or future labels.
                        state = state_from_live_info(info, prev_xy=prev_xy)
                        prev_xy = state[: n_beads * 2].reshape(n_beads, 2)
                        state_hist.append(state)
                        hist_states = pad_history(state_hist, th).reshape(-1)
                        if action_hist:
                            hist_actions = pad_history(action_hist, th).reshape(-1)
                        else:
                            hist_actions = np.zeros((th * action_dim,), dtype=np.float32)
                        if baseline == "paper_state":
                            model_x = hist_states.reshape(1, -1)
                        else:
                            model_x = np.concatenate([hist_states, hist_actions], axis=0).reshape(1, -1)
                        samples = state_model.sample(model_x, n_samples=int(args.samples_per_step), seed=visible_seed + step)[:, 0, :]
                        mean_future = np.mean(samples, axis=0, keepdims=True)
                        idm_x = np.concatenate([hist_states.reshape(1, -1), mean_future], axis=1)
                        raw_action_vec = idm.predict(idm_x)[0]
                        ood_scores.append(float(idm.ood_score(raw_action_vec.reshape(1, -1))[0]))
                        lo = idm.train_action_mean - float(args.action_clip_std) * idm.train_action_std
                        hi = idm.train_action_mean + float(args.action_clip_std) * idm.train_action_std
                        action_vec = np.clip(raw_action_vec, lo, hi).astype(np.float32)
                        action = codec.decode(action_vec)
                        obs, reward, done, info = env.step(action)
                        action_hist.append(action_vec.astype(np.float32))
                        if done:
                            break
                    success = bool(info.get("extras", {}).get("task.done", done))
                    final_fraction = final_fraction_from_info(info)
                    try:
                        final_state = state_from_live_info(info)
                        final_curve = curve_metric_from_state(final_state, n_beads)
                    except Exception:
                        pass
                except Exception as exc:
                    failure = repr(exc)
                finally:
                    close_env_safely(env)

                row = {
                    "baseline": baseline,
                    "checkpoint_dir": str(ckpt),
                    "condition": condition,
                    "visible_seed": visible_seed,
                    "episode_idx": episode_idx,
                    "success": int(success),
                    "final_fraction": final_fraction,
                    "final_curve": final_curve,
                    "num_steps": len(action_hist),
                    "mean_action_ood_score": float(np.mean(ood_scores)) if ood_scores else float("nan"),
                    "predicted_branch": "nearest_branch_proxy_unavailable",
                    "failure_reason": failure,
                    "primary_hidden_condition": args.primary_hidden_condition,
                    "diagnostic_hidden_condition": args.diagnostic_hidden_condition,
                    "primary_pair": primary_pair,
                    "diagnostic_pair": diagnostic_pair,
                    "selected_recoverable_config": args.selected_recoverable_config,
                    "rollout_runtime": "tf_free_learned_policy",
                }
                trials.append(row)
                print("[Phase3.4] rollout", row, flush=True)

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    if not out_json.is_absolute():
        out_json = root / out_json
    if not out_md.is_absolute():
        out_md = root / out_md
    write_outputs(trials, out_csv, out_json, out_md, args)
    print("[Phase3.4] wrote", out_csv)
    print("[Phase3.4] wrote", out_json)
    print("[Phase3.4] wrote", out_md)


if __name__ == "__main__":
    main()
