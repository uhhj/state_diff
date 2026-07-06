#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PRIMARY = "hidden_breakaway_pin"
FORBIDDEN_PREFIXES = ["tensorflow", "ravens.agents", "ravens.models", "ravens.datasets"]


def assert_gate() -> None:
    if os.environ.get("PHASE3_ALLOW_GEOMETRY_IDM_TRAIN", "0") != "1":
        raise SystemExit("[Phase3.9][BLOCKED] Set PHASE3_ALLOW_GEOMETRY_IDM_TRAIN=1")
    if os.environ.get("PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED", "0") != "1":
        raise SystemExit("[Phase3.9][BLOCKED] Set PHASE3_GEOMETRY_IDM_TRAIN_CONFIRMED=1")


def assert_no_forbidden(stage: str) -> None:
    bad: List[str] = []
    for name in sys.modules:
        for prefix in FORBIDDEN_PREFIXES:
            if name == prefix or name.startswith(prefix + "."):
                bad.append(name)
    if bad:
        raise RuntimeError(f"[Phase3.9][FAIL] forbidden modules loaded at {stage}: {bad[:20]}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_windows(root: Path, path: str) -> Tuple[Any, Dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    data = np.load(p, allow_pickle=True)
    meta: Dict[str, Any] = {}
    if "meta_json" in data.files:
        raw = data["meta_json"]
        try:
            text = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
            meta = json.loads(text)
        except Exception:
            meta = {}
    return data, meta


def load_codec(root: Path, data: Any, fallback: str) -> Any:
    candidates: List[Path] = []
    if "action_template_json_or_pickle_path" in data.files:
        raw = data["action_template_json_or_pickle_path"]
        val = str(raw.item() if getattr(raw, "shape", ()) == () else raw.reshape(-1)[0])
        candidates.append(Path(val))
    candidates.append(Path(fallback))
    for cand in candidates:
        p = cand if cand.is_absolute() else root / cand
        if not p.exists():
            continue
        try:
            from ccda_phase3.data_io import load_action_codec_from_template
            return load_action_codec_from_template(p)
        except Exception:
            with p.open("rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("action template not found")


def codec_paths(codec: Any) -> List[str]:
    if hasattr(codec, "summary"):
        s = codec.summary()
        return [str(x) for x in s.get("paths", [])]
    return [str(x) for x in getattr(codec, "path_strings", getattr(codec, "paths", []))]


def idx_for(paths: List[str], suffix: str) -> int:
    for i, p in enumerate(paths):
        if p == suffix or p.endswith(suffix):
            return i
    raise KeyError(f"missing codec path {suffix}")


def make_indices(paths: List[str]) -> Dict[str, Any]:
    return {
        "p0xy": [idx_for(paths, "params/pose0/0/0"), idx_for(paths, "params/pose0/0/1")],
        "p1xy": [idx_for(paths, "params/pose1/0/0"), idx_for(paths, "params/pose1/0/1")],
        "p0z": [idx_for(paths, "params/pose0/0/2")],
        "p1z": [idx_for(paths, "params/pose1/0/2")],
        "q0": [idx_for(paths, f"params/pose0/1/{i}") for i in range(4)],
        "q1": [idx_for(paths, f"params/pose1/1/{i}") for i in range(4)],
    }


def first_checkpoint(root: Path, checkpoint_root: str, baseline: str) -> Path:
    base = root / checkpoint_root / baseline
    for cand in sorted(p for p in base.glob("fold_*_seed_*") if p.is_dir()):
        if (cand / "state_model.pt").exists() and (cand / "inverse_dynamics.pt").exists():
            return cand
    raise FileNotFoundError(f"No complete checkpoint under {base}")


def infer_future_key_from_old_idm(root: Path, data: Any, old_inverse_path: Path) -> str:
    from ccda_phase3.train_utils import load_inverse_model

    idm = load_inverse_model(old_inverse_path)
    idm_x_dim = int(idm.x_std.mean.size)
    paper_dim = int(np.asarray(data["paper_x"]).shape[1])
    future_dim = idm_x_dim - paper_dim
    if future_dim <= 0:
        raise RuntimeError(f"bad inferred future_dim={future_dim}; idm_x_dim={idm_x_dim}, paper_dim={paper_dim}")

    for key in ["y_state", "y_final_state", "y_future_state", "future_state"]:
        if key in data.files:
            arr = np.asarray(data[key])
            if int(np.prod(arr.shape[1:])) == future_dim:
                return key
    available = {k: list(np.asarray(data[k]).shape) for k in data.files if k.startswith("y")}
    raise RuntimeError(f"No future key matches old IDM future_dim={future_dim}. Available={available}")


def build_xy_idm_dataset(data: Any, future_key: str, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    paper_x = np.asarray(data["paper_x"], dtype=np.float32)
    future = np.asarray(data[future_key], dtype=np.float32).reshape(len(paper_x), -1)
    y = np.asarray(data["y_action"], dtype=np.float32)
    x = np.concatenate([paper_x, future], axis=1).astype(np.float32)
    return x[mask], y[mask]


def split_masks(data: Any) -> Tuple[np.ndarray, np.ndarray]:
    n = int(len(data["y_action"]))
    if "split_name" not in data.files:
        train = np.ones(n, dtype=bool)
        val = np.zeros(n, dtype=bool)
        return train, val
    split = np.asarray([str(x) for x in data["split_name"]])
    train = split == "train"
    val = split == "heldout"
    if not np.any(train):
        train = np.ones(n, dtype=bool)
    return train, val


def mse_t(x):
    import torch
    return torch.mean(x * x)


def train_geometry_idm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    idxs: Dict[str, Any],
    config: Dict[str, Any],
    seed: int,
) -> Tuple[Any, Dict[str, Any]]:
    import torch

    from ccda_phase3.models import InverseDynamicsMLP
    from ccda_phase3.train_utils import Standardizer, TorchInverseDynamics

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x_std = Standardizer.fit(x_train, std_floor=float(config.get("idm_x_std_floor", 0.1)))
    y_std = Standardizer.fit(y_train)

    xz = x_std.transform(x_train).astype(np.float32)
    yz = y_std.transform(y_train).astype(np.float32)

    hidden_dim = int(config.get("idm_hidden_dim", 256))
    model = InverseDynamicsMLP(xz.shape[1], yz.shape[1], hidden_dim=hidden_dim).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("idm_lr", 1e-3)),
        weight_decay=float(config.get("idm_weight_decay", 1e-3)),
    )

    xt = torch.from_numpy(xz).float()
    yt_z = torch.from_numpy(yz).float()
    yt_raw = torch.from_numpy(y_train.astype(np.float32)).float()

    y_mean = torch.from_numpy(y_std.mean.astype(np.float32)).to(device)
    y_std_t = torch.from_numpy(y_std.std.astype(np.float32)).to(device)

    p0xy = torch.as_tensor(idxs["p0xy"], dtype=torch.long, device=device)
    p1xy = torch.as_tensor(idxs["p1xy"], dtype=torch.long, device=device)
    p0z = torch.as_tensor(idxs["p0z"], dtype=torch.long, device=device)
    p1z = torch.as_tensor(idxs["p1z"], dtype=torch.long, device=device)
    quat = torch.as_tensor(idxs["q0"] + idxs["q1"], dtype=torch.long, device=device)

    n = len(xt)
    batch_size = max(1, min(int(config.get("batch_size", 128)), n))
    epochs = int(config.get("epochs", 300))

    w_action_z = float(config.get("action_z_weight", 1.0))
    w_p0 = float(config.get("pose0_xy_weight", 8.0))
    w_p1 = float(config.get("pose1_xy_weight", 8.0))
    w_coupled = float(config.get("coupled_xy_weight", 8.0))
    w_pull = float(config.get("pull_xy_weight", 4.0))
    w_z = float(config.get("z_weight", 0.25))
    w_quat = float(config.get("quat_weight", 0.02))

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n)
        losses = []
        geom_losses = []
        for start in range(0, n, batch_size):
            ids = perm[start:start + batch_size]
            xb = xt[ids].to(device)
            yb_z = yt_z[ids].to(device)
            yb_raw = yt_raw[ids].to(device)

            pred_z = model(xb)
            pred_raw = pred_z * y_std_t + y_mean

            action_z_loss = mse_t(pred_z - yb_z)
            p0_loss = mse_t(pred_raw[:, p0xy] - yb_raw[:, p0xy])
            p1_loss = mse_t(pred_raw[:, p1xy] - yb_raw[:, p1xy])
            coupled_loss = mse_t(torch.cat([pred_raw[:, p0xy], pred_raw[:, p1xy]], dim=1) - torch.cat([yb_raw[:, p0xy], yb_raw[:, p1xy]], dim=1))
            pull_loss = mse_t((pred_raw[:, p1xy] - pred_raw[:, p0xy]) - (yb_raw[:, p1xy] - yb_raw[:, p0xy]))
            z_loss = mse_t(torch.cat([pred_raw[:, p0z], pred_raw[:, p1z]], dim=1) - torch.cat([yb_raw[:, p0z], yb_raw[:, p1z]], dim=1))
            quat_loss = mse_t(pred_raw[:, quat] - yb_raw[:, quat])

            loss = (
                w_action_z * action_z_loss
                + w_p0 * p0_loss
                + w_p1 * p1_loss
                + w_coupled * coupled_loss
                + w_pull * pull_loss
                + w_z * z_loss
                + w_quat * quat_loss
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 1.0)))
            opt.step()

            losses.append(float(loss.detach().cpu()))
            geom_losses.append(float((p0_loss + p1_loss + pull_loss).detach().cpu()))

        if epoch == 1 or epoch % int(config.get("log_every", 25)) == 0 or epoch == epochs:
            history.append({
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_geom_loss": float(np.mean(geom_losses)),
            })

    final_config = dict(config)
    final_config.update({
        "phase": "phase3_9_geometry_aware_idm",
        "geometry_loss_used": True,
        "future_ddpm_trained": False,
        "phase4_or_cps": False,
        "idxs": idxs,
        "train_history": history,
    })

    wrapper = TorchInverseDynamics(
        model=model,
        x_std=x_std,
        y_std=y_std,
        train_action_mean=np.mean(y_train, axis=0),
        train_action_std=np.std(y_train, axis=0),
        config=final_config,
        device=device,
    )

    metrics: Dict[str, Any] = {
        "history": history,
        "train_rows": int(len(x_train)),
        "val_rows": int(len(x_val)),
    }

    if len(x_val) > 0:
        pred_val = wrapper.predict(x_val)
        metrics["val_action_mae"] = float(np.mean(np.abs(pred_val - y_val)))
        metrics["val_pose0_xy_mae"] = float(np.mean(np.abs(pred_val[:, idxs["p0xy"]] - y_val[:, idxs["p0xy"]])))
        metrics["val_pose1_xy_mae"] = float(np.mean(np.abs(pred_val[:, idxs["p1xy"]] - y_val[:, idxs["p1xy"]])))
        metrics["val_pull_xy_mae"] = float(np.mean(np.abs((pred_val[:, idxs["p1xy"]] - pred_val[:, idxs["p0xy"]]) - (y_val[:, idxs["p1xy"]] - y_val[:, idxs["p0xy"]]))))
    return wrapper, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/state_diff2")
    parser.add_argument("--windows", default="data/phase3_state_diff_windows/phase3_windows.npz")
    parser.add_argument("--action_template", default="data/phase3_state_diff_windows/phase3_action_template.pkl")
    parser.add_argument("--old_checkpoint_root", default="checkpoints/phase3")
    parser.add_argument("--baseline", default="state_action")
    parser.add_argument("--out_dir", default="checkpoints/phase3_9_geometry_idm/state_action/fold_phase3_9_seed_390000")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--idm_hidden_dim", type=int, default=256)
    parser.add_argument("--idm_lr", type=float, default=1e-3)
    parser.add_argument("--idm_weight_decay", type=float, default=1e-3)
    parser.add_argument("--action_z_weight", type=float, default=1.0)
    parser.add_argument("--pose0_xy_weight", type=float, default=8.0)
    parser.add_argument("--pose1_xy_weight", type=float, default=8.0)
    parser.add_argument("--coupled_xy_weight", type=float, default=8.0)
    parser.add_argument("--pull_xy_weight", type=float, default=4.0)
    parser.add_argument("--z_weight", type=float, default=0.25)
    parser.add_argument("--quat_weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=390000)
    parser.add_argument("--summary_json", default="reports/phase3_9_geometry_idm_train_summary.json")
    parser.add_argument("--summary_md", default="reports/phase3_9_geometry_idm_train_report.md")
    args = parser.parse_args()

    assert_gate()
    root = Path(args.root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    data, meta = load_windows(root, args.windows)
    if meta.get("primary_hidden_condition") != PRIMARY:
        raise SystemExit(f"[Phase3.9][FAIL] bad primary hidden condition: {meta.get('primary_hidden_condition')}")

    codec = load_codec(root, data, args.action_template)
    paths = codec_paths(codec)
    if len(paths) != 14:
        raise SystemExit(f"[Phase3.9][FAIL] action codec must have 14 paths, got {len(paths)}")
    idxs = make_indices(paths)

    old_ckpt = first_checkpoint(root, args.old_checkpoint_root, args.baseline)
    future_key = infer_future_key_from_old_idm(root, data, old_ckpt / "inverse_dynamics.pt")

    train_mask, val_mask = split_masks(data)
    x_train, y_train = build_xy_idm_dataset(data, future_key, train_mask)
    x_val, y_val = build_xy_idm_dataset(data, future_key, val_mask)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "idm_hidden_dim": args.idm_hidden_dim,
        "idm_lr": args.idm_lr,
        "idm_weight_decay": args.idm_weight_decay,
        "action_z_weight": args.action_z_weight,
        "pose0_xy_weight": args.pose0_xy_weight,
        "pose1_xy_weight": args.pose1_xy_weight,
        "coupled_xy_weight": args.coupled_xy_weight,
        "pull_xy_weight": args.pull_xy_weight,
        "z_weight": args.z_weight,
        "quat_weight": args.quat_weight,
        "seed": args.seed,
        "baseline": args.baseline,
        "future_key": future_key,
        "old_checkpoint": str(old_ckpt),
        "note": "Phase3.9 trains inverse dynamics only; no future DDPM, no CPS.",
    }

    wrapper, metrics = train_geometry_idm(x_train, y_train, x_val, y_val, idxs, config, args.seed)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    inverse_path = out_dir / "inverse_dynamics.pt"
    wrapper.save(inverse_path)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    payload = {
        "verdict": "PASS",
        "scope": "phase3_9_geometry_aware_idm_train_no_phase4_no_cps",
        "inverse_dynamics_path": str(inverse_path),
        "config_path": str(config_path),
        "old_checkpoint": str(old_ckpt),
        "baseline": args.baseline,
        "future_key": future_key,
        "codec_paths": paths,
        "idxs": idxs,
        "config": config,
        "metrics": metrics,
        "important_note": "Checkpoint is local diagnostic artifact and must not be committed.",
    }

    summary_json = root / args.summary_json
    summary_md = root / args.summary_md
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True))

    lines = [
        "# Phase3.9 Geometry-Aware IDM Train Report",
        "",
        "## Verdict",
        "",
        "- Verdict: `PASS`",
        f"- Baseline: `{args.baseline}`",
        f"- Future key: `{future_key}`",
        f"- Train rows: `{metrics['train_rows']}`",
        f"- Val rows: `{metrics['val_rows']}`",
        f"- Inverse dynamics path: `{inverse_path}`",
        "",
        "## Validation Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["val_action_mae", "val_pose0_xy_mae", "val_pose1_xy_mae", "val_pull_xy_mae"]:
        if key in metrics:
            lines.append(f"| `{key}` | `{float(metrics[key]):.6f}` |")

    lines += [
        "",
        "## Scope",
        "",
        "- Inverse dynamics only.",
        "- No future DDPM training.",
        "- No Phase4.",
        "- No CPS.",
        "- Checkpoint must not be committed.",
    ]
    summary_md.write_text("\n".join(lines) + "\n")

    print(json.dumps(payload, indent=2, sort_keys=True))
    assert_no_forbidden("end")


if __name__ == "__main__":
    main()
