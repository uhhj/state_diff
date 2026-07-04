#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DDPM_MODEL_TYPE = "torch_conditional_ddpm_future_state"
SCHEDULER_TYPE = "diffusers.DDPMScheduler"
PAPER_ALIGNMENT_LEVEL = "ddpm_scheduler_aligned_mlp_denoiser"


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


class Standardizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray, std_floor: float = 1e-6):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        floor = float(std_floor)
        self.std = np.where(self.std < floor, floor, self.std).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, std_floor: float = 1e-6) -> "Standardizer":
        return cls(np.mean(x, axis=0), np.std(x, axis=0), std_floor=std_floor)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Standardizer":
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))


def make_ddpm_scheduler(
    num_train_timesteps: int,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "epsilon",
    variance_type: str = "fixed_small",
    clip_sample: bool = True,
):
    try:
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    except Exception as exc:
        raise RuntimeError(
            "diffusers is required for Phase3 paper-aligned DDPM scheduler. "
            "Install it in coord_bimanual: python -m pip install diffusers"
        ) from exc

    return DDPMScheduler(
        num_train_timesteps=int(num_train_timesteps),
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule=str(beta_schedule),
        variance_type=str(variance_type),
        clip_sample=bool(clip_sample),
        prediction_type=str(prediction_type),
    )


def normalize_ddpm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(config)
    diffusion_steps = int(cfg.get("diffusion_steps", cfg.get("num_train_timesteps", 100)))
    cfg.update({
        "future_model_type": DDPM_MODEL_TYPE,
        "ddpm_used": True,
        "scheduler_type": SCHEDULER_TYPE,
        "beta_schedule": str(cfg.get("beta_schedule", "squaredcos_cap_v2")),
        "prediction_type": str(cfg.get("prediction_type", "epsilon")),
        "variance_type": str(cfg.get("variance_type", "fixed_small")),
        "clip_sample": bool(cfg.get("clip_sample", True)),
        "diffusion_steps": diffusion_steps,
        "num_train_timesteps": int(cfg.get("num_train_timesteps", diffusion_steps)),
        "num_inference_steps": int(cfg.get("num_inference_steps", diffusion_steps)),
        "sample_temperature": float(cfg.get("sample_temperature", 1.0)),
        "denoiser_arch": str(cfg.get("denoiser_arch", "mlp")),
        "conditional_unet1d_used": bool(cfg.get("conditional_unet1d_used", False)),
        "paper_alignment_level": str(cfg.get("paper_alignment_level", PAPER_ALIGNMENT_LEVEL)),
    })
    if cfg["scheduler_type"] != SCHEDULER_TYPE:
        raise ValueError(f"scheduler_type must be {SCHEDULER_TYPE}, got {cfg['scheduler_type']}")
    if cfg["beta_schedule"] != "squaredcos_cap_v2":
        raise ValueError("beta_schedule must be squaredcos_cap_v2")
    if cfg["prediction_type"] != "epsilon":
        raise ValueError("prediction_type must be epsilon")
    if cfg["variance_type"] == "learned_range":
        raise ValueError("learned_range requires 2*y_dim output and is not enabled for the pre-medium MLP denoiser")
    if cfg["denoiser_arch"] != "mlp":
        raise ValueError("only denoiser_arch=mlp is enabled in this pre-medium alignment commit")
    if cfg["conditional_unet1d_used"]:
        raise ValueError("conditional_unet1d_used must be false for this pre-medium MLP denoiser")
    return cfg


def ridge_fit(x: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    eye = np.eye(xb.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    w = np.linalg.solve(xb.T @ xb + l2 * eye, xb.T @ y)
    return w.astype(np.float32)


def ridge_predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.asarray(x, dtype=np.float32), np.ones((len(x), 1), dtype=np.float32)], axis=1)
    return xb @ np.asarray(w, dtype=np.float32)


class NumpyInverseDynamics:
    def __init__(self, x_std: Standardizer, y_std: Standardizer, weights: np.ndarray, train_action_mean: np.ndarray, train_action_std: np.ndarray, config: Dict[str, Any]):
        self.x_std = x_std
        self.y_std = y_std
        self.weights = weights
        self.train_action_mean = np.asarray(train_action_mean, dtype=np.float32)
        self.train_action_std = np.where(np.asarray(train_action_std, dtype=np.float32) < 1e-6, 1.0, train_action_std).astype(np.float32)
        self.config = dict(config)

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_z = ridge_predict(self.x_std.transform(x), self.weights)
        return self.y_std.inverse(pred_z)

    def ood_score(self, action: np.ndarray) -> np.ndarray:
        z = (np.asarray(action, dtype=np.float32) - self.train_action_mean) / self.train_action_std
        return np.sqrt(np.mean(z * z, axis=-1))

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump({
                "backend": "numpy_ridge_inverse_dynamics",
                "x_std": self.x_std.to_dict(),
                "y_std": self.y_std.to_dict(),
                "weights": self.weights,
                "train_action_mean": self.train_action_mean,
                "train_action_std": self.train_action_std,
                "config": self.config,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "NumpyInverseDynamics":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return cls(Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["weights"], obj["train_action_mean"], obj["train_action_std"], obj.get("config", {}))


def train_numpy_inverse_model(x: np.ndarray, y_action: np.ndarray, config: Dict[str, Any]) -> NumpyInverseDynamics:
    x_std = Standardizer.fit(x, std_floor=float(config.get("idm_x_std_floor", 0.1)))
    y_std = Standardizer.fit(y_action)
    w = ridge_fit(x_std.transform(x), y_std.transform(y_action), l2=float(config.get("ridge_l2", 1e-4)))
    return NumpyInverseDynamics(x_std, y_std, w, np.mean(y_action, axis=0), np.std(y_action, axis=0), config)


def grouped_folds(seeds: Sequence[int], folds: int) -> List[Tuple[List[int], List[int]]]:
    unique = sorted({int(s) for s in seeds})
    if not unique:
        return []
    folds = max(1, min(int(folds), len(unique)))
    out = []
    for k in range(folds):
        val = unique[k::folds]
        train = [s for s in unique if s not in set(val)]
        if not train:
            train = val
        out.append((train, val))
    return out


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


class TorchDDPMFutureModel:
    def __init__(self, model, config: Dict[str, Any], device: str = "cpu"):
        self.model = model.to(device)
        self.config = normalize_ddpm_config(config)
        self.device = device
        self.model.eval()

    def sample(self, x: np.ndarray, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        import torch
        scheduler = make_ddpm_scheduler(
            num_train_timesteps=int(self.config.get("num_train_timesteps", self.config.get("diffusion_steps", 100))),
            beta_schedule=str(self.config.get("beta_schedule", "squaredcos_cap_v2")),
            prediction_type=str(self.config.get("prediction_type", "epsilon")),
            variance_type=str(self.config.get("variance_type", "fixed_small")),
            clip_sample=bool(self.config.get("clip_sample", True)),
        )
        num_inference_steps = int(self.config.get("num_inference_steps", self.config.get("diffusion_steps", 100)))
        scheduler.set_timesteps(num_inference_steps)

        torch.manual_seed(int(seed))
        device = self.device
        x_t = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        batch = x_t.shape[0]
        x_rep = x_t.repeat_interleave(n_samples, dim=0)

        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        y_z = torch.randn((batch * n_samples, self.model.y_dim), device=device, generator=generator)
        y_z = y_z * float(self.config.get("sample_temperature", 1.0))

        self.model.eval()
        with torch.no_grad():
            for t in scheduler.timesteps:
                tt = torch.full((y_z.shape[0],), int(t), device=device, dtype=torch.long)
                eps = self.model(x_rep, y_z, tt)
                try:
                    step_out = scheduler.step(eps, t, y_z, generator=generator)
                except TypeError:
                    step_out = scheduler.step(eps, t, y_z)
                y_z = step_out.prev_sample

        y_raw = self.model.unstandardize_y(y_z)
        return y_raw.detach().cpu().numpy().reshape(batch, n_samples, self.model.y_dim).transpose(1, 0, 2)

    def predict_mean(self, x: np.ndarray, n_samples: int = 16, seed: int = 0) -> np.ndarray:
        samples = self.sample(x, n_samples=n_samples, seed=seed)
        return np.mean(samples, axis=0)

    def save(self, path: Path) -> None:
        import torch
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = normalize_ddpm_config(self.config)
        torch.save(
            {
                "backend": "torch",
                "model_type": DDPM_MODEL_TYPE,
                "ddpm_used": True,
                "scheduler_type": cfg["scheduler_type"],
                "beta_schedule": cfg["beta_schedule"],
                "prediction_type": cfg["prediction_type"],
                "variance_type": cfg["variance_type"],
                "clip_sample": cfg["clip_sample"],
                "num_train_timesteps": cfg["num_train_timesteps"],
                "num_inference_steps": cfg["num_inference_steps"],
                "sample_temperature": cfg["sample_temperature"],
                "denoiser_arch": cfg["denoiser_arch"],
                "paper_alignment_level": cfg["paper_alignment_level"],
                "conditional_unet1d_used": cfg["conditional_unet1d_used"],
                "state_dict": self.model.cpu().state_dict(),
                "x_dim": self.model.x_dim,
                "y_dim": self.model.y_dim,
                "hidden_dim": self.model.hidden_dim,
                "time_dim": self.model.time_dim,
                "diffusion_steps": self.model.diffusion_steps,
                "x_mean": self.model.x_mean.detach().cpu().numpy(),
                "x_std": self.model.x_std.detach().cpu().numpy(),
                "y_mean": self.model.y_mean.detach().cpu().numpy(),
                "y_std": self.model.y_std.detach().cpu().numpy(),
                "config": cfg,
            },
            path,
        )
        self.model.to(self.device)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TorchDDPMFutureModel":
        import torch
        from ccda_phase3.models import ConditionalStateDDPM
        obj = torch.load(path, map_location=device)
        if obj.get("model_type") != DDPM_MODEL_TYPE or not bool(obj.get("ddpm_used", False)):
            raise RuntimeError("Old or non-DDPM future checkpoint is not allowed in Phase3.")
        cfg = normalize_ddpm_config(obj.get("config", {}))
        if obj.get("scheduler_type", cfg.get("scheduler_type")) != SCHEDULER_TYPE:
            raise RuntimeError("Checkpoint does not use diffusers.DDPMScheduler; retrain Phase3.")
        if obj.get("beta_schedule", cfg.get("beta_schedule")) != "squaredcos_cap_v2":
            raise RuntimeError("Checkpoint beta_schedule is not squaredcos_cap_v2; retrain Phase3.")
        model = ConditionalStateDDPM(
            x_dim=int(obj["x_dim"]),
            y_dim=int(obj["y_dim"]),
            hidden_dim=int(obj.get("hidden_dim", 512)),
            time_dim=int(obj.get("time_dim", 128)),
            diffusion_steps=int(obj.get("diffusion_steps", cfg.get("diffusion_steps", 100))),
            scheduler_type=cfg["scheduler_type"],
            beta_schedule=cfg["beta_schedule"],
            prediction_type=cfg["prediction_type"],
            variance_type=cfg["variance_type"],
            clip_sample=cfg["clip_sample"],
            denoiser_arch=cfg["denoiser_arch"],
        ).to(device)
        model.load_state_dict(obj["state_dict"])
        model.set_standardizers(obj["x_mean"], obj["x_std"], obj["y_mean"], obj["y_std"])
        return cls(model, cfg, device=device)


class TorchInverseDynamics:
    def __init__(self, model, x_std: Standardizer, y_std: Standardizer, train_action_mean: np.ndarray, train_action_std: np.ndarray, config: Dict[str, Any], device: str = "cpu"):
        self.model = model
        self.x_std = x_std
        self.y_std = y_std
        self.train_action_mean = np.asarray(train_action_mean, dtype=np.float32)
        self.train_action_std = np.where(np.asarray(train_action_std, dtype=np.float32) < 1e-6, 1.0, train_action_std).astype(np.float32)
        self.config = dict(config)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch
        xz = self.x_std.transform(x).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(xz).to(self.device)
            pred = self.model(xt).cpu().numpy()
        return self.y_std.inverse(pred)

    def ood_score(self, action: np.ndarray) -> np.ndarray:
        z = (np.asarray(action, dtype=np.float32) - self.train_action_mean) / self.train_action_std
        return np.sqrt(np.mean(z * z, axis=-1))

    def save(self, path: Path) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backend": "torch",
            "model_type": "torch_inverse_dynamics_mlp",
            "state_dict": self.model.cpu().state_dict(),
            "x_dim": int(self.x_std.mean.size),
            "y_dim": int(self.y_std.mean.size),
            "hidden_dim": int(self.config.get("idm_hidden_dim", self.config.get("hidden_dim", 64))),
            "x_std": self.x_std.to_dict(),
            "y_std": self.y_std.to_dict(),
            "train_action_mean": self.train_action_mean,
            "train_action_std": self.train_action_std,
            "config": self.config,
        }, path)
        self.model.to(self.device)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TorchInverseDynamics":
        import torch
        from ccda_phase3.models import InverseDynamicsMLP
        obj = torch.load(path, map_location=device)
        hidden_dim = int(obj.get("hidden_dim", obj.get("config", {}).get("idm_hidden_dim", 64)))
        model = InverseDynamicsMLP(int(obj["x_dim"]), int(obj["y_dim"]), hidden_dim=hidden_dim)
        model.load_state_dict(obj["state_dict"])
        return cls(model, Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["train_action_mean"], obj["train_action_std"], obj.get("config", {}), device=device)


def train_torch_ddpm_future_model(x: np.ndarray, y: np.ndarray, config: Dict[str, Any], epochs: int, batch_size: int, seed: int) -> TorchDDPMFutureModel:
    import torch
    from ccda_phase3.models import ConditionalStateDDPM

    cfg = normalize_ddpm_config(config)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = np.asarray(x, dtype=np.float32)
    y_flat = np.asarray(y, dtype=np.float32).reshape(len(y), -1)

    x_std = Standardizer.fit(x)
    y_std = Standardizer.fit(y_flat)

    model = ConditionalStateDDPM(
        x_dim=x.shape[1],
        y_dim=y_flat.shape[1],
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        time_dim=int(cfg.get("time_dim", 128)),
        diffusion_steps=int(cfg.get("num_train_timesteps", cfg.get("diffusion_steps", 100))),
        scheduler_type=cfg["scheduler_type"],
        beta_schedule=cfg["beta_schedule"],
        prediction_type=cfg["prediction_type"],
        variance_type=cfg["variance_type"],
        clip_sample=cfg["clip_sample"],
        denoiser_arch=cfg["denoiser_arch"],
    ).to(device)
    model.set_standardizers(x_std.mean, x_std.std, y_std.mean, y_std.std)

    scheduler = make_ddpm_scheduler(
        num_train_timesteps=int(cfg["num_train_timesteps"]),
        beta_schedule=cfg["beta_schedule"],
        prediction_type=cfg["prediction_type"],
        variance_type=cfg["variance_type"],
        clip_sample=cfg["clip_sample"],
    )

    xt = torch.from_numpy(x).float()
    y0_z = torch.from_numpy(y_std.transform(y_flat)).float()

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-3)), weight_decay=1e-4)
    batch_size = max(1, min(int(batch_size), len(x)))
    n = len(x)

    losses = []
    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = xt[idx].to(device)
            yb = y0_z[idx].to(device)
            b = xb.shape[0]
            t = torch.randint(0, scheduler.config.num_train_timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(yb)
            y_noisy = scheduler.add_noise(yb, noise, t)
            pred = model(xb, y_noisy, t)
            loss = torch.mean((pred - noise) ** 2)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % 50 == 0 or epoch == int(epochs):
            losses.append({"epoch": epoch, "ddpm_noise_mse": float(np.mean(epoch_losses))})

    cfg["ddpm_train_loss_history"] = losses
    return TorchDDPMFutureModel(model, cfg, device=device)


def train_torch_inverse_model(x: np.ndarray, y_action: np.ndarray, config: Dict[str, Any], epochs: int, batch_size: int, seed: int) -> TorchInverseDynamics:
    import torch
    from ccda_phase3.models import InverseDynamicsMLP

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = np.asarray(x, dtype=np.float32)
    y_action = np.asarray(y_action, dtype=np.float32)
    x_std = Standardizer.fit(x, std_floor=float(config.get("idm_x_std_floor", 0.1)))
    y_std = Standardizer.fit(y_action)
    xz = x_std.transform(x).astype(np.float32)
    yz = y_std.transform(y_action).astype(np.float32)

    hidden_dim = int(config.get("idm_hidden_dim", config.get("hidden_dim", 64)))
    model = InverseDynamicsMLP(xz.shape[1], yz.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(config.get("idm_lr", 1e-3)), weight_decay=float(config.get("idm_weight_decay", 1e-3)))
    loss_fn = torch.nn.MSELoss()
    xt = torch.from_numpy(xz)
    yt = torch.from_numpy(yz)
    n = len(xt)
    batch_size = max(1, min(int(batch_size), n))
    for _ in range(max(1, int(epochs))):
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return TorchInverseDynamics(model, x_std, y_std, np.mean(y_action, axis=0), np.std(y_action, axis=0), config, device=device)


def _torch_checkpoint_obj(path: Path):
    import torch
    return torch.load(Path(path), map_location="cpu")


def load_future_model(path: Path):
    try:
        obj = _torch_checkpoint_obj(Path(path))
    except Exception as exc:
        raise RuntimeError("Old or non-DDPM future checkpoint is not allowed in Phase3.") from exc
    if obj.get("model_type") != DDPM_MODEL_TYPE or not bool(obj.get("ddpm_used", False)):
        raise RuntimeError("Old or non-DDPM future checkpoint is not allowed in Phase3.")
    if obj.get("scheduler_type") != SCHEDULER_TYPE:
        raise RuntimeError("Future checkpoint does not use diffusers.DDPMScheduler; retrain Phase3.")
    if obj.get("beta_schedule") != "squaredcos_cap_v2":
        raise RuntimeError("Future checkpoint beta_schedule is not squaredcos_cap_v2; retrain Phase3.")
    return TorchDDPMFutureModel.load(Path(path), device="cuda" if _torch_cuda_available() else "cpu")


def load_inverse_model(path: Path):
    try:
        obj = _torch_checkpoint_obj(Path(path))
        if obj.get("backend") == "torch":
            return TorchInverseDynamics.load(Path(path), device="cuda" if _torch_cuda_available() else "cpu")
    except Exception:
        pass
    return NumpyInverseDynamics.load(Path(path))


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
