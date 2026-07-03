import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


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
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std = np.where(self.std < 1e-6, 1.0, self.std).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "Standardizer":
        return cls(np.mean(x, axis=0), np.std(x, axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float32) * self.std + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Standardizer":
        return cls(np.asarray(d["mean"], dtype=np.float32), np.asarray(d["std"], dtype=np.float32))


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


class NumpyFutureModel:
    def __init__(self, x_std: Standardizer, y_std: Standardizer, weights: np.ndarray, residual_std: np.ndarray, config: Dict[str, Any]):
        self.x_std = x_std
        self.y_std = y_std
        self.weights = weights
        self.residual_std = np.asarray(residual_std, dtype=np.float32)
        self.config = dict(config)

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        pred_z = ridge_predict(self.x_std.transform(x), self.weights)
        return self.y_std.inverse(pred_z)

    def sample(self, x: np.ndarray, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        mean = self.predict_mean(x)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.residual_std.reshape(1, 1, -1), size=(n_samples, mean.shape[0], mean.shape[1]))
        return mean.reshape(1, mean.shape[0], mean.shape[1]) + noise.astype(np.float32)

    def save(self, path: Path) -> None:
        with Path(path).open("wb") as f:
            pickle.dump({
                "backend": "numpy_ridge_diffusion_surrogate",
                "x_std": self.x_std.to_dict(),
                "y_std": self.y_std.to_dict(),
                "weights": self.weights,
                "residual_std": self.residual_std,
                "config": self.config,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "NumpyFutureModel":
        with Path(path).open("rb") as f:
            obj = pickle.load(f)
        return cls(Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["weights"], obj["residual_std"], obj.get("config", {}))


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


def train_numpy_future_model(x: np.ndarray, y: np.ndarray, config: Dict[str, Any]) -> NumpyFutureModel:
    y_flat = y.reshape(len(y), -1).astype(np.float32)
    x_std = Standardizer.fit(x)
    y_std = Standardizer.fit(y_flat)
    xz = x_std.transform(x)
    yz = y_std.transform(y_flat)
    w = ridge_fit(xz, yz, l2=float(config.get("ridge_l2", 1e-4)))
    pred = ridge_predict(xz, w)
    residual = yz - pred
    residual_std = np.std(residual, axis=0).astype(np.float32)
    residual_std = np.maximum(residual_std, 1e-4)
    return NumpyFutureModel(x_std, y_std, w, residual_std, config)


def train_numpy_inverse_model(x: np.ndarray, y_action: np.ndarray, config: Dict[str, Any]) -> NumpyInverseDynamics:
    x_std = Standardizer.fit(x)
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




class TorchFutureModel:
    def __init__(self, model, x_std: Standardizer, y_std: Standardizer, residual_std: np.ndarray, config: Dict[str, Any], device: str = "cpu"):
        self.model = model
        self.x_std = x_std
        self.y_std = y_std
        self.residual_std = np.asarray(residual_std, dtype=np.float32)
        self.config = dict(config)
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict_mean(self, x: np.ndarray) -> np.ndarray:
        import torch
        xz = self.x_std.transform(x).astype(np.float32)
        with torch.no_grad():
            xt = torch.from_numpy(xz).to(self.device)
            pred = self.model(xt).cpu().numpy()
        return self.y_std.inverse(pred)

    def sample(self, x: np.ndarray, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        mean = self.predict_mean(x)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, self.residual_std.reshape(1, 1, -1), size=(n_samples, mean.shape[0], mean.shape[1]))
        return mean.reshape(1, mean.shape[0], mean.shape[1]) + noise.astype(np.float32)

    def save(self, path: Path) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "backend": "torch",
            "model_type": "torch_mlp_future_state",
            "state_dict": self.model.cpu().state_dict(),
            "x_dim": int(self.x_std.mean.size),
            "y_dim": int(self.y_std.mean.size),
            "x_std": self.x_std.to_dict(),
            "y_std": self.y_std.to_dict(),
            "residual_std": self.residual_std,
            "config": self.config,
        }, path)
        self.model.to(self.device)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "TorchFutureModel":
        import torch
        obj = torch.load(path, map_location=device)
        model = make_mlp(int(obj["x_dim"]), int(obj["y_dim"]), int(obj.get("hidden_dim", obj.get("config", {}).get("hidden_dim", 256))))
        model.load_state_dict(obj["state_dict"])
        return cls(model, Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["residual_std"], obj.get("config", {}), device=device)


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
            "model_type": "torch_mlp_inverse_dynamics",
            "state_dict": self.model.cpu().state_dict(),
            "x_dim": int(self.x_std.mean.size),
            "y_dim": int(self.y_std.mean.size),
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
        obj = torch.load(path, map_location=device)
        model = make_mlp(int(obj["x_dim"]), int(obj["y_dim"]), int(obj.get("hidden_dim", obj.get("config", {}).get("hidden_dim", 256))))
        model.load_state_dict(obj["state_dict"])
        return cls(model, Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["train_action_mean"], obj["train_action_std"], obj.get("config", {}), device=device)


def make_mlp(x_dim: int, y_dim: int, hidden_dim: int = 256):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(x_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, y_dim),
    )


def _train_torch_regressor(x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, seed: int, hidden_dim: int = 256, lr: float = 1e-3):
    import torch
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_std = Standardizer.fit(x)
    y_std = Standardizer.fit(y)
    xz = x_std.transform(x).astype(np.float32)
    yz = y_std.transform(y).astype(np.float32)
    model = make_mlp(xz.shape[1], yz.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xt.to(device)).cpu().numpy()
    residual = yz - pred
    residual_std = np.maximum(np.std(residual, axis=0).astype(np.float32), 1e-4)
    return model, x_std, y_std, residual_std, device


def train_torch_future_model(x: np.ndarray, y: np.ndarray, config: Dict[str, Any], epochs: int, batch_size: int, seed: int) -> TorchFutureModel:
    y_flat = y.reshape(len(y), -1).astype(np.float32)
    model, x_std, y_std, residual_std, device = _train_torch_regressor(x, y_flat, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))
    return TorchFutureModel(model, x_std, y_std, residual_std, config, device=device)


def train_torch_inverse_model(x: np.ndarray, y_action: np.ndarray, config: Dict[str, Any], epochs: int, batch_size: int, seed: int) -> TorchInverseDynamics:
    model, x_std, y_std, _residual_std, device = _train_torch_regressor(x, y_action, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))
    return TorchInverseDynamics(model, x_std, y_std, np.mean(y_action, axis=0), np.std(y_action, axis=0), config, device=device)


def _torch_checkpoint_backend(path: Path) -> str:
    try:
        import torch
        obj = torch.load(path, map_location="cpu")
        return str(obj.get("backend", ""))
    except Exception:
        return ""


def load_future_model(path: Path):
    if _torch_checkpoint_backend(Path(path)) == "torch":
        return TorchFutureModel.load(Path(path), device="cuda" if _torch_cuda_available() else "cpu")
    return NumpyFutureModel.load(Path(path))


def load_inverse_model(path: Path):
    if _torch_checkpoint_backend(Path(path)) == "torch":
        return TorchInverseDynamics.load(Path(path), device="cuda" if _torch_cuda_available() else "cpu")
    return NumpyInverseDynamics.load(Path(path))


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False
