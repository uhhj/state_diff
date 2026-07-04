# Phase3 DDPM Code Audit

- Timestamp: `2026-07-04T19:42:19.102794`
- Branch: `Experiment1`
- Local HEAD: `8b199bc763b526e4dd672e107fc3d49452a941f1`
- origin/Experiment1: `8b199bc763b526e4dd672e107fc3d49452a941f1`
- Local HEAD equals origin/Experiment1: `true`
- Working tree clean before DDPM fix: `true`

## Findings

- Action codec has `ExecutableActionCodec`: `true`
- Action codec still mentions `camera_config` guard: `true`
- Future model still has `_train_torch_regressor`: `true`
- Future model still has residual sampling path: `true`
- Conditional DDPM present before fix: `true`
- Branch reference still keyed only by visible seed: `true`
- Branch reference uses `window_t`: `true`
- DeformableRavens submodule branch: `ccda-cable`
- DeformableRavens submodule HEAD: `b95180593d3366b754a4d8a4488d723bc9d61be5`
- DeformableRavens has `robot_pose_proxy` logging: `true`

## Raw Audit Output

```text
# Phase3 DDPM Code Audit Raw
timestamp: 2026-07-04T19:42:07+08:00

=== git fetch origin ===

=== main repo ===
branch: Experiment1
head: 8b199bc763b526e4dd672e107fc3d49452a941f1
origin_experiment1: 8b199bc763b526e4dd672e107fc3d49452a941f1
status:

=== ActionCodec ===
34:class ExecutableActionCodec:
ccda_phase3/action_codec.py:5:camera_config. Those fields are not executable robot actions and must never
ccda_phase3/action_codec.py:38:      - camera_config
ccda_phase3/action_codec.py:66:        bad = [p for p in self.path_strings if "camera_config" in p]
ccda_phase3/action_codec.py:68:            raise ValueError(f"ExecutableActionCodec must not encode camera_config paths: {bad[:5]}")
ccda_phase3/action_codec.py:208:            "num_camera_config_paths": sum("camera_config" in p for p in self.path_strings),

=== DDPM vs MLP ===
ccda_phase3/models.py:35:    class ConditionalStateDDPM(nn.Module):
ccda_phase3/models.py:104:    class ConditionalStateDDPM:  # type: ignore
ccda_phase3/models.py:106:            raise ImportError("torch is required for ConditionalStateDDPM")
ccda_phase3/train_utils.py:286:def _train_torch_regressor(x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, seed: int, hidden_dim: int = 256, lr: float = 1e-3):
ccda_phase3/train_utils.py:321:    model, x_std, y_std, residual_std, device = _train_torch_regressor(x, y_flat, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))
ccda_phase3/train_utils.py:326:    model, x_std, y_std, _residual_std, device = _train_torch_regressor(x, y_action, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))
ccda_phase3/train_utils.py:207:            "model_type": "torch_mlp_future_state",
ccda_phase3/train_utils.py:256:            "model_type": "torch_mlp_inverse_dynamics",
ccda_phase3/train_utils.py:63:    def __init__(self, x_std: Standardizer, y_std: Standardizer, weights: np.ndarray, residual_std: np.ndarray, config: Dict[str, Any]):
ccda_phase3/train_utils.py:67:        self.residual_std = np.asarray(residual_std, dtype=np.float32)
ccda_phase3/train_utils.py:77:        noise = rng.normal(0.0, self.residual_std.reshape(1, 1, -1), size=(n_samples, mean.shape[0], mean.shape[1]))
ccda_phase3/train_utils.py:87:                "residual_std": self.residual_std,
ccda_phase3/train_utils.py:95:        return cls(Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["weights"], obj["residual_std"], obj.get("config", {}))
ccda_phase3/train_utils.py:143:    residual_std = np.std(residual, axis=0).astype(np.float32)
ccda_phase3/train_utils.py:144:    residual_std = np.maximum(residual_std, 1e-4)
ccda_phase3/train_utils.py:145:    return NumpyFutureModel(x_std, y_std, w, residual_std, config)
ccda_phase3/train_utils.py:178:    def __init__(self, model, x_std: Standardizer, y_std: Standardizer, residual_std: np.ndarray, config: Dict[str, Any], device: str = "cpu"):
ccda_phase3/train_utils.py:182:        self.residual_std = np.asarray(residual_std, dtype=np.float32)
ccda_phase3/train_utils.py:199:        noise = rng.normal(0.0, self.residual_std.reshape(1, 1, -1), size=(n_samples, mean.shape[0], mean.shape[1]))
ccda_phase3/train_utils.py:213:            "residual_std": self.residual_std,
ccda_phase3/train_utils.py:224:        return cls(model, Standardizer.from_dict(obj["x_std"]), Standardizer.from_dict(obj["y_std"]), obj["residual_std"], obj.get("config", {}), device=device)
ccda_phase3/train_utils.py:315:    residual_std = np.maximum(np.std(residual, axis=0).astype(np.float32), 1e-4)
ccda_phase3/train_utils.py:316:    return model, x_std, y_std, residual_std, device
ccda_phase3/train_utils.py:321:    model, x_std, y_std, residual_std, device = _train_torch_regressor(x, y_flat, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))
ccda_phase3/train_utils.py:322:    return TorchFutureModel(model, x_std, y_std, residual_std, config, device=device)
ccda_phase3/train_utils.py:326:    model, x_std, y_std, _residual_std, device = _train_torch_regressor(x, y_action, epochs, batch_size, seed, hidden_dim=int(config.get("hidden_dim", 256)))

=== branch reference ===
55:    refs = defaultdict(dict)
57:        refs[int(visible_seed[i])][str(cond_name[i])] = y_final[i]
94:                ref = refs[int(visible_seed[i])]
109:                    "window_t": int(data["window_t"][i]),

=== DeformableRavens submodule ===
 b95180593d3366b754a4d8a4488d723bc9d61be5 external/deformable-ravens (heads/ccda-cable)
branch: ccda-cable
head: b95180593d3366b754a4d8a4488d723bc9d61be5
106:    def _robot_pose_proxy(self):
176:            "robot_pose_proxy": self._robot_pose_proxy(),

```
