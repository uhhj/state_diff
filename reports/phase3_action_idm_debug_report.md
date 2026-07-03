# Phase3 Action Codec / Inverse Dynamics Debug Report

## Verdict

- Verdict: `FAIL`

## Input Dimensions

- `paper_x_shape`: `[48, 405]`
- `state_action_x_shape`: `[48, 636]`
- `y_action_shape`: `[48, 77]`

## Dataset Counts

- Condition counts: `{'free': 16, 'hidden_high_friction': 16, 'hidden_pin': 16}`
- Split counts: `{'heldout': 18, 'train': 30}`

## Action Target Stats

| Metric | Value |
|---|---:|
| `shape` | `[48, 77]` |
| `mean` | `101.03858689128964` |
| `std` | `191.68543282869206` |
| `min` | `-0.8681627511978149` |
| `max` | `640.0` |
| `mean_abs` | `101.11478402784071` |
| `p95_abs` | `480.0` |
| `p99_abs` | `640.0` |
| `num_nan` | `0` |
| `num_inf` | `0` |

## Action Dimension Diagnostics

- Zero-variance dims: `71`
- Near-zero-variance dims: `71`
- Large-scale dims count: `18`
- Zero-variance dims head: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Near-zero dims head: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]`
- Large-scale dims head: `[0, 1, 2, 4, 6, 7, 21, 22, 23, 25, 27, 28, 42, 43, 44, 46, 48, 49]`

## State-Action Extra Block

- `shape`: `[48, 231]`
- `mean`: `0.0`
- `std`: `0.0`
- `min`: `0.0`
- `max`: `0.0`
- `mean_abs`: `0.0`
- `p95_abs`: `0.0`
- `p99_abs`: `0.0`
- `num_nan`: `0`
- `num_inf`: `0`

## Action Template

- `path`: `/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl`
- `num_paths`: `77`
- `num_camera_config_paths`: `63`
- `num_param_paths`: `14`
- `camera_config_paths_head`: `['camera_config/0/image_size/0', 'camera_config/0/image_size/1', 'camera_config/0/intrinsics/0', 'camera_config/0/intrinsics/1', 'camera_config/0/intrinsics/2', 'camera_config/0/intrinsics/3', 'camera_config/0/intrinsics/4', 'camera_config/0/intrinsics/5', 'camera_config/0/intrinsics/6', 'camera_config/0/intrinsics/7', 'camera_config/0/intrinsics/8', 'camera_config/0/noise', 'camera_config/0/position/0', 'camera_config/0/position/1', 'camera_config/0/position/2', 'camera_config/0/rotation/0', 'camera_config/0/rotation/1', 'camera_config/0/rotation/2', 'camera_config/0/rotation/3', 'camera_config/0/zrange/0']`

## IDM Heldout Reconstruction

| Checkpoint | Backend | Raw MSE | Normalized MSE | Pred OOD mean | Pred OOD max | Standardized target |
|---|---|---:|---:|---:|---:|---|
| `paper_state/fold_0_seed_0` | `torch` | `1018.7034912109375` | `1091.6448974609375` | `15.354827880859375` | `79.51719665527344` | `True` |
| `paper_state/fold_1_seed_0` | `torch` | `0.010322703048586845` | `0.17466022074222565` | `0.1514567732810974` | `0.28031817078590393` | `True` |
| `state_action/fold_0_seed_0` | `torch` | `1018.7034912109375` | `1091.6448974609375` | `15.354827880859375` | `79.51719665527344` | `True` |
| `state_action/fold_1_seed_0` | `torch` | `0.010322703048586845` | `0.17466022074222565` | `0.1514567732810974` | `0.28031817078590393` | `True` |

## Checkpoint Backends

| Checkpoint | Backend |
|---|---|
| `/data/state_diff2/checkpoints/phase3/paper_state/fold_0_seed_0` | `torch` |
| `/data/state_diff2/checkpoints/phase3/paper_state/fold_1_seed_0` | `torch` |
| `/data/state_diff2/checkpoints/phase3/state_action/fold_0_seed_0` | `torch` |
| `/data/state_diff2/checkpoints/phase3/state_action/fold_1_seed_0` | `torch` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `state_action_extra_block_constant` | state_action baseline has limited additional information because the current primitive dataset contains very short action histories. |
| `WARN` | `large_raw_action_scale` | y_action p99 abs=640.000000. Raw action MSE may be dominated by unnormalized coordinates or wrong fields. |
| `WARN` | `many_near_zero_action_dims` | 71/77 action dims have std < 1e-8. |
| `FAIL` | `action_codec_encodes_camera_config` | Action codec target includes 63 camera_config numeric paths. These are observation metadata, not executable pick-place action parameters. |
| `FAIL` | `idm_raw_mse_too_large` | heldout IDM raw-space MSE max=1018.703491 > 10.0. |
| `FAIL` | `idm_normalized_mse_too_large` | heldout IDM normalized MSE max=1091.644897 > 5.0. |
| `FAIL` | `idm_pred_ood_too_large` | heldout IDM predicted action OOD mean max=15.354828 > 5.0. |

## Interpretation

- If this report is FAIL, do not trust policy rollout.
- High raw action scale may make raw MSE misleading, but high OOD still requires action normalization/codec diagnosis.
- If `state_action_x` is not larger than `paper_x`, the state_action baseline is not implemented correctly.