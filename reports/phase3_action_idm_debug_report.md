# Phase3 Action Codec / Inverse Dynamics Debug Report

## Verdict

- Verdict: `WARN`

## Input Dimensions

- `paper_x_shape`: `[75, 405]`
- `state_action_x_shape`: `[75, 447]`
- `y_action_shape`: `[75, 14]`
- `episode_action_len_max`: `20`

## Dataset Counts

- Condition counts: `{'free': 25, 'hidden_high_friction': 25, 'hidden_pin': 25}`
- Split counts: `{'heldout': 21, 'train': 54}`

## Action Target Stats

| Metric | Value |
|---|---:|
| `shape` | `[75, 14]` |
| `mean` | `0.20927994590111276` |
| `std` | `0.3774403436945198` |
| `min` | `-0.4781250059604645` |
| `max` | `1.0` |
| `mean_abs` | `0.2399181101241404` |
| `p95_abs` | `1.0` |
| `p99_abs` | `1.0` |
| `num_nan` | `0` |
| `num_inf` | `0` |

## Action Dimension Diagnostics

- Zero-variance dims: `8`
- Near-zero-variance dims: `8`
- Large-scale dims count: `0`
- Zero-variance dims head: `[3, 4, 5, 6, 10, 11, 12, 13]`
- Near-zero dims head: `[3, 4, 5, 6, 10, 11, 12, 13]`
- Large-scale dims head: `[]`

## State-Action Extra Block

- `shape`: `[75, 42]`
- `mean`: `0.03358640093195607`
- `std`: `0.1718082709216301`
- `min`: `-0.47187501192092896`
- `max`: `1.0`
- `mean_abs`: `0.0396039736318025`
- `p95_abs`: `0.3503100872039795`
- `p99_abs`: `1.0`
- `num_nan`: `0`
- `num_inf`: `0`

## Action Template

- `path`: `/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl`
- `class`: `ExecutableActionCodec`
- `dim`: `14`
- `num_paths`: `14`
- `num_camera_config_paths`: `0`
- `num_param_paths`: `14`
- `roundtrip_error`: `0.0`

## IDM Heldout Reconstruction

- Feature mode: `['cable_xy_history_future']`

| Checkpoint | Backend | Raw MSE | Normalized MSE | Pred OOD mean | Pred OOD max | Standardized target |
|---|---|---:|---:|---:|---:|---|
| `paper_state/fold_0_seed_0` | `torch` | `0.014688277617096901` | `2.3499228954315186` | `0.5697910785675049` | `0.9521216750144958` | `True` |
| `paper_state/fold_1_seed_0` | `torch` | `0.008442241698503494` | `0.8672045469284058` | `0.5028700232505798` | `1.0105563402175903` | `True` |
| `state_action/fold_0_seed_0` | `torch` | `0.014688277617096901` | `2.3499228954315186` | `0.5697910785675049` | `0.9521216750144958` | `True` |
| `state_action/fold_1_seed_0` | `torch` | `0.008442241698503494` | `0.8672045469284058` | `0.5028700232505798` | `1.0105563402175903` | `True` |

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
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

## Interpretation

- If this report is FAIL, do not trust policy rollout.
- High raw action scale may make raw MSE misleading, but high OOD still requires action normalization/codec diagnosis.
- If `state_action_x` is not larger than `paper_x`, the state_action baseline is not implemented correctly.