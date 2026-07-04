# Phase3 Action Codec / Inverse Dynamics Debug Report

## Verdict

- Verdict: `WARN`

## Input Dimensions

- `paper_x_shape`: `[75, 405]`
- `state_action_x_shape`: `[75, 447]`
- `y_action_shape`: `[75, 14]`
- `episode_action_len_max`: `20`

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

## Action-History Leakage

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
- `state_action_extra_dim`: `42`
- `expected_extra_dim`: `42`
- `action_history_target_exact_match_rate`: `0.0`
- `action_history_target_corr_max`: `0.44612507259263456`

## Action Template

- `path`: `/data/state_diff2/data/phase3_state_diff_windows/phase3_action_template.pkl`
- `class`: `ExecutableActionCodec`
- `dim`: `14`
- `num_paths`: `14`
- `num_camera_config_paths`: `0`
- `num_param_paths`: `14`
- `roundtrip_error`: `0.0`

## IDM Heldout Reconstruction

| Checkpoint | Backend | Future model | DDPM | IDM features | Raw MSE | Normalized MSE | Pred OOD mean | Pred OOD max |
|---|---|---|---|---|---:|---:|---:|---:|
| `paper_state/fold_0_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0049493820406496525` | `2.189709186553955` | `0.5352833271026611` | `0.9634959101676941` |
| `paper_state/fold_1_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.004737044218927622` | `0.8170351386070251` | `0.5460922718048096` | `1.0564866065979004` |
| `state_action/fold_0_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0049493820406496525` | `2.189709186553955` | `0.5352833271026611` | `0.9634959101676941` |
| `state_action/fold_1_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.004737044218927622` | `0.8170351386070251` | `0.5460922718048096` | `1.0564866065979004` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

## Interpretation

- IDM diagnostics use `paper_x + future_state` full-state inputs, matching the formal Phase3 action pipeline.
- If this report is FAIL, do not trust policy rollout or move to Phase4.
