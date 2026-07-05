# Phase3 Action Codec / Inverse Dynamics Debug Report

## Verdict

- Verdict: `WARN`

## Input Dimensions

- `paper_x_shape`: `[956, 405]`
- `state_action_x_shape`: `[956, 447]`
- `y_action_shape`: `[956, 14]`
- `episode_action_len_max`: `20`

## Action Target Stats

| Metric | Value |
|---|---:|
| `shape` | `[956, 14]` |
| `mean` | `0.2150564457340395` |
| `std` | `0.3757316787428618` |
| `min` | `-0.5` |
| `max` | `1.0` |
| `mean_abs` | `0.24065064310687637` |
| `p95_abs` | `1.0` |
| `p99_abs` | `1.0` |
| `num_nan` | `0` |
| `num_inf` | `0` |

## Action-History Leakage

- `shape`: `[956, 42]`
- `mean`: `0.06875332409319714`
- `std`: `0.23386660931195236`
- `min`: `-0.47187501192092896`
- `max`: `1.0`
- `mean_abs`: `0.07618896553246736`
- `p95_abs`: `0.6499999761581421`
- `p99_abs`: `1.0`
- `num_nan`: `0`
- `num_inf`: `0`
- `state_action_extra_dim`: `42`
- `expected_extra_dim`: `42`
- `action_history_target_exact_match_rate`: `0.0`
- `action_history_target_corr_max`: `0.3478860831495951`

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
| `paper_state/fold_0_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.000865520560182631` | `0.4086143374443054` | `0.6494226455688477` | `1.2100106477737427` |
| `paper_state/fold_0_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008737480966374278` | `0.4016678035259247` | `0.6525393724441528` | `1.2250148057937622` |
| `paper_state/fold_1_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008904627757146955` | `0.4215887486934662` | `0.6427599787712097` | `1.451646089553833` |
| `paper_state/fold_1_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008858043584041297` | `0.4284796416759491` | `0.6655601859092712` | `1.2709057331085205` |
| `paper_state/fold_2_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.002674034796655178` | `0.7782579660415649` | `0.8030984997749329` | `3.3164453506469727` |
| `paper_state/fold_2_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0016221808036789298` | `0.8460516929626465` | `0.8441181778907776` | `2.567943811416626` |
| `state_action/fold_0_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.000865520560182631` | `0.4086143374443054` | `0.6494226455688477` | `1.2100106477737427` |
| `state_action/fold_0_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008737480966374278` | `0.4016678035259247` | `0.6525393724441528` | `1.2250148057937622` |
| `state_action/fold_1_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008904627757146955` | `0.4215887486934662` | `0.6427599787712097` | `1.451646089553833` |
| `state_action/fold_1_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0008858043584041297` | `0.4284796416759491` | `0.6655601859092712` | `1.2709057331085205` |
| `state_action/fold_2_seed_0` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.002674034796655178` | `0.7782579660415649` | `0.8030984997749329` | `3.3164453506469727` |
| `state_action/fold_2_seed_1` | `torch` | `torch_conditional_ddpm_future_state` | `True` | `paper_full_state_history_future` | `0.0016221808036789298` | `0.8460516929626465` | `0.8441181778907776` | `2.567943811416626` |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

## Interpretation

- IDM diagnostics use `paper_x + future_state` full-state inputs, matching the formal Phase3 action pipeline.
- If this report is FAIL, do not trust policy rollout or move to Phase4.
