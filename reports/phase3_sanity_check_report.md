# Phase3 Sanity Check Report

## Verdict

- Verdict: `WARN`
- Prediction rows: `84`

## Backend Counts

| Backend | Count |
|---|---:|
| `torch` | 84 |

## DDPM Metadata Counts

### future_model_type_counts

| Value | Count |
|---|---:|
| `torch_conditional_ddpm_future_state` | 84 |

### ddpm_used_counts

| Value | Count |
|---|---:|
| `true` | 84 |

### branch_reference_mode_counts

| Value | Count |
|---|---:|
| `split_visible_seed_window_t` | 84 |

## Metric Summary

| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_state | free | 14 | 0.193080 | 0.188033 | -0.027737 | 0.005387 | 0.443694 |
| paper_state | hidden_high_friction | 14 | NA | 0.189616 | -0.027737 | 0.005450 | 0.443694 |
| paper_state | hidden_pin | 14 | 0.806920 | 0.227296 | -0.027737 | 0.005257 | 0.443694 |
| state_action | free | 14 | 0.186384 | 0.187472 | -0.029107 | 0.005495 | 0.442982 |
| state_action | hidden_high_friction | 14 | NA | 0.189024 | -0.029107 | 0.005558 | 0.442982 |
| state_action | hidden_pin | 14 | 0.813616 | 0.227020 | -0.029107 | 0.005373 | 0.442982 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `small_prediction_table` | Only 84 prediction rows. This is smoke-scale, not paper-scale. |
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.1880333838718278 vs 0.18747245307479585 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_free` | paper_state and state_action differ by <1e-3 for action_mse_mean on free: 0.005387358267658523 vs 0.005494523743566658 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.18961642788989203 vs 0.18902442497866495 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_high_friction: 0.005450264858414552 vs 0.005558272624122245 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_pin: 0.22729624594960893 vs 0.22701956225293024 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_pin: 0.005257259921303817 vs 0.005373392079491168 |

## Interpretation

- `FAIL` means do not run policy rollout or Phase4 until fixed.
- `WARN` means acceptable for smoke, but inspect before medium/full.
- High wrong-branch on `hidden_pin` is expected. Cascaded action MSE/OOD is a warning; pure IDM health is checked separately.