# Phase3 Sanity Check Report

## Verdict

- Verdict: `WARN`
- Prediction rows: `84`

## Backend Counts

| Backend | Count |
|---|---:|
| `torch` | 84 |

## Scheduler Metadata Counts

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

### scheduler_type_counts

| Value | Count |
|---|---:|
| `diffusers.DDPMScheduler` | 84 |

### beta_schedule_counts

| Value | Count |
|---|---:|
| `squaredcos_cap_v2` | 84 |

### prediction_type_counts

| Value | Count |
|---|---:|
| `epsilon` | 84 |

### variance_type_counts

| Value | Count |
|---|---:|
| `fixed_small` | 84 |

### denoiser_arch_counts

| Value | Count |
|---|---:|
| `mlp` | 84 |

### conditional_unet1d_used_counts

| Value | Count |
|---|---:|
| `false` | 84 |

### paper_alignment_level_counts

| Value | Count |
|---|---:|
| `ddpm_scheduler_aligned_mlp_denoiser` | 84 |

### eval_sample_seed_mode_counts

| Value | Count |
|---|---:|
| `paired_shared_visible_seed` | 84 |

## Metric Summary

| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_state | free | 14 | 0.256696 | 0.084582 | -0.035603 | 0.005552 | 0.439413 |
| paper_state | hidden_high_friction | 14 | NA | 0.085631 | -0.035603 | 0.005619 | 0.439413 |
| paper_state | hidden_pin | 14 | 0.743304 | 0.139249 | -0.035603 | 0.005434 | 0.439413 |
| state_action | free | 14 | 0.248884 | 0.084281 | -0.035258 | 0.005597 | 0.438207 |
| state_action | hidden_high_friction | 14 | NA | 0.085245 | -0.035258 | 0.005665 | 0.438207 |
| state_action | hidden_pin | 14 | 0.751116 | 0.139187 | -0.035258 | 0.005486 | 0.438207 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `small_prediction_table` | Only 84 prediction rows. This is smoke-scale, not paper-scale. |
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.08458239611770425 vs 0.08428095360951764 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_free` | paper_state and state_action differ by <1e-3 for averaging_score_mean on free: -0.03560307728392737 vs -0.035258366859384944 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_free` | paper_state and state_action differ by <1e-3 for action_mse_mean on free: 0.005551980517338961 vs 0.005597225190805537 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.0856306217610836 vs 0.08524539933672973 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_high_friction: -0.03560307728392737 vs -0.035258366859384944 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_high_friction: 0.005619003811651575 vs 0.005664670565498194 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_pin: 0.1392485275864601 vs 0.13918697887233325 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_pin: -0.03560307728392737 vs -0.035258366859384944 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_pin: 0.005434069774180118 vs 0.005485800535617662 |

## Interpretation

- `FAIL` means do not run policy rollout or Phase4 until fixed.
- `WARN` means acceptable for smoke, but inspect before medium/full.
- High wrong-branch on `hidden_pin` is expected. Cascaded action MSE/OOD is a warning; pure IDM health is checked separately.