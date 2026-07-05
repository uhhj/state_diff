# Phase3 Sanity Check Report

## Verdict

- Verdict: `WARN`
- Prediction rows: `2256`
- Primary hidden condition: `hidden_breakaway_pin`
- Diagnostic hidden condition: `hidden_pin`

## Backend Counts

| Backend | Count |
|---|---:|
| `torch` | 2256 |

## Scheduler Metadata Counts

### future_model_type_counts

| Value | Count |
|---|---:|
| `torch_conditional_ddpm_future_state` | 2256 |

### ddpm_used_counts

| Value | Count |
|---|---:|
| `true` | 2256 |

### branch_reference_mode_counts

| Value | Count |
|---|---:|
| `split_visible_seed_window_t` | 2256 |

### scheduler_type_counts

| Value | Count |
|---|---:|
| `diffusers.DDPMScheduler` | 2256 |

### beta_schedule_counts

| Value | Count |
|---|---:|
| `squaredcos_cap_v2` | 2256 |

### prediction_type_counts

| Value | Count |
|---|---:|
| `epsilon` | 2256 |

### variance_type_counts

| Value | Count |
|---|---:|
| `fixed_small` | 2256 |

### denoiser_arch_counts

| Value | Count |
|---|---:|
| `mlp` | 2256 |

### conditional_unet1d_used_counts

| Value | Count |
|---|---:|
| `false` | 2256 |

### paper_alignment_level_counts

| Value | Count |
|---|---:|
| `ddpm_scheduler_aligned_mlp_denoiser` | 2256 |

### eval_sample_seed_mode_counts

| Value | Count |
|---|---:|
| `paired_shared_visible_seed` | 2256 |

## Metric Summary

| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_state | free | 282 | 0.667941 | 0.082353 | -0.002221 | 0.004543 | 0.549511 |
| paper_state | hidden_breakaway_pin | 282 | 0.332059 | 0.081501 | -0.002221 | 0.004811 | 0.549511 |
| paper_state | hidden_high_friction | 282 | NA | 0.080742 | -0.002221 | 0.005216 | 0.549511 |
| paper_state | hidden_pin | 282 | 0.332059 | 0.206809 | -0.002221 | 0.004314 | 0.549511 |
| state_action | free | 282 | 0.661181 | 0.081804 | -0.002307 | 0.004474 | 0.549975 |
| state_action | hidden_breakaway_pin | 282 | 0.338819 | 0.081025 | -0.002307 | 0.004749 | 0.549975 |
| state_action | hidden_high_friction | 282 | NA | 0.080252 | -0.002307 | 0.005144 | 0.549975 |
| state_action | hidden_pin | 282 | 0.338819 | 0.205365 | -0.002307 | 0.004265 | 0.549975 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.0823527677843334 vs 0.08180416024006004 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_free` | paper_state and state_action differ by <1e-3 for averaging_score_mean on free: -0.002221443049662502 vs -0.0023073571344428027 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_free` | paper_state and state_action differ by <1e-3 for action_mse_mean on free: 0.004543299423155032 vs 0.00447397438218891 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_breakaway_pin` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_breakaway_pin: 0.08150097301084522 vs 0.08102517872256168 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_breakaway_pin` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_breakaway_pin: -0.002221443049662502 vs -0.0023073571344428027 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_breakaway_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_breakaway_pin: 0.004810759173430617 vs 0.004748823288330686 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.08074241530493642 vs 0.08025192390096948 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_high_friction: -0.002221443049662502 vs -0.0023073571344428027 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_high_friction: 0.00521646897418545 vs 0.005143958617485081 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_pin: -0.002221443049662502 vs -0.0023073571344428027 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_pin: 0.004313946771868901 vs 0.004264966830524949 |
| `WARN` | `low_future_error_contrast_paper_state` | hidden_breakaway_pin and free future errors are close: hidden-free=-0.000852. This may indicate mean prediction collapse or an overly coarse metric. |
| `WARN` | `low_future_error_contrast_state_action` | hidden_breakaway_pin and free future errors are close: hidden-free=-0.000779. This may indicate mean prediction collapse or an overly coarse metric. |

## Interpretation

- `FAIL` means do not run policy rollout or Phase4 until fixed.
- `WARN` means acceptable for smoke, but inspect before medium/full.
- High wrong-branch on `hidden_breakaway_pin` is the primary CCDA signal. `hidden_pin` remains a hard diagnostic branch.
- Cascaded action MSE/OOD is a warning; pure IDM health is checked separately.
