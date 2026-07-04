# Phase3 PyTorch StateDiff CCDA Baseline Evaluation

## Scope

Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.

## Runtime Backend

| Step | Python | Conda Env | Torch | Ravens |
|---|---|---|---|---|
| generate | `/root/miniforge3/envs/defravens37/bin/python` | `defravens37` | not required | OK |
| train_eval | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |
| rollout | not run in current report | NA | NA | NA |
| aggregate | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |

## Future Model

| Field | Value |
|---|---|
| future_model_type | torch_conditional_ddpm_future_state |
| ddpm_used | true |
| simplified_mlp_removed | true |
| branch_reference_mode | split_visible_seed_window_t |

This Phase3 run uses a conditional DDPM future-state predictor. It no longer uses the previous PyTorch MLP residual future-state surrogate.

## Paper Alignment Status

| Field | Value |
|---|---|
| scheduler_type | diffusers.DDPMScheduler |
| beta_schedule | squaredcos_cap_v2 |
| prediction_type | epsilon |
| variance_type | fixed_small |
| clip_sample | true |
| denoiser_arch | mlp |
| conditional_unet1d_used | false |
| alignment_level | ddpm_scheduler_aligned_mlp_denoiser |

This run aligns the DDPM scheduler with the original StateDiff configuration but still uses an MLP denoiser over low-dimensional future states. It is not yet an architecture-identical ConditionalUnet1D reproduction.

### Paper Alignment Metadata Counts

- scheduler_type_counts: `{'diffusers.DDPMScheduler': 84}`
- beta_schedule_counts: `{'squaredcos_cap_v2': 84}`
- prediction_type_counts: `{'epsilon': 84}`
- variance_type_counts: `{'fixed_small': 84}`
- clip_sample_counts: `{'true': 84}`
- denoiser_arch_counts: `{'mlp': 84}`
- conditional_unet1d_used_counts: `{'false': 84}`
- paper_alignment_level_counts: `{'ddpm_scheduler_aligned_mlp_denoiser': 84}`

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 84 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.256696 | 0.743304 | 0.0845824 | -0.0356031 |
| paper_state | hidden_high_friction | NA | NA | 0.0856306 | -0.0356031 |
| paper_state | hidden_pin | 0.743304 | 0.256696 | 0.139249 | -0.0356031 |
| state_action | free | 0.248884 | 0.751116 | 0.084281 | -0.0352584 |
| state_action | hidden_high_friction | NA | NA | 0.0852454 | -0.0352584 |
| state_action | hidden_pin | 0.751116 | 0.248884 | 0.139187 | -0.0352584 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 0.00555198 | 0.439413 |
| paper_state | hidden_high_friction | 0.005619 | 0.439413 |
| paper_state | hidden_pin | 0.00543407 | 0.439413 |
| state_action | free | 0.00559723 | 0.438207 |
| state_action | hidden_high_friction | 0.00566467 | 0.438207 |
| state_action | hidden_pin | 0.0054858 | 0.438207 |

## Policy Execution

Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.

## Phase3 Conclusion

Phase3 PyTorch DDPM smoke passed after replacing the simplified MLP residual future predictor. The executable action codec removes camera_config leakage; y_action contains only pick-place pose parameters. Paper-level evidence still requires MODE=medium or MODE=full.

Across the current folds and random seeds, the contact-blind baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. This preserves the offline wrong-branch signal, but rollout, medium/full runs, and Phase4 decisions must obey the sanity diagnostics below.

## Sanity Diagnostics

| Diagnostic | Verdict |
|---|---|
| Phase3 metric sanity | `WARN` |
| Action/IDM debug | `WARN` |

### Metric Sanity Issues

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

### Action/IDM Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

**Diagnostics contain WARN items. This is acceptable for smoke, but should be resolved or explicitly discussed before medium/full paper-level runs.**
