# Phase3 PyTorch StateDiff CCDA Baseline Evaluation

## Scope

Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, recoverability parameters, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.

## Condition Scope

- Conditions: `free hidden_pin hidden_high_friction hidden_breakaway_pin`
- Primary branch pair: `free_vs_hidden_breakaway_pin`
- Diagnostic branch pair: `free_vs_hidden_pin`
- Policy rollout is intentionally excluded unless explicitly run with `PHASE3_ALLOW_ROLLOUT=1`.

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

- scheduler_type_counts: `{'diffusers.DDPMScheduler': 2256}`
- beta_schedule_counts: `{'squaredcos_cap_v2': 2256}`
- prediction_type_counts: `{'epsilon': 2256}`
- variance_type_counts: `{'fixed_small': 2256}`
- clip_sample_counts: `{'true': 2256}`
- denoiser_arch_counts: `{'mlp': 2256}`
- conditional_unet1d_used_counts: `{'false': 2256}`
- paper_alignment_level_counts: `{'ddpm_scheduler_aligned_mlp_denoiser': 2256}`

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 2256 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.667941 | 0.332059 | 0.0823528 | -0.00222144 |
| paper_state | hidden_breakaway_pin | 0.332059 | 0.667941 | 0.081501 | -0.00222144 |
| paper_state | hidden_high_friction | NA | NA | 0.0807424 | -0.00222144 |
| paper_state | hidden_pin | 0.332059 | 0.667941 | 0.206809 | -0.00222144 |
| state_action | free | 0.661181 | 0.338819 | 0.0818042 | -0.00230736 |
| state_action | hidden_breakaway_pin | 0.338819 | 0.661181 | 0.0810252 | -0.00230736 |
| state_action | hidden_high_friction | NA | NA | 0.0802519 | -0.00230736 |
| state_action | hidden_pin | 0.338819 | 0.661181 | 0.205365 | -0.00230736 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 0.0045433 | 0.549511 |
| paper_state | hidden_breakaway_pin | 0.00481076 | 0.549511 |
| paper_state | hidden_high_friction | 0.00521647 | 0.549511 |
| paper_state | hidden_pin | 0.00431395 | 0.549511 |
| state_action | free | 0.00447397 | 0.549975 |
| state_action | hidden_breakaway_pin | 0.00474882 | 0.549975 |
| state_action | hidden_high_friction | 0.00514396 | 0.549975 |
| state_action | hidden_pin | 0.00426497 | 0.549975 |

## Policy Execution

Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.

## Phase3 Conclusion

Phase3-medium passed with PyTorch DDPM backend, leakage-checked matched inputs, executable action targets, and multi-fold/multi-seed statistics using primary pair `free_vs_hidden_breakaway_pin`. Policy execution remains excluded unless torch-enabled DeformableRavens rollout is run explicitly and action diagnostics pass.

Across the current folds and random seeds, the contact-blind baselines exhibit wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_breakaway_pin` CCDA subset. `hidden_pin` remains a hard diagnostic branch, not the primary CPS success-improvement branch. Rollout and Phase4 decisions must obey the sanity diagnostics below.

## Sanity Diagnostics

| Diagnostic | Verdict |
|---|---|
| Phase3 metric sanity | `WARN` |
| Action/IDM debug | `WARN` |

### Metric Sanity Issues

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

### Action/IDM Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

**Diagnostics contain WARN items. They are documented in this report and do not introduce a blocking FAIL for the current offline medium run; rollout, Phase4, and paper-level claims still require separate review of these WARN items.**
