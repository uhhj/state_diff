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

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 84 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.19308 | 0.80692 | 0.188033 | -0.0277367 |
| paper_state | hidden_high_friction | NA | NA | 0.189616 | -0.0277367 |
| paper_state | hidden_pin | 0.80692 | 0.19308 | 0.227296 | -0.0277367 |
| state_action | free | 0.186384 | 0.813616 | 0.187472 | -0.0291065 |
| state_action | hidden_high_friction | NA | NA | 0.189024 | -0.0291065 |
| state_action | hidden_pin | 0.813616 | 0.186384 | 0.22702 | -0.0291065 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 0.00538736 | 0.443694 |
| paper_state | hidden_high_friction | 0.00545026 | 0.443694 |
| paper_state | hidden_pin | 0.00525726 | 0.443694 |
| state_action | free | 0.00549452 | 0.442982 |
| state_action | hidden_high_friction | 0.00555827 | 0.442982 |
| state_action | hidden_pin | 0.00537339 | 0.442982 |

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
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.1880333838718278 vs 0.18747245307479585 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_free` | paper_state and state_action differ by <1e-3 for action_mse_mean on free: 0.005387358267658523 vs 0.005494523743566658 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.18961642788989203 vs 0.18902442497866495 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_high_friction: 0.005450264858414552 vs 0.005558272624122245 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_pin: 0.22729624594960893 vs 0.22701956225293024 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_pin: 0.005257259921303817 vs 0.005373392079491168 |

### Action/IDM Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

**Diagnostics contain WARN items. This is acceptable for smoke, but should be resolved or explicitly discussed before medium/full paper-level runs.**
