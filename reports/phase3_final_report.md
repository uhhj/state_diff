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

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 84 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.285714 | 0.714286 | 0.532394 | -0.0135989 |
| paper_state | hidden_high_friction | NA | NA | 0.53375 | -0.0135989 |
| paper_state | hidden_pin | 0.714286 | 0.285714 | 0.561706 | -0.0135989 |
| state_action | free | 0.31808 | 0.68192 | 0.533286 | -0.0136526 |
| state_action | hidden_high_friction | NA | NA | 0.534667 | -0.0136526 |
| state_action | hidden_pin | 0.68192 | 0.31808 | 0.561484 | -0.0136526 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 0.0133526 | 0.568227 |
| paper_state | hidden_high_friction | 0.0134342 | 0.568227 |
| paper_state | hidden_pin | 0.0133955 | 0.568227 |
| state_action | free | 0.013896 | 0.581839 |
| state_action | hidden_high_friction | 0.0139855 | 0.581839 |
| state_action | hidden_pin | 0.0138171 | 0.581839 |

## Policy Execution

Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.

## Phase3 Conclusion

Phase3 PyTorch smoke passed after fixing the executable action codec. The previous camera_config leakage into y_action was removed; y_action now contains only executable pick-place pose parameters. This validates the corrected offline state prediction and inverse dynamics pipeline at smoke scale. Paper-level evidence still requires MODE=medium or MODE=full.

The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.

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
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.5323939727885383 vs 0.5332859030791691 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_free` | paper_state and state_action differ by <1e-3 for averaging_score_mean on free: -0.01359889842569828 vs -0.013652579858899117 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_free` | paper_state and state_action differ by <1e-3 for action_mse_mean on free: 0.013352553459948726 vs 0.013896014192141593 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.5337501849446978 vs 0.5346673130989075 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_high_friction: -0.01359889842569828 vs -0.013652579858899117 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_high_friction: 0.013434154846306359 vs 0.01398549198971263 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_pin: 0.5617055020162037 vs 0.5614840494734901 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_pin: -0.01359889842569828 vs -0.013652579858899117 |
| `WARN` | `baseline_nearly_identical_action_mse_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for action_mse_mean on hidden_pin: 0.01339553639159671 vs 0.013817112709927772 |

### Action/IDM Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `many_near_zero_action_dims` | 8/14 action dims have std < 1e-8. |

**Diagnostics contain WARN items. This is acceptable for smoke, but should be resolved or explicitly discussed before medium/full paper-level runs.**
