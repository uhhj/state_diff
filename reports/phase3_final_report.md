# Phase3 PyTorch StateDiff CCDA Baseline Evaluation

## Scope

Phase3 evaluates contact-blind StateDiff-style baselines on `hidden-contact-cable-line` without hidden condition labels, pin ids, hidden contact metadata, success labels, contact concatenation, CPS guidance, or action feasibility classifiers.

## Runtime Backend

| Step | Python | Conda Env | Torch | Ravens |
|---|---|---|---|---|
| generate | not run in current report | NA | NA | NA |
| train_eval | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |
| rollout | not run in current report | NA | NA | NA |
| aggregate | `/miniforge3/envs/coord_bimanual/bin/python` | `coord_bimanual` | OK | not required |

## Training Backend

| Backend | Count |
|---|---:|
| `torch` | 72 |

## Offline State Prediction

| Baseline | Condition | Wrong-Branch mean | Branch-Accuracy mean | Future Error mean | Averaging Score mean |
|---|---|---:|---:|---:|---:|
| paper_state | free | 0.372396 | 0.627604 | 0.516468 | 0.00090625 |
| paper_state | hidden_high_friction | NA | NA | 0.516261 | 0.00090625 |
| paper_state | hidden_pin | 0.627604 | 0.372396 | 0.533711 | 0.00090625 |
| state_action | free | 0.354167 | 0.645833 | 0.51585 | 0.00125573 |
| state_action | hidden_high_friction | NA | NA | 0.515737 | 0.00125573 |
| state_action | hidden_pin | 0.645833 | 0.354167 | 0.535677 | 0.00125573 |

## Inverse Dynamics

| Baseline | Condition | Action MSE mean | Action OOD mean |
|---|---|---:|---:|
| paper_state | free | 283.149 | 17.3679 |
| paper_state | hidden_high_friction | 283.149 | 17.3679 |
| paper_state | hidden_pin | 283.149 | 17.3679 |
| state_action | free | 283.087 | 17.3638 |
| state_action | hidden_high_friction | 283.087 | 17.3638 |
| state_action | hidden_pin | 283.087 | 17.3638 |

## Policy Execution

Policy rollout was not rerun for the current PyTorch train/eval report, so any older rollout CSV is not counted as current execution evidence.

## Phase3 Conclusion

This is a PyTorch smoke/medium validation, not the final full 5-fold x 3-seed result unless MODE=full was run.

The input consistency and leakage checks verify that paired `free` and `hidden_pin` samples have matched visible/proprio/action inputs, and probe classifiers cannot reliably recover hidden condition from the model inputs. Therefore, the branch ambiguity is not caused by accidental input leakage.

Across the current folds and random seeds, the contact-blind baselines exhibit elevated wrong-branch rate and/or branch ambiguity on the primary `free` vs `hidden_pin` CCDA subset. This preserves the offline wrong-branch signal, but rollout, medium/full runs, and Phase4 decisions must obey the sanity diagnostics below.

## Sanity Diagnostics

| Diagnostic | Verdict |
|---|---|
| Phase3 metric sanity | `FAIL` |
| Action/IDM debug | `FAIL` |

### Metric Sanity Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `small_prediction_table` | Only 72 prediction rows. This is smoke-scale, not paper-scale. |
| `WARN` | `baseline_nearly_identical_future_error_mean_free` | paper_state and state_action differ by <1e-3 for future_error_mean on free: 0.51646805057923 vs 0.5158502335349718 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_free` | paper_state and state_action differ by <1e-3 for averaging_score_mean on free: 0.0009062504395842552 vs 0.0012557267521818478 |
| `WARN` | `baseline_nearly_identical_future_error_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for future_error_mean on hidden_high_friction: 0.5162610212961832 vs 0.5157370045781136 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_high_friction` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_high_friction: 0.0009062504395842552 vs 0.0012557267521818478 |
| `WARN` | `baseline_nearly_identical_averaging_score_mean_hidden_pin` | paper_state and state_action differ by <1e-3 for averaging_score_mean on hidden_pin: 0.0009062504395842552 vs 0.0012557267521818478 |
| `WARN` | `low_future_error_contrast_paper_state` | hidden_pin and free future errors are close: pin-free=0.017243. This may indicate mean prediction collapse or an overly coarse metric. |
| `WARN` | `low_future_error_contrast_state_action` | hidden_pin and free future errors are close: pin-free=0.019826. This may indicate mean prediction collapse or an overly coarse metric. |
| `FAIL` | `action_mse_too_large` | max action MSE=283.148601 > 10.0. Do not trust rollout until action codec / normalization / IDM target are diagnosed. |
| `FAIL` | `action_ood_too_large` | max action OOD=17.367888 > 5.0. Predicted actions are far from expert action distribution. |

### Action/IDM Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `state_action_extra_block_constant` | state_action baseline has limited additional information because the current primitive dataset contains very short action histories. |
| `WARN` | `large_raw_action_scale` | y_action p99 abs=640.000000. Raw action MSE may be dominated by unnormalized coordinates or wrong fields. |
| `WARN` | `many_near_zero_action_dims` | 71/77 action dims have std < 1e-8. |
| `FAIL` | `action_codec_encodes_camera_config` | Action codec target includes 63 camera_config numeric paths. These are observation metadata, not executable pick-place action parameters. |
| `FAIL` | `idm_raw_mse_too_large` | heldout IDM raw-space MSE max=1018.703491 > 10.0. |
| `FAIL` | `idm_normalized_mse_too_large` | heldout IDM normalized MSE max=1091.644897 > 5.0. |
| `FAIL` | `idm_pred_ood_too_large` | heldout IDM predicted action OOD mean max=15.354828 > 5.0. |

**Execution-level Phase3 evidence is blocked by sanity diagnostics. Do not count policy rollout or move to Phase4 until the FAIL items are fixed.**
