# Phase3 Sanity Check Report

## Verdict

- Verdict: `WARN`
- Prediction rows: `84`

## Backend Counts

| Backend | Count |
|---|---:|
| `torch` | 84 |

## Metric Summary

| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_state | free | 14 | 0.285714 | 0.532394 | -0.013599 | 0.013353 | 0.568227 |
| paper_state | hidden_high_friction | 14 | NA | 0.533750 | -0.013599 | 0.013434 | 0.568227 |
| paper_state | hidden_pin | 14 | 0.714286 | 0.561706 | -0.013599 | 0.013396 | 0.568227 |
| state_action | free | 14 | 0.318080 | 0.533286 | -0.013653 | 0.013896 | 0.581839 |
| state_action | hidden_high_friction | 14 | NA | 0.534667 | -0.013653 | 0.013985 | 0.581839 |
| state_action | hidden_pin | 14 | 0.681920 | 0.561484 | -0.013653 | 0.013817 | 0.581839 |

## Issues

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

## Interpretation

- `FAIL` means do not run policy rollout or Phase4 until fixed.
- `WARN` means acceptable for smoke, but inspect before medium/full.
- High wrong-branch on `hidden_pin` is expected. Cascaded action MSE/OOD is a warning; pure IDM health is checked separately.