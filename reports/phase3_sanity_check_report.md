# Phase3 Sanity Check Report

## Verdict

- Verdict: `FAIL`
- Prediction rows: `72`

## Backend Counts

| Backend | Count |
|---|---:|
| `torch` | 72 |

## Metric Summary

| Baseline | Condition | Count | Wrong-Branch | Future Error | Averaging Score | Action MSE | Action OOD |
|---|---|---:|---:|---:|---:|---:|---:|
| paper_state | free | 12 | 0.372396 | 0.516468 | 0.000906 | 283.148592 | 17.367888 |
| paper_state | hidden_high_friction | 12 | NA | 0.516261 | 0.000906 | 283.148594 | 17.367888 |
| paper_state | hidden_pin | 12 | 0.627604 | 0.533711 | 0.000906 | 283.148601 | 17.367888 |
| state_action | free | 12 | 0.354167 | 0.515850 | 0.001256 | 283.087039 | 17.363845 |
| state_action | hidden_high_friction | 12 | NA | 0.515737 | 0.001256 | 283.087035 | 17.363845 |
| state_action | hidden_pin | 12 | 0.645833 | 0.535677 | 0.001256 | 283.087049 | 17.363845 |

## Issues

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

## Interpretation

- `FAIL` means do not run policy rollout or Phase4 until fixed.
- `WARN` means acceptable for smoke, but inspect before medium/full.
- High wrong-branch on `hidden_pin` is expected; high action MSE/OOD is not expected.