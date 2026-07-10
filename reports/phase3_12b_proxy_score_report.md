# Phase3.12b Condition Proxy / Score Ablation Diagnostic Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase312b_proxy_score_not_supported_or_inconclusive`
- Episode rows: `84`
- Step rows: `1344`
- Progress status: `completed`

## Per-Condition Selector Comparison

| Condition | DDPM mean | Condition upper | Proxy action | Proxy state | Proxy combined | Proxy topK geom | Proxy topK no-geom | Cond Δ | Action Δ | State Δ | Combined Δ | TopK geom Δ | TopK no-geom Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.2396 | 0.1146 | 0.0938 | 0.0521 | 0.0417 | 0.0729 | 0.0000 | -0.1250 | -0.1458 | -0.1875 | -0.1979 | -0.1667 | -0.2396 |
| `hidden_breakaway_pin` | 0.1979 | 0.0938 | 0.0208 | 0.1354 | 0.0000 | 0.0312 | 0.0417 | -0.1042 | -0.1771 | -0.0625 | -0.1979 | -0.1667 | -0.1562 |
| `hidden_high_friction` | 0.1875 | 0.0729 | 0.0000 | 0.0000 | 0.0729 | 0.0625 | 0.0833 | -0.1146 | -0.1875 | -0.1875 | -0.1146 | -0.1250 | -0.1042 |

## Pull / Future-Match Diagnostics

| Condition | DDPM pull | Condition pull | Proxy action pull | Proxy state pull | Proxy combined pull | TopK geom pull | TopK no-geom pull | DDPM match | Condition match | Proxy combined match | TopK geom match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.0742 | 0.0951 | 0.2253 | 0.1311 | 0.1843 | 0.2270 | 0.2197 | 0.000 | 1.000 | 0.344 | 0.188 |
| `hidden_breakaway_pin` | 0.0866 | 0.1412 | 0.2038 | 0.1483 | 0.1868 | 0.2246 | 0.1881 | 0.188 | 1.000 | 0.328 | 0.141 |
| `hidden_high_friction` | 0.1322 | 0.0858 | 0.1461 | 0.1899 | 0.1480 | 0.2419 | 0.1873 | 0.781 | 1.000 | 0.156 | 0.281 |

## Episode Summary

| Selector | Condition | Rows | OK | Timeout | Uses condition | Final | Δ | Future match | Proxy dist | Pull | OOD | Action MAE | Pull diff | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddpm_mean` | `free` | 4 | 4 | 0 | 0 | 0.2396 | 0.2396 | 0.000 | nan | 0.0742 | 0.258 | nan | nan | 0 |
| `ddpm_mean` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.1979 | 0.1979 | 0.188 | nan | 0.0866 | 0.268 | nan | nan | 0 |
| `ddpm_mean` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.1875 | 0.1875 | 0.781 | nan | 0.1322 | 0.357 | nan | nan | 0 |
| `condition_nearest_upper` | `free` | 4 | 4 | 0 | 1 | 0.1146 | 0.1146 | 1.000 | 0.0523 | 0.0951 | 0.581 | nan | nan | 0 |
| `condition_nearest_upper` | `hidden_breakaway_pin` | 4 | 4 | 0 | 1 | 0.0938 | 0.0938 | 1.000 | 0.0593 | 0.1412 | 0.497 | nan | nan | 0 |
| `condition_nearest_upper` | `hidden_high_friction` | 4 | 4 | 0 | 1 | 0.0729 | 0.0729 | 1.000 | 0.0572 | 0.0858 | 0.488 | nan | nan | 0 |
| `proxy_action_nn` | `free` | 4 | 4 | 0 | 0 | 0.0938 | 0.0938 | 0.391 | 0.6582 | 0.2253 | 0.501 | nan | nan | 0 |
| `proxy_action_nn` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.0208 | 0.0208 | 0.453 | 0.6344 | 0.2038 | 0.524 | nan | nan | 0 |
| `proxy_action_nn` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.0000 | 0.0000 | 0.109 | 0.5021 | 0.1461 | 0.477 | nan | nan | 0 |
| `proxy_state_motion_nn` | `free` | 4 | 4 | 0 | 0 | 0.0521 | -0.0938 | 0.312 | 0.4860 | 0.1311 | 0.412 | nan | nan | 0 |
| `proxy_state_motion_nn` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.1354 | 0.0104 | 0.312 | 0.4802 | 0.1483 | 0.410 | nan | nan | 0 |
| `proxy_state_motion_nn` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.0000 | 0.0000 | 0.141 | 0.8605 | 0.1899 | 0.535 | nan | nan | 0 |
| `proxy_combined_nn` | `free` | 4 | 4 | 0 | 0 | 0.0417 | 0.0417 | 0.344 | 0.6565 | 0.1843 | 0.535 | nan | nan | 0 |
| `proxy_combined_nn` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.0000 | 0.0000 | 0.328 | 0.8666 | 0.1868 | 0.553 | nan | nan | 0 |
| `proxy_combined_nn` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.0729 | 0.0521 | 0.156 | 0.9052 | 0.1480 | 0.522 | nan | nan | 0 |
| `proxy_combined_topk_action_geom` | `free` | 4 | 4 | 0 | 0 | 0.0729 | 0.0729 | 0.188 | 0.8086 | 0.2270 | 0.465 | 0.0082 | 0.0205 | 0 |
| `proxy_combined_topk_action_geom` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.0312 | 0.0312 | 0.141 | 0.9665 | 0.2246 | 0.512 | 0.0094 | 0.0254 | 0 |
| `proxy_combined_topk_action_geom` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.0625 | 0.0625 | 0.281 | 0.9090 | 0.2419 | 0.492 | 0.0076 | 0.0242 | 0 |
| `proxy_combined_topk_no_action_geom` | `free` | 4 | 4 | 0 | 0 | 0.0000 | 0.0000 | 0.156 | 1.0410 | 0.2197 | 0.495 | 0.0187 | 0.1298 | 0 |
| `proxy_combined_topk_no_action_geom` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0 | 0.0417 | -0.0625 | 0.062 | 0.8546 | 0.1881 | 0.414 | 0.0142 | 0.1051 | 0 |
| `proxy_combined_topk_no_action_geom` | `hidden_high_friction` | 4 | 4 | 0 | 0 | 0.0833 | 0.0833 | 0.141 | 0.9829 | 0.1873 | 0.391 | 0.0126 | 0.0907 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `proxy_score_not_clearly_supported` | {'condition': 'hidden_breakaway_pin', 'ddpm_mean_final': 0.19791666666666666, 'condition_nearest_upper_final': 0.09375, 'proxy_action_nn_final': 0.020833333333333332, 'proxy_state_motion_nn_final': 0.13541666666666669, 'proxy_combined_nn_final': 0.0, 'proxy_combined_topk_action_geom_final': 0.03125, 'proxy_combined_topk_no_action_geom_final': 0.041666666666666664, 'ddpm_mean_pull': 0.0865726404445013, 'condition_nearest_upper_pull': 0.14115068709361367, 'proxy_action_nn_pull': 0.20380853138340171, 'proxy_state_motion_nn_pull': 0.14833938222727738, 'proxy_combined_nn_pull': 0.1867915685143089, 'proxy_topk_action_geom_pull': 0.22456504177534953, 'proxy_topk_no_action_geom_pull': 0.18811730202287436, 'ddpm_match': 0.1875, 'condition_upper_match': 1.0, 'proxy_action_match': 0.453125, 'proxy_state_match': 0.3125, 'proxy_combined_match': 0.328125, 'proxy_topk_action_geom_match': 0.140625, 'proxy_topk_no_action_geom_match': 0.0625, 'condition_nearest_upper_improvement': -0.10416666666666666, 'proxy_action_nn_improvement': -0.17708333333333331, 'proxy_state_motion_nn_improvement': -0.06249999999999997, 'proxy_combined_nn_improvement': -0.19791666666666666, 'proxy_combined_topk_action_geom_improvement': -0.16666666666666666, 'proxy_combined_topk_no_action_geom_improvement': -0.15625} |

## Interpretation

- This is condition proxy / score ablation diagnostic only.
- No model training was run.
- No future DDPM was trained.
- No Phase4 or CPS was run.
- `condition_nearest_upper` uses condition labels and is an upper bound, not deployable policy evidence.
- If observable proxy selectors do not improve primary, the next step is contact/proprio proxy design, not full CPS.
