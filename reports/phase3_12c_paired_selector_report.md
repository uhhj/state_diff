# Phase3.12c Matched-Reset Paired Selector Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase312c_no_selector_robustly_improves_ddpm_under_paired_reset`
- Matched-reset integrity passed: `True`
- Episode rows: `96`
- Step rows: `1536`
- Paired rows: `96`
- Progress status: `completed`

## Paired selector summary

| Condition | Selector | Pairs | Mean Δ | Baseline mean Δ | Paired mean gain | Median gain | 95% CI | Positive | Negative | Robust |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `free` | `ddpm_mean` | 8 | -0.0156 | -0.0156 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | `False` |
| `free` | `condition_nearest_upper` | 8 | -0.0312 | -0.0156 | -0.0156 | 0.0000 | [-0.0729, 0.0365] | 1 | 2 | `False` |
| `free` | `proxy_state_motion_nn` | 8 | 0.0625 | -0.0156 | 0.0781 | 0.0625 | [0.0365, 0.1250] | 6 | 0 | `True` |
| `free` | `proxy_combined_topk_action_geom` | 8 | -0.0052 | -0.0156 | 0.0104 | 0.0000 | [-0.0521, 0.0677] | 3 | 1 | `False` |
| `hidden_breakaway_pin` | `ddpm_mean` | 8 | 0.0104 | 0.0104 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | `False` |
| `hidden_breakaway_pin` | `condition_nearest_upper` | 8 | -0.0052 | 0.0104 | -0.0156 | 0.0000 | [-0.0990, 0.0417] | 2 | 1 | `False` |
| `hidden_breakaway_pin` | `proxy_state_motion_nn` | 8 | 0.0000 | 0.0104 | -0.0104 | 0.0000 | [-0.1094, 0.0781] | 1 | 1 | `False` |
| `hidden_breakaway_pin` | `proxy_combined_topk_action_geom` | 8 | -0.0260 | 0.0104 | -0.0365 | 0.0000 | [-0.1094, 0.0000] | 0 | 1 | `False` |
| `hidden_high_friction` | `ddpm_mean` | 8 | 0.0677 | 0.0677 | 0.0000 | 0.0000 | [0.0000, 0.0000] | 0 | 0 | `False` |
| `hidden_high_friction` | `condition_nearest_upper` | 8 | 0.0052 | 0.0677 | -0.0625 | 0.0000 | [-0.1667, 0.0208] | 1 | 2 | `False` |
| `hidden_high_friction` | `proxy_state_motion_nn` | 8 | -0.0052 | 0.0677 | -0.0729 | 0.0000 | [-0.2552, 0.0625] | 2 | 2 | `False` |
| `hidden_high_friction` | `proxy_combined_topk_action_geom` | 8 | 0.0573 | 0.0677 | -0.0104 | 0.0000 | [-0.0833, 0.0625] | 2 | 2 | `False` |

## Seed-cohort comparison

| Cohort | Condition | Selector | Pairs | Paired mean gain | Median gain | Positive | Negative |
|---|---|---|---:|---:|---:|---:|---:|
| `phase312_seed_block` | `free` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312_seed_block` | `free` | `condition_nearest_upper` | 4 | -0.0625 | -0.0417 | 0 | 2 |
| `phase312_seed_block` | `free` | `proxy_state_motion_nn` | 4 | 0.0417 | 0.0417 | 3 | 0 |
| `phase312_seed_block` | `free` | `proxy_combined_topk_action_geom` | 4 | 0.0104 | 0.0208 | 2 | 1 |
| `phase312_seed_block` | `hidden_breakaway_pin` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312_seed_block` | `hidden_breakaway_pin` | `condition_nearest_upper` | 4 | -0.0521 | 0.0000 | 1 | 1 |
| `phase312_seed_block` | `hidden_breakaway_pin` | `proxy_state_motion_nn` | 4 | -0.0729 | 0.0000 | 0 | 1 |
| `phase312_seed_block` | `hidden_breakaway_pin` | `proxy_combined_topk_action_geom` | 4 | -0.0729 | 0.0000 | 0 | 1 |
| `phase312_seed_block` | `hidden_high_friction` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312_seed_block` | `hidden_high_friction` | `condition_nearest_upper` | 4 | -0.1458 | -0.1042 | 0 | 2 |
| `phase312_seed_block` | `hidden_high_friction` | `proxy_state_motion_nn` | 4 | -0.2083 | -0.1042 | 0 | 2 |
| `phase312_seed_block` | `hidden_high_friction` | `proxy_combined_topk_action_geom` | 4 | -0.0833 | -0.0625 | 0 | 2 |
| `phase312b_seed_block` | `free` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312b_seed_block` | `free` | `condition_nearest_upper` | 4 | 0.0312 | 0.0000 | 1 | 0 |
| `phase312b_seed_block` | `free` | `proxy_state_motion_nn` | 4 | 0.1146 | 0.1250 | 3 | 0 |
| `phase312b_seed_block` | `free` | `proxy_combined_topk_action_geom` | 4 | 0.0104 | 0.0000 | 1 | 0 |
| `phase312b_seed_block` | `hidden_breakaway_pin` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312b_seed_block` | `hidden_breakaway_pin` | `condition_nearest_upper` | 4 | 0.0208 | 0.0000 | 1 | 0 |
| `phase312b_seed_block` | `hidden_breakaway_pin` | `proxy_state_motion_nn` | 4 | 0.0521 | 0.0000 | 1 | 0 |
| `phase312b_seed_block` | `hidden_breakaway_pin` | `proxy_combined_topk_action_geom` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312b_seed_block` | `hidden_high_friction` | `ddpm_mean` | 4 | 0.0000 | 0.0000 | 0 | 0 |
| `phase312b_seed_block` | `hidden_high_friction` | `condition_nearest_upper` | 4 | 0.0208 | 0.0000 | 1 | 0 |
| `phase312b_seed_block` | `hidden_high_friction` | `proxy_state_motion_nn` | 4 | 0.0625 | 0.0625 | 2 | 0 |
| `phase312b_seed_block` | `hidden_high_friction` | `proxy_combined_topk_action_geom` | 4 | 0.0625 | 0.0625 | 2 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `no_selector_robustly_improves_primary_under_paired_reset` | {'condition_upper': {'condition': 'hidden_breakaway_pin', 'selector': 'condition_nearest_upper', 'num_pairs': 8, 'mean_delta': -0.005208333333333334, 'baseline_mean_delta': 0.010416666666666668, 'paired_mean_gain': -0.015625000000000007, 'paired_median_gain': 0.0, 'bootstrap_ci_low': -0.09895833333333334, 'bootstrap_ci_high': 0.041666666666666664, 'positive_pairs': 2, 'negative_pairs': 1, 'ties': 5, 'required_positive_pairs': 6, 'robust_improvement': False, 'mean_selector_pull': 0.0882601479315781, 'mean_baseline_pull': 0.11600226505834144, 'mean_selector_future_match': 1.0, 'mean_baseline_future_match': 0.3125}, 'state_proxy': {'condition': 'hidden_breakaway_pin', 'selector': 'proxy_state_motion_nn', 'num_pairs': 8, 'mean_delta': 0.0, 'baseline_mean_delta': 0.010416666666666668, 'paired_mean_gain': -0.010416666666666668, 'paired_median_gain': 0.0, 'bootstrap_ci_low': -0.109375, 'bootstrap_ci_high': 0.078125, 'positive_pairs': 1, 'negative_pairs': 1, 'ties': 6, 'required_positive_pairs': 6, 'robust_improvement': False, 'mean_selector_pull': 0.22273222723742947, 'mean_baseline_pull': 0.11600226505834144, 'mean_selector_future_match': 0.421875, 'mean_baseline_future_match': 0.3125}, 'compat_proxy': {'condition': 'hidden_breakaway_pin', 'selector': 'proxy_combined_topk_action_geom', 'num_pairs': 8, 'mean_delta': -0.026041666666666668, 'baseline_mean_delta': 0.010416666666666668, 'paired_mean_gain': -0.036458333333333336, 'paired_median_gain': 0.0, 'bootstrap_ci_low': -0.109375, 'bootstrap_ci_high': 0.0, 'positive_pairs': 0, 'negative_pairs': 1, 'ties': 7, 'required_positive_pairs': 6, 'robust_improvement': False, 'mean_selector_pull': 0.2624665732146241, 'mean_baseline_pull': 0.11600226505834144, 'mean_selector_future_match': 0.1015625, 'mean_baseline_future_match': 0.3125}} |

## Interpretation

- Initial state equivalence is a hard prerequisite for selector comparison.
- Main gains are paired differences in `delta_fraction`, not unpaired final fractions.
- The two historical seed blocks are reported separately.
- `condition_nearest_upper` uses true condition labels and is not deployable.
- No model training was run.
- No future DDPM training was run.
- No Phase4 or CPS was run.
