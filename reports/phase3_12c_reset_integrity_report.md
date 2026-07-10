# Phase3.12c Matched-Reset Integrity Report

## Verdict

- Verdict: `PASS`
- Root cause: `phase312c_matched_reset_integrity_passed`
- Reset rows: `96`
- Pair rows: `96`
- Maximum initial state absolute difference: `0.0`
- Maximum initial fraction difference: `0.0`
- Paired selector rollout allowed: `True`

## Integrity thresholds

| Threshold | Value |
|---|---:|
| Initial state max abs | 1e-06 |
| Initial state MAE | 1e-07 |
| Initial fraction diff | 1e-09 |
| Initial curve diff | 1e-07 |

## Pair summary

| Condition | Seed | Selector | Max abs | MAE | Fraction diff | Curve diff | Hash equal | Pass |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `free` | 312000 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312000 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312000 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312000 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312001 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312001 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312001 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312001 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312002 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312002 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312002 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312002 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312003 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312003 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312003 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312003 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312500 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312500 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312500 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312500 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312501 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312501 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312501 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312501 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312502 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312502 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312502 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312502 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312503 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312503 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312503 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `free` | 312503 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312000 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312000 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312000 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312000 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312001 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312001 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312001 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312001 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312002 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312002 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312002 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312002 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312003 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312003 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312003 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312003 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312500 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312500 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312500 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312500 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312501 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312501 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312501 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312501 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312502 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312502 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312502 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312502 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312503 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312503 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312503 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_breakaway_pin` | 312503 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312000 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312000 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312000 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312000 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312001 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312001 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312001 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312001 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312002 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312002 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312002 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312002 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312003 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312003 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312003 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312003 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312500 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312500 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312500 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312500 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312501 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312501 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312501 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312501 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312502 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312502 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312502 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312502 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312503 | `ddpm_mean` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312503 | `condition_nearest_upper` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312503 | `proxy_state_motion_nn` | 0 | 0 | 0 | 0 | 1 | 1 |
| `hidden_high_friction` | 312503 | `proxy_combined_topk_action_geom` | 0 | 0 | 0 | 0 | 1 | 1 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `reset_did_not_reach_static_threshold` | 96 |

## Interpretation

- Selector quality must not be analyzed unless this report is PASS.
- Reset groups are keyed by `condition + visible_seed`.
- Pair group is selector-independent and condition-independent.
- Wall-clock cable settling was replaced with deterministic PyBullet stepping.
- No Phase4 or CPS was run.
