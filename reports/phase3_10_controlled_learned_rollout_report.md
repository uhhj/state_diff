# Phase3.10 Controlled Learned Rollout Retry Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase310_repaired_rollout_not_supported`
- Rows: `36`
- Progress status: `completed`
- Improved default conditions: `['hidden_high_friction']`
- Improved best conditions: `[]`
- Primary default improved: `False`
- Primary best improved: `False`

## Per-Condition Learned Rollout

| Condition | Old final fraction | Default final fraction | Best final fraction | Default improvement | Best improvement | Default improved | Best improved |
|---|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.2292 | 0.2604 | 0.1875 | 0.0312 | -0.0417 | `False` | `False` |
| `hidden_breakaway_pin` | 0.0729 | 0.0208 | 0.1146 | -0.0521 | 0.0417 | `False` | `False` |
| `hidden_high_friction` | 0.1667 | 0.2188 | 0.0000 | 0.0521 | -0.1667 | `True` | `False` |

## Summary Table

| Policy | Condition | Rows | OK | Timeout | Success | Final fraction | Steps | Action OOD | Future std | Pull len | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `old_state_action` | `free` | 4 | 4 | 0 | 0.000 | 0.2292 | 16.00 | 0.235 | 0.0386 | 0.0848 | 0 |
| `old_state_action` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.0729 | 16.00 | 0.454 | 0.0359 | 0.2238 | 0 |
| `old_state_action` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.1667 | 16.00 | 0.460 | 0.0359 | 0.1752 | 0 |
| `phase39_default_geometry` | `free` | 4 | 4 | 0 | 0.000 | 0.2604 | 16.00 | 0.238 | 0.0388 | 0.0756 | 0 |
| `phase39_default_geometry` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.0208 | 16.00 | 0.263 | 0.0386 | 0.0789 | 0 |
| `phase39_default_geometry` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.2188 | 16.00 | 0.266 | 0.0342 | 0.0742 | 0 |
| `phase39b_xy_only_high_weight` | `free` | 4 | 4 | 0 | 0.000 | 0.1875 | 16.00 | 0.339 | 0.0391 | 0.1525 | 0 |
| `phase39b_xy_only_high_weight` | `hidden_breakaway_pin` | 4 | 4 | 0 | 0.000 | 0.1146 | 16.00 | 0.318 | 0.0384 | 0.1271 | 0 |
| `phase39b_xy_only_high_weight` | `hidden_high_friction` | 4 | 4 | 0 | 0.000 | 0.0000 | 16.00 | 0.511 | 0.0356 | 0.2179 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `phase310_repaired_rollout_not_supported` | {'best': [], 'default': ['hidden_high_friction']} |

## Interpretation

- This tests DDPM predicted future + repaired IDM in a small closed-loop runtime.
- It does not run Phase4 or CPS.
- It does not prove paper-level policy success.
- If repaired rollout improves primary branch, next step is still Phase3.10b repeat/error audit, not CPS.
