# Phase3.9 Geometry-Aware IDM Controlled Retry Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase39_geometry_idm_repair_supported`
- Rows: `18`
- OK rows: `18`
- Timeout rows: `0`
- Failed rows: `0`
- Progress status: `completed`
- Improved conditions: `['free', 'hidden_breakaway_pin', 'hidden_high_friction']`
- Primary improved: `True`

## Per-Condition Retry

| Condition | GT Δ | Old IDM Δ | Phase3.9 IDM Δ | Improvement | Ratio to GT | Improved |
|---|---:|---:|---:|---:|---:|---:|
| `free` | 0.4583 | 0.0000 | 0.4583 | 0.4583 | 1.0000 | `True` |
| `hidden_breakaway_pin` | 0.3750 | 0.0208 | 0.4375 | 0.4167 | 1.1667 | `True` |
| `hidden_high_friction` | 0.8333 | 0.0833 | 0.8750 | 0.7917 | 1.0500 | `True` |

## Summary Table

| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `gt_reference` | `free` | 2 | 2 | 0 | 0.000 | 0.4583 | 0.4583 | 0.0474 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.3750 | 0.3750 | 0.0474 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.8333 | 0.8333 | 0.0222 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `old_idm_gt_future` | `free` | 2 | 2 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0474 | 0.0106 | 0.0590 | 0.0473 | 15.18 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.0208 | 0.0208 | 0.0474 | 0.0107 | 0.0514 | 0.0587 | 13.73 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.0833 | 0.0833 | 0.0225 | 0.0088 | 0.0392 | 0.0490 | 14.89 | 0 |
| `state_action` | `phase39_idm_gt_future` | `free` | 2 | 2 | 0 | 0.000 | 0.4583 | 0.4583 | 0.0474 | 0.0019 | 0.0079 | 0.0094 | 1.44 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.4375 | 0.4375 | 0.0474 | 0.0022 | 0.0115 | 0.0076 | 1.95 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.8750 | 0.8750 | 0.0225 | 0.0017 | 0.0069 | 0.0069 | 1.18 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase39_geometry_idm_repair_supported` | ['free', 'hidden_breakaway_pin', 'hidden_high_friction'] |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- It does not run Phase4 or CPS.
- It does not prove deployable closed-loop policy success.
- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.
