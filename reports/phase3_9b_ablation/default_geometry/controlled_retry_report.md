# Phase3.9 Geometry-Aware IDM Controlled Retry Report

## Verdict

- Verdict: `WARN`
- Root cause: `phase39_geometry_idm_repair_supported`
- Rows: `27`
- OK rows: `27`
- Timeout rows: `0`
- Failed rows: `0`
- Progress status: `completed`
- Improved conditions: `['free', 'hidden_breakaway_pin', 'hidden_high_friction']`
- Primary improved: `True`

## Per-Condition Retry

| Condition | GT Δ | Old IDM Δ | Phase3.9 IDM Δ | Improvement | Ratio to GT | Improved |
|---|---:|---:|---:|---:|---:|---:|
| `free` | 0.2917 | 0.0000 | 0.2500 | 0.2500 | 0.8571 | `True` |
| `hidden_breakaway_pin` | 0.2361 | 0.0556 | 0.1250 | 0.0694 | 0.5294 | `True` |
| `hidden_high_friction` | 0.1875 | 0.0833 | 0.2500 | 0.1667 | 1.3333 | `True` |

## Summary Table

| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `gt_reference` | `free` | 3 | 3 | 0 | 0.000 | 0.2917 | 0.2917 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.2361 | 0.2361 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_high_friction` | 3 | 3 | 0 | 0.333 | 0.1875 | 0.6042 | 0.0331 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `old_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0652 | 0.0158 | 0.1194 | 0.0434 | 23.37 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0556 | 0.0556 | 0.0651 | 0.0132 | 0.0418 | 0.0985 | 18.97 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.0833 | 0.0833 | 0.0337 | 0.0114 | 0.0331 | 0.0822 | 21.00 | 0 |
| `state_action` | `phase39_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.2500 | 0.2500 | 0.0651 | 0.0018 | 0.0092 | 0.0084 | 6.59 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.1250 | 0.1250 | 0.0652 | 0.0018 | 0.0108 | 0.0086 | 4.49 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.333 | 0.2500 | 0.7292 | 0.0333 | 0.0018 | 0.0076 | 0.0108 | 4.69 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase39_geometry_idm_repair_supported` | ['free', 'hidden_breakaway_pin', 'hidden_high_friction'] |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- It does not run Phase4 or CPS.
- It does not prove deployable closed-loop policy success.
- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.
