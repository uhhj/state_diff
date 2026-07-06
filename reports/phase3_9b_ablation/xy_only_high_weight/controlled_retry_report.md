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
| `free` | 0.2778 | 0.0000 | 0.2917 | 0.2917 | 1.0500 | `True` |
| `hidden_breakaway_pin` | 0.1944 | 0.0417 | 0.2778 | 0.2361 | 1.4286 | `True` |
| `hidden_high_friction` | 0.9167 | 0.0000 | 0.4167 | 0.4167 | 0.4545 | `True` |

## Summary Table

| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `gt_reference` | `free` | 3 | 3 | 0 | 0.000 | 0.2778 | 0.2778 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.1944 | 0.1944 | 0.0651 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.9167 | 0.9167 | 0.0330 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `old_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0651 | 0.0158 | 0.1194 | 0.0434 | 23.37 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0417 | 0.0417 | 0.0651 | 0.0132 | 0.0418 | 0.0985 | 18.97 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.0000 | 0.0000 | 0.0332 | 0.0114 | 0.0331 | 0.0822 | 21.00 | 0 |
| `state_action` | `phase39_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.2917 | 0.2917 | 0.0651 | 0.0011 | 0.0051 | 0.0047 | 1.44 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.2778 | 0.2778 | 0.0651 | 0.0012 | 0.0064 | 0.0052 | 1.18 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.333 | 0.4167 | 0.7292 | 0.0340 | 0.0011 | 0.0049 | 0.0047 | 2.53 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase39_geometry_idm_repair_supported` | ['free', 'hidden_breakaway_pin', 'hidden_high_friction'] |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- It does not run Phase4 or CPS.
- It does not prove deployable closed-loop policy success.
- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.
