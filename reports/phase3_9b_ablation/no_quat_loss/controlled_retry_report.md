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
| `free` | 0.2917 | 0.0000 | 0.1944 | 0.1944 | 0.6667 | `True` |
| `hidden_breakaway_pin` | 0.2222 | 0.0417 | 0.2639 | 0.2222 | 1.1875 | `True` |
| `hidden_high_friction` | 0.8750 | 0.0833 | 1.0000 | 0.9167 | 1.1429 | `True` |

## Summary Table

| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `gt_reference` | `free` | 3 | 3 | 0 | 0.000 | 0.2917 | 0.2917 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.2222 | 0.2222 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.8750 | 0.8750 | 0.0330 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `old_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0651 | 0.0158 | 0.1194 | 0.0434 | 23.37 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0417 | 0.0417 | 0.0652 | 0.0132 | 0.0418 | 0.0985 | 18.97 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.333 | 0.0833 | 0.5625 | 0.0335 | 0.0114 | 0.0331 | 0.0822 | 21.00 | 0 |
| `state_action` | `phase39_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.1944 | 0.1944 | 0.0652 | 0.0008 | 0.0030 | 0.0030 | 4.64 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.2639 | 0.2639 | 0.0654 | 0.0010 | 0.0049 | 0.0038 | 1.77 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 1.000 | 1.0000 | 1.0000 | 0.0334 | 0.0012 | 0.0063 | 0.0042 | 3.34 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `phase39_geometry_idm_repair_supported` | ['free', 'hidden_breakaway_pin', 'hidden_high_friction'] |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- It does not run Phase4 or CPS.
- It does not prove deployable closed-loop policy success.
- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.
