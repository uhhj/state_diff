# Phase3.9 Geometry-Aware IDM Controlled Retry Report

## Verdict

- Verdict: `FAIL`
- Root cause: `phase39_repair_not_supported`
- Rows: `27`
- OK rows: `27`
- Timeout rows: `0`
- Failed rows: `0`
- Progress status: `completed`
- Improved conditions: `['hidden_high_friction']`
- Primary improved: `False`

## Per-Condition Retry

| Condition | GT Δ | Old IDM Δ | Phase3.9 IDM Δ | Improvement | Ratio to GT | Improved |
|---|---:|---:|---:|---:|---:|---:|
| `free` | 0.2917 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | `False` |
| `hidden_breakaway_pin` | 0.1806 | 0.0694 | 0.0000 | -0.0694 | 0.0000 | `False` |
| `hidden_high_friction` | 0.7083 | 0.0417 | 0.4792 | 0.4375 | 0.6765 | `True` |

## Summary Table

| Baseline | Action | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `gt_reference` | `free` | 3 | 3 | 0 | 0.000 | 0.2917 | 0.2917 | 0.0651 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.1806 | 0.1806 | 0.0652 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `gt_reference` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.7083 | 0.7083 | 0.0336 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `old_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0651 | 0.0158 | 0.1194 | 0.0434 | 23.37 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0694 | 0.0694 | 0.0652 | 0.0132 | 0.0418 | 0.0985 | 18.97 | 0 |
| `state_action` | `old_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.667 | 0.0417 | 0.0417 | 0.0336 | 0.0114 | 0.0331 | 0.0822 | 21.00 | 0 |
| `state_action` | `phase39_idm_gt_future` | `free` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0651 | 0.0015 | 0.0067 | 0.0059 | 1.09 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_breakaway_pin` | 3 | 3 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0654 | 0.0017 | 0.0090 | 0.0084 | 5.13 | 0 |
| `state_action` | `phase39_idm_gt_future` | `hidden_high_friction` | 3 | 3 | 0 | 0.333 | 0.4792 | 0.9375 | 0.0282 | 0.0019 | 0.0100 | 0.0079 | 1.65 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `FAIL` | `phase39_repair_not_supported` | ['hidden_high_friction'] |

## Interpretation

- This is one-step matched-prefix diagnostic only.
- It does not run Phase4 or CPS.
- It does not prove deployable closed-loop policy success.
- If Phase3.9 IDM improves hidden_breakaway_pin, the next step is still a controlled retry/ablation, not CPS.
