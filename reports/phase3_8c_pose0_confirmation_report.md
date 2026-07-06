# Phase3.8c Pose0-XY Repair Confirmation Report

## Verdict

- Verdict: `WARN`
- Root cause: `coupled_pick_place_xy_geometry_blocker_supported`
- Rows: `30`
- OK rows: `30`
- Timeout rows: `0`
- Failed rows: `0`
- Progress status: `completed`
- Pose0 supported conditions: `['free']`
- Pose1 supported conditions: `['hidden_breakaway_pin', 'hidden_high_friction']`
- Coupled xy supported conditions: `['free', 'hidden_breakaway_pin', 'hidden_high_friction']`

## Per-Condition Confirmation

| Condition | GT Δ | IDM original Δ | Pose0 XY Δ | Pose1 XY Δ | Pose0+Pose1 XY Δ | Pose0 effective | Pose1 effective | Both XY effective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 0.4375 | 0.0000 | 0.3333 | 0.0000 | 0.4583 | `True` | `False` | `True` |
| `hidden_breakaway_pin` | 0.3542 | 0.0000 | 0.0208 | 0.3542 | 0.2708 | `False` | `True` | `True` |
| `hidden_high_friction` | 0.7500 | 0.0833 | 0.0000 | 0.2292 | 0.8750 | `False` | `True` | `True` |

## Summary Table

| Baseline | Source | Variant | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `idm_gt_future` | `gt_reference` | `free` | 2 | 2 | 0 | 0.000 | 0.4375 | 0.4375 | 0.0474 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `gt_reference` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.3542 | 0.3542 | 0.0474 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `gt_reference` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.7500 | 0.7500 | 0.0222 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_pose1_xy` | `free` | 2 | 2 | 0 | 0.000 | 0.4583 | 0.4583 | 0.0474 | 0.0008 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_pose1_xy` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.2708 | 0.2708 | 0.0474 | 0.0009 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_pose1_xy` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.8750 | 0.8750 | 0.0225 | 0.0008 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_xy` | `free` | 2 | 2 | 0 | 0.000 | 0.3333 | 0.3333 | 0.0474 | 0.0049 | 0.0000 | 0.0473 | 5.13 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_xy` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.0208 | 0.0208 | 0.0474 | 0.0058 | 0.0000 | 0.0587 | 8.89 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_xy` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.0000 | 0.0000 | 0.0228 | 0.0049 | 0.0000 | 0.0490 | 8.19 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose1_xy` | `free` | 2 | 2 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0474 | 0.0065 | 0.0590 | 0.0000 | 17.96 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose1_xy` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.3542 | 0.3542 | 0.0474 | 0.0058 | 0.0514 | 0.0000 | 12.83 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose1_xy` | `hidden_high_friction` | 2 | 2 | 0 | 0.000 | 0.2292 | 0.7083 | 0.0220 | 0.0047 | 0.0392 | 0.0000 | 13.85 | 0 |
| `state_action` | `idm_gt_future` | `idm_original` | `free` | 2 | 2 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0474 | 0.0106 | 0.0590 | 0.0473 | 15.18 | 0 |
| `state_action` | `idm_gt_future` | `idm_original` | `hidden_breakaway_pin` | 2 | 2 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0474 | 0.0107 | 0.0514 | 0.0587 | 13.73 | 0 |
| `state_action` | `idm_gt_future` | `idm_original` | `hidden_high_friction` | 2 | 2 | 0 | 0.500 | 0.0833 | 0.0833 | 0.0221 | 0.0088 | 0.0392 | 0.0490 | 14.89 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `coupled_xy_repair_supported` | ['free', 'hidden_breakaway_pin', 'hidden_high_friction'] |

## Interpretation

- This confirms or rejects the Phase3.8b early-stop pose0_xy finding across multiple conditions.
- GT-blended pose0/pose1 repairs are diagnostic upper bounds only.
- If pose0_xy is effective on hidden_breakaway_pin, the next repair target is pose0/pick-point geometry.
- If only pose0+pose1_xy is effective, use coupled decoded-pose-space loss.
- This is not Phase4 and not CPS evidence.
