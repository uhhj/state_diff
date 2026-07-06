# Phase3.8b Narrowed Pose1-XY Repair Report

## Verdict

- Verdict: `WARN`
- Root cause: `pose0_pick_xy_geometry_blocker_supported`
- Rows: `4`
- OK rows: `4`
- Timeout rows: `0`
- Failed rows: `0`
- Progress status: `stopped_after_first_effective`
- GT reference Δ final_fraction: `0.875000`
- IDM original Δ final_fraction: `0.000000`
- pose1_xy repair Δ final_fraction: `0.000000`
- pose0_xy repair Δ final_fraction: `0.666667`
- pose0+pose1_xy repair Δ final_fraction: `nan`
- pull_direction_gt_len repair Δ final_fraction: `nan`

## Effective GT-Blended Repairs

| Repair | Mean Δ final_fraction |
|---|---:|
| `pose0_xy` | 0.666667 |

## Summary Table

| Baseline | Source | Variant | Condition | Rows | OK | Timeout | Success | Δ final_fraction | Final fraction | Prefix MAE | Action MAE | Pose0 dist | Pose1 dist | Pull angle | Failures |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_action` | `idm_gt_future` | `gt_reference` | `free` | 1 | 1 | 0 | 0.000 | 0.8750 | 0.8750 | 0.0038 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose0_xy` | `free` | 1 | 1 | 0 | 0.000 | 0.6667 | 0.6667 | 0.0038 | 0.0028 | 0.0000 | 0.0157 | 1.71 | 0 |
| `state_action` | `idm_gt_future` | `idm_gt_pose1_xy` | `free` | 1 | 1 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0038 | 0.0048 | 0.0335 | 0.0000 | 3.59 | 0 |
| `state_action` | `idm_gt_future` | `idm_original` | `free` | 1 | 1 | 0 | 0.000 | 0.0000 | 0.0000 | 0.0038 | 0.0062 | 0.0335 | 0.0157 | 5.31 | 0 |

## Issues

| Level | Name | Detail |
|---|---|---|
| `WARN` | `narrowed_gt_blended_repair_restores_progress` | {'pose0_xy': 0.6666666666666666} |

## Interpretation

- This is a narrowed diagnostic, not a full Phase3.8 repair matrix.
- GT-blended variants are diagnostic upper bounds only.
- If pose1_xy restores progress, pose1/place endpoint geometry is supported as the first repair target.
- If pose0_pose1_xy is required, the blocker is coupled pick/place geometry.
- This is not Phase4 and not CPS evidence.
