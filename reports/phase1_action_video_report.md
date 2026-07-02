# Phase1 Action Execution Video Report

## Purpose

This report summarizes action-step observation videos generated from saved DeformableRavens RGB observations.
Each MP4 shows four hidden-contact conditions in a 2x2 grid for the same visible seed.

## Important Note

These are dataset replay videos, not continuous PyBullet substep recordings.
Each frame corresponds to a saved observation before or after an action step.

## Videos

| Seed | Status | Frames | Max Obs Steps | Path |
|---|---|---:|---:|---|
| 0 | written | 14 | 10 | `/data/state_diff2/reports/phase1_action_videos/seed_0_action_steps.mp4` |
| 1 | written | 11 | 7 | `/data/state_diff2/reports/phase1_action_videos/seed_1_action_steps.mp4` |
| 2 | written | 13 | 9 | `/data/state_diff2/reports/phase1_action_videos/seed_2_action_steps.mp4` |
| 3 | written | 25 | 21 | `/data/state_diff2/reports/phase1_action_videos/seed_3_action_steps.mp4` |
| 4 | written | 13 | 9 | `/data/state_diff2/reports/phase1_action_videos/seed_4_action_steps.mp4` |

## Interpretation

- Use these videos to inspect whether the same visible seed produces different action-step outcomes under different hidden-contact conditions.
- For formal CCDA proof, combine this video check with RGB-D difference metrics, bead trajectory divergence, success difference, and later threshold-based pair mining.
