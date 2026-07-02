# Phase1.1 Continuous PyBullet Rollout Video Report

## Purpose

This report summarizes continuous PyBullet execution videos for the hidden-contact cable task.
Unlike the earlier action-step dataset replay videos, these MP4 files are recorded while `Environment.pick_place()` is executing, so the UR5 approach, descent, grasp, movement, release, and settle phases are visible.

## Camera

- View: side-top debug camera, not the policy RGB-D camera
- Camera position: `[0.55, -0.75, 0.45]`
- Camera target: `[0.5, 0.0, 0.02]`
- Image size: `960x720`

## Videos

| Condition | Seed | Status | Action Steps | Frames | Reward | Done | Path |
|---|---:|---|---:|---:|---:|---|---|
| free | 0 | written | 1 | 129 | 0.0000 | True | `/data/state_diff2/reports/phase1_continuous_videos/seed_0_free.mp4` |
| hidden_pin | 0 | written | 1 | 126 | 0.0000 | True | `/data/state_diff2/reports/phase1_continuous_videos/seed_0_hidden_pin.mp4` |
| hidden_high_friction | 0 | written | 1 | 128 | 0.0000 | True | `/data/state_diff2/reports/phase1_continuous_videos/seed_0_hidden_high_friction.mp4` |
| hidden_side_jam | 0 | written | 1 | 132 | 0.0000 | True | `/data/state_diff2/reports/phase1_continuous_videos/seed_0_hidden_side_jam.mp4` |

## Interpretation

- These videos are intended for qualitative inspection of continuous primitive execution and contact timing.
- They complement, but do not replace, the Phase1 RGB-D checks, bead trajectory overlays, and Phase2 threshold-based CCDA pair mining.
- This report does not claim that CCDA has been fully proven.
