# Phase3.8c Pose0-XY Repair Confirmation Runtime Config

- Timestamp: `2026-07-06T19:12:54+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `e7bcdc0e15512509cee47699ed9435f9e07ebc7e`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Baselines: `state_action`
- Sources: `idm_gt_future`
- Variants: `gt_reference idm_original idm_gt_pose0_xy idm_gt_pose1_xy idm_gt_pose0_pose1_xy`
- Samples per condition: `2`
- Max rows: `30`
- Row timeout sec: `240`
- Total timeout sec: `3600`
- Early stop: `disabled`

Scope:
- pose0-XY confirmation diagnostic only
- reuses Phase3.8b row-by-row CSV flush and per-row worker timeout
- GT-blended variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
