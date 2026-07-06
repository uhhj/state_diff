# Phase3.8b Narrowed Pose1-XY Repair Runtime Config

- Timestamp: `2026-07-06T18:51:53+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `a3288f51c1a080a75a2e452ccd7ccde19166af9a`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Baselines: `state_action`
- Sources: `idm_gt_future`
- Variants: `gt_reference idm_original idm_gt_pose1_xy idm_gt_pose0_xy idm_gt_pose0_pose1_xy idm_gt_pull_dir_gt_len`
- Samples per condition: `1`
- Max rows: `18`
- Row timeout sec: `240`
- Total timeout sec: `1800`
- Stop after first effective: `1`

Scope:
- narrowed pose1-XY repair diagnostic only
- row-by-row CSV flush
- per-row worker timeout
- GT-blended variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
