# Phase3.9 Geometry-Aware IDM Runtime Config

- Timestamp: `2026-07-06T19:35:37+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `4be9aaf5ffeab648e110e8f847347b598d43d447`
- Baseline: `state_action`
- Out dir: `checkpoints/phase3_9_geometry_idm/state_action/fold_phase3_9_seed_390000`
- Epochs: `300`
- Batch size: `128`
- Hidden dim: `256`
- LR: `0.001`
- action_z_weight: `1.0`
- pose0_xy_weight: `8.0`
- pose1_xy_weight: `8.0`
- coupled_xy_weight: `8.0`
- pull_xy_weight: `4.0`
- z_weight: `0.25`
- quat_weight: `0.02`

Scope:
- train inverse dynamics only
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoint is local diagnostic artifact and should not be committed
