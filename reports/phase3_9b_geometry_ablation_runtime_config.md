# Phase3.9b Geometry IDM Ablation Runtime Config

- Timestamp: `2026-07-06T19:57:47+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `85cd6b578ffee68484c426ac208268fe280c5f71`
- Ablations: `default_geometry no_pull_loss no_coupled_xy_loss xy_only_high_weight no_quat_loss`
- Baseline: `state_action`
- Epochs: `250`
- Samples per condition: `4`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Actions: `gt_reference old_idm_gt_future phase39_idm_gt_future`
- Row timeout sec: `240`
- Total timeout sec per ablation: `7200`
- Max rows per ablation: `36`

Scope:
- geometry-aware inverse dynamics ablation only
- one-step matched-prefix controlled retry only
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- checkpoints are local-only and must not be committed
