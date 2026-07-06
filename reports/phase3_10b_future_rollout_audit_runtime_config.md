# Phase3.10b Future Rollout Audit Runtime Config

- Timestamp: `2026-07-06T23:35:33+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `87a2a01e2963c2832e5300cfb2e8c62a78b7447a`
- Policies: `old_state_action phase39_default_geometry phase39b_xy_only_high_weight`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Episodes per condition: `4`
- Max steps: `16`
- Samples per step: `16`
- Motion timeout: `15.0`
- Action clip std: `3.0`
- Row timeout sec: `900`
- Total timeout sec: `14400`
- Max rows: `36`
- Best ablation: `xy_only_high_weight`

Scope:
- learned future quality + rollout error audit only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
