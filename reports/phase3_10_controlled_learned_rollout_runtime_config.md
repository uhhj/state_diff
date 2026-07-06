# Phase3.10 Controlled Learned Rollout Runtime Config

- Timestamp: `2026-07-06T22:00:21+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `ae76cb48ce4460459a6e097df1982ad98650902f`
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
- controlled learned rollout retry only
- DDPM predicted future + repaired IDM
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
