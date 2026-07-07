# Phase3.11b Retrieval Feasibility Runtime Config

- Timestamp: `2026-07-07T12:52:20+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `8dbca469cf89f915953e5c0524e9211c9a75d26f`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Candidates per condition: `3`
- Variants: `source_gt_action source_old_idm_future source_repaired_idm_future live_retrieved_gt_action live_repaired_idm_retrieved_future live_ddpm_mean_repaired_idm live_condition_retrieval_repaired_idm`
- Samples per step: `32`
- Motion timeout: `15.0`
- Action clip std: `3.0`
- Row timeout sec: `360`
- Total timeout sec: `14400`
- Max rows: `63`
- Best ablation: `xy_only_high_weight`

Scope:
- retrieval feasibility audit only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
