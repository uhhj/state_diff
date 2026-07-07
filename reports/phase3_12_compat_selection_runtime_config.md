# Phase3.12 Compatibility-Aware Future Selection Runtime Config

- Timestamp: `2026-07-07T14:56:23+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `04ff1504eb7fc84e4d710caf967ba17bafc0d3c2`
- Selectors: `ddpm_mean input_nearest condition_nearest compat_global_topk compat_condition_topk compat_condition_action_geom`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Episodes per condition: `3`
- Max steps: `16`
- Samples per step: `32`
- Top-K: `32`
- Motion timeout: `15.0`
- Action clip std: `3.0`
- Row timeout sec: `900`
- Total timeout sec: `21600`
- Max rows: `54`
- Best ablation: `xy_only_high_weight`

Score weights:
- context: `1.0`
- action_ood: `0.15`
- clip: `2.0`
- action_mae: `4.0`
- pull_diff: `2.0`
- small_pull: `2.0`
- min_pull: `0.08`

Scope:
- compatibility-aware future target selection diagnostic only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- condition-aware selectors are diagnostic upper bounds only
