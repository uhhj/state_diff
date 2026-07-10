# Phase3.12b Condition Proxy / Score Ablation Runtime Config

- Timestamp: `2026-07-10T09:59:58+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `55fa480797f693a7477ba607f9fa5ddc0ddc1a89`
- Selectors: `ddpm_mean condition_nearest_upper proxy_action_nn proxy_state_motion_nn proxy_combined_nn proxy_combined_topk_action_geom proxy_combined_topk_no_action_geom`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Episodes per condition: `4`
- Max steps: `16`
- Samples per step: `32`
- Top-K: `64`
- Motion timeout: `15.0`
- Action clip std: `3.0`
- Row timeout sec: `900`
- Total timeout sec: `28800`
- Max rows: `84`
- Best ablation: `xy_only_high_weight`

Score weights:
- proxy: `1.0`
- action_ood: `0.15`
- clip: `2.0`
- action_mae: `4.0`
- pull_diff: `2.0`
- small_pull: `2.0`
- min_pull: `0.08`

Scope:
- observable condition-proxy and score-ablation diagnostic only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- condition_nearest_upper is diagnostic upper bound only
