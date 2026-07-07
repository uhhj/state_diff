# Phase3.11 Future-Source Swap Runtime Config

- Timestamp: `2026-07-07T11:09:21+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `8555def3a8a36e19d353b0dbf70f246a69755bf8`
- IDM policies: `phase39b_xy_only_high_weight`
- Future sources: `ddpm_mean ddpm_best_of_k_by_train_nn global_input_retrieval condition_matched_retrieval`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Episodes per condition: `3`
- Max steps: `16`
- Samples per step: `32`
- Motion timeout: `15.0`
- Action clip std: `3.0`
- Row timeout sec: `900`
- Total timeout sec: `14400`
- Max rows: `36`
- Best ablation: `xy_only_high_weight`

Scope:
- future-source swap diagnostic only
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
- condition_matched_retrieval is diagnostic upper bound only
