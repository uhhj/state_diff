# Phase3.12c Matched-Reset Runtime Config

- Timestamp: `2026-07-10T12:04:05+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `3401fd7164669804511d1df66ac6bd9e8dd00c5c`
- Submodule: ` 388a3f5582c8fdffaecb06448f285d693a45b83c external/deformable-ravens (heads/ccda-cable)`
- Selectors: `ddpm_mean condition_nearest_upper proxy_state_motion_nn proxy_combined_topk_action_geom`
- Conditions: `free hidden_breakaway_pin hidden_high_friction`
- Visible seeds: `312000 312001 312002 312003 312500 312501 312502 312503`
- Baseline selector: `ddpm_mean`
- Primary condition: `hidden_breakaway_pin`
- Expected reset rows: `96`
- Expected rollout rows: `96`
- Maximum step rows: `1536`
- Max rollout steps: `16`
- DDPM samples per step: `32`
- Top-K: `64`

Deterministic reset:
- minimum settle steps: `540`
- maximum settle steps: `2400`
- static checks required: `8`
- static check interval: `10`
- rounded state decimals: `7`
- wall-clock settling disabled: `True`
- explicit Python/NumPy/Torch seeding: `True`
- selector-independent pair group: `True`
- condition-independent pair group: `True`

Matched-reset thresholds:
- initial state max abs: `1e-6`
- initial state MAE: `1e-7`
- initial fraction diff: `1e-9`
- initial curve diff: `1e-7`

Paired improvement criteria:
- mean paired gain threshold: `0.05`
- minimum pairs: `6`
- minimum positive pair fraction: `0.75`
- bootstrap samples: `10000`
- bootstrap CI lower bound must be above zero: `True`

Scope:
- no model training
- no future DDPM training
- no Phase4
- no CPS
- no medium/full rollout
