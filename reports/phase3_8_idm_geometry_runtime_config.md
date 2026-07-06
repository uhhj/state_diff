# Phase3.8 IDM Geometry Runtime Config

- Timestamp: `2026-07-06T16:24:42+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `bf652354c925f3865f0485544cbdcc94710ff089`
- Conditions: `free hidden_pin hidden_high_friction hidden_breakaway_pin`
- Primary hidden condition: `hidden_breakaway_pin`
- Diagnostic hidden condition: `hidden_pin`
- Baselines: `paper_state state_action`
- Sources: `idm_gt_future idm_pred_future`
- Samples per condition: `3`
- Pred samples: `16`
- Motion timeout: `15.0`
- Max prefix actions: `20`
- Action clip std: `3.0`

Scope:
- IDM action geometry repair diagnostic only
- GT-blended repair variants are diagnostic upper bounds only
- no Phase4
- no CPS
- no retraining
