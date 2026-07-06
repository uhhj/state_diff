# Phase3.7 Learned Action Alignment Runtime Config

- Timestamp: `2026-07-06T12:35:18+08:00`
- Python: `/miniforge3/envs/coord_bimanual/bin/python`
- Conda env: `coord_bimanual`
- Main HEAD: `45aea45571ed2c2392d60ffc45c0c0fe2f7b71fe`
- Conditions: `free hidden_pin hidden_high_friction hidden_breakaway_pin`
- Primary hidden condition: `hidden_breakaway_pin`
- Diagnostic hidden condition: `hidden_pin`
- Samples per condition: `3`
- Baselines: `paper_state state_action`
- Pred samples: `16`
- Execute actions: `1`
- Motion timeout: `15.0`
- Max prefix actions: `20`
- Action clip std: `3.0`

Scope:
- learned action alignment diagnostic only
- no Phase4
- no CPS
- no paper-level claims
