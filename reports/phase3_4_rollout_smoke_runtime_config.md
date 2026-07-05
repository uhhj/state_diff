# Phase3.4 Rollout Smoke Runtime Config

- Timestamp: 2026-07-05T22:54:58+08:00
- Python: /miniforge3/envs/coord_bimanual/bin/python
- Conda env: coord_bimanual
- Main HEAD: 0f825485c173cedd42a1c9810603f85966622f76
- Conditions: free hidden_pin hidden_high_friction hidden_breakaway_pin
- Primary hidden condition: hidden_breakaway_pin
- Diagnostic hidden condition: hidden_pin
- Max episodes per condition: 4
- Max steps: 16
- Rollout models: paper_state state_action
- CCDA_BREAKAWAY_FORCE: 2.6
- CCDA_BREAKAWAY_DISP: 0.045
- CCDA_BREAKAWAY_BEAD_RATIO: 0.45
- CCDA_ORACLE_BREAKAWAY_PULL_DIST: 0.36

Scope:
- learned rollout smoke only
- no Phase4
- no CPS
- no paper-level claims
