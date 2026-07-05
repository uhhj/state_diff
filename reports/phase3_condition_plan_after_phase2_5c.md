# Phase3 Condition Plan After Phase2.5c

- Phase2.5c verdict: `PASS`
- hard diagnostic condition: `hidden_pin`
- weak/auxiliary condition: `hidden_high_friction`
- selected recoverable CPS branch: `hidden_breakaway_pin`
- selected recoverable config: `breakaway_force_2p6_disp_0p045_pull_0p36`

## Selected Env Vars

- `CCDA_BREAKAWAY_FORCE=2.6`
- `CCDA_BREAKAWAY_DISP=0.045`
- `CCDA_BREAKAWAY_BEAD_RATIO=0.45`
- `CCDA_ORACLE_BREAKAWAY_PULL_DIST=0.36`

## Phase3 medium should use

conditions:
- `free`
- `hidden_pin`
- `hidden_high_friction`
- `hidden_breakaway_pin`

primary_branch_pair:
- `free` vs `hidden_breakaway_pin`

diagnostic_branch_pair:
- `free` vs `hidden_pin`
