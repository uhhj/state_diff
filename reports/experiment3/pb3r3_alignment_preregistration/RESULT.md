# PB3-R3 Alignment Re-Preregistration

Verdict: `PB3R3_ALIGNMENT_RULE_PREREGISTERED`

## Scope

- CPU-only: Yes
- Genesis/GPU worker executed: No
- Future suffix executed: No
- Gate 4 executed: No
- Formal PB3 outcomes used for threshold: No

## New targeted-replay rope alignment rule

- Formal metric: coordinate-wise rope XYZ RMSE
- Calibration states: 20 independent states
- Repeats/state: 3
- State statistic: median of 3 reconstruction RMSEs
- Envelope: maximum of 20 state medians
- Frozen threshold: 61.945756736 um
- Rope max-abs: diagnostic only

## Unchanged

- EE max abs: 5.000000000e-05 m
- Motor qpos max abs: 5.000000000e-05 rad
- Snapshot restore rope max-abs: 5.000000000e-05 m
- Frozen winding-index equality: required
- Gate 4: 1 mm absolute effect and 5x repeat floor

## Future live pair revalidation

- All 10 formal pairs must pass original PB2-C 3-frame semantics live
- Pair dropping: forbidden
- Pair replacement: forbidden
- Any failure verdict: `PB3_R3_LIVE_BRANCH_ALIGNMENT_FAILED`

## Stop

Commit this rule before any PB3 Resume. Do not run formal PB3 in this phase.
