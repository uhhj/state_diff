# PB3-R2 Independent Replay Floor Calibration

Verdict: `PB3R2_REPLAY_FLOOR_CALIBRATED`

## Scope

- Formal PB3 remains blocked: Yes
- Future suffix executed: No
- Gate 4 executed: No
- Frozen 50 um alignment reference modified: No
- New alignment threshold selected: No
- Formal PB3 shortlist modified: No

## Calibration design

- Calibration states: 20
- Formal-shortlist overlap: 0 rollouts
- Fresh-process repeats/state: 3
- Frozen reconstruction samples: 60
- Fresh-process repeat-pair samples: 60
- Fresh Genesis processes: 12
- All 60 target samples finite: True

## Frozen reconstruction — rope max abs

- median: 2.447143197e-05 m
- P95: 3.953114152e-04 m
- P99: 5.167424679e-04 m
- max: 5.167424679e-04 m

## Fresh-process repeat — rope max abs

- median: 9.458744898e-10 m
- P95: 5.960464478e-08 m
- P99: 1.652538776e-05 m
- max: 1.652538776e-05 m

## Original 50 um rope-only reference

- Pass rate: 36/60 (60.00%)
- Exceedances: 24

## Next action

Use this independent calibration distribution to pre-register PB3-R3 alignment handling. Do not choose the new rule from the formal 10-pair PB3 outcomes, and do not resume Gate 4 before that rule is committed.
