# Phase 0B-R2-R1 restricted low-stiffness result

## Verdict

- Final verdict: `PHASE0B_R2R1_ENGINEERING_BLOCKED`.
- Starting SHA: `ce88eddf0b05a500e79e03ff4ec4bfd4984e5a84`.
- Ending implementation SHA: `9f46781181ff077a77bf35ec2bbb117bf8a8a96c`.
- Frozen gitlink: `282b93535b125d1a4487df85ad24aa41551957af`.
- Tests: 46 passed, 0 failed.
- Known pip conflict only: yes.

## Family confirmation

- Profile: `kv_r2r1_a1p50_z025`.
- M8/M16 verdict: `PHASE0B_R2R1_FAMILY_CONFIRMATION_BLOCKED`.
- Maximum relative error: `0.0190176636072027`.
- M retained for calibration: `None`; prior M8 selection reopened: no.

## Calibration

- Profiles run: ``.
- Axial verdict: `not run`; peak primary: `not run` m.
- Shear verdict: `not run`; peak primary: `not run` m.
- Table verdict: `not run`.
- Material frozen: `False`.
- Pair run: no; training run: no.

Failure cause: `A1.5 M8 two-node energy increase fraction 0.020876826722338204 exceeded the fixed 0.02 gate`.

Next permitted task: stop and address only the reported calibration gate; do not run Pair.
