# Phase 0B-R2-R1 restricted low-stiffness result

## Verdict

- Final verdict: `PHASE0B_R2R1_NO_PROFILE_PASSED`.
- Starting SHA: `a02f7393686d0dfcd62a6044299942b6bde168b6`.
- Ending implementation SHA: `cdc3a22d77d8539bad1a128b66f8aa2d31a31eb3`.
- Frozen gitlink: `282b93535b125d1a4487df85ad24aa41551957af`.
- Tests: 31 passed, 0 failed.
- Known pip conflict only: yes.

## Family confirmation

- Profile: `kv_r2r1_a1p50_z025`.
- M8/M16 verdict: `PHASE0B_R2R1_FAMILY_CONFIRMATION_COMPLETE`.
- Energy measurement: post-step aligned.
- Maximum relative error: `0.0190176636072027`.
- M retained for calibration: `8`; prior M8 selection reopened: no.

## Calibration

- Profiles run: `kv_r2r1_a1p25_z025`.
- Axial verdict: `PHASE0B_R2_AXIAL_COMPLETE`; peak primary: `0.002796818638032087` m.
- Shear verdict: `PHASE0B_R2_SHEAR_UNSTABLE`; peak primary: `0.03381663321323485` m.
- Table verdict: `not run`.
- Material frozen: `False`.
- Pair run: no; training run: no.

Failure cause: `engineering_or_stability_failure`.

Next permitted task: stop and address only the reported calibration gate; do not run Pair.
