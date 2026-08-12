# PB3-R1 Wrapping Replay Alignment Root-Cause Audit

Verdict: `PB3R1_ACQUISITION_HISTORY_NOT_CONFIRMED_REPLAY_FLOOR_CALIBRATION_REQUIRED`

## Scope

- Formal PB3 remains blocked: Yes
- Future suffix executed: No
- Gate 4 executed: No
- Frozen PB3 threshold changed: No
- Frozen PB3 shortlist changed: No

## Target

- Rollout: 46
- Batch/env/seed: 1 / 14 / 124
- Time index: 13

## Fresh-process modes

### m0_sequential_history

- Repeats: 3
- t0 alignment valid for all repeats: True
- Original 50 um alignment passes: 3 / 3
- Median coordinate RMSE: 7.886020456e-06 m
- Median rope max-abs: 4.667043686e-05 m
- Median max vertex L2: 5.059761134e-05 m

### m1_isolated_collector

- Repeats: 3
- t0 alignment valid for all repeats: True
- Original 50 um alignment passes: 3 / 3
- Median coordinate RMSE: 8.120306056e-06 m
- Median rope max-abs: 4.723668098e-05 m
- Median max vertex L2: 5.169861932e-05 m

### m2_build_state_restore

- Repeats: 1
- t0 alignment valid for all repeats: True
- Original 50 um alignment passes: 1 / 1
- Median coordinate RMSE: 8.120306056e-06 m
- Median rope max-abs: 4.723668098e-05 m
- Median max vertex L2: 5.169861932e-05 m

## Decision

- M0/M1 coordinate-RMSE ratio: 0.971148
- M0/M1 rope-max-abs ratio: 0.988013
- Material reduction criterion: <= 0.500

## Next action

Proceed to PB3-R2 independent replay reconstruction-floor calibration using official-valid non-shortlist PB2-C states. Do not change the frozen PB3 50 um alignment threshold before that calibration.
