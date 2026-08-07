Verdict: PHASE0C_R1_SENSOR_NOT_SEPARABLE
Scientific status: sensor_not_separable

Repository:
- Starting SHA: fc83c62c43f140fd1c293cf5d816a625992e1d0f
- Ending SHA: b3d79e32470df9992bb5ede108b6564ef1422206
- Branch: Experiment3
- Clean before result: True
- Remote tip: b3d79e32470df9992bb5ede108b6564ef1422206
- Remote tip matches: True
- Submodule gitlink: 282b93535b125d1a4487df85ad24aa41551957af

Frozen mechanics:
- Material: kv_ccda_v41
- Outer dt: 0.004166666666666667 s
- Manual microsteps: 8
- Micro dt: 0.0005208333333333333 s
- Bullet numSubSteps: 1

State:
- State dim: 74
- Peak joint-state RMSE: 0.005187078667648407 m
- Peak deformable RMSE: 0.00525862796021349 m
- Peak EE-XY RMSE: 3.8047644435082514e-06 m
- Repeat floor: 0.0 m
- Threshold: 0.0006142376578788483 m
- State onset: 235 outer steps / 979.1666666666667 ms

Sensor:
- Sensor dim: 45
- Recording rate: 240 Hz
- Causal window: 24 outer samples per policy interval
- Formal pre-step sample: No
- Sensor onset: None outer steps / None ms
- Sensor lead: None outer steps / None ms
- Trigger: None
- Peak fused gap: 0.2867526575126078

Engineering:
- Gate: {'actions_equal': True, 'ee_targets_equal': True, 'finite': True, 'initial_state': True, 'joint_commands_equal': True, 'no_action_stable': True, 'oracle_intervention': True, 'phases_equal': True, 'spring_force_cap': True}
- Oracle intervention: True

Control relevance:
- Executed only if observability complete: Yes
- Verdict: not run
- Progress matrix: not run
- Cross-condition regret low: not run
- Cross-condition regret high: not run

Tests:
- Passed: 38
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- StateDiff B0: No
- StateDiff-FT B1: No
- CFPM B2: No
- B3: No

Failure cause: PHASE0C_R1_SENSOR_NOT_SEPARABLE
Next permitted task: Stop at the reported Phase 0C-R1 gate; do not train models.
