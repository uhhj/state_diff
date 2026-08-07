Verdict: PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED
Scientific status: hidden_dynamics_not_established

Repository:
- Starting SHA: 536d4fa2e4d5565eb5b46283caa47b661485f6e6
- Ending SHA: 3b3d592818892fb55e2ae31170a4bf5892bdfb8d
- Branch: Experiment3
- Clean: True
- Remote tip matches: True
- Submodule gitlink: 282b93535b125d1a4487df85ad24aa41551957af

Cleanup:
- Removed old Phase0B script trees: 6
- Removed old material profile families: 5
- Removed coupon/angle runtime modules: 7
- Removed obsolete tests: 21
- Archived old code: No
- Obsolete runtime references remaining: 0

Active mechanics:
- Material: kv_ccda_v41
- Outer dt: 1/240 s
- Manual microsteps: 8
- Micro dt: 1/1920 s
- Bullet numSubSteps: 1
- Force recomputed each microstep: Yes

State API:
- State Diff state dim: 74
- State Diff state channels: top-layer 24 XYZ + EE XY
- Contact sensor dim: 45
- Contact sensor channels: motor torque 6 + joint reaction wrench 36 + tracking error 3
- Extended proprio dim: 24
- Oracle leakage: No

Tests:
- Passed: 32
- Failed: 0
- pip check: known multiprocess 0.70.14 / dill 0.3.5.1 conflict only

Strict pair:
- Initial state max difference: 0.0
- Joint command arrays equal: True
- EE target arrays equal: True
- Action arrays equal: True
- No-action drift: {'right_local_high': 0.00012343000925384265, 'uniform_low': 0.00012343000925384265}
- Repeat/noise floor: 0.0
- Oracle intervention: True
- Formal sensor onset: 14
- State divergence onset: 10
- Sensor lead: -4
- Peak future state RMSE: 0.005183604637318026
- Pair verdict: PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED

Control relevance:
- Candidates: not run
- Best low: not run
- Best high: not run
- Progress matrix: not run
- Success matrix: not run
- Cross regret low: not run
- Cross regret high: not run
- Verdict: not run

Deformation diagnostics:
- Optional only: Yes
- Used as hard gate: No

Training:
- StateDiff B0: No
- StateDiff-FT B1: No
- CFPM B2: No
- B3: No

Failure cause: PHASE0C_HIDDEN_DYNAMICS_NOT_ESTABLISHED
Next permitted task: Stop at the reported Phase 0C gate; do not train models.
