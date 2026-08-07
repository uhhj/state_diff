Verdict: PHASE0D_SENSOR_NOT_OBSERVABLE

Repository:
- Main start: 1f41cd4d93a1c6de0ed8378bbee41e0a0c7e0204
- Main end: 036f0b542b0313399887394738bcfbca561d5dd2
- Branch: Experiment3
- Clean before result: True
- Remote tip: 036f0b542b0313399887394738bcfbca561d5dd2
- Submodule start: 7704cc7c5a413971deacf73544d773a51891b5c9
- Submodule end: 7704cc7c5a413971deacf73544d773a51891b5c9

Repair:
- Canonical hold reassert after restore: True

Frozen:
- Probe amplitude: 1.0 mm
- Jam clearance: 0.5 mm
- Post-probe settle: 240 steps / 1.0 s
- State: 51D
- Sensor: 6D

Five gates:
- Initial observable equivalence: True
- Post-probe observable equivalence: True
- Sensor observability: False
- Same-action future divergence: False
- Control relevance: None

Key values:
- Initial 51D RMSE: 0.0 m
- Post-probe 51D RMSE: 0.00014188584723631365 m
- Post-probe keypoint RMSE: 0.00012501552188295682 m
- Post-probe EE RMSE: 0.00030360336064467753 m
- Immediate-return 51D RMSE: 0.0 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 7.357269418751122e-05, 'keypoint_rmse_m': 3.426413630148166e-05, 'rmse_51d_m': 3.7727691432306965e-05}, 'free_repeat': {'ee_rmse_m': 7.357269418751122e-05, 'keypoint_rmse_m': 3.426413630148166e-05, 'rmse_51d_m': 3.7727691432306965e-05}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 0.00030360336064467753, 'keypoint_rmse_m': 0.00012501552188295682, 'rmse_51d_m': 0.00014188584723631365}, 'free_repeat': {'ee_rmse_m': 0.00030360336064467753, 'keypoint_rmse_m': 0.00012501552188295682, 'rmse_51d_m': 0.00014188584723631365}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 0.00014188584723631365 m
- Final branch excess over repeat: 0.0 m
- Final repeat/FREE-JAM fraction: 1.0
- Final branch-excess/FREE-JAM fraction: 0.0
- Recovery fraction: None
- JAM post-probe latch contact samples: 0
- JAM post-probe latch contact fraction: 0.0
- JAM post-probe latch peak force: 0.0 N
- Sensor onset: None / None
- Sensor trigger: None
- Peak fused sensor gap: 0.0
- Future visible RMSE peak: 0.0025298769691095313 m
- Repeat floor: 0.00029334395346416613 m
- Future/repeat ratio: 8.624268334948217
- FREE/JAM extraction progress: {'free': 0.009215277998608329, 'free_repeat': 0.009210609552002591, 'jam_right': 0.014601403435143767}
- Repeat first phase above equivalence threshold: None
- Repeat phase diagnostics: [{'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'no_action', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_forward', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_hold', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_return', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.00014188584723631365, 'peak_51d_rmse_m': 0.0001433725659557809, 'peak_ee_rmse_m': 0.000306042359901989, 'peak_keypoint_rmse_m': 0.00012643794162411307, 'phase': 'post_probe', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.963062116747649e-05, 'peak_51d_rmse_m': 0.0002879641526718603, 'peak_ee_rmse_m': 0.00032212128807745135, 'peak_keypoint_rmse_m': 0.00029334395346416613, 'phase': 'test_pull', 'start_51d_rmse_m': 0.00014339417697791665}, {'end_51d_rmse_m': 3.256898518094861e-05, 'peak_51d_rmse_m': 6.84586241038932e-05, 'peak_ee_rmse_m': 0.0001369510275836894, 'peak_keypoint_rmse_m': 6.170308475126064e-05, 'phase': 'post_test', 'start_51d_rmse_m': 6.84586241038932e-05}]
- Control verdict: not run

Tests:
- Passed: 16
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: Keep the 1.0 mm probe and 240-step settle fixed; reduce initial jam clearance from 0.50 mm to 0.25 mm and rerun Gate 2/3.
