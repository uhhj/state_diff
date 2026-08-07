Verdict: PHASE0D_SENSOR_NOT_OBSERVABLE

Repository:
- Main start: c1dd6493c1289766be04d613c27efae47f550055
- Main end: 936ac21a6c0f133ed207e85587aac0c8535c2386
- Branch: Experiment3
- Clean before result: True
- Remote tip: 936ac21a6c0f133ed207e85587aac0c8535c2386
- Submodule start: 7704cc7c5a413971deacf73544d773a51891b5c9
- Submodule end: 567f64eca2af4956658d141ca625ebed6a24780e

Repair:
- Canonical hold reassert after restore: True

Frozen:
- Probe amplitude: 1.0 mm
- Jam clearance: 0.25 mm
- Post-probe settle: 240 steps / 1.0 s
- State: 51D
- Sensor: 9D
- Sensor representation: gripper_wrench_plus_surface_tactile_force
- Sensor trace field: formal_sensor

Probe engagement:
- FREE probe latch contact samples: 0
- JAM probe latch contact samples: 72
- JAM probe latch contact fraction: 1.0
- JAM probe latch peak force: 0.4395324444133227 N

Five gates:
- Initial observable equivalence: True
- Post-probe observable equivalence: True
- Sensor observability: False
- Same-action future divergence: False
- Control relevance: None

Key values:
- Initial 51D RMSE: 0.0 m
- Post-probe 51D RMSE: 6.637808892143848e-05 m
- Post-probe keypoint RMSE: 6.0135285441198936e-05 m
- Post-probe EE RMSE: 0.00013054815521158905 m
- Immediate-return 51D RMSE: 5.927373587294412e-06 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 5.927373587294412e-06, 'free_jam': {'ee_rmse_m': 6.161051064774432e-07, 'keypoint_rmse_m': 6.107855057397475e-06, 'rmse_51d_m': 5.927373587294412e-06}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 2.2326231939614253e-05, 'free_jam': {'ee_rmse_m': 3.043177172455996e-06, 'keypoint_rmse_m': 2.300077420879257e-05, 'rmse_51d_m': 2.2326231939614253e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 2.0524725167348972e-05, 'free_jam': {'ee_rmse_m': 2.3776714416339e-06, 'keypoint_rmse_m': 2.114805031325905e-05, 'rmse_51d_m': 2.0524725167348972e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 1.6900740467580756e-05, 'free_jam': {'ee_rmse_m': 2.4218645086937556e-06, 'keypoint_rmse_m': 1.741035980080251e-05, 'rmse_51d_m': 1.6900740467580756e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 0.0, 'free_jam': {'ee_rmse_m': 0.00013054815521158905, 'keypoint_rmse_m': 6.0135285441198936e-05, 'rmse_51d_m': 6.637808892143848e-05}, 'free_repeat': {'ee_rmse_m': 0.0001289755277041796, 'keypoint_rmse_m': 6.087576798347961e-05, 'rmse_51d_m': 6.683096886387268e-05}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 6.683096886387268e-05 m
- Final branch excess over repeat: 0.0 m
- Final repeat/FREE-JAM fraction: 1.0068227324677907
- Final branch-excess/FREE-JAM fraction: 0.0
- Recovery fraction: -10.198566775632779
- JAM post-probe latch contact samples: 240
- JAM post-probe latch contact fraction: 1.0
- JAM post-probe latch peak force: 0.3052588358041752 N
- Sensor onset: None / None
- Sensor trigger: None
- Grasp-only peak fused gap: 0.05934637578044022
- Tactile-only peak fused gap: 0.3069940509174422
- Tactile raw force-gap peak: 0.02677513183562985 N
- Tactile probe contact samples: {'free': 72, 'jam_right': 72}
- Tactile contact-count peak gap: 0.0
- Combined formal peak fused gap: 0.3069940509174422
- Future visible RMSE peak: 0.0026621585774327717 m
- Repeat floor: 0.0003163165519357304 m
- Future/repeat ratio: 8.416121638723707
- FREE/JAM extraction progress: {'free': 0.009474010769790542, 'free_repeat': 0.00961386967304756, 'jam_right': 0.015138446928295746}
- Repeat first phase above equivalence threshold: None
- Repeat phase diagnostics: [{'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'no_action', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_forward', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_hold', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_return', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.683096886387268e-05, 'peak_51d_rmse_m': 6.70602574775536e-05, 'peak_ee_rmse_m': 0.0001295240903839642, 'peak_keypoint_rmse_m': 6.10705730438436e-05, 'phase': 'post_probe', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.866758660336229e-05, 'peak_51d_rmse_m': 0.00030790956243841337, 'peak_ee_rmse_m': 0.00014509853261499795, 'peak_keypoint_rmse_m': 0.0003163165519357304, 'phase': 'test_pull', 'start_51d_rmse_m': 6.638570377971164e-05}, {'end_51d_rmse_m': 7.844724115941172e-05, 'peak_51d_rmse_m': 0.0001044716867527471, 'peak_ee_rmse_m': 7.390582941985888e-05, 'peak_keypoint_rmse_m': 0.00010688959361410294, 'phase': 'post_test', 'start_51d_rmse_m': 6.769959301575476e-05}]
- Control verdict: not run

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Hidden latch/contact used as formal sensor: No

Tests:
- Passed: 20
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: Gripper-surface contact exists but the minimal 3D tactile resultant is still not observable. Freeze task physics and prepare a small spatial gripper-tactile representation.
