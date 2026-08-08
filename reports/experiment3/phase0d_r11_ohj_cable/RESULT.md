Verdict: PHASE0D_FUTURE_BRANCH_NOT_ESTABLISHED

Repository:
- Main start: 4f2859cb5ccfbe02010c6d3838d62e607b2eba23
- Main end: b8a04480c2ea57430f62aa89d93d0b3578c7d3bb
- Branch: Experiment3
- Clean before result: True
- Remote tip: b8a04480c2ea57430f62aa89d93d0b3578c7d3bb
- Submodule start: 0ab2ea0ff70e4dc7f7d7bef1c45600bb6b91ff54
- Submodule end: 0ab2ea0ff70e4dc7f7d7bef1c45600bb6b91ff54

Repair:
- Canonical hold reassert after restore: True

Frozen:
- Probe amplitude: 1.0 mm
- Probe delta XYZ: [0.0, -0.001, 0.0]
- Probe direction unit: [0.0, -1.0, 0.0]
- Probe forward/hold/return: 24/24/24
- Probe speed: 0.01 m/s
- Jam clearance: 0.25 mm
- Post-probe settle: 240 steps / 1.0 s
- State: 51D
- Sensor: 18D
- Sensor representation: gripper_wrench_plus_4patch_surface_tactile
- Sensor trace field: formal_sensor_spatial

Probe engagement:
- FREE probe latch contact samples: 0
- JAM probe latch contact samples: 72
- JAM probe latch contact fraction: 1.0
- JAM probe latch peak force: 0.4395324238951634 N

Five gates:
- Initial observable equivalence: True
- Post-probe observable equivalence: True
- Sensor observability: True
- Same-action future divergence: False
- Control relevance: None

Key values:
- Initial 51D RMSE: 0.0 m
- Post-probe 51D RMSE: 8.982128272931303e-05 m
- Post-probe keypoint RMSE: 8.241566323887439e-05 m
- Post-probe EE RMSE: 0.000168748935924673 m
- Immediate-return 51D RMSE: 1.1600444674511073e-05 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 1.1600444674511073e-05, 'free_jam': {'ee_rmse_m': 3.371762236857243e-06, 'keypoint_rmse_m': 1.1927716154001288e-05, 'rmse_51d_m': 1.1600444674511073e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 1.3654247051184611e-05, 'free_jam': {'ee_rmse_m': 2.666564396030935e-06, 'keypoint_rmse_m': 1.4058679030702753e-05, 'rmse_51d_m': 1.3654247051184611e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 1.4117086018806314e-05, 'free_jam': {'ee_rmse_m': 2.5035943700927846e-06, 'keypoint_rmse_m': 1.4538092239635596e-05, 'rmse_51d_m': 1.4117086018806314e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 2.22202142308758e-05, 'free_jam': {'ee_rmse_m': 2.0074419848417787e-06, 'keypoint_rmse_m': 2.2898573679106843e-05, 'rmse_51d_m': 2.22202142308758e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 9.403724867943614e-06, 'free_jam': {'ee_rmse_m': 0.000168748935924673, 'keypoint_rmse_m': 8.241566323887439e-05, 'rmse_51d_m': 8.982128272931303e-05}, 'free_repeat': {'ee_rmse_m': 0.00017128502826042652, 'keypoint_rmse_m': 7.097541850193573e-05, 'rmse_51d_m': 8.041755786136942e-05}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 8.041755786136942e-05 m
- Final branch excess over repeat: 9.403724867943614e-06 m
- Final repeat/FREE-JAM fraction: 0.8953062728319875
- Final branch-excess/FREE-JAM fraction: 0.10469372716801253
- Recovery fraction: -6.742917211326535
- JAM post-probe latch contact samples: 240
- JAM post-probe latch contact fraction: 1.0
- JAM post-probe latch peak force: 0.23961607318803074 N
- Sensor onset: 80 / probe_hold
- Sensor trigger: {'channel': 'patch2_Fy', 'channel_index': 13, 'normalized_gap': 1.4060754537167597}
- Grasp-only peak fused gap: 0.2398164334718064
- Tactile-only peak fused gap: 2.6147160295345095
- Aggregate 3D tactile peak: 2.034814634315811
- Spatial tactile patch metrics: [{'contact_count_peak_gap': 0.0, 'patch': 0, 'peak_fused_gap': 0.16969282855490367, 'probe_contact_samples': {'free': 72, 'jam_right': 72}, 'raw_force_gap_peak_n': 0.03808596807143397}, {'contact_count_peak_gap': 0.0, 'patch': 1, 'peak_fused_gap': 0.0, 'probe_contact_samples': {'free': 0, 'jam_right': 0}, 'raw_force_gap_peak_n': 0.0}, {'contact_count_peak_gap': 0.0, 'patch': 2, 'peak_fused_gap': 2.6147160295345095, 'probe_contact_samples': {'free': 43, 'jam_right': 43}, 'raw_force_gap_peak_n': 0.1340027162137773}, {'contact_count_peak_gap': 0.0, 'patch': 3, 'peak_fused_gap': 2.0543956338770557, 'probe_contact_samples': {'free': 72, 'jam_right': 72}, 'raw_force_gap_peak_n': 0.13043592034115153}]
- Tactile raw force-gap peak: 0.1340027162137773 N
- Tactile probe contact samples: {'free': 72, 'jam_right': 72}
- Tactile contact-count peak gap: 0.0
- Combined formal peak fused gap: 2.6147160295345095
- Future visible RMSE peak: 0.00403363901868536 m
- Repeat floor: 0.0004646102384433032 m
- Future/repeat ratio: 8.681769545587809
- FREE/JAM extraction progress: {'free': 0.03322656906399396, 'free_repeat': 0.03281747155592113, 'jam_right': 0.0379283250337234}
- Repeat first phase above equivalence threshold: None
- Repeat phase diagnostics: [{'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'no_action', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_forward', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_hold', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_return', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 8.041755786136942e-05, 'peak_51d_rmse_m': 8.113769519627918e-05, 'peak_ee_rmse_m': 0.00017243476014722596, 'peak_keypoint_rmse_m': 7.166885149027517e-05, 'phase': 'post_probe', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 8.051824890159804e-05, 'peak_51d_rmse_m': 0.00033138578000013524, 'peak_ee_rmse_m': 0.00019053187456078368, 'peak_keypoint_rmse_m': 0.00034011900097921205, 'phase': 'test_pull', 'start_51d_rmse_m': 8.230614515579854e-05}, {'end_51d_rmse_m': 0.00026507871993898064, 'peak_51d_rmse_m': 0.00045075213083623086, 'peak_ee_rmse_m': 0.00015136485965277257, 'peak_keypoint_rmse_m': 0.0004646102384433032, 'phase': 'post_test', 'start_51d_rmse_m': 8.068170865226452e-05}]
- Control verdict: not run

Spatial repeat-corrected load-path audit:
- Method: spatial_repeat_corrected_pair_level_load_profile
- Interpretation limit: pair-level oracle mechanism diagnostic; not a mutual-information or held-out prediction claim
- Actual JAM probe contact bead indices: [15]
- Contact-adjacent constraint indices: [14, 15]
- Contact-region branch-specific constraint indices: [14, 15]
- Contact-region peak segment/excess: 14 / 0.3009355396283823 N
- Global peak segment/excess: 14 / 0.3009355396283823 N
- Global peak is contact-adjacent: True
- All branch-specific segment indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
- Furthest branch-specific segment toward gripper: 30
- Gripper-proximal constraint index: 30
- Proximal branch-specific: True
- Proximal repeat-corrected excess: 0.24669686990680706 N
- Proximal mechanically large: True
- Proximal retention ratio: 0.8197664862430233
- Route hint: repair_physical_grasp_sensing_coupling

Segment rows:
- Segment 0: no_action branch/repeat sustained 0.02314885596427919/0.0 N; no_action excess 0.02314885596427919 N; probe branch/repeat sustained 0.03633838138512644/0.0 N; probe excess 0.03633838138512644 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.01318952542084725 N
- Segment 1: no_action branch/repeat sustained 0.019348607690578477/0.0 N; no_action excess 0.019348607690578477 N; probe branch/repeat sustained 0.033014697619784934/0.0 N; probe excess 0.033014697619784934 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013666089929206458 N
- Segment 2: no_action branch/repeat sustained 0.019647284080077845/0.0 N; no_action excess 0.019647284080077845 N; probe branch/repeat sustained 0.03284644331859957/0.0 N; probe excess 0.03284644331859957 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013199159238521729 N
- Segment 3: no_action branch/repeat sustained 0.019845232107699985/0.0 N; no_action excess 0.019845232107699985 N; probe branch/repeat sustained 0.035837610424882226/0.0 N; probe excess 0.035837610424882226 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.01599237831718224 N
- Segment 4: no_action branch/repeat sustained 0.01930369678807796/0.0 N; no_action excess 0.01930369678807796 N; probe branch/repeat sustained 0.028469454258788284/0.0 N; probe excess 0.028469454258788284 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.009165757470710325 N
- Segment 5: no_action branch/repeat sustained 0.004991874019521593/0.0 N; no_action excess 0.004991874019521593 N; probe branch/repeat sustained 0.012821687231108419/0.0 N; probe excess 0.012821687231108419 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.007829813211586826 N
- Segment 6: no_action branch/repeat sustained 0.010058159405937626/0.0 N; no_action excess 0.010058159405937626 N; probe branch/repeat sustained 0.015056930936956802/0.0 N; probe excess 0.015056930936956802 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.004998771531019176 N
- Segment 7: no_action branch/repeat sustained 0.0056570064689243195/0.0 N; no_action excess 0.0056570064689243195 N; probe branch/repeat sustained 0.006143449385000203/0.0 N; probe excess 0.006143449385000203 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.0004864429160758837 N
- Segment 8: no_action branch/repeat sustained 0.004050002145592819/0.0 N; no_action excess 0.004050002145592819 N; probe branch/repeat sustained 0.00822680912763545/0.0 N; probe excess 0.00822680912763545 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.004176806982042632 N
- Segment 9: no_action branch/repeat sustained 0.01156485185808385/0.0 N; no_action excess 0.01156485185808385 N; probe branch/repeat sustained 0.03796344617080924/0.0 N; probe excess 0.03796344617080924 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.02639859431272539 N
- Segment 10: no_action branch/repeat sustained 0.05714686148762226/0.0 N; no_action excess 0.05714686148762226 N; probe branch/repeat sustained 0.06533306524152573/0.0 N; probe excess 0.06533306524152573 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.008186203753903463 N
- Segment 11: no_action branch/repeat sustained 0.0542583065075688/0.0 N; no_action excess 0.0542583065075688 N; probe branch/repeat sustained 0.04710109946508849/0.0 N; probe excess 0.04710109946508849 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess -0.00715720704248031 N
- Segment 12: no_action branch/repeat sustained 0.031862625670064565/0.0 N; no_action excess 0.031862625670064565 N; probe branch/repeat sustained 0.09724778946050726/0.0 N; probe excess 0.09724778946050726 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0653851637904427 N
- Segment 13: no_action branch/repeat sustained 0.17679803058732924/0.0 N; no_action excess 0.17679803058732924 N; probe branch/repeat sustained 0.19021553297982083/0.0 N; probe excess 0.19021553297982083 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.013417502392491593 N
- Segment 14: no_action branch/repeat sustained 0.3521055969148146/0.0 N; no_action excess 0.3521055969148146 N; probe branch/repeat sustained 0.3009355396283823/0.0 N; probe excess 0.3009355396283823 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.05117005728643226 N
- Segment 15: no_action branch/repeat sustained 0.15355827813607234/0.0 N; no_action excess 0.15355827813607234 N; probe branch/repeat sustained 0.08022388330255294/0.0 N; probe excess 0.08022388330255294 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.0733343948335194 N
- Segment 16: no_action branch/repeat sustained 0.1709887824558825/0.0 N; no_action excess 0.1709887824558825 N; probe branch/repeat sustained 0.05933876795870423/0.0 N; probe excess 0.05933876795870423 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.11165001449717828 N
- Segment 17: no_action branch/repeat sustained 0.04100634487510897/0.0 N; no_action excess 0.04100634487510897 N; probe branch/repeat sustained 0.05436809365993183/0.0 N; probe excess 0.05436809365993183 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.01336174878482286 N
- Segment 18: no_action branch/repeat sustained 0.03550675910805615/0.0 N; no_action excess 0.03550675910805615 N; probe branch/repeat sustained 0.05344874654194105/0.0 N; probe excess 0.05344874654194105 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0179419874338849 N
- Segment 19: no_action branch/repeat sustained 0.043770846915164224/0.0 N; no_action excess 0.043770846915164224 N; probe branch/repeat sustained 0.08029204109587434/0.0 N; probe excess 0.08029204109587434 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03652119418071012 N
- Segment 20: no_action branch/repeat sustained 0.04119599117567114/0.0 N; no_action excess 0.04119599117567114 N; probe branch/repeat sustained 0.07416056198048/0.0 N; probe excess 0.07416056198048 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.032964570804808864 N
- Segment 21: no_action branch/repeat sustained 0.03012149775032703/0.0 N; no_action excess 0.03012149775032703 N; probe branch/repeat sustained 0.06693804260381889/0.0 N; probe excess 0.06693804260381889 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03681654485349185 N
- Segment 22: no_action branch/repeat sustained 0.043073937202897206/0.0 N; no_action excess 0.043073937202897206 N; probe branch/repeat sustained 0.07264878247190851/0.0 N; probe excess 0.07264878247190851 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0295748452690113 N
- Segment 23: no_action branch/repeat sustained 0.12381144087242116/0.0 N; no_action excess 0.12381144087242116 N; probe branch/repeat sustained 0.13430537505467327/0.0 N; probe excess 0.13430537505467327 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.010493934182252113 N
- Segment 24: no_action branch/repeat sustained 0.08692536366684958/0.0 N; no_action excess 0.08692536366684958 N; probe branch/repeat sustained 0.07062373293107051/0.0 N; probe excess 0.07062373293107051 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.016301630735779074 N
- Segment 25: no_action branch/repeat sustained 0.04692864482245199/0.0 N; no_action excess 0.04692864482245199 N; probe branch/repeat sustained 0.13676511917689327/0.0 N; probe excess 0.13676511917689327 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.08983647435444128 N
- Segment 26: no_action branch/repeat sustained 0.012161132769669555/0.0 N; no_action excess 0.012161132769669555 N; probe branch/repeat sustained 0.13717989454668492/0.0 N; probe excess 0.13717989454668492 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.12501876177701537 N
- Segment 27: no_action branch/repeat sustained 0.011077755141593763/0.0 N; no_action excess 0.011077755141593763 N; probe branch/repeat sustained 0.21501545563285906/0.0 N; probe excess 0.21501545563285906 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.2039377004912653 N
- Segment 28: no_action branch/repeat sustained 0.018594867653556914/0.0 N; no_action excess 0.018594867653556914 N; probe branch/repeat sustained 0.22315830497604797/0.0 N; probe excess 0.22315830497604797 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.20456343732249105 N
- Segment 29: no_action branch/repeat sustained 0.026247582529669945/0.0 N; no_action excess 0.026247582529669945 N; probe branch/repeat sustained 0.24038624525555544/0.0 N; probe excess 0.24038624525555544 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.2141386627258855 N
- Segment 30: no_action branch/repeat sustained 0.011631645034722323/0.0 N; no_action excess 0.011631645034722323 N; probe branch/repeat sustained 0.24669686990680706/0.0 N; probe excess 0.24669686990680706 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.23506522487208473 N

Fixed-speed probe amplification comparison: not configured

Orthogonal contact-loading probe comparison:
- Report only: True
- Stop after this trial: True
- Direction dot baseline: 0.0
- Baseline: {'contact_reference_peak_excess_n': 0.3009539902806676, 'formal_sensor_peak_fused_gap': 0.3338189115908537, 'future_peak_visible_rmse_m': 0.0026621585774327717, 'phase_name': 'phase0d-r7-spatial-repeat-corrected-load-path-audit', 'post_probe_jam_latch_contact_fraction': 1.0, 'post_probe_rmse_m': 6.637808892143848e-05, 'probe_amplitude_m': 0.001, 'probe_delta_xyz_m': [0.001, 0.0, 0.0], 'probe_forward_steps': 24, 'probe_return_steps': 24, 'probe_speed_m_s': 0.01, 'proximal_excess_n': 0.022166104111013166, 'proximal_retention_ratio': 0.07365280018497582, 'repeat_peak_visible_rmse_m': 0.0003163165519357304, 'result_sha': 'aa5600408d835f344b580f155e61780c737930d1'}
- Current: {'contact_reference_peak_excess_n': 0.3009355396283823, 'formal_sensor_peak_fused_gap': 2.6147160295345095, 'future_peak_visible_rmse_m': 0.00403363901868536, 'post_probe_jam_latch_contact_fraction': 1.0, 'post_probe_rmse_m': 8.982128272931303e-05, 'probe_amplitude_m': 0.001, 'probe_delta_xyz_m': [0.0, -0.001, 0.0], 'probe_direction_unit': [0.0, -1.0, 0.0], 'probe_forward_steps': 24, 'probe_return_steps': 24, 'probe_speed_m_s': 0.01, 'proximal_excess_n': 0.24669686990680706, 'proximal_retention_ratio': 0.8197664862430233}
- Gain ratios: {'contact_reference_peak_excess': 0.9999386927806869, 'formal_sensor_peak': 7.832737866988331, 'future_peak': 1.5151760878854794, 'post_probe_jam_latch_contact_fraction': 1.0, 'post_probe_rmse': 1.353176691115959, 'probe_amplitude': 1.0, 'probe_speed': 1.0, 'proximal_excess': 11.129464549624506, 'proximal_retention': 11.130146907981981}

Final future-horizon sufficiency audit:
- Diagnostic only: True
- Gates unchanged: True
- Stop after this trial: True
- Future samples/span: 1296 / 5.395833333333333 s
- Post-test observation: 1200 steps / 5.0 s
- Future threshold: 0.005 m
- Threshold reached: False
- First threshold crossing: None
- Endpoint / peak: 0.004032283777229083 / 0.00403363901868536 m
- Peak index/time: 1287 / 5.3625 s
- Endpoint is global max: False
- Repeat peak: 0.0004646102384433032 m
- Future/repeat: 8.681769545587809
- Remaining margin: 0.0009663609813146401 m
- Tail windows: [{'delta_m': 4.50514900341651e-05, 'least_squares_slope_m_s': 9.943415419231034e-05, 'positive_step_differences': 90, 'samples': 120, 'total_step_differences': 119, 'window_seconds': 0.5}, {'delta_m': 7.620507460631091e-05, 'least_squares_slope_m_s': 8.254108722271933e-05, 'positive_step_differences': 181, 'samples': 240, 'total_step_differences': 239, 'window_seconds': 1.0}, {'delta_m': 0.00013846486791877985, 'least_squares_slope_m_s': 6.452633312795914e-05, 'positive_step_differences': 349, 'samples': 480, 'total_step_differences': 479, 'window_seconds': 2.0}]
- All tail slopes positive: True
- Tail still rising at horizon end: False
- Interpretation: threshold_not_reached_without_strong_endpoint_right_censoring

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Internal cable constraint force used for Gate 3: No
- Internal cable constraint force used only as oracle mechanism diagnostic: Yes
- Latch-contact bead mask used as formal sensor: No
- Hidden latch/contact used as formal sensor: No

Tests:
- Passed: 34
- Failed: 0
- pip check: No_broken_requirements_found

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: The final fixed 5 s post-action observation does not reach the existing Gate 4 threshold and no longer shows the strong endpoint right-censoring signature used in R10. Keep the R9 probe and 18D sensor frozen. Do not lower Gate 4 and do not extend the horizon again automatically; prepare a single-purpose future-consequence task/topology repair.
