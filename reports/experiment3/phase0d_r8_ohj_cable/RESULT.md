Verdict: PHASE0D_SENSOR_NOT_OBSERVABLE

Repository:
- Main start: aa5600408d835f344b580f155e61780c737930d1
- Main end: 114b36f8b48059fee1b837a72cda3e4207568fb0
- Branch: Experiment3
- Clean before result: True
- Remote tip: 114b36f8b48059fee1b837a72cda3e4207568fb0
- Submodule start: 0ab2ea0ff70e4dc7f7d7bef1c45600bb6b91ff54
- Submodule end: 0ab2ea0ff70e4dc7f7d7bef1c45600bb6b91ff54

Repair:
- Canonical hold reassert after restore: True

Frozen:
- Probe amplitude: 2.0 mm
- Probe forward/hold/return: 48/24/48
- Jam clearance: 0.25 mm
- Post-probe settle: 240 steps / 1.0 s
- State: 51D
- Sensor: 18D
- Sensor representation: gripper_wrench_plus_4patch_surface_tactile
- Sensor trace field: formal_sensor_spatial

Probe engagement:
- FREE probe latch contact samples: 0
- JAM probe latch contact samples: 120
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
- Post-probe 51D RMSE: 7.134265109489577e-05 m
- Post-probe keypoint RMSE: 6.792581585745833e-05 m
- Post-probe EE RMSE: 0.00011270977043182699 m
- Immediate-return 51D RMSE: 7.464619965755167e-06 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 7.464619965755167e-06, 'free_jam': {'ee_rmse_m': 2.066089304272102e-06, 'keypoint_rmse_m': 7.67699748416631e-06, 'rmse_51d_m': 7.464619965755167e-06}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 2.01859928819131e-05, 'free_jam': {'ee_rmse_m': 1.985713965005928e-06, 'keypoint_rmse_m': 2.0801322353903624e-05, 'rmse_51d_m': 2.01859928819131e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 2.2638779583445625e-05, 'free_jam': {'ee_rmse_m': 4.0732233365091376e-07, 'keypoint_rmse_m': 2.333529768128866e-05, 'rmse_51d_m': 2.2638779583445625e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 2.4805513874417662e-05, 'free_jam': {'ee_rmse_m': 1.1947141548941378e-06, 'keypoint_rmse_m': 2.5567193913593717e-05, 'rmse_51d_m': 2.4805513874417662e-05}, 'free_repeat': {'ee_rmse_m': 0.0, 'keypoint_rmse_m': 0.0, 'rmse_51d_m': 0.0}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 1.2210057276201468e-05, 'free_jam': {'ee_rmse_m': 0.00011270977043182699, 'keypoint_rmse_m': 6.792581585745833e-05, 'rmse_51d_m': 7.134265109489577e-05}, 'free_repeat': {'ee_rmse_m': 0.00011148304962257707, 'keypoint_rmse_m': 5.4207247973750953e-05, 'rmse_51d_m': 5.91325938186943e-05}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 5.91325938186943e-05 m
- Final branch excess over repeat: 1.2210057276201468e-05 m
- Final repeat/FREE-JAM fraction: 0.8288533284253711
- Final branch-excess/FREE-JAM fraction: 0.17114667157462896
- Recovery fraction: -8.557439149238498
- JAM post-probe latch contact samples: 240
- JAM post-probe latch contact fraction: 1.0
- JAM post-probe latch peak force: 0.3053324236281954 N
- Sensor onset: None / None
- Sensor trigger: None
- Grasp-only peak fused gap: 0.06673373706763241
- Tactile-only peak fused gap: 0.3243398252070224
- Aggregate 3D tactile peak: 0.3000769012145558
- Spatial tactile patch metrics: [{'contact_count_peak_gap': 0.0, 'patch': 0, 'peak_fused_gap': 0.03054599307305918, 'probe_contact_samples': {'free': 120, 'jam_right': 120}, 'raw_force_gap_peak_n': 0.006841495934303615}, {'contact_count_peak_gap': 0.0, 'patch': 1, 'peak_fused_gap': 0.0, 'probe_contact_samples': {'free': 0, 'jam_right': 0}, 'raw_force_gap_peak_n': 0.0}, {'contact_count_peak_gap': 0.0, 'patch': 2, 'peak_fused_gap': 0.0, 'probe_contact_samples': {'free': 0, 'jam_right': 0}, 'raw_force_gap_peak_n': 0.0}, {'contact_count_peak_gap': 0.0, 'patch': 3, 'peak_fused_gap': 0.3243398252070224, 'probe_contact_samples': {'free': 120, 'jam_right': 120}, 'raw_force_gap_peak_n': 0.02478854464202563}]
- Tactile raw force-gap peak: 0.02478854464202563 N
- Tactile probe contact samples: {'free': 120, 'jam_right': 120}
- Tactile contact-count peak gap: 0.0
- Combined formal peak fused gap: 0.3243398252070224
- Future visible RMSE peak: 0.002592416125579732 m
- Repeat floor: 0.000308874598157777 m
- Future/repeat ratio: 8.393102382137277
- FREE/JAM extraction progress: {'free': 0.009484305099483281, 'free_repeat': 0.009611124812695937, 'jam_right': 0.015000755606220884}
- Repeat first phase above equivalence threshold: None
- Repeat phase diagnostics: [{'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'no_action', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_forward', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_hold', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_return', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 5.91325938186943e-05, 'peak_51d_rmse_m': 5.9428553554285764e-05, 'peak_ee_rmse_m': 0.00011191827924871387, 'peak_keypoint_rmse_m': 5.4494322748128594e-05, 'phase': 'post_probe', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.237542067045315e-05, 'peak_51d_rmse_m': 0.00030055808704893125, 'peak_ee_rmse_m': 0.00013006280683242792, 'peak_keypoint_rmse_m': 0.000308874598157777, 'phase': 'test_pull', 'start_51d_rmse_m': 6.113036171189722e-05}, {'end_51d_rmse_m': 7.214267956755018e-05, 'peak_51d_rmse_m': 9.633275549501617e-05, 'peak_ee_rmse_m': 6.753168544118592e-05, 'peak_keypoint_rmse_m': 9.858583531227534e-05, 'phase': 'post_test', 'start_51d_rmse_m': 6.072303780096149e-05}]
- Control verdict: not run

Spatial repeat-corrected load-path audit:
- Method: spatial_repeat_corrected_pair_level_load_profile
- Interpretation limit: pair-level oracle mechanism diagnostic; not a mutual-information or held-out prediction claim
- Actual JAM probe contact bead indices: [15]
- Contact-adjacent constraint indices: [14, 15]
- Contact-region branch-specific constraint indices: [14, 15]
- Contact-region peak segment/excess: 14 / 0.3009539902806676 N
- Global peak segment/excess: 14 / 0.3009539902806676 N
- Global peak is contact-adjacent: True
- All branch-specific segment indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
- Furthest branch-specific segment toward gripper: 30
- Gripper-proximal constraint index: 30
- Proximal branch-specific: True
- Proximal repeat-corrected excess: 0.022166104111013166 N
- Proximal mechanically large: False
- Proximal retention ratio: 0.07365280018497582
- Route hint: amplify_probe_or_mechanical_signal

Segment rows:
- Segment 0: no_action branch/repeat sustained 0.02314885596427919/0.0 N; no_action excess 0.02314885596427919 N; probe branch/repeat sustained 0.036338379199330084/0.0 N; probe excess 0.036338379199330084 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013189523235050894 N
- Segment 1: no_action branch/repeat sustained 0.019348607690578477/0.0 N; no_action excess 0.019348607690578477 N; probe branch/repeat sustained 0.03301468866372558/0.0 N; probe excess 0.03301468866372558 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013666080973147105 N
- Segment 2: no_action branch/repeat sustained 0.019647284080077845/0.0 N; no_action excess 0.019647284080077845 N; probe branch/repeat sustained 0.03284643567508965/0.0 N; probe excess 0.03284643567508965 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013199151595011806 N
- Segment 3: no_action branch/repeat sustained 0.019845232107699985/0.0 N; no_action excess 0.019845232107699985 N; probe branch/repeat sustained 0.03583788171624701/0.0 N; probe excess 0.03583788171624701 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.015992649608547023 N
- Segment 4: no_action branch/repeat sustained 0.01930369678807796/0.0 N; no_action excess 0.01930369678807796 N; probe branch/repeat sustained 0.028469092378105944/0.0 N; probe excess 0.028469092378105944 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.009165395590027985 N
- Segment 5: no_action branch/repeat sustained 0.004991874019521593/0.0 N; no_action excess 0.004991874019521593 N; probe branch/repeat sustained 0.0150292645314121/0.0 N; probe excess 0.0150292645314121 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.010037390511890508 N
- Segment 6: no_action branch/repeat sustained 0.010058159405937626/0.0 N; no_action excess 0.010058159405937626 N; probe branch/repeat sustained 0.01799489980798191/0.0 N; probe excess 0.01799489980798191 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.007936740402044283 N
- Segment 7: no_action branch/repeat sustained 0.0056570064689243195/0.0 N; no_action excess 0.0056570064689243195 N; probe branch/repeat sustained 0.012254081377207904/0.0 N; probe excess 0.012254081377207904 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.006597074908283584 N
- Segment 8: no_action branch/repeat sustained 0.004050002145592819/0.0 N; no_action excess 0.004050002145592819 N; probe branch/repeat sustained 0.017583942046601915/0.0 N; probe excess 0.017583942046601915 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013533939901009095 N
- Segment 9: no_action branch/repeat sustained 0.01156485185808385/0.0 N; no_action excess 0.01156485185808385 N; probe branch/repeat sustained 0.04097777541664058/0.0 N; probe excess 0.04097777541664058 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.02941292355855673 N
- Segment 10: no_action branch/repeat sustained 0.05714686148762226/0.0 N; no_action excess 0.05714686148762226 N; probe branch/repeat sustained 0.06533258453056996/0.0 N; probe excess 0.06533258453056996 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0081857230429477 N
- Segment 11: no_action branch/repeat sustained 0.0542583065075688/0.0 N; no_action excess 0.0542583065075688 N; probe branch/repeat sustained 0.04913617830048427/0.0 N; probe excess 0.04913617830048427 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess -0.005122128207084527 N
- Segment 12: no_action branch/repeat sustained 0.031862625670064565/0.0 N; no_action excess 0.031862625670064565 N; probe branch/repeat sustained 0.12833775833269914/0.0 N; probe excess 0.12833775833269914 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.09647513266263458 N
- Segment 13: no_action branch/repeat sustained 0.17679803058732924/0.0 N; no_action excess 0.17679803058732924 N; probe branch/repeat sustained 0.2421452358272931/0.0 N; probe excess 0.2421452358272931 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.06534720523996387 N
- Segment 14: no_action branch/repeat sustained 0.3521055969148146/0.0 N; no_action excess 0.3521055969148146 N; probe branch/repeat sustained 0.3009539902806676/0.0 N; probe excess 0.3009539902806676 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.05115160663414697 N
- Segment 15: no_action branch/repeat sustained 0.15355827813607234/0.0 N; no_action excess 0.15355827813607234 N; probe branch/repeat sustained 0.09426269602720455/0.0 N; probe excess 0.09426269602720455 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.05929558210886779 N
- Segment 16: no_action branch/repeat sustained 0.1709887824558825/0.0 N; no_action excess 0.1709887824558825 N; probe branch/repeat sustained 0.05816159797613951/0.0 N; probe excess 0.05816159797613951 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.11282718447974299 N
- Segment 17: no_action branch/repeat sustained 0.04100634487510897/0.0 N; no_action excess 0.04100634487510897 N; probe branch/repeat sustained 0.0665396648449216/0.0 N; probe excess 0.0665396648449216 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.025533319969812632 N
- Segment 18: no_action branch/repeat sustained 0.03550675910805615/0.0 N; no_action excess 0.03550675910805615 N; probe branch/repeat sustained 0.05472307780221803/0.0 N; probe excess 0.05472307780221803 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.019216318694161877 N
- Segment 19: no_action branch/repeat sustained 0.043770846915164224/0.0 N; no_action excess 0.043770846915164224 N; probe branch/repeat sustained 0.07128596880672278/0.0 N; probe excess 0.07128596880672278 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.027515121891558553 N
- Segment 20: no_action branch/repeat sustained 0.04119599117567114/0.0 N; no_action excess 0.04119599117567114 N; probe branch/repeat sustained 0.07611463150958198/0.0 N; probe excess 0.07611463150958198 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03491864033391084 N
- Segment 21: no_action branch/repeat sustained 0.03012149775032703/0.0 N; no_action excess 0.03012149775032703 N; probe branch/repeat sustained 0.07205684905104671/0.0 N; probe excess 0.07205684905104671 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.04193535130071968 N
- Segment 22: no_action branch/repeat sustained 0.043073937202897206/0.0 N; no_action excess 0.043073937202897206 N; probe branch/repeat sustained 0.08609923075929367/0.0 N; probe excess 0.08609923075929367 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.04302529355639646 N
- Segment 23: no_action branch/repeat sustained 0.12381144087242116/0.0 N; no_action excess 0.12381144087242116 N; probe branch/repeat sustained 0.14628151233698833/0.0 N; probe excess 0.14628151233698833 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.02247007146456717 N
- Segment 24: no_action branch/repeat sustained 0.08692536366684958/0.0 N; no_action excess 0.08692536366684958 N; probe branch/repeat sustained 0.11183999573916092/0.0 N; probe excess 0.11183999573916092 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.024914632072311343 N
- Segment 25: no_action branch/repeat sustained 0.04692864482245199/0.0 N; no_action excess 0.04692864482245199 N; probe branch/repeat sustained 0.08740702552164034/0.0 N; probe excess 0.08740702552164034 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.040478380699188354 N
- Segment 26: no_action branch/repeat sustained 0.012161132769669555/0.0 N; no_action excess 0.012161132769669555 N; probe branch/repeat sustained 0.055354812408950155/0.0 N; probe excess 0.055354812408950155 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0431936796392806 N
- Segment 27: no_action branch/repeat sustained 0.011077755141593763/0.0 N; no_action excess 0.011077755141593763 N; probe branch/repeat sustained 0.03746534273504092/0.0 N; probe excess 0.03746534273504092 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.02638758759344716 N
- Segment 28: no_action branch/repeat sustained 0.018594867653556914/0.0 N; no_action excess 0.018594867653556914 N; probe branch/repeat sustained 0.03629310610092807/0.0 N; probe excess 0.03629310610092807 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.017698238447371156 N
- Segment 29: no_action branch/repeat sustained 0.026247582529669945/0.0 N; no_action excess 0.026247582529669945 N; probe branch/repeat sustained 0.03270232936839213/0.0 N; probe excess 0.03270232936839213 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.006454746838722186 N
- Segment 30: no_action branch/repeat sustained 0.011631645034722323/0.0 N; no_action excess 0.011631645034722323 N; probe branch/repeat sustained 0.022166104111013166/0.0 N; probe excess 0.022166104111013166 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.010534459076290843 N

Fixed-speed probe amplification comparison:
- Report only: True
- Stop after this trial: True
- Baseline: {'contact_reference_peak_excess_n': 0.3009539902806676, 'formal_sensor_peak_fused_gap': 0.3338189115908537, 'phase_name': 'phase0d-r7-spatial-repeat-corrected-load-path-audit', 'post_probe_rmse_m': 6.637808892143848e-05, 'probe_amplitude_m': 0.001, 'probe_forward_steps': 24, 'probe_return_steps': 24, 'probe_speed_m_s': 0.01, 'proximal_excess_n': 0.022166104111013166, 'proximal_retention_ratio': 0.07365280018497582, 'result_sha': 'aa5600408d835f344b580f155e61780c737930d1'}
- Current: {'contact_reference_peak_excess_n': 0.3009539902806676, 'formal_sensor_peak_fused_gap': 0.3243398252070224, 'post_probe_rmse_m': 7.134265109489577e-05, 'probe_amplitude_m': 0.002, 'probe_forward_steps': 48, 'probe_return_steps': 48, 'probe_speed_m_s': 0.01, 'proximal_excess_n': 0.022166104111013166, 'proximal_retention_ratio': 0.07365280018497582}
- Gain ratios: {'contact_reference_peak_excess': 1.0, 'formal_sensor_peak': 0.971604106134498, 'post_probe_rmse': 1.0747921829947993, 'probe_amplitude': 2.0, 'probe_speed': 1.0, 'proximal_excess': 1.0, 'proximal_retention': 1.0}

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Internal cable constraint force used for Gate 3: No
- Internal cable constraint force used only as oracle mechanism diagnostic: Yes
- Latch-contact bead mask used as formal sensor: No
- Hidden latch/contact used as formal sensor: No

Tests:
- Passed: 29
- Failed: 0
- pip check: No broken requirements found.

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: The one-shot 2 mm fixed-speed probe still leaves the gripper-proximal branch-specific excess below the mechanical reference scale and Gate 3 remains false. Stop scalar amplitude escalation; do not try 3/4/5 mm. Prepare a different minimal mechanical information-gathering probe/coupling repair while preserving Gate 2.
