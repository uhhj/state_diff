Verdict: PHASE0D_SENSOR_NOT_OBSERVABLE

Repository:
- Main start: 20421f9c30b3291cf7a0b4c4c0b9c2cd3a68cf51
- Main end: 0ae6475bfff688be6b3b8d40579498c78dd3eab6
- Branch: Experiment3
- Clean before result: True
- Remote tip: 0ae6475bfff688be6b3b8d40579498c78dd3eab6
- Submodule start: 4e2316fd93e9403be19fd2a9c621f8d6d0acdb6f
- Submodule end: 0ab2ea0ff70e4dc7f7d7bef1c45600bb6b91ff54

Repair:
- Canonical hold reassert after restore: True

Frozen:
- Probe amplitude: 1.0 mm
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
- Tactile-only peak fused gap: 0.3338189115908537
- Aggregate 3D tactile peak: 0.30699405091745485
- Spatial tactile patch metrics: [{'contact_count_peak_gap': 0.0, 'patch': 0, 'peak_fused_gap': 0.03054599307305918, 'probe_contact_samples': {'free': 72, 'jam_right': 72}, 'raw_force_gap_peak_n': 0.006841495934303615}, {'contact_count_peak_gap': 0.0, 'patch': 1, 'peak_fused_gap': 0.0, 'probe_contact_samples': {'free': 0, 'jam_right': 0}, 'raw_force_gap_peak_n': 0.0}, {'contact_count_peak_gap': 0.0, 'patch': 2, 'peak_fused_gap': 0.0, 'probe_contact_samples': {'free': 0, 'jam_right': 0}, 'raw_force_gap_peak_n': 0.0}, {'contact_count_peak_gap': 0.0, 'patch': 3, 'peak_fused_gap': 0.3338189115908537, 'probe_contact_samples': {'free': 72, 'jam_right': 72}, 'raw_force_gap_peak_n': 0.02478854464202563}]
- Tactile raw force-gap peak: 0.02478854464202563 N
- Tactile probe contact samples: {'free': 72, 'jam_right': 72}
- Tactile contact-count peak gap: 0.0
- Combined formal peak fused gap: 0.3338189115908537
- Future visible RMSE peak: 0.0026621585774327717 m
- Repeat floor: 0.0003163165519357304 m
- Future/repeat ratio: 8.416121638723707
- FREE/JAM extraction progress: {'free': 0.009474010769790542, 'free_repeat': 0.00961386967304756, 'jam_right': 0.015138446928295746}
- Repeat first phase above equivalence threshold: None
- Repeat phase diagnostics: [{'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'no_action', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_forward', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_hold', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 0.0, 'peak_51d_rmse_m': 0.0, 'peak_ee_rmse_m': 0.0, 'peak_keypoint_rmse_m': 0.0, 'phase': 'probe_return', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.683096886387268e-05, 'peak_51d_rmse_m': 6.70602574775536e-05, 'peak_ee_rmse_m': 0.0001295240903839642, 'peak_keypoint_rmse_m': 6.10705730438436e-05, 'phase': 'post_probe', 'start_51d_rmse_m': 0.0}, {'end_51d_rmse_m': 6.866758660336229e-05, 'peak_51d_rmse_m': 0.00030790956243841337, 'peak_ee_rmse_m': 0.00014509853261499795, 'peak_keypoint_rmse_m': 0.0003163165519357304, 'phase': 'test_pull', 'start_51d_rmse_m': 6.638570377971164e-05}, {'end_51d_rmse_m': 7.844724115941172e-05, 'peak_51d_rmse_m': 0.0001044716867527471, 'peak_ee_rmse_m': 7.390582941985888e-05, 'peak_keypoint_rmse_m': 0.00010688959361410294, 'phase': 'post_test', 'start_51d_rmse_m': 6.769959301575476e-05}]
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
- Segment 0: no_action branch/repeat sustained 0.02314885596427919/0.0 N; no_action excess 0.02314885596427919 N; probe branch/repeat sustained 0.043732659613521525/0.0 N; probe excess 0.043732659613521525 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.020583803649242335 N
- Segment 1: no_action branch/repeat sustained 0.019348607690578477/0.0 N; no_action excess 0.019348607690578477 N; probe branch/repeat sustained 0.03301468866372558/0.0 N; probe excess 0.03301468866372558 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013666080973147105 N
- Segment 2: no_action branch/repeat sustained 0.019647284080077845/0.0 N; no_action excess 0.019647284080077845 N; probe branch/repeat sustained 0.03284643567508965/0.0 N; probe excess 0.03284643567508965 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013199151595011806 N
- Segment 3: no_action branch/repeat sustained 0.019845232107699985/0.0 N; no_action excess 0.019845232107699985 N; probe branch/repeat sustained 0.03583788171624701/0.0 N; probe excess 0.03583788171624701 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.015992649608547023 N
- Segment 4: no_action branch/repeat sustained 0.01930369678807796/0.0 N; no_action excess 0.01930369678807796 N; probe branch/repeat sustained 0.03293922427894854/0.0 N; probe excess 0.03293922427894854 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.013635527490870582 N
- Segment 5: no_action branch/repeat sustained 0.004991874019521593/0.0 N; no_action excess 0.004991874019521593 N; probe branch/repeat sustained 0.016880330380217686/0.0 N; probe excess 0.016880330380217686 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.011888456360696093 N
- Segment 6: no_action branch/repeat sustained 0.010058159405937626/0.0 N; no_action excess 0.010058159405937626 N; probe branch/repeat sustained 0.013138372139576755/0.0 N; probe excess 0.013138372139576755 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.003080212733639129 N
- Segment 7: no_action branch/repeat sustained 0.0056570064689243195/0.0 N; no_action excess 0.0056570064689243195 N; probe branch/repeat sustained 0.008124444185306568/0.0 N; probe excess 0.008124444185306568 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.002467437716382249 N
- Segment 8: no_action branch/repeat sustained 0.004050002145592819/0.0 N; no_action excess 0.004050002145592819 N; probe branch/repeat sustained 0.009657511313579068/0.0 N; probe excess 0.009657511313579068 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.005607509167986249 N
- Segment 9: no_action branch/repeat sustained 0.01156485185808385/0.0 N; no_action excess 0.01156485185808385 N; probe branch/repeat sustained 0.03190404425580129/0.0 N; probe excess 0.03190404425580129 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.02033919239771744 N
- Segment 10: no_action branch/repeat sustained 0.05714686148762226/0.0 N; no_action excess 0.05714686148762226 N; probe branch/repeat sustained 0.06533258453056996/0.0 N; probe excess 0.06533258453056996 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.0081857230429477 N
- Segment 11: no_action branch/repeat sustained 0.0542583065075688/0.0 N; no_action excess 0.0542583065075688 N; probe branch/repeat sustained 0.047099261331568804/0.0 N; probe excess 0.047099261331568804 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess -0.007159045175999995 N
- Segment 12: no_action branch/repeat sustained 0.031862625670064565/0.0 N; no_action excess 0.031862625670064565 N; probe branch/repeat sustained 0.12832282907776765/0.0 N; probe excess 0.12832282907776765 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.09646020340770309 N
- Segment 13: no_action branch/repeat sustained 0.17679803058732924/0.0 N; no_action excess 0.17679803058732924 N; probe branch/repeat sustained 0.21244503805094317/0.0 N; probe excess 0.21244503805094317 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03564700746361393 N
- Segment 14: no_action branch/repeat sustained 0.3521055969148146/0.0 N; no_action excess 0.3521055969148146 N; probe branch/repeat sustained 0.3009539902806676/0.0 N; probe excess 0.3009539902806676 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.05115160663414697 N
- Segment 15: no_action branch/repeat sustained 0.15355827813607234/0.0 N; no_action excess 0.15355827813607234 N; probe branch/repeat sustained 0.09426269602720455/0.0 N; probe excess 0.09426269602720455 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.05929558210886779 N
- Segment 16: no_action branch/repeat sustained 0.1709887824558825/0.0 N; no_action excess 0.1709887824558825 N; probe branch/repeat sustained 0.05816159797613951/0.0 N; probe excess 0.05816159797613951 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess -0.11282718447974299 N
- Segment 17: no_action branch/repeat sustained 0.04100634487510897/0.0 N; no_action excess 0.04100634487510897 N; probe branch/repeat sustained 0.0665396648449216/0.0 N; probe excess 0.0665396648449216 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.025533319969812632 N
- Segment 18: no_action branch/repeat sustained 0.03550675910805615/0.0 N; no_action excess 0.03550675910805615 N; probe branch/repeat sustained 0.05473495095012134/0.0 N; probe excess 0.05473495095012134 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.019228191842065187 N
- Segment 19: no_action branch/repeat sustained 0.043770846915164224/0.0 N; no_action excess 0.043770846915164224 N; probe branch/repeat sustained 0.07128956412529139/0.0 N; probe excess 0.07128956412529139 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.027518717210127164 N
- Segment 20: no_action branch/repeat sustained 0.04119599117567114/0.0 N; no_action excess 0.04119599117567114 N; probe branch/repeat sustained 0.07611463150958198/0.0 N; probe excess 0.07611463150958198 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03491864033391084 N
- Segment 21: no_action branch/repeat sustained 0.03012149775032703/0.0 N; no_action excess 0.03012149775032703 N; probe branch/repeat sustained 0.058863138611572385/0.0 N; probe excess 0.058863138611572385 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.028741640861245355 N
- Segment 22: no_action branch/repeat sustained 0.043073937202897206/0.0 N; no_action excess 0.043073937202897206 N; probe branch/repeat sustained 0.07073939337246456/0.0 N; probe excess 0.07073939337246456 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.02766545616956735 N
- Segment 23: no_action branch/repeat sustained 0.12381144087242116/0.0 N; no_action excess 0.12381144087242116 N; probe branch/repeat sustained 0.14628151233698833/0.0 N; probe excess 0.14628151233698833 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.02247007146456717 N
- Segment 24: no_action branch/repeat sustained 0.08692536366684958/0.0 N; no_action excess 0.08692536366684958 N; probe branch/repeat sustained 0.08942204064786317/0.0 N; probe excess 0.08942204064786317 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.002496676981013593 N
- Segment 25: no_action branch/repeat sustained 0.04692864482245199/0.0 N; no_action excess 0.04692864482245199 N; probe branch/repeat sustained 0.06422790261476728/0.0 N; probe excess 0.06422790261476728 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.017299257792315295 N
- Segment 26: no_action branch/repeat sustained 0.012161132769669555/0.0 N; no_action excess 0.012161132769669555 N; probe branch/repeat sustained 0.060843244383853995/0.0 N; probe excess 0.060843244383853995 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.04868211161418444 N
- Segment 27: no_action branch/repeat sustained 0.011077755141593763/0.0 N; no_action excess 0.011077755141593763 N; probe branch/repeat sustained 0.06881881062494294/0.0 N; probe excess 0.06881881062494294 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.05774105548334918 N
- Segment 28: no_action branch/repeat sustained 0.018594867653556914/0.0 N; no_action excess 0.018594867653556914 N; probe branch/repeat sustained 0.05152833018153289/0.0 N; probe excess 0.05152833018153289 N; branch/repeat ratio None; branch-specific True; mechanically large True; probe emergence excess 0.03293346252797598 N
- Segment 29: no_action branch/repeat sustained 0.026247582529669945/0.0 N; no_action excess 0.026247582529669945 N; probe branch/repeat sustained 0.03712997587472568/0.0 N; probe excess 0.03712997587472568 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.010882393345055738 N
- Segment 30: no_action branch/repeat sustained 0.011631645034722323/0.0 N; no_action excess 0.011631645034722323 N; probe branch/repeat sustained 0.022166104111013166/0.0 N; probe excess 0.022166104111013166 N; branch/repeat ratio None; branch-specific True; mechanically large False; probe emergence excess 0.010534459076290843 N

Leakage statement:
- Internal cable constraint force used as formal sensor: No
- Internal cable constraint force used for Gate 3: No
- Internal cable constraint force used only as oracle mechanism diagnostic: Yes
- Latch-contact bead mask used as formal sensor: No
- Hidden latch/contact used as formal sensor: No

Tests:
- Passed: 27
- Failed: 0
- pip check: No broken requirements found.

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: A repeat-corrected branch-specific internal signal reaches the gripper-proximal segment, but its mechanical excess is below the reference force scale. Preserve Gate 2 and prepare one minimal probe/mechanical-signal amplification repair.
