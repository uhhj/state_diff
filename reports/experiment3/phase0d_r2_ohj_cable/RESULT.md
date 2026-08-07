Verdict: PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL

Repository:
- Main start: 37cadd94c4ad257bb2dbe2eb3b366f128ba89413
- Main end: 281c18cfb6945f38a49c66f7480a7b58d03b84fa
- Branch: Experiment3
- Clean before result: True
- Remote tip: 281c18cfb6945f38a49c66f7480a7b58d03b84fa
- Submodule start: 7704cc7c5a413971deacf73544d773a51891b5c9
- Submodule end: 7704cc7c5a413971deacf73544d773a51891b5c9

Frozen:
- Probe amplitude: 1.0 mm
- Jam clearance: 0.5 mm
- Post-probe settle: 240 steps / 1.0 s
- State: 51D
- Sensor: 6D

Five gates:
- Initial observable equivalence: True
- Post-probe observable equivalence: False
- Sensor observability: False
- Same-action future divergence: False
- Control relevance: None

Key values:
- Initial 51D RMSE: 0.0 m
- Post-probe 51D RMSE: 0.006522088299712606 m
- Post-probe keypoint RMSE: 0.006720430214703735 m
- Post-probe EE RMSE: 0.0007161648455284883 m
- Immediate-return 51D RMSE: 0.007130506884792735 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 0.00037697852090995757, 'free_jam': {'ee_rmse_m': 0.005701795302051411, 'keypoint_rmse_m': 0.0072104079015718955, 'rmse_51d_m': 0.007130506884792735}, 'free_repeat': {'ee_rmse_m': 0.005758526149983215, 'keypoint_rmse_m': 0.006810891313413188, 'rmse_51d_m': 0.006753528363882777}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 0.0003800017016346441, 'free_jam': {'ee_rmse_m': 0.004349887889592295, 'keypoint_rmse_m': 0.007233603659172663, 'rmse_51d_m': 0.007096485770808801}, 'free_repeat': {'ee_rmse_m': 0.004393645172997926, 'keypoint_rmse_m': 0.006835502781291887, 'rmse_51d_m': 0.006716484069174157}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 0.00038847720848416735, 'free_jam': {'ee_rmse_m': 0.003380458875208739, 'keypoint_rmse_m': 0.007231631446031685, 'rmse_51d_m': 0.007063457727448683}, 'free_repeat': {'ee_rmse_m': 0.0034144446568193025, 'keypoint_rmse_m': 0.006827255906799995, 'rmse_51d_m': 0.006674980518964516}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 0.0005676432188586695, 'free_jam': {'ee_rmse_m': 0.0017799842584216928, 'keypoint_rmse_m': 0.0071731279440179685, 'rmse_51d_m': 0.006972334303939907}, 'free_repeat': {'ee_rmse_m': 0.0017231227333244312, 'keypoint_rmse_m': 0.006587734826299956, 'rmse_51d_m': 0.006404691085081238}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 0.0005874024400898525, 'free_jam': {'ee_rmse_m': 0.0007161648455284883, 'keypoint_rmse_m': 0.006720430214703735, 'rmse_51d_m': 0.006522088299712606}, 'free_repeat': {'ee_rmse_m': 0.0006197518878194196, 'keypoint_rmse_m': 0.006115371737926335, 'rmse_51d_m': 0.0059346858596227535}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 0.0059346858596227535 m
- Final branch excess over repeat: 0.0005874024400898525 m
- Final repeat/FREE-JAM fraction: 0.9099364477914633
- Final branch-excess/FREE-JAM fraction: 0.09006355220853668
- Recovery fraction: 0.08532613387944499
- JAM post-probe latch contact samples: 240
- JAM post-probe latch contact fraction: 1.0
- JAM post-probe latch peak force: 0.9360980028638053 N
- Sensor onset: None / None
- Sensor trigger: None
- Peak fused sensor gap: 0.7869381745321995
- Future visible RMSE peak: 0.0067192108113922925 m
- Repeat floor: 0.006114374373936565 m
- Future/repeat ratio: 1.0989204128608698
- FREE/JAM extraction progress: {'free': 0.009250455468371332, 'free_repeat': 0.00989820229103433, 'jam_right': 0.015086696494846508}
- Control verdict: not run

Tests:
- Passed: 12
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: Stop. Do not automatically reduce the probe below 1.0 mm. Compare FREE-repeat against FREE/JAM and the JAM post-probe contact fraction. If repeat variability dominates, prepare a same-condition repeatability repair; if JAM-specific excess and persistent latch contact dominate, prepare a separate 0.5 mm probe repair.
