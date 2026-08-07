Verdict: PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL

Repository:
- Main start: 2c1d6acc6606f13ef09d3fced663486c1d5d95cd
- Main end: 6c16ec350e906e656440b506d974acb39423db02
- Branch: Experiment3
- Clean before result: True
- Remote tip: 6c16ec350e906e656440b506d974acb39423db02
- Submodule start: 7704cc7c5a413971deacf73544d773a51891b5c9
- Submodule end: 7704cc7c5a413971deacf73544d773a51891b5c9

Frozen:
- Probe amplitude: 2.0 mm
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
- Post-probe 51D RMSE: 0.0064859658973046635 m
- Post-probe keypoint RMSE: 0.006683185053004547 m
- Post-probe EE RMSE: 0.0007158298077965741 m
- Immediate-return 51D RMSE: 0.0070871609350366994 m
- Post-probe recovery curve: [{'branch_excess_over_repeat_m': 0.00038465206293618066, 'free_jam': {'ee_rmse_m': 0.005725700121056984, 'keypoint_rmse_m': 0.007163666179707682, 'rmse_51d_m': 0.0070871609350366994}, 'free_repeat': {'ee_rmse_m': 0.005782410491871794, 'keypoint_rmse_m': 0.006755855594285671, 'rmse_51d_m': 0.006702508872100519}, 'post_probe_steps': 0, 'time_s': 0.0}, {'branch_excess_over_repeat_m': 0.00039088293311286253, 'free_jam': {'ee_rmse_m': 0.004363298466966666, 'keypoint_rmse_m': 0.007152846799532533, 'rmse_51d_m': 0.007019510000131812}, 'free_repeat': {'ee_rmse_m': 0.004407017226603366, 'keypoint_rmse_m': 0.006743219022914322, 'rmse_51d_m': 0.006628627067018949}, 'post_probe_steps': 24, 'time_s': 0.1}, {'branch_excess_over_repeat_m': 0.0003872153811959964, 'free_jam': {'ee_rmse_m': 0.0033492781960724025, 'keypoint_rmse_m': 0.007197474635900448, 'rmse_51d_m': 0.007029667896873058}, 'free_repeat': {'ee_rmse_m': 0.003382956770458266, 'keypoint_rmse_m': 0.006794448955216252, 'rmse_51d_m': 0.006642452515677062}, 'post_probe_steps': 48, 'time_s': 0.2}, {'branch_excess_over_repeat_m': 0.0005415095738995797, 'free_jam': {'ee_rmse_m': 0.0017530882033501092, 'keypoint_rmse_m': 0.007139782324651651, 'rmse_51d_m': 0.006939643932818826}, 'free_repeat': {'ee_rmse_m': 0.0017125244229143897, 'keypoint_rmse_m': 0.006581134760492642, 'rmse_51d_m': 0.0063981343589192465}, 'post_probe_steps': 120, 'time_s': 0.5}, {'branch_excess_over_repeat_m': 0.0005904636081768165, 'free_jam': {'ee_rmse_m': 0.0007158298077965741, 'keypoint_rmse_m': 0.006683185053004547, 'rmse_51d_m': 0.0064859658973046635}, 'free_repeat': {'ee_rmse_m': 0.0005991306572362969, 'keypoint_rmse_m': 0.006075098484564151, 'rmse_51d_m': 0.005895502289127847}, 'post_probe_steps': 240, 'time_s': 1.0}]
- Final FREE-repeat post-probe RMSE: 0.005895502289127847 m
- Final branch excess over repeat: 0.0005904636081768165 m
- Recovery fraction: 0.08482875487699404
- JAM post-probe latch contact samples: 240
- JAM post-probe latch peak force: 1.196222479612289 N
- Sensor onset: None / None
- Sensor trigger: None
- Peak fused sensor gap: 0.776681021320769
- Future visible RMSE peak: 0.006681974398106151 m
- Repeat floor: 0.006075247901766967 m
- Future/repeat ratio: 1.099868598969059
- FREE/JAM extraction progress: {'free': 0.00926062896785862, 'free_repeat': 0.009729416541548463, 'jam_right': 0.014944793719758265}
- Control verdict: not run

Tests:
- Passed: 11
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: Reduce zero-net probe amplitude from 2.0 mm to 1.0 mm; keep post-probe settle at 240 steps and jam clearance at 0.5 mm.
