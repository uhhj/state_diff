Verdict: PHASE0D_OBSERVABLE_EQUIVALENCE_FAIL

Repository:
- Main start: fba086cd2841386bdc04472f0413ddb9a8e623e6
- Main end: ccced9710bd790ab68e8c06812a852f230941fba
- Branch: Experiment3
- Clean before result: True
- Remote tip: ccced9710bd790ab68e8c06812a852f230941fba
- Submodule start: 282b93535b125d1a4487df85ad24aa41551957af
- Submodule end: 7704cc7c5a413971deacf73544d773a51891b5c9

Five gates:
- Initial observable equivalence: True
- Post-probe observable equivalence: False
- Sensor observability: False
- Same-action future divergence: False
- Control relevance: None

Key values:
- Initial 51D RMSE: 0.0 m
- Post-probe 51D RMSE: 0.007029667896873058 m
- Sensor onset: None / None
- Sensor trigger: None
- Peak fused sensor gap: 0.776681021320769
- Future visible RMSE peak: 0.007196196734190705 m
- Repeat floor: 0.00679308331500146 m
- FREE/JAM extraction progress: {'free': 0.009665048651547825, 'free_repeat': 0.011465949241629914, 'jam_right': 0.016774614636455853}
- Control verdict: not run

Tests:
- Passed: 26
- Failed: 0
- pip check: known multiprocess/dill conflict only

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Next task: Stop at this Phase 0D gate and follow its single allowed repair.
