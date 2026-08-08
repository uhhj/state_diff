Verdict: PHASE0F0_DHR_ACTION_SWITCH_FAIL

Scientific status:
- Formal Phase 0: NOT RUN
- Model training: NOT RUN
- Control-first benchmark admission: FAIL

Best action:
- FREE: right_release
- JAM-R: right_release
- Required FREE: straight
- Required JAM-R: left_release
- Best-action switch: False

Classifier invariants:
- Legacy control relevance: False
- Same candidate commands: True
- Branch-local IK: False

FREE:
- straight: {'success': False, 'final_progress_m': 0.03254703806914505, 'peak_force_n': 39.49469380534878, 'path_length_m': 0.06000000000000005, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}
- left_release: {'success': False, 'final_progress_m': 0.030474001735192713, 'peak_force_n': 39.28266255879315, 'path_length_m': 0.08099999999999995, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}
- right_release: {'success': False, 'final_progress_m': 0.030940235046256204, 'peak_force_n': 39.12418952647861, 'path_length_m': 0.08099999999999993, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}

JAM-R:
- straight: {'success': False, 'final_progress_m': 0.03332728815649544, 'peak_force_n': 39.35760903677638, 'path_length_m': 0.059994098084595146, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}
- left_release: {'success': False, 'final_progress_m': 0.0323509042187175, 'peak_force_n': 39.29825322150412, 'path_length_m': 0.0810059093134022, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}
- right_release: {'success': False, 'final_progress_m': 0.030297013336692658, 'peak_force_n': 39.2573181421928, 'path_length_m': 0.0810059093134022, 'grasp_retained': True, 'pocket_contact_fraction': 0.0, 'pocket_peak_contact_force_n': 0.0, 'pocket_contact_beads': []}

Key comparison:
- JAM left progress advantage: -0.000976383937777936 m
- Pre-action visible RMSE: 0.0001713931562056371 m

Next task:
Reject this DHR benchmark candidate. Do not tune pocket, route, actions, or thresholds; review the benchmark concept at task level.
