Verdict: PHASE0E_SCIENTIFIC_AUDIT_COMPLETE

Route: OHJ_CONTROL_STRUCTURE_INSUFFICIENT

Source benchmark:
- Config: configs/experiment3/phase0/ohj_cable_phase0d_r13_lslot.json
- Source pair verdict: PHASE0D_FUTURE_BRANCH_NOT_ESTABLISHED
- Formal Phase-0 verdict changed: No
- Formal Gate 5 executed: No

Gate-4 definition audit:
- Future peak: 0.004053722490296285 m
- Repeat peak: 0.0006030962301782327 m
- Future/repeat: 6.7215185362679035
- Absolute floor: 0.005 m
- Repeat multiplier: 3.0
- Repeat-relative floor: 0.001809288690534698 m
- Formal threshold: 0.005 m
- Absolute pass: False
- Repeat-relative pass: True
- Formal pass: False
- Absolute margin: -0.0009462775097037147 m
- Repeat-relative margin: 0.0022444337997615874 m

Frozen control-relevance audit:
- Diagnostic only: True
- Existing classifier reused: True
- Control relevant under frozen rule: False
- Same candidate commands across branches: True
- Best candidate: {'free': 'straight', 'jam_right': 'straight'}
- JAM left progress advantage: -0.006015747440475727 m
- FREE matrix: {'left_release': {'success': False, 'final_progress_m': 0.02649305441743338, 'peak_force_n': 73.74483281338996, 'path_length_m': 0.08100000000000006, 'grasp_retained': True}, 'right_release': {'success': False, 'final_progress_m': 0.027220273638235815, 'peak_force_n': 66.39430900294147, 'path_length_m': 0.08100000000000004, 'grasp_retained': True}, 'straight': {'success': False, 'final_progress_m': 0.030228728993286214, 'peak_force_n': 38.91656134968954, 'path_length_m': 0.05999999999999994, 'grasp_retained': True}}
- JAM-R matrix: {'left_release': {'success': False, 'final_progress_m': 0.027264952787161068, 'peak_force_n': 75.10789424334611, 'path_length_m': 0.08100000000000006, 'grasp_retained': True}, 'right_release': {'success': False, 'final_progress_m': 0.035636008251868856, 'peak_force_n': 66.52107209393886, 'path_length_m': 0.08100000000000004, 'grasp_retained': True}, 'straight': {'success': False, 'final_progress_m': 0.033280700227636795, 'peak_force_n': 63.47884812689331, 'path_length_m': 0.05999999999999994, 'grasp_retained': True}}

Training:
- B0: No
- B1: No
- CFPM: No
- IDM: No

Interpretation:
- This audit does not lower or rewrite Gate 4.
- This audit does not count as formal Gate 5.
- If control relevance is present while only the absolute Gate-4 floor fails,
  the next step is an independent scientific justification/preregistration of
  Gate 4 before any training.
- If control relevance is absent, stop local OHJ tuning and review the
  benchmark structure instead of training around the problem.
