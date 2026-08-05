# Phase 0K — Fixed Same-End Tension Extension

- Topology: `delayed_z_latch_v1`
- Intervention: `same_end_tension_extension_v1`
- Fixed path: `0.080 m + 0.040 m`, same grasp, same direction
- Seeds: `[71001, 71002, 71003]`
- Official outcome gate: `mean_cable_progress_gap >= 0.01 m`
- Tension diagnostics are gates: `False`
- Tension motion valid: `True`
- All Cartesian stages successful: `True`
- Stage-1 progress gap median: `0.001557005`
- Stage-2 progress gap median: `-0.009299656`
- Final progress gap median: `-0.009326184`
- Failed checks: `['outcome_progress_gap']`
- Verdict: `HIDDEN_TENSION_EXTENSION_SMOKE_BLOCKED`
- Training performed: `False`
- Geometry search performed: `False`
- Action search performed: `False`
- Extension search performed: `False`
- Status: fixed 3-seed Stage M0 smoke, not Scientific PASS.
